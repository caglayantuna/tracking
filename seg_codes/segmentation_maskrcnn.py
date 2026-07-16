"""
Fine-tune a pretrained Mask R-CNN on a custom COCO-format dataset.
No pycocotools required — works on Apple Silicon (MPS) and Linux (CUDA/CPU).

Requirements:
    pip install torch torchvision pillow

Dataset folder structure:
    data/
        images/
            train/   *.jpg / *.png
            val/     *.jpg / *.png
        train_maskrcnn.json   (COCO-format annotation file)
        val_maskrcnn.json

Usage:
    python train_maskrcnn.py \
        --train_json   data/train_maskrcnn.json \
        --train_imgs   data/images/train \
        --val_json     data/val_maskrcnn.json \
        --val_imgs     data/images/val \
        --num_classes  2 \
        --epochs       20 \
        --batch_size   2 \
        --output_dir   runs/maskrcnn
"""

import csv
import json
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


# ─────────────────────────── COCO JSON loader ────────────────────────────────

class CocoAnnotations:
    """
    Lightweight COCO-format JSON reader.
    Replaces pycocotools.coco.COCO for the subset of features we need.
    """

    def __init__(self, json_path: str):
        with open(json_path) as f:
            data = json.load(f)

        self.imgs = {img["id"]: img for img in data["images"]}

        # Map image_id -> list of annotations
        self._img_to_anns: dict[int, list] = {}
        for ann in data.get("annotations", []):
            self._img_to_anns.setdefault(ann["image_id"], []).append(ann)

    def get_anns(self, image_id: int) -> list:
        return self._img_to_anns.get(image_id, [])


# ─────────────────────────── Polygon → mask ──────────────────────────────────

def polygons_to_mask(segmentation: list, height: int, width: int) -> np.ndarray:
    """
    Convert a COCO polygon segmentation to a binary H×W uint8 mask.
    Works with multiple polygons per annotation (merged via OR).

    Args:
        segmentation: list of polygon coordinate lists, e.g. [[x1,y1,x2,y2,...], ...]
        height: image height in pixels
        width:  image width in pixels

    Returns:
        Binary mask as numpy uint8 array of shape (H, W).
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for poly in segmentation:
        # poly is a flat list [x1, y1, x2, y2, ...] — convert to (x, y) tuples
        xy = list(zip(poly[0::2], poly[1::2]))
        if len(xy) >= 3:
            draw.polygon(xy, outline=1, fill=1)
    return np.array(mask, dtype=np.uint8)


# ─────────────────────────── Dataset ─────────────────────────────────────────

class CocoSegmentationDataset(torch.utils.data.Dataset):
    """
    Reads a COCO-format JSON and returns (image_tensor, target_dict) pairs
    compatible with torchvision Mask R-CNN.
    """

    def __init__(self, json_path: str, images_dir: str, transforms=None):
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        self.coco = CocoAnnotations(json_path)
        self.ids = sorted(self.coco.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = self.images_dir / img_info["file_name"]

        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        anns = self.coco.get_anns(img_id)
        boxes, labels, masks = [], [], []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

            m = polygons_to_mask(ann["segmentation"], H, W)
            masks.append(torch.as_tensor(m, dtype=torch.uint8))

        if not boxes:
            target = {
                "boxes":    torch.zeros((0, 4), dtype=torch.float32),
                "labels":   torch.zeros((0,),   dtype=torch.int64),
                "masks":    torch.zeros((0, H, W), dtype=torch.uint8),
                "image_id": torch.tensor([img_id]),
                "area":     torch.zeros((0,),   dtype=torch.float32),
                "iscrowd":  torch.zeros((0,),   dtype=torch.int64),
            }
        else:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            target = {
                "boxes":    boxes_t,
                "labels":   torch.as_tensor(labels, dtype=torch.int64),
                "masks":    torch.stack(masks),
                "image_id": torch.tensor([img_id]),
                "area":     (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0]),
                "iscrowd":  torch.zeros((len(boxes),), dtype=torch.int64),
            }

        image = TF.to_tensor(image)

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target


# ─────────────────────────── Augmentation ────────────────────────────────────
# Applied only to the training set. Each transform receives (image_tensor, target)
# and must keep boxes/masks consistent with the image. Val stays un-augmented.

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHFlip:
    """Horizontal flip; mirrors boxes (x) and masks."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = image.flip(-1)
            W = image.shape[-1]
            if target["boxes"].numel():
                b = target["boxes"].clone()
                b[:, [0, 2]] = W - b[:, [2, 0]]
                target["boxes"] = b
            if target["masks"].numel():
                target["masks"] = target["masks"].flip(-1)
        return image, target


class RandomVFlip:
    """Vertical flip; safe here because larvae have no canonical up/down orientation."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = image.flip(-2)
            H = image.shape[-2]
            if target["boxes"].numel():
                b = target["boxes"].clone()
                b[:, [1, 3]] = H - b[:, [3, 1]]
                target["boxes"] = b
            if target["masks"].numel():
                target["masks"] = target["masks"].flip(-2)
        return image, target


class PhotometricJitter:
    """Brightness/contrast jitter — models microscopy exposure/illumination variation.
    Geometry-free, so boxes/masks are untouched."""
    def __init__(self, brightness=0.2, contrast=0.2):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, image, target):
        if self.brightness:
            image = TF.adjust_brightness(image, 1.0 + random.uniform(-self.brightness, self.brightness))
        if self.contrast:
            image = TF.adjust_contrast(image, 1.0 + random.uniform(-self.contrast, self.contrast))
        return image.clamp(0.0, 1.0), target


def build_train_transforms():
    return Compose([RandomHFlip(0.5), RandomVFlip(0.5), PhotometricJitter(0.2, 0.2)])


# ─────────────────────────── Model ───────────────────────────────────────────

def build_model(num_classes: int):
    """
    Load COCO-pretrained Mask R-CNN and replace the classification/mask
    heads with new ones for `num_classes` (including background).
    """
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    # Box head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Mask head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    return model


# ─────────────────────────── Collate ─────────────────────────────────────────

def collate_fn(batch):
    return tuple(zip(*batch))


# ─────────────────────────── Training / eval loops ───────────────────────────

def train_one_epoch(model, optimizer, loader, device, epoch):
    model.train()
    total_loss = 0.0

    for step, (images, targets) in enumerate(loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += losses.item()

        if (step + 1) % 10 == 0:
            detail = "  ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
            print(f"  Epoch [{epoch}] Step [{step+1}/{len(loader)}]  loss: {losses.item():.4f}  {detail}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Compute average validation loss.
    Mask R-CNN only returns losses in training mode, so we temporarily
    keep the model in train mode and skip gradient computation.
    """
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        total_loss += sum(loss_dict.values()).item()
    model.eval()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_segmentation(model, loader, device, score_thr=0.5, iou_thr=0.5, mask_thr=0.5):
    """
    Real mask-quality metric (no pycocotools) for model selection / early stopping.

    Val *loss* is a weak proxy for mask quality, so we run the model in eval mode,
    greedily match predicted masks to GT by IoU (highest-score first, one GT each),
    and report:
        F1     — detection quality at IoU>=iou_thr (RQ)
        SQ     — mean IoU over matched (TP) pairs (localisation quality)
        PQ     — SQ * F1, the single number to select the best model on.
    For full cross-model comparability, run eval_utils.evaluate_predictions separately.
    """
    model.eval()
    tp = fp = fn = 0
    iou_sum = 0.0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            gt_masks = (target["masks"] > 0).to(device).float()          # [G, H, W]

            keep = output["scores"] >= score_thr
            pred_masks = (output["masks"][keep, 0] >= mask_thr).float()   # [P, H, W]
            order = output["scores"][keep].argsort(descending=True)
            pred_masks = pred_masks[order]

            matched = set()
            for pm in pred_masks:
                best_iou, best_gi = 0.0, -1
                for gi in range(gt_masks.shape[0]):
                    if gi in matched:
                        continue
                    inter = (pm * gt_masks[gi]).sum()
                    union = pm.sum() + gt_masks[gi].sum() - inter
                    iou = (inter / union).item() if union > 0 else 0.0
                    if iou > best_iou:
                        best_iou, best_gi = iou, gi
                if best_iou >= iou_thr:
                    tp += 1
                    matched.add(best_gi)
                    iou_sum += best_iou
                else:
                    fp += 1
            fn += gt_masks.shape[0] - len(matched)

    eps = 1e-16
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    sq = iou_sum / tp if tp else 0.0
    pq = sq * f1
    return {"PQ": pq, "F1": f1, "SQ": sq, "precision": precision, "recall": recall,
            "TP": tp, "FP": fp, "FN": fn}


# ─────────────────────────── Main ────────────────────────────────────────────

def main(args):
    # Device selection: MPS (Apple Silicon) → CUDA → CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Datasets & loaders (augment train only; val stays clean for honest metrics)
    train_tfms = None if args.no_aug else build_train_transforms()
    train_ds = CocoSegmentationDataset(args.train_json, args.train_imgs, transforms=train_tfms)
    val_ds   = CocoSegmentationDataset(args.val_json,   args.val_imgs)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )

    # Model
    model = build_model(num_classes=args.num_classes)
    model.to(device)

    # Optimiser (only fine-tune trainable parameters)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Select the best model on mask PQ (higher is better), not proxy val loss.
    best_pq = -1.0
    best_epoch = 0
    epochs_no_improve = 0

    csv_path = output_dir / "losses.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_PQ", "val_F1", "val_SQ", "lr"])

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}  (lr={optimizer.param_groups[0]['lr']:.6f})")

        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss   = evaluate(model, val_loader, device)
        metrics    = evaluate_segmentation(model, val_loader, device,
                                           score_thr=args.score_thr, iou_thr=args.iou_thr)
        lr_scheduler.step()

        print(f"  → Train loss: {train_loss:.4f}  |  Val loss: {val_loss:.4f}")
        print(f"  → Val  PQ: {metrics['PQ']:.4f}  F1: {metrics['F1']:.4f}  SQ: {metrics['SQ']:.4f}  "
              f"(TP={metrics['TP']} FP={metrics['FP']} FN={metrics['FN']})")

        current_lr = optimizer.param_groups[0]["lr"]
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                                    f"{metrics['PQ']:.6f}", f"{metrics['F1']:.6f}",
                                    f"{metrics['SQ']:.6f}", f"{current_lr:.8f}"])

        # Rolling "last" checkpoint (weights + optimizer, so training is resumable)
        # instead of one file per epoch — avoids ~35 GB of redundant checkpoints.
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_metrics": metrics,
            },
            output_dir / "maskrcnn_last.pth",
        )

        if metrics["PQ"] > best_pq:
            best_pq = metrics["PQ"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_dir / "maskrcnn_best.pth")
            print(f"  ✅ New best model saved (val PQ={best_pq:.4f})")
        else:
            epochs_no_improve += 1
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f"\n⏹  Early stopping: no PQ improvement for {args.patience} epochs "
                      f"(best PQ={best_pq:.4f} @ epoch {best_epoch}).")
                break

    print("\n🏁 Training complete.")
    print(f"   Best validation PQ   : {best_pq:.4f} (epoch {best_epoch})")
    print(f"   Checkpoints saved in : {output_dir}")


# ─────────────────────────── Inference helper ────────────────────────────────

def load_trained_model(weights_path: str, num_classes: int):
    """Reload a saved model for inference."""
    model = build_model(num_classes)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model


# ─────────────────────────── CLI ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Mask R-CNN (no pycocotools)")
    parser.add_argument("--train_json",  required=True)
    parser.add_argument("--train_imgs",  required=True)
    parser.add_argument("--val_json",    required=True)
    parser.add_argument("--val_imgs",    required=True)
    parser.add_argument("--num_classes", type=int, required=True,
                        help="Number of classes INCLUDING background (e.g. 2 for 1 class)")
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch_size",  type=int,   default=2)
    parser.add_argument("--lr",          type=float, default=0.001)
    parser.add_argument("--num_workers", type=int,   default=0)
    parser.add_argument("--output_dir",  default="runs/maskrcnn")
    parser.add_argument("--no_aug",      action="store_true",
                        help="Disable train-time augmentation (flips + photometric jitter)")
    parser.add_argument("--patience",    type=int,   default=8,
                        help="Early-stop after N epochs with no val-PQ improvement (0 disables)")
    parser.add_argument("--score_thr",   type=float, default=0.5,
                        help="Confidence threshold for the validation PQ/F1 metric")
    parser.add_argument("--iou_thr",     type=float, default=0.5,
                        help="Mask-IoU threshold counting a detection as TP for the metric")
    args = parser.parse_args()
    main(args)