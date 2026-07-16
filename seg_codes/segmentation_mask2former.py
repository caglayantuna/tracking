"""
Fine-tune Mask2Former for instance segmentation on a COCO-format dataset.

Requirements:
    pip install torch torchvision transformers pillow

Dataset structure:
    data/
        images/train/   *.jpg / *.png
        images/val/     *.jpg / *.png
        train.json
        val.json

Usage:
    # Frozen backbone (faster, less VRAM, good when data is similar to COCO)
    python segmentation_mask2former.py \
        --train_json   data/train.json \
        --train_imgs   data/images/train \
        --val_json     data/val.json \
        --val_imgs     data/images/val \
        --num_classes  1 \
        --freeze_backbone \
        --output_dir   runs/mask2former_frozen

    # Unfrozen backbone (slower, more VRAM, better when data looks different from COCO)
    python segmentation_mask2former.py \
        --train_json   data/train.json \
        --train_imgs   data/images/train \
        --val_json     data/val.json \
        --val_imgs     data/images/val \
        --num_classes  1 \
        --output_dir   runs/mask2former_full

Outputs (in --output_dir):
    best_model/            best checkpoint by val loss
    checkpoint-epochXXX/   per-epoch checkpoints
    losses.csv             epoch-level train/val loss and lr
"""

import csv
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)

PRETRAINED = "facebook/mask2former-swin-tiny-coco-instance"
ACCUM_STEPS = 4   # gradient accumulation — effective batch = batch_size × ACCUM_STEPS


# ─────────────────────────── COCO loader ─────────────────────────────────────

class CocoAnnotations:
    def __init__(self, json_path: str):
        with open(json_path) as f:
            data = json.load(f)
        self.imgs = {img["id"]: img for img in data["images"]}
        self.categories = {c["id"]: c for c in data.get("categories", [])}
        self._img_to_anns: dict[int, list] = {}
        for ann in data.get("annotations", []):
            self._img_to_anns.setdefault(ann["image_id"], []).append(ann)

    def get_anns(self, image_id: int) -> list:
        return self._img_to_anns.get(image_id, [])


# ─────────────────────────── Polygon → mask ──────────────────────────────────

def polygon_to_mask(segmentation: list, height: int, width: int) -> np.ndarray:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for poly in segmentation:
        xy = list(zip(poly[0::2], poly[1::2]))
        if len(xy) >= 3:
            draw.polygon(xy, outline=1, fill=1)
    return np.array(mask, dtype=np.uint8)


# ─────────────────────────── Dataset ─────────────────────────────────────────

class CocoInstanceDataset(Dataset):
    def __init__(self, json_path: str, images_dir: str,
                 processor: Mask2FormerImageProcessor):
        self.images_dir = Path(images_dir)
        self.processor  = processor
        self.coco       = CocoAnnotations(json_path)
        self.ids        = sorted(self.coco.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id   = self.ids[idx]
        img_info = self.coco.imgs[img_id]
        image    = Image.open(self.images_dir / img_info["file_name"]).convert("RGB")
        W, H     = image.size

        masks, class_ids = [], []
        for ann in self.coco.get_anns(img_id):
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            m = polygon_to_mask(ann["segmentation"], H, W)
            if m.sum() == 0:
                continue
            masks.append(m)
            class_ids.append(ann["category_id"] - 1)  # 0-indexed

        return _encode(image, masks, class_ids, self.processor)


# ─────────────────────────── Encoding ────────────────────────────────────────

def _encode(image: Image.Image, masks: list, class_ids: list,
            processor: Mask2FormerImageProcessor) -> dict:
    enc          = processor(images=image, return_tensors="pt")
    pixel_values = enc["pixel_values"].squeeze(0)  # [C, H, W]
    pixel_mask   = enc["pixel_mask"].squeeze(0)    # [H, W] bool
    _, Hp, Wp    = pixel_values.shape

    if masks:
        resized = [
            np.array(
                Image.fromarray(m.astype(np.uint8)).resize((Wp, Hp), resample=Image.NEAREST),
                dtype=np.uint8,
            ) for m in masks
        ]
        mask_labels  = torch.tensor(np.stack(resized), dtype=torch.float32)  # [N, H, W]
        class_labels = torch.tensor(class_ids, dtype=torch.int64)             # [N]
    else:
        mask_labels  = torch.zeros((0, Hp, Wp), dtype=torch.float32)
        class_labels = torch.zeros((0,), dtype=torch.int64)

    return {
        "pixel_values": pixel_values,
        "pixel_mask":   pixel_mask,
        "mask_labels":  mask_labels,
        "class_labels": class_labels,
    }


# ─────────────────────────── Collate ─────────────────────────────────────────

def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    pixel_masks  = [item["pixel_mask"]   for item in batch]

    max_h = max(x.shape[1] for x in pixel_values)
    max_w = max(x.shape[2] for x in pixel_values)

    padded_pixels, padded_masks = [], []
    for x, m in zip(pixel_values, pixel_masks):
        _, h, w = x.shape
        padded_pixels.append(F.pad(x, (0, max_w - w, 0, max_h - h)))
        pm = torch.zeros((max_h, max_w), dtype=torch.bool)
        pm[:h, :w] = m
        padded_masks.append(pm)

    return {
        "pixel_values": torch.stack(padded_pixels),
        "pixel_mask":   torch.stack(padded_masks),
        "mask_labels":  [item["mask_labels"]  for item in batch],
        "class_labels": [item["class_labels"] for item in batch],
    }


# ─────────────────────────── Model ───────────────────────────────────────────

def build_model(num_classes: int,
                freeze_backbone: bool) -> Mask2FormerForUniversalSegmentation:
    id2label = {i: str(i) for i in range(num_classes)}
    label2id = {str(i): i for i in range(num_classes)}

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        PRETRAINED,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "model.pixel_level_module.encoder" in name:
                param.requires_grad = False

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Backbone frozen : {freeze_backbone}", flush=True)
    print(f"Parameters      : {n_trainable:,} trainable / {n_total:,} total", flush=True)

    return model


# ─────────────────────────── Training ────────────────────────────────────────

def run_epoch(model, loader, device, optimizer=None, scaler=None) -> float:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for step, batch in enumerate(loader, 1):
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask   = batch["pixel_mask"].to(device)
            mask_labels  = [m.to(device) for m in batch["mask_labels"]]
            class_labels = [c.to(device) for c in batch["class_labels"]]

            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                outputs = model(
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                    mask_labels=mask_labels,
                    class_labels=class_labels,
                )

            loss = outputs.loss / ACCUM_STEPS

            if is_train:
                scaler.scale(loss).backward()
                if step % ACCUM_STEPS == 0 or step == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            loss_val    = loss.item() * ACCUM_STEPS
            total_loss += loss_val
            tag = "train" if is_train else "val"
            print(f"  [{tag}] step {step:>4}/{len(loader)}  loss: {loss_val:.4f}", flush=True)

    return total_loss / max(len(loader), 1)


# ─────────────────────────── Main ────────────────────────────────────────────

def main(args):
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device : {device}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Processor
    processor = Mask2FormerImageProcessor.from_pretrained(
        PRETRAINED,
        do_resize=True,
        size={"shortest_edge": 800, "longest_edge": 1333},
        ignore_index=255,
        reduce_labels=False,
    )

    # Datasets & loaders
    train_ds = CocoInstanceDataset(args.train_json, args.train_imgs, processor)
    val_ds   = CocoInstanceDataset(args.val_json,   args.val_imgs,   processor)
    print(f"Train : {len(train_ds)} images  |  Val : {len(val_ds)} images", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )

    # Model
    model = build_model(args.num_classes, freeze_backbone=args.freeze_backbone)
    model.to(device)

    # Optimizer — only pass parameters that need updating
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    # LR schedule: linear warmup for first 10% of epochs, then cosine decay
    warmup_epochs = max(1, args.epochs // 10)
    lr_scheduler  = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.05, end_factor=1.0, total_iters=warmup_epochs
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs - warmup_epochs)
            ),
        ],
        milestones=[warmup_epochs],
    )

    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # CSV logging
    csv_path = output_dir / "losses.csv"
    csv_file = open(csv_path, "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
    writer.writeheader()

    best_val_loss = float("inf")

    try:
        for epoch in range(1, args.epochs + 1):
            lr = optimizer.param_groups[0]["lr"]
            print(f"\n{'='*60}", flush=True)
            print(f"Epoch {epoch}/{args.epochs}   lr={lr:.2e}", flush=True)

            train_loss = run_epoch(model, train_loader, device, optimizer, scaler)
            val_loss   = run_epoch(model, val_loader,   device)
            lr_scheduler.step()

            print(f"\n  → Train loss: {train_loss:.4f}  |  Val loss: {val_loss:.4f}", flush=True)

            writer.writerow({
                "epoch":      epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_loss":   f"{val_loss:.6f}",
                "lr":         f"{lr:.2e}",
            })
            csv_file.flush()

            # Save checkpoint every epoch
            ckpt_dir = output_dir / f"checkpoint-epoch{epoch:03d}"
            model.save_pretrained(ckpt_dir)
            processor.save_pretrained(ckpt_dir)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_pretrained(output_dir / "best_model")
                processor.save_pretrained(output_dir / "best_model")
                print(f"  ✓ New best  val_loss={best_val_loss:.4f}", flush=True)

    finally:
        csv_file.close()

    print(f"\nTraining complete.", flush=True)
    print(f"  Best val loss : {best_val_loss:.4f}", flush=True)
    print(f"  Best model    : {output_dir / 'best_model'}", flush=True)
    print(f"  Losses CSV    : {csv_path}", flush=True)


# ─────────────────────────── Inference helper ────────────────────────────────

def load_trained_model(model_dir: str):
    """Reload a trained model for inference."""
    processor = Mask2FormerImageProcessor.from_pretrained(model_dir)
    model     = Mask2FormerForUniversalSegmentation.from_pretrained(model_dir)
    model.eval()
    return model, processor


# ─────────────────────────── CLI ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune Mask2Former for instance segmentation"
    )

    # Dataset
    parser.add_argument("--train_json",  required=True, help="COCO train annotation JSON")
    parser.add_argument("--val_json",    required=True, help="COCO val annotation JSON")
    parser.add_argument("--train_imgs",  required=True, help="Train images directory")
    parser.add_argument("--val_imgs",    required=True, help="Val images directory")
    parser.add_argument("--num_classes", type=int, required=True,
                        help="Number of classes (excluding background)")

    # Model
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze the Swin backbone and train only the head. "
                             "Faster and uses less VRAM. Good starting point.")

    # Training
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=2)
    parser.add_argument("--lr",          type=float, default=1e-5,
                        help="Peak LR after warmup (default 1e-5). "
                             "Use 1e-5 for frozen backbone, 5e-6 for unfrozen.")
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--output_dir",  default="runs/mask2former")

    main(parser.parse_args())
