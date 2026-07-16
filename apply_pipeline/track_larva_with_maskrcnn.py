"""
Robust larva tracker using a torchvision Mask R-CNN detector (instead of
RF-DETR) with the same custom tracker designed for top-view larva videos
where:
  - Larva count is roughly constant (no exits unless near border)
  - Larva area is nearly constant per individual
  - Larvae occasionally overlap or touch, causing the model to segment them as one
  - Larvae may be missed for a few frames but should keep their ID on return
"""

import argparse
import datetime
import os
import subprocess
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from skimage import measure
from skimage.morphology import skeletonize
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from larva_head_tail import process_spine_file


def build_model(num_classes: int) -> torch.nn.Module:
    """
    Build a Mask R-CNN with the box/mask heads resized for `num_classes`.
    Inlined here (rather than imported from segmentation_maskrcnn.py) so this
    script has no dependency on where the training code lives on a given
    machine — it only needs torchvision.
    """
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    return model


# ========== CONFIG ==========
MASKRCNN_WEIGHTS = os.path.expanduser(
     "/pasteur/helix/users/ctuna/my_models/maskrcnn_best.pth"
)
NUM_CLASSES         = 2      # background + larva
DETECTION_THRESHOLD = 0.5    # score threshold to keep a detection
MASK_THRESHOLD      = 0.5    # threshold to binarize the soft Mask R-CNN mask

_OBJ_COLOR           = (0, 0, 255)   # red (BGR)
OVERLAY_JPEG_QUALITY = 85

# Tracker hyperparameters
MAX_MISSING_FRAMES   = 20    # keep a ghost track alive this many frames
AREA_HISTORY_LEN     = 30    # rolling window for per-track area median
AREA_RATIO_MIN       = 0.40  # match rejected if det_area / track_area < this
AREA_RATIO_MAX       = 2.50  # match rejected if det_area / track_area > this
MERGE_AREA_RATIO     = 1.50  # detection flagged as merge if area > this × track_area
MAX_DIST_PX          = 200   # centroid distance gate (pixels)
IOU_WEIGHT           = 0.55  # weight for IoU term in cost
DIST_WEIGHT          = 0.45  # weight for distance term in cost
BORDER_MARGIN        = 40    # pixels from border to consider a larva as exited
WARMUP_FRAMES        = 10    # frames before count is locked


# ========== DEVICE ==========

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        for i in range(torch.cuda.device_count()):
            print(f"[GPU {i}] {torch.cuda.get_device_name(i)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[GPU] Apple Metal (MPS) — Mac Silicon")
    else:
        device = torch.device("cpu")
        print("[CPU] No GPU — running on CPU")
    print(f"[Device] Using: {device}")
    return device


device = get_device()


# ========== MASK R-CNN LOADING / INFERENCE ==========

def load_maskrcnn(weights_path: str, num_classes: int) -> torch.nn.Module:
    model = build_model(num_classes)
    state_dict = torch.load(weights_path, map_location="cpu")
    # Support both a raw state_dict and a training checkpoint dict.
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_maskrcnn(model: torch.nn.Module, frame_rgb: np.ndarray, threshold: float) -> List[np.ndarray]:
    """
    Run Mask R-CNN on a single RGB frame and return a list of binary
    (float32, 0/1) H×W masks that passed the score threshold.
    """
    img_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.to(device)

    output = model([img_tensor])[0]
    scores = output["scores"]
    keep = scores >= threshold

    masks = output["masks"][keep]  # (N, 1, H, W) soft masks in [0, 1]

    det_masks: List[np.ndarray] = []
    for i in range(masks.shape[0]):
        soft = masks[i, 0].detach().cpu().numpy()
        det_masks.append((soft >= MASK_THRESHOLD).astype(np.float32))
    return det_masks


# ========== GEOMETRY HELPERS ==========

def mask_centroid(binary: np.ndarray) -> Optional[Tuple[int, int]]:
    M = cv2.moments(binary)
    if M["m00"] == 0:
        return None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def is_near_border(cx: int, cy: int, h: int, w: int, margin: int = BORDER_MARGIN) -> bool:
    return cx < margin or cy < margin or cx > w - margin or cy > h - margin


# ========== TRACK STATE ==========

@dataclass
class TrackState:
    track_id: int
    mask: np.ndarray                          # last confirmed binary mask
    centroid: Tuple[int, int]                 # (cx, cy) last confirmed
    area_history: deque = field(default_factory=lambda: deque(maxlen=AREA_HISTORY_LEN))
    frames_missing: int = 0
    active: bool = True

    @property
    def expected_area(self) -> float:
        """Median area from recent history, or last mask area as fallback."""
        if self.area_history:
            return float(np.median(self.area_history))
        return float(self.mask.sum())

    def update(self, mask: np.ndarray, centroid: Tuple[int, int]) -> None:
        self.mask = mask.astype(bool)
        self.centroid = centroid
        area = int(self.mask.sum())
        if area > 0:
            self.area_history.append(area)
        self.frames_missing = 0

    def tick_missing(self) -> None:
        self.frames_missing += 1


# ========== ROBUST LARVA TRACKER ==========

class LarvaTracker:
    """
    Custom tracker for top-view larva videos.

    Key behaviours:
      1. Hungarian matching on combined IoU + centroid-distance cost.
      2. Area gating: reject matches where the detected area is wildly different
         from the track's historical area (catches split/merge artefacts).
      3. Merge detection: if a single detection's area ≈ sum of two nearby
         track areas, treat it as a merge event — suppress the detection and
         keep both tracks as ghosts so their IDs are not lost.
      4. Ghost propagation: a track that is not matched keeps its last mask for
         up to MAX_MISSING_FRAMES frames.  During this window the track still
         appears in the output with the same ID.
      5. Exit gate: a track is retired only after MAX_MISSING_FRAMES of
         consecutive misses AND its centroid was near the image border.
    """

    def __init__(self, warmup_frames: int = 10) -> None:
        self.tracks: Dict[int, TrackState] = {}
        self._next_id: int = 1
        self._img_diag: float = 1.0
        self._warmup_frames: int = warmup_frames
        self._frame_count: int = 0
        self._expected_n: Optional[int] = None  # locked after warmup

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        det_masks: List[np.ndarray],
        frame_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Parameters
        ----------
        det_masks : list of binary (bool/uint8) H×W arrays from the detector
        frame_shape : (H, W)

        Returns
        -------
        labeled_mask : uint16 H×W array where pixel value = track_id (0 = background)
        """
        H, W = frame_shape
        self._img_diag = float(np.hypot(H, W))
        self._frame_count += 1

        active_ids    = [tid for tid, t in self.tracks.items() if t.active]
        active_tracks = [self.tracks[tid] for tid in active_ids]

        # Pre-process detections
        det_centroids: List[Optional[Tuple[int, int]]] = []
        det_areas:     List[float]                     = []
        det_masks_bool: List[np.ndarray]               = []

        for m in det_masks:
            mb = (m > 0.5).astype(bool)
            det_masks_bool.append(mb)
            det_areas.append(float(mb.sum()))
            det_centroids.append(mask_centroid(mb.astype(np.uint8)))

        n_tracks = len(active_tracks)
        n_dets   = len(det_masks_bool)

        matched_track_idx: Dict[int, int] = {}   # track_idx → det_idx
        matched_det_idx:   set            = set()
        merge_det_idx:     set            = set()

        if n_tracks > 0 and n_dets > 0:
            cost = self._build_cost_matrix(
                active_tracks, det_masks_bool, det_centroids, det_areas
            )
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= 0.95:
                    continue
                matched_track_idx[r] = c
                matched_det_idx.add(c)

        # ── Merge detection ───────────────────────────────────────────
        unmatched_det = [i for i in range(n_dets) if i not in matched_det_idx]
        for di in unmatched_det:
            if det_centroids[di] is None:
                continue
            if self._is_merge_detection(di, det_areas, det_centroids, active_tracks):
                merge_det_idx.add(di)

        # ── Apply updates to matched tracks ───────────────────────────
        for ti, track in enumerate(active_tracks):
            if ti in matched_track_idx:
                di = matched_track_idx[ti]
                cen = det_centroids[di]
                if cen is not None:
                    track.update(det_masks_bool[di], cen)
            else:
                track.tick_missing()
                if track.frames_missing > MAX_MISSING_FRAMES:
                    cx, cy = track.centroid
                    # Only retire if larva is near border (genuine exit)
                    if is_near_border(cx, cy, H, W):
                        track.active = False
                        if self._expected_n is not None:
                            self._expected_n -= 1
                    elif track.frames_missing > MAX_MISSING_FRAMES * 3:
                        # Very long disappearance in the middle — retire anyway
                        track.active = False

        # ── Lock expected count after warmup ──────────────────────────
        if self._frame_count == self._warmup_frames:
            self._expected_n = sum(1 for t in self.tracks.values() if t.active)
            print(f"[Tracker] Count locked at {self._expected_n} larvae after {self._warmup_frames} warmup frames")

        current_active = sum(1 for t in self.tracks.values() if t.active)
        count_locked   = self._expected_n is not None

        # ── Handle unmatched detections ───────────────────────────────
        new_det = [i for i in unmatched_det if i not in merge_det_idx]
        for di in new_det:
            if det_centroids[di] is None:
                continue
            # Always try to re-link to a ghost first
            reassigned = self._try_reassign_ghost(di, det_masks_bool, det_centroids, det_areas)
            if reassigned:
                continue
            # Only spawn a new track if we're still in warmup OR count allows it
            if count_locked and current_active >= self._expected_n:
                # Count is full — this detection is noise or a split of an existing larva; ignore
                continue
            self._spawn_track(det_masks_bool[di], det_centroids[di])
            current_active += 1

        # ── Build output labeled mask ─────────────────────────────────
        labeled = np.zeros((H, W), dtype=np.uint16)
        for track in self.tracks.values():
            if not track.active:
                continue
            if track.mask.shape == (H, W):
                labeled[track.mask] = track.track_id
            else:
                # Resize ghost mask to current frame shape if needed
                resized = cv2.resize(
                    track.mask.astype(np.uint8), (W, H),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                labeled[resized] = track.track_id

        return labeled

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_cost_matrix(
        self,
        active_tracks: List[TrackState],
        det_masks_bool: List[np.ndarray],
        det_centroids: List[Optional[Tuple[int, int]]],
        det_areas: List[float],
    ) -> np.ndarray:
        n_t = len(active_tracks)
        n_d = len(det_masks_bool)
        cost = np.ones((n_t, n_d), dtype=np.float32)

        for ti, track in enumerate(active_tracks):
            tcx, tcy = track.centroid
            exp_area  = track.expected_area

            for di in range(n_d):
                # Distance gate
                cen = det_centroids[di]
                if cen is None:
                    continue
                dx = tcx - cen[0]
                dy = tcy - cen[1]
                dist = float(np.hypot(dx, dy))
                if dist > MAX_DIST_PX:
                    continue   # leave as 1.0 (infeasible)

                # Area gate — skip if area ratio is wildly off
                if exp_area > 0:
                    ratio = det_areas[di] / exp_area
                    if ratio < AREA_RATIO_MIN or ratio > AREA_RATIO_MAX:
                        continue

                iou  = mask_iou(track.mask, det_masks_bool[di])
                dist_cost = min(dist / MAX_DIST_PX, 1.0)
                cost[ti, di] = DIST_WEIGHT * dist_cost + IOU_WEIGHT * (1.0 - iou)

        return cost

    def _is_merge_detection(
        self,
        di: int,
        det_areas: List[float],
        det_centroids: List[Optional[Tuple[int, int]]],
        active_tracks: List[TrackState],
    ) -> bool:
        """
        Return True if detection di looks like two merged larvae:
          - Its area > MERGE_AREA_RATIO × the median expected area of nearby tracks
          - At least two active tracks have centroids within MAX_DIST_PX of the det
        """
        if not active_tracks:
            return False

        cen = det_centroids[di]
        if cen is None:
            return False

        nearby_tracks = []
        for track in active_tracks:
            tcx, tcy = track.centroid
            dist = float(np.hypot(tcx - cen[0], tcy - cen[1]))
            if dist < MAX_DIST_PX:
                nearby_tracks.append(track)

        if len(nearby_tracks) < 2:
            return False

        # Expected total area if two larvae merged
        areas = [t.expected_area for t in nearby_tracks if t.expected_area > 0]
        if not areas:
            return False
        median_single = float(np.median(areas))

        return det_areas[di] > MERGE_AREA_RATIO * median_single

    def _try_reassign_ghost(
        self,
        di: int,
        det_masks_bool: List[np.ndarray],
        det_centroids: List[Optional[Tuple[int, int]]],
        det_areas: List[float],
    ) -> bool:
        """
        Try to re-link an unmatched detection to a recently-lost ghost track.
        Returns True if reassignment happened (no new track needed).
        """
        cen = det_centroids[di]
        if cen is None:
            return False

        best_cost = 0.5   # threshold — only reassign if cost is low enough
        best_track = None

        for track in self.tracks.values():
            if not track.active or track.frames_missing == 0:
                continue   # only consider ghost tracks (frames_missing > 0)

            tcx, tcy = track.centroid
            dist = float(np.hypot(tcx - cen[0], tcy - cen[1]))
            if dist > MAX_DIST_PX:
                continue

            exp_area = track.expected_area
            if exp_area > 0:
                ratio = det_areas[di] / exp_area
                if ratio < AREA_RATIO_MIN or ratio > AREA_RATIO_MAX:
                    continue

            iou = mask_iou(track.mask, det_masks_bool[di])
            dist_cost = min(dist / MAX_DIST_PX, 1.0)
            c = DIST_WEIGHT * dist_cost + IOU_WEIGHT * (1.0 - iou)
            if c < best_cost:
                best_cost = c
                best_track = track

        if best_track is not None:
            new_cen = det_centroids[di]
            if new_cen is not None:
                best_track.update(det_masks_bool[di], new_cen)
            return True
        return False

    def _spawn_track(self, mask: np.ndarray, centroid: Tuple[int, int]) -> None:
        tid   = self._next_id
        self._next_id += 1
        state = TrackState(track_id=tid, mask=mask.astype(bool), centroid=centroid)
        state.area_history.append(int(mask.sum()))
        self.tracks[tid] = state


# ========== DRAW / WRITE HELPERS ==========

def order_skeleton_points(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    ordered   = [points[0]]
    remaining = points[1:].tolist()
    while remaining:
        last    = ordered[-1]
        arr     = np.asarray(remaining)
        nearest = int(np.argmin(np.sum((arr - last) ** 2, axis=1)))
        ordered.append(remaining.pop(nearest))
    return np.asarray(ordered)


def resample_to_n_points(points: np.ndarray, n: int = 11) -> np.ndarray:
    if len(points) < 2:
        return points
    seg_len = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumlen  = np.concatenate([[0.0], np.cumsum(seg_len)])
    total   = cumlen[-1]
    if total == 0:
        return np.tile(points[0], (n, 1))
    t = np.linspace(0, total, n)
    return np.column_stack([
        np.interp(t, cumlen, points[:, 0]),
        np.interp(t, cumlen, points[:, 1]),
    ])


def draw_overlay(frame_bgr: np.ndarray,
                 labeled_mask: np.ndarray,
                 ghost_ids: set,
                 real_frame_idx: int) -> np.ndarray:
    overlay     = frame_bgr.copy()
    color_layer = np.zeros_like(frame_bgr)

    obj_ids = np.unique(labeled_mask)
    obj_ids = obj_ids[obj_ids != 0]

    for obj_id in obj_ids:
        binary = (labeled_mask == obj_id).astype(np.uint8)
        is_ghost = int(obj_id) in ghost_ids
        color = (100, 100, 255) if is_ghost else _OBJ_COLOR   # lighter red for ghosts

        color_layer[binary > 0] = color
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        thickness = 1 if not is_ghost else 1
        line_type  = cv2.LINE_AA
        cv2.drawContours(overlay, contours, -1, color, thickness, line_type)

        M = cv2.moments(binary)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            text = str(int(obj_id)) + ("?" if is_ghost else "")
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(overlay, (cx - 2, cy - th - 2), (cx + tw + 2, cy + 2), (0, 0, 0), -1)
            cv2.putText(overlay, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.addWeighted(color_layer, 0.35, overlay, 0.65, 0, overlay)
    cv2.putText(
        overlay,
        f"Frame:{real_frame_idx}  Objects:{len(obj_ids)}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
    )
    return overlay


def _write_mask_data(binary, obj_id, time_value, date_str, outline_lines, spine_lines):
    oid    = int(obj_id)
    padded = f"{oid:05d}"
    outline_lines.setdefault(oid, [])
    spine_lines.setdefault(oid, [])

    contours = measure.find_contours(binary.astype(float), 0.5)
    if contours:
        contour = max(contours, key=len)
        parts   = [f"{date_str} {padded} {time_value:.3f}"]
        for pt in contour:
            parts.append(f"{pt[1]:.4f} {pt[0]:.4f}")
        outline_lines[oid].append(" ".join(parts))

    skeleton = skeletonize(binary > 0)
    if np.any(skeleton):
        ys, xs   = np.where(skeleton)
        skel_pts = order_skeleton_points(np.column_stack([xs, ys]))
        skel_pts = resample_to_n_points(skel_pts, n=11)
        parts    = [f"{date_str} {padded} {time_value:.3f}"]
        for sx, sy in skel_pts:
            parts.append(f"{sx:.4f} {sy:.4f}")
        spine_lines[oid].append(" ".join(parts))


# ========== VIDEO WRITERS ==========

def _open_ffmpeg_writer(path: str, fps: float, w: int, h: int):
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", "h264_videotoolbox",
        "-b:v", "8M", "-pix_fmt", "yuv420p",
        path,
    ]
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print("[Video] ffmpeg + h264_videotoolbox (hardware)")
        return proc
    except FileNotFoundError:
        print("[Video] ffmpeg not found — falling back to OpenCV")
        return None


def _open_cv_writer(path: str, fps: float, w: int, h: int):
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        if writer.isOpened():
            print(f"[Video] OpenCV VideoWriter ({fourcc})")
            return writer
    return None


# ========== MAIN ==========

def main(video_path: str):
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    t_start  = datetime.datetime.now()

    video_stem  = os.path.splitext(os.path.basename(video_path))[0]
    OUTPUT_DIR  = os.path.expanduser(f"./output_maskrcnn_track_robust/{video_stem}/")
    OVERLAY_DIR = os.path.join(OUTPUT_DIR, "overlay")
    os.makedirs(OUTPUT_DIR,  exist_ok=True)
    os.makedirs(OVERLAY_DIR, exist_ok=True)

    cap          = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"[Start]  {video_path}")
    print(f"[Init]   FPS={fps:.1f}  Frames={total_frames}")

    first_frame_out = os.path.join(OUTPUT_DIR, f"{video_stem}_frame0.jpg")
    outline_file    = os.path.join(OUTPUT_DIR, f"{video_stem}.outline")
    spine_file      = os.path.join(OUTPUT_DIR, f"{video_stem}.spine")
    tracked_video   = os.path.join(OUTPUT_DIR, f"{video_stem}_tracked_maskrcnn.mp4")

    model   = load_maskrcnn(MASKRCNN_WEIGHTS, NUM_CLASSES)
    tracker = LarvaTracker(warmup_frames=WARMUP_FRAMES)

    save_pool = ThreadPoolExecutor(max_workers=4)

    def _save_jpeg(img: np.ndarray, path: str) -> None:
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, OVERLAY_JPEG_QUALITY])

    outline_lines: dict = {}
    spine_lines:   dict = {}

    ffmpeg_proc      = None
    cv_writer        = None
    writer_ready     = False
    first_frame_done = False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_idx = 0
    while True:
        success, frame_bgr = cap.read()
        if not success:
            break

        if frame_idx % 100 == 0:
            elapsed    = (datetime.datetime.now() - t_start).total_seconds()
            throughput = frame_idx / elapsed if elapsed > 0 else 0.0
            n_active   = sum(1 for t in tracker.tracks.values() if t.active)
            print(f"[{frame_idx:>6}/{total_frames}] {elapsed:6.1f}s  {throughput:.1f} fps  tracks={n_active}")

        h, w       = frame_bgr.shape[:2]
        time_value = frame_idx / fps

        if not first_frame_done:
            cv2.imwrite(first_frame_out, frame_bgr)
            first_frame_done = True

        # ── Detect ────────────────────────────────────────────────────
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        det_masks = run_maskrcnn(model, frame_rgb, DETECTION_THRESHOLD)

        # Resize masks to frame size if the model output doesn't match (safety net)
        det_masks = [
            m if m.shape == (h, w)
            else cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            for m in det_masks
        ]

        # ── Custom tracking ───────────────────────────────────────────
        labeled_mask = tracker.update(det_masks, (h, w))

        # Ghost IDs: active tracks that are currently missing (frames_missing > 0)
        ghost_ids = {
            tid for tid, t in tracker.tracks.items()
            if t.active and t.frames_missing > 0
        }

        # ── Outline + spine data ──────────────────────────────────────
        # Write only for confidently detected (non-ghost) IDs to keep data clean
        for tid in np.unique(labeled_mask):
            if tid == 0:
                continue
            if int(tid) in ghost_ids:
                continue   # skip ghost frames — no new contour data
            binary = (labeled_mask == tid).astype(np.uint8)
            if binary.any():
                _write_mask_data(binary, tid, time_value, date_str, outline_lines, spine_lines)

        # ── Overlay ───────────────────────────────────────────────────
        overlay = draw_overlay(frame_bgr, labeled_mask, ghost_ids, frame_idx)

        save_pool.submit(
            _save_jpeg,
            overlay.copy(),
            os.path.join(OVERLAY_DIR, f"{frame_idx:05d}.jpg"),
        )

        # ── Init video writer once ────────────────────────────────────
        if not writer_ready:
            ffmpeg_proc = _open_ffmpeg_writer(tracked_video, fps, w, h)
            if ffmpeg_proc is None:
                cv_writer = _open_cv_writer(tracked_video, fps, w, h)
            writer_ready = True

        # ── Write frame ───────────────────────────────────────────────
        if ffmpeg_proc is not None:
            try:
                ffmpeg_proc.stdin.write(overlay.tobytes())
            except BrokenPipeError:
                print("[Warning] ffmpeg pipe broken — switching to OpenCV")
                ffmpeg_proc = None
                cv_writer   = _open_cv_writer(tracked_video, fps, w, h)
                if cv_writer:
                    cv_writer.write(overlay)
        elif cv_writer is not None:
            cv_writer.write(overlay)

        frame_idx += 1

    cap.release()

    # ── Teardown ──────────────────────────────────────────────────────
    save_pool.shutdown(wait=True)

    if ffmpeg_proc is not None:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
    if cv_writer is not None:
        cv_writer.release()

    print(f"[Video]  → {tracked_video}")

    with open(outline_file, "w") as f:
        for oid in sorted(outline_lines):
            for line in outline_lines[oid]:
                f.write(line + "\n")

    with open(spine_file, "w") as f:
        for oid in sorted(spine_lines):
            for line in spine_lines[oid]:
                f.write(line + "\n")

    print(f"[Output] outline → {outline_file}")
    print(f"[Output] spine   → {spine_file}")

    print("[Spine]  Running head/tail correction…")
    process_spine_file(outline_file, spine_file)

    total = (datetime.datetime.now() - t_start).total_seconds()
    print(f"[Done]   {total_frames} frames in {total:.1f}s ({total/60:.2f} min)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track larvae with Mask R-CNN segmentation + robust custom tracker."
    )
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument(
        "--weights",
        default=MASKRCNN_WEIGHTS,
        help="Path to Mask R-CNN checkpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=NUM_CLASSES,
        help="Number of classes including background (default: %(default)s)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DETECTION_THRESHOLD,
        help="Detection confidence threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--max-missing",
        type=int,
        default=MAX_MISSING_FRAMES,
        help="Frames a track can be missing before retirement (default: %(default)s)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=WARMUP_FRAMES,
        help="Frames before larva count is locked (default: %(default)s)",
    )
    args = parser.parse_args()

    MASKRCNN_WEIGHTS    = args.weights
    NUM_CLASSES         = args.num_classes
    DETECTION_THRESHOLD = args.threshold
    MAX_MISSING_FRAMES  = args.max_missing
    WARMUP_FRAMES       = args.warmup

    main(args.video_path)
