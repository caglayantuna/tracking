import argparse
import datetime
import gc
import os
import sys
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty

import cv2
import numpy as np
import torch
from skimage import measure
from skimage.morphology import skeletonize
from ultralytics import YOLO

# ── Make the EfficientTAM package importable ──────────────────────────────────
# Preferred: `pip install -e .` the EfficientTAM repo into the env so the plain
# import works. Fallback: point EFFICIENTTAM_ROOT (or the default below) at the
# cloned repo directory that contains the `efficient_track_anything/` package.
try:
    from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor
except ModuleNotFoundError:
    _ETAM_ROOT = os.environ.get(
        "EFFICIENTTAM_ROOT", "/pasteur/helix/users/ctuna/EfficientTAM"
    )
    if _ETAM_ROOT not in sys.path:
        sys.path.insert(0, _ETAM_ROOT)
    from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor

from larva_head_tail import process_spine_file



# ========== CONFIG ==========
YOLO_WEIGHTS = os.path.expanduser(
    "/pasteur/helix/users/ctuna/my_models/yolo_weights.pt"
)
CHECKPOINT   = os.path.expanduser("/pasteur/helix/users/ctuna/pretrained_models/efficienttam_s.pt")
MODEL_CFG    = "configs/efficienttam/efficienttam_s.yaml"
YOLO_CONF    = 0.5          # first-frame seed threshold; real larvae score ~0.9,
                            # so this rejects spurious low-confidence detections
CHUNK_SIZE   = 800          # larger = fewer re-inits = faster
N_IO_WORKERS = 4            # async PNG/overlay writers
OVERLAY_JPEG_QUALITY = 90   # JPEG instead of PNG: ~3× faster writes

# Single colour used for all objects (BGR)
_OBJ_COLOR = (0, 0, 255)   # red


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
        print("[GPU] No GPU found — running on CPU")
    print(f"[Device] Using: {device}")
    return device


device = get_device()


def _obj_color(obj_id: int):
    return _OBJ_COLOR


# ========== ASYNC DISK WRITER ==========
# Offloads cv2.imwrite calls to a background thread pool so the main loop
# never blocks on disk I/O.

class AsyncWriter:
    """Thread-pool-backed image writer. Call .write() then .flush() at end."""

    def __init__(self, n_workers: int = N_IO_WORKERS):
        self._pool = ThreadPoolExecutor(max_workers=n_workers)
        self._futures = []

    def write(self, path: str, img: np.ndarray, params=None):
        future = self._pool.submit(cv2.imwrite, path, img, params or [])
        self._futures.append(future)
        # Prune completed futures to avoid unbounded list growth
        if len(self._futures) > 200:
            self._futures = [f for f in self._futures if not f.done()]

    def flush(self):
        """Wait for all pending writes to finish."""
        for f in self._futures:
            f.result()
        self._futures.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.flush()
        self._pool.shutdown(wait=True)


# ========== HELPERS ==========

def order_skeleton_points(points: np.ndarray) -> np.ndarray:
    """Connect skeleton pixels into a single ordered path via nearest-neighbor."""
    if len(points) == 0:
        return points
    ordered = [points[0]]
    remaining = list(points[1:])
    while remaining:
        last = ordered[-1]
        dists = np.linalg.norm(np.array(remaining) - last, axis=1)
        nearest_idx = int(np.argmin(dists))
        ordered.append(remaining.pop(nearest_idx))
    return np.array(ordered)


def resample_to_n_points(points: np.ndarray, n: int = 11) -> np.ndarray:
    """Resample an ordered polyline to exactly n evenly-spaced points."""
    if len(points) < 2:
        return points
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumlen = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_len = cumlen[-1]
    if total_len == 0:
        return np.tile(points[0], (n, 1))
    target = np.linspace(0, total_len, n)
    return np.column_stack([
        np.interp(target, cumlen, points[:, 0]),
        np.interp(target, cumlen, points[:, 1]),
    ])


def extract_frames(video_path: str, frames_dir: str) -> list[str]:
    """Extract all frames from video into frames_dir. Returns list of filenames."""
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame_names = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fname = f"{frame_idx:05d}.jpg"
        # JPEG is ~3× faster than PNG and plenty for EfficientTAM inputs
        cv2.imwrite(
            os.path.join(frames_dir, fname),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        frame_names.append(fname)
        frame_idx += 1
    cap.release()
    print(f"[Frames] Extracted {frame_idx} frames → {frames_dir}")
    return frame_names


def segment_frame_yolo(yolo_model, frame_path: str) -> np.ndarray:
    """
    Run YOLO segmentation on a single frame.
    Returns labeled mask (H×W uint8): pixel = object ID (1-indexed), 0 = background.
    """
    frame = cv2.imread(frame_path)
    results = yolo_model(frame, conf=YOLO_CONF, verbose=False)
    h, w = frame.shape[:2]
    labeled_mask = np.zeros((h, w), dtype=np.uint8)
    result = results[0]
    if result.masks is None:
        return labeled_mask
    for obj_id, mask in enumerate(result.masks.data.cpu().numpy(), start=1):
        if mask.shape != (h, w):
            mask = cv2.resize(
                mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST
            )
        labeled_mask[mask > 0.5] = obj_id
    return labeled_mask


# ========== MASK DATA (CPU-bound, runs in thread pool) ==========

def _write_mask_data(
    binary: np.ndarray,
    obj_id: int,
    time_value: float,
    date_str: str,
    outline_lines: dict,
    spine_lines: dict,
    lock: threading.Lock,
):
    """
    Extract outline contour + spine from a binary mask.
    Thread-safe via lock when appending to shared dicts.
    """
    oid = int(obj_id)
    padded_label = f"{oid:05d}"

    # Outline
    contours = measure.find_contours(binary.astype(float), 0.5)
    outline_line = None
    if contours:
        contour = max(contours, key=len)
        line = f"{date_str} {padded_label} {time_value:.3f}"
        for point in contour:
            x, y = point[1], point[0]
            line += f" {x:.4f} {y:.4f}"
        outline_line = line

    # Spine
    spine_line = None
    skeleton = skeletonize(binary > 0)
    if np.any(skeleton):
        skel_pts = np.column_stack(np.where(skeleton > 0))
        skel_pts = np.array([[x, y] for y, x in skel_pts])
        skel_pts = order_skeleton_points(skel_pts)
        skel_pts = resample_to_n_points(skel_pts, n=11)
        sl = f"{date_str} {padded_label} {time_value:.3f}"
        for sx, sy in skel_pts:
            sl += f" {sx:.4f} {sy:.4f}"
        spine_line = sl

    # Append under lock (multiple threads write to shared dicts)
    with lock:
        if outline_line:
            outline_lines[oid].append(outline_line)
        if spine_line:
            spine_lines[oid].append(spine_line)


# ========== OVERLAY ==========

def get_mask_center(mask: np.ndarray):
    """Return (cx, cy) integer centroid of a boolean/uint8 mask, or None if empty."""
    binary = mask.squeeze().astype(np.uint8)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return None
    return int(xs.mean()), int(ys.mean())


def build_overlay(frame_bgr: np.ndarray, obj_masks: dict, real_frame_idx: int) -> np.ndarray:
    """
    Render coloured mask fills + contours + ID labels onto frame_bgr.
    Returns the overlay image (does NOT write to disk).
    """
    overlay = frame_bgr.copy()
    color_layer = np.zeros_like(frame_bgr)

    for obj_id, mask in obj_masks.items():
        binary = mask.squeeze().astype(np.uint8)
        if not binary.any():
            continue
        color = _obj_color(obj_id)
        color_layer[binary > 0] = color
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 1)
        center = get_mask_center(mask)
        if center:
            cx, cy = center
            text = str(int(obj_id))
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(overlay, (cx - 2, cy - th - 2), (cx + tw + 2, cy + 2), (0, 0, 0), -1)
            cv2.putText(overlay, text, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.addWeighted(color_layer, 0.35, overlay, 0.65, 0, overlay)
    cv2.putText(overlay, f"Frame:{real_frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return overlay


# ========== CHUNK ==========

def run_chunk(
    predictor,
    chunk_frame_names: list[str],
    chunk_start: int,
    fps: float,
    date_str: str,
    seed_mask: np.ndarray,
    frames_dir: str,
    outline_lines: dict,
    spine_lines: dict,
    overlay_dir: str,
    chunk_dir: str,
    writer: AsyncWriter,
    mask_executor: ThreadPoolExecutor,
    dict_lock: threading.Lock,
) -> np.ndarray:
    """
    Run EfficientTAM propagation over one chunk of frames.
    - Mask data (skeletonize/contour) is offloaded to mask_executor (CPU threads).
    - Overlay frames are written asynchronously via writer.
    Returns final frame mask for seeding next chunk.
    """
    # Build symlinks for this chunk
    if os.path.exists(chunk_dir):
        shutil.rmtree(chunk_dir)
    os.makedirs(chunk_dir, exist_ok=True)
    for fname in chunk_frame_names:
        os.symlink(
            os.path.abspath(os.path.join(frames_dir, fname)),
            os.path.join(chunk_dir, fname),
        )

    # Init EfficientTAM state
    inference_state = predictor.init_state(
        video_path=chunk_dir,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )
    predictor.reset_state(inference_state)

    # Seed all objects from the mask
    for obj_id in np.unique(seed_mask):
        if obj_id == 0:
            continue
        mask = (seed_mask == obj_id)
        if not mask.any():
            continue
        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=int(obj_id),
            mask=mask,
        )

    last_frame_masks: dict = {}
    mask_futures = []

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state=inference_state
    ):
        real_frame_idx = chunk_start + out_frame_idx
        time_value = real_frame_idx / fps

        obj_masks = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

        # Submit CPU-heavy mask processing to thread pool (non-blocking)
        for obj_id, mask in obj_masks.items():
            binary = mask.squeeze().astype(np.uint8)
            if not binary.any():
                continue
            f = mask_executor.submit(
                _write_mask_data,
                binary, obj_id, time_value, date_str,
                outline_lines, spine_lines, dict_lock,
            )
            mask_futures.append(f)

        # Build overlay and schedule async write
        orig_frame = cv2.imread(os.path.join(frames_dir, chunk_frame_names[out_frame_idx]))
        overlay_img = build_overlay(orig_frame, obj_masks, real_frame_idx)
        out_path = os.path.join(overlay_dir, f"{real_frame_idx:05d}.jpg")
        writer.write(out_path, overlay_img, [cv2.IMWRITE_JPEG_QUALITY, OVERLAY_JPEG_QUALITY])

        last_frame_masks = obj_masks.copy()

    # Wait for all mask CPU jobs from this chunk before returning
    for f in mask_futures:
        f.result()

    # Build seed mask for next chunk
    if last_frame_masks:
        sample = next(iter(last_frame_masks.values()))
        h, w = sample.squeeze().shape
        final_mask = np.zeros((h, w), dtype=np.uint8)
        for obj_id, mask in last_frame_masks.items():
            if mask.any():
                final_mask[mask.squeeze()] = obj_id
    else:
        final_mask = seed_mask

    return final_mask


# ========== MAIN ==========

def main(video_path: str):
    date_str   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_stem = os.path.splitext(os.path.basename(video_path))[0]

    OUTPUT_DIR  = os.path.expanduser(f"./output_yolo_efficienttam/{video_stem}/")
    OVERLAY_DIR = os.path.join(OUTPUT_DIR, "overlay")
    FRAMES_DIR  = os.path.join(OUTPUT_DIR, "frames_temp")
    CHUNK_DIR   = os.path.join(FRAMES_DIR, "chunk_symlinks")

    t_start = datetime.datetime.now()
    print(f"[Start] {video_path}  at {t_start.isoformat()}")

    all_frame_names = extract_frames(video_path, FRAMES_DIR)
    total_frames    = len(all_frame_names)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    os.makedirs(OUTPUT_DIR,  exist_ok=True)
    os.makedirs(OVERLAY_DIR, exist_ok=True)

    first_frame_out = os.path.join(OUTPUT_DIR, f"{video_stem}_frame0.jpg")
    outline_file    = os.path.join(OUTPUT_DIR, f"{video_stem}.outline")
    spine_file      = os.path.join(OUTPUT_DIR, f"{video_stem}.spine")
    tracked_video   = os.path.join(OUTPUT_DIR, f"{video_stem}_tracked.mp4")

    cv2.imwrite(first_frame_out, cv2.imread(os.path.join(FRAMES_DIR, all_frame_names[0])))

    # YOLO: detect first frame
    yolo_model   = YOLO(YOLO_WEIGHTS)
    labeled_mask = segment_frame_yolo(yolo_model, os.path.join(FRAMES_DIR, all_frame_names[0]))
    obj_ids      = np.unique(labeled_mask)
    obj_ids      = obj_ids[obj_ids != 0]

    if len(obj_ids) == 0:
        raise RuntimeError("No objects detected by YOLO — cannot proceed with tracking.")

    print(f"[Init] Detected {len(obj_ids)} object(s) in first frame.")

    outline_lines: dict = {int(oid): [] for oid in obj_ids}
    spine_lines:   dict = {int(oid): [] for oid in obj_ids}
    current_mask        = labeled_mask.copy()

    predictor = build_efficienttam_video_predictor(MODEL_CFG, CHECKPOINT, device=device)

    dict_lock = threading.Lock()

    # Shared async resources (live for the full run)
    with AsyncWriter(n_workers=N_IO_WORKERS) as writer, \
         ThreadPoolExecutor(max_workers=os.cpu_count()) as mask_executor:

        for chunk_start in range(0, total_frames, CHUNK_SIZE):
            chunk_end         = min(chunk_start + CHUNK_SIZE, total_frames)
            chunk_frame_names = all_frame_names[chunk_start:chunk_end]
            t_chunk           = datetime.datetime.now()
            print(f"[Chunk] frames {chunk_start}–{chunk_end - 1}")

            current_mask = run_chunk(
                predictor         = predictor,
                chunk_frame_names = chunk_frame_names,
                chunk_start       = chunk_start,
                fps               = fps,
                date_str          = date_str,
                seed_mask         = current_mask,
                frames_dir        = FRAMES_DIR,
                outline_lines     = outline_lines,
                spine_lines       = spine_lines,
                overlay_dir       = OVERLAY_DIR,
                chunk_dir         = CHUNK_DIR,
                writer            = writer,
                mask_executor     = mask_executor,
                dict_lock         = dict_lock,
            )

            # Free GPU/MPS memory after each chunk
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()
            gc.collect()

            chunk_secs = (datetime.datetime.now() - t_chunk).total_seconds()
            print(f"[Chunk] done in {chunk_secs:.1f}s")

        # Ensure all overlay frames are flushed before building video
        print("[IO] Flushing overlay writes…")
        writer.flush()

    # Write output files
    with open(outline_file, "w") as f:
        for oid in sorted(outline_lines):
            for line in outline_lines[oid]:
                f.write(line + "\n")

    with open(spine_file, "w") as f:
        for oid in sorted(spine_lines):
            for line in spine_lines[oid]:
                f.write(line + "\n")

    print("[Spine]  Running head/tail correction…")
    process_spine_file(outline_file, spine_file)

    # Compile overlay frames into tracked video
    print("\n🎬 Creating tracked video…")
    first_overlay = os.path.join(OVERLAY_DIR, "00000.jpg")
    if os.path.exists(first_overlay):
        sample = cv2.imread(first_overlay)
        h, w = sample.shape[:2]
        writer_vid = cv2.VideoWriter(
            tracked_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        for frame_idx in range(total_frames):
            fpath = os.path.join(OVERLAY_DIR, f"{frame_idx:05d}.jpg")
            if os.path.exists(fpath):
                writer_vid.write(cv2.imread(fpath))
            else:
                writer_vid.write(np.zeros((h, w, 3), dtype=np.uint8))
        writer_vid.release()
        print(f"[Video] Saved → {tracked_video}")
    else:
        print("[Video] No overlay frames found — skipping video creation.")

    # Cleanup temp frames
    if os.path.exists(FRAMES_DIR):
        shutil.rmtree(FRAMES_DIR)

    total_secs = (datetime.datetime.now() - t_start).total_seconds()
    print(f"[Done]  total wall time: {total_secs:.1f}s  ({total_secs/60:.2f} min)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track larvae in video with YOLO + EfficientTAM and export outline/spine files."
    )
    parser.add_argument("video_path", help="Path to input video file")
    args = parser.parse_args()
    main(args.video_path)
