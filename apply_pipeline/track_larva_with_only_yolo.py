import argparse
import datetime
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from skimage import measure
from skimage.morphology import skeletonize
from ultralytics import YOLO
import torch
from larva_head_tail import process_spine_file


# ========== CONFIG ==========
YOLO_WEIGHTS = os.path.expanduser(
    "/pasteur/helix/users/ctuna/my_models/yolo_weights.pt"
)

_OBJ_COLOR          = (0, 0, 255)   # red (BGR)
OVERLAY_JPEG_QUALITY = 85            # lower → smaller file, faster write


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


# ========== HELPERS ==========

def _obj_color(_obj_id: int):
    return _OBJ_COLOR


def order_skeleton_points(points: np.ndarray) -> np.ndarray:
    """Connect skeleton pixels into an ordered path (nearest-neighbour)."""
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
    """Resample an ordered polyline to exactly n evenly-spaced points."""
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


def get_mask_center(binary: np.ndarray):
    """Fast centroid via cv2.moments instead of np.where + mean."""
    M = cv2.moments(binary)
    if M["m00"] == 0:
        return None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


def draw_overlay(frame_bgr: np.ndarray,
                 labeled_mask: np.ndarray,
                 real_frame_idx: int) -> np.ndarray:
    """
    Render coloured fills + contours + ID labels.
    Returns the overlay frame directly — NO disk write here.
    (Saving is handled asynchronously in the main loop.)
    """
    overlay     = frame_bgr.copy()
    color_layer = np.zeros_like(frame_bgr)

    obj_ids = np.unique(labeled_mask)
    obj_ids = obj_ids[obj_ids != 0]

    for obj_id in obj_ids:
        binary = (labeled_mask == obj_id).astype(np.uint8)
        color  = _obj_color(obj_id)
        color_layer[binary > 0] = color

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, 1)

        center = get_mask_center(binary)
        if center:
            cx, cy = center
            text   = str(int(obj_id))
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
            )
            cv2.rectangle(
                overlay,
                (cx - 2, cy - th - 2), (cx + tw + 2, cy + 2),
                (0, 0, 0), -1,
            )
            cv2.putText(
                overlay, text, (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
            )

    cv2.addWeighted(color_layer, 0.35, overlay, 0.65, 0, overlay)
    cv2.putText(
        overlay,
        f"Frame:{real_frame_idx}  Objects:{len(obj_ids)}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
    )
    return overlay


def _write_mask_data(
    binary, obj_id, time_value, date_str, outline_lines, spine_lines
):
    """Extract outline contour + spine from a binary mask."""
    oid    = int(obj_id)
    padded = f"{oid:05d}"
    outline_lines.setdefault(oid, [])
    spine_lines.setdefault(oid, [])

    # Outline — use list + join (faster than string concatenation)
    contours = measure.find_contours(binary.astype(float), 0.5)
    if contours:
        contour = max(contours, key=len)
        parts   = [f"{date_str} {padded} {time_value:.3f}"]
        for pt in contour:
            parts.append(f"{pt[1]:.4f} {pt[0]:.4f}")
        outline_lines[oid].append(" ".join(parts))

    # Spine
    skeleton = skeletonize(binary > 0)
    if np.any(skeleton):
        ys, xs  = np.where(skeleton)
        skel_pts = order_skeleton_points(np.column_stack([xs, ys]))
        skel_pts = resample_to_n_points(skel_pts, n=11)
        parts    = [f"{date_str} {padded} {time_value:.3f}"]
        for sx, sy in skel_pts:
            parts.append(f"{sx:.4f} {sy:.4f}")
        spine_lines[oid].append(" ".join(parts))


# ========== VIDEO WRITERS ==========

def _open_ffmpeg_writer(path: str, fps: float, w: int, h: int):
    """
    Hardware H.264 via Apple VideoToolbox.
    Install ffmpeg:  brew install ffmpeg
    """
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", "h264_videotoolbox",   # Apple Silicon hardware encoder
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
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h)
        )
        if writer.isOpened():
            print(f"[Video] OpenCV VideoWriter ({fourcc})")
            return writer
    return None


# ========== MAIN ==========

def main(video_path: str, tracker: str = "bytetrack"):
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    t_start  = datetime.datetime.now()

    video_stem  = os.path.splitext(os.path.basename(video_path))[0]
    OUTPUT_DIR  = os.path.expanduser(f"./output_yolo_track/{video_stem}/")
    OVERLAY_DIR = os.path.join(OUTPUT_DIR, "overlay")
    os.makedirs(OUTPUT_DIR,  exist_ok=True)
    os.makedirs(OVERLAY_DIR, exist_ok=True)

    cap         = cv2.VideoCapture(video_path)
    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"[Start]  {video_path}")
    print(f"[Init]   FPS={fps:.1f}  Frames={total_frames}  Tracker={tracker}")

    first_frame_out = os.path.join(OUTPUT_DIR, f"{video_stem}_frame0.jpg")
    outline_file    = os.path.join(OUTPUT_DIR, f"{video_stem}.outline")
    spine_file      = os.path.join(OUTPUT_DIR, f"{video_stem}.spine")
    tracked_video   = os.path.join(OUTPUT_DIR,
                                   f"{video_stem}_tracked_yolo_track.mp4")

    yolo_model = YOLO(YOLO_WEIGHTS)

    # FP16 inference — big speedup on MPS and CUDA
    # If you see MPS dtype errors, set use_half = False
    use_half = device.type in ("cuda", "mps")

    # Normalise tracker name: accept "bytetrack" or "bytetrack.yaml"
    tracker_yaml = tracker if tracker.endswith(".yaml") else f"{tracker}.yaml"

    stream = yolo_model.track(
        source  = video_path,
        stream  = True,
        persist = True,
        tracker = tracker_yaml,
        verbose = True,     
        device  = device,
        half    = use_half,  # ← FP16 for ~2× inference speedup on MPS
        workers = 0,         # ← avoids fork() issues on macOS
    )

    outline_lines: dict = {}
    spine_lines:   dict = {}

    # Async JPEG saves — overlay writes never block the inference loop
    save_pool = ThreadPoolExecutor(max_workers=4)

    def _save_jpeg(img: np.ndarray, path: str) -> None:
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, OVERLAY_JPEG_QUALITY])

    ffmpeg_proc      = None
    cv_writer        = None
    writer_ready     = False
    first_frame_done = False

    for frame_idx, result in enumerate(stream):

        if frame_idx % 100 == 0:
            elapsed    = (datetime.datetime.now() - t_start).total_seconds()
            throughput = frame_idx / elapsed if elapsed > 0 else 0.0
            print(
                f"[{frame_idx:>6}/{total_frames}] "
                f"{elapsed:6.1f}s  {throughput:.1f} fps"
            )

        frame_bgr  = result.orig_img
        h, w       = frame_bgr.shape[:2]
        time_value = frame_idx / fps

        if not first_frame_done:
            cv2.imwrite(first_frame_out, frame_bgr)
            first_frame_done = True

        # ── Build labeled mask at YOLO's native resolution ────────────
        track_ids: list = []

        if result.masks is not None and result.boxes.id is not None:
            track_ids    = result.boxes.id.int().cpu().tolist()
            masks_data   = result.masks.data.cpu().numpy()
            mh, mw       = masks_data.shape[1], masks_data.shape[2]
            labeled_mask = np.zeros((mh, mw), dtype=np.uint16)
            for tid, mask in zip(track_ids, masks_data):
                labeled_mask[mask > 0.5] = tid
        else:
            mh, mw       = h, w
            labeled_mask = np.zeros((mh, mw), dtype=np.uint16)

        # ── Outline + spine data ──────────────────────────────────────
        for tid in track_ids:
            binary = (labeled_mask == tid).astype(np.uint8)
            if binary.any():
                _write_mask_data(
                    binary, tid, time_value,
                    date_str, outline_lines, spine_lines,
                )

        # ── Overlay rendered in-memory (no blocking disk write) ───────
        # Scale the frame to match the mask resolution so the mask is untouched
        frame_canvas = cv2.resize(frame_bgr, (mw, mh)) if (mh, mw) != (h, w) else frame_bgr
        overlay = draw_overlay(frame_canvas, labeled_mask, frame_idx)

        # Async JPEG save — does not block inference loop
        save_pool.submit(
            _save_jpeg,
            overlay.copy(),   # copy so the worker owns its buffer
            os.path.join(OVERLAY_DIR, f"{frame_idx:05d}.jpg"),
        )

        # ── Initialise video writer once ──────────────────────────────
        if not writer_ready:
            ffmpeg_proc = _open_ffmpeg_writer(tracked_video, fps, mw, mh)
            if ffmpeg_proc is None:
                cv_writer = _open_cv_writer(tracked_video, fps, mw, mh)
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

    process_spine_file(outline_file, spine_file)


    print(f"[Output] outline → {outline_file}")
    print(f"[Output] spine   → {spine_file}")

    total = (datetime.datetime.now() - t_start).total_seconds()
    print(f"[Done]   {total_frames} frames in {total:.1f}s ({total/60:.2f} min)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track larvae with YOLO — optimised for Apple Silicon."
    )
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument(
        "--tracker",
        default="bytetrack",
        choices=["bytetrack", "botrack"],
        help="Tracker to use (default: bytetrack)",
    )
    args = parser.parse_args()
    main(args.video_path, tracker=args.tracker)