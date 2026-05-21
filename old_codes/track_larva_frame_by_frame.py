import argparse
import datetime
import gc
import os
import shutil

import cv2
import numpy as np
from skimage import measure
from skimage.morphology import skeletonize
from ultralytics import YOLO


# ========== CONFIG ==========
YOLO_WEIGHTS = os.path.expanduser("./larvahub/yolo_weights.pt")
FRAMES_DIR   = os.path.expanduser("./frames_temp_frame_by_frame/")
OUTPUT_DIR   = os.path.expanduser("./output_frame_by_frame/")
OVERLAY_DIR  = os.path.join(OUTPUT_DIR, "overlay")

# Single colour used for all objects (BGR)
_OBJ_COLOR = (0, 0, 255)  # red

def _obj_color(obj_id: int):
    return _OBJ_COLOR


# ========== HELPERS ==========


def compute_iou(mask1, mask2):
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

def match_masks_iou(prev_tracks, current_masks, iou_threshold=0.3):
    """
    Match current masks to previous tracks using IoU.

    prev_tracks: dict {track_id: mask}
    current_masks: list of binary masks

    Returns:
        assigned_ids: list of track_ids (same order as current_masks)
        new_tracks: updated dict
        next_track_id: updated counter
    """
    assigned_ids = [-1] * len(current_masks)
    used_prev = set()

    new_tracks = {}

    for i, curr_mask in enumerate(current_masks):
        best_iou = 0
        best_id = None

        for track_id, prev_mask in prev_tracks.items():
            if track_id in used_prev:
                continue

            iou = compute_iou(curr_mask, prev_mask)

            if iou > best_iou:
                best_iou = iou
                best_id = track_id

        if best_iou > iou_threshold:
            assigned_ids[i] = best_id
            used_prev.add(best_id)
            new_tracks[best_id] = curr_mask
        else:
            # will assign new ID later
            pass

    return assigned_ids, new_tracks

def order_skeleton_points(points):
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


def resample_to_n_points(points, n=11):
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


def extract_frames(video_path, frames_dir):
    """Extract all frames from video into frames_dir. Returns list of frame filenames."""
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
        cv2.imwrite(os.path.join(frames_dir, fname), frame)
        frame_names.append(fname)
        frame_idx += 1
    cap.release()
    return frame_names


def segment_frame_yolo(yolo_model, frame_path):
    """
    Run YOLO segmentation on a single frame.
    Returns a labeled mask (H x W uint8) where pixel value = object ID (1-indexed),
    0 = background.
    """
    frame = cv2.imread(frame_path)
    results = yolo_model(frame, verbose=False)
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


def get_mask_center(binary):
    """Return (cx, cy) integer centroid of a binary mask, or None if empty."""
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return None
    return int(xs.mean()), int(ys.mean())


def draw_overlay(frame_bgr, labeled_mask, real_frame_idx, overlay_dir):
    """
    Render coloured mask fills + contours + ID labels onto frame_bgr and save to disk.
    """
    overlay = frame_bgr.copy()
    color_layer = np.zeros_like(frame_bgr)

    obj_ids = np.unique(labeled_mask)
    obj_ids = obj_ids[obj_ids != 0]

    for obj_id in obj_ids:
        binary = (labeled_mask == obj_id).astype(np.uint8)
        color = _obj_color(obj_id)

        # Semi-transparent filled region
        color_layer[binary > 0] = color

        # Contour outline
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 1)

        # ID label at centroid
        center = get_mask_center(binary)
        if center:
            cx, cy = center
            text = str(int(obj_id))
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(overlay, (cx - 2, cy - th - 2), (cx + tw + 2, cy + 2),
                          (0, 0, 0), -1)
            cv2.putText(overlay, text, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Blend colour layer
    cv2.addWeighted(color_layer, 0.35, overlay, 0.65, 0, overlay)

    # HUD
    hud = f"Frame:{real_frame_idx}  Objects:{len(obj_ids)}"
    cv2.putText(overlay, hud, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out_path = os.path.join(overlay_dir, f"{real_frame_idx:05d}.png")
    cv2.imwrite(out_path, overlay)


def _write_mask_data(binary, obj_id, time_value, date_str, outline_lines, spine_lines):
    """Extract outline contour + spine from a binary mask and append to output lists."""
    oid = int(obj_id)

    if oid not in outline_lines:
        outline_lines[oid] = []
    if oid not in spine_lines:
        spine_lines[oid] = []

    padded_label = f"{oid:05d}"

    # Outline
    contours = measure.find_contours(binary.astype(float), 0.5)
    if contours:
        contour = max(contours, key=len)
        line = f"{date_str} {padded_label} {time_value:.3f}"
        for point in contour:
            x, y = point[1], point[0]
            line += f" {x:.4f} {y:.4f}"
        outline_lines[oid].append(line)

    # Spine
    skeleton = skeletonize(binary > 0)
    if np.any(skeleton):
        skel_pts = np.column_stack(np.where(skeleton > 0))
        skel_pts = np.array([[x, y] for y, x in skel_pts])
        skel_pts = order_skeleton_points(skel_pts)
        skel_pts = resample_to_n_points(skel_pts, n=11)
        spine_line = f"{date_str} {padded_label} {time_value:.3f}"
        for sx, sy in skel_pts:
            spine_line += f" {sx:.4f} {sy:.4f}"
        spine_lines[oid].append(spine_line)


# ========== MAIN ==========

def main(video_path):
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract frames
    all_frame_names = extract_frames(video_path, FRAMES_DIR)
    total_frames = len(all_frame_names)
    print(f"[Init] Extracted {total_frames} frames.")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    # Output paths
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OVERLAY_DIR, exist_ok=True)
    video_stem      = os.path.splitext(os.path.basename(video_path))[0]
    first_frame_out = os.path.join(OUTPUT_DIR, f"{video_stem}_frame0.jpg")
    outline_file    = os.path.join(OUTPUT_DIR, f"{video_stem}.outline")
    spine_file      = os.path.join(OUTPUT_DIR, f"{video_stem}.spine")
    tracked_video   = os.path.join(OUTPUT_DIR, f"{video_stem}_tracked_frame_by_frame.mp4")

    # Save first frame
    cv2.imwrite(first_frame_out, cv2.imread(os.path.join(FRAMES_DIR, all_frame_names[0])))

    yolo_model    = YOLO(YOLO_WEIGHTS)
    outline_lines = {}
    spine_lines   = {}
    next_track_id = 1
    prev_tracks = {}

    # ---- Process every frame independently with YOLO ----
    for frame_idx, fname in enumerate(all_frame_names):
        if frame_idx % 100 == 0:
            print(f"[Progress] frame {frame_idx}/{total_frames}")

        frame_path  = os.path.join(FRAMES_DIR, fname)
        time_value  = frame_idx / fps
        labeled_mask = segment_frame_yolo(yolo_model, frame_path)

        # Extract current masks
        obj_ids = np.unique(labeled_mask)
        obj_ids = obj_ids[obj_ids != 0]

        current_masks = [(labeled_mask == oid).astype(np.uint8) for oid in obj_ids]

        # Match with previous frame
        assigned_ids, new_tracks = match_masks_iou(prev_tracks, current_masks)

        # Assign new IDs where needed
        for i, track_id in enumerate(assigned_ids):
            if track_id == -1:
                track_id = next_track_id
                next_track_id += 1
                assigned_ids[i] = track_id
                new_tracks[track_id] = current_masks[i]

        # Save data using TRACK IDs
        for mask, track_id in zip(current_masks, assigned_ids):
            _write_mask_data(mask, track_id, time_value, date_str,
                             outline_lines, spine_lines)

        # Update tracks
        prev_tracks = new_tracks

        # Overlay
        frame_bgr = cv2.imread(frame_path)
        draw_overlay(frame_bgr, labeled_mask, frame_idx, OVERLAY_DIR)

    print(f"[Done] Processed {total_frames} frames.")

    # Write output files
    with open(outline_file, 'w') as f:
        for oid in sorted(outline_lines):
            for line in outline_lines[oid]:
                f.write(line + "\n")

    with open(spine_file, 'w') as f:
        for oid in sorted(spine_lines):
            for line in spine_lines[oid]:
                f.write(line + "\n")

    print(f"[Output] outline → {outline_file}")
    print(f"[Output] spine   → {spine_file}")

    # Compile overlay frames into tracked video
    print("\n🎬 Creating tracked video…")
    first_overlay = os.path.join(OVERLAY_DIR, "00000.png")
    if os.path.exists(first_overlay):
        sample = cv2.imread(first_overlay)
        h, w = sample.shape[:2]
        writer = cv2.VideoWriter(
            tracked_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        for frame_idx in range(total_frames):
            fpath = os.path.join(OVERLAY_DIR, f"{frame_idx:05d}.png")
            if os.path.exists(fpath):
                writer.write(cv2.imread(fpath))
            else:
                writer.write(np.zeros((h, w, 3), dtype=np.uint8))
        writer.release()
        print(f"[Video] Saved → {tracked_video}")
    else:
        print("[Video] No overlay frames found — skipping video creation.")

    # Cleanup
    if os.path.exists(FRAMES_DIR):
        shutil.rmtree(FRAMES_DIR)
    print("[Done]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track larvae frame-by-frame using YOLO segmentation only."
    )
    parser.add_argument("video_path", help="Path to input video file")
    args = parser.parse_args()
    main(args.video_path)
