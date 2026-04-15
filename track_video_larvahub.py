import argparse
import datetime
import gc
import os
import shutil

import cv2
import numpy as np
import torch
from skimage import measure
from skimage.morphology import skeletonize
from sam2.build_sam import build_sam2_video_predictor
from ultralytics import YOLO


# ========== CONFIG ==========
YOLO_WEIGHTS = os.path.expanduser("./yolo_weights.pt")
CHECKPOINT   = os.path.expanduser("./sam2_hiera_small.pt")
MODEL_CFG    = "sam2_hiera_s.yaml"
FRAMES_DIR   = os.path.expanduser("./frames_temp")
CHUNK_DIR    = os.path.expanduser("./frames_temp/chunk_symlinks")
OUTPUT_DIR   = os.path.expanduser("./output/")
CHUNK_SIZE   = 300


# ========== HELPERS ==========

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


def segment_first_frame_yolo(yolo_model, frame_path):
    """
    Run YOLO segmentation on the first frame.
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


# ========== MAIN ==========

def main(video_path):
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract frames
    all_frame_names = extract_frames(video_path, FRAMES_DIR)
    total_frames = len(all_frame_names)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    # Output paths
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_stem      = os.path.splitext(os.path.basename(video_path))[0]
    first_frame_out = os.path.join(OUTPUT_DIR, f"{video_stem}_frame0.jpg")
    outline_file    = os.path.join(OUTPUT_DIR, f"{video_stem}.outline")
    spine_file      = os.path.join(OUTPUT_DIR, f"{video_stem}.spine")

    # Save first frame
    cv2.imwrite(first_frame_out, cv2.imread(os.path.join(FRAMES_DIR, all_frame_names[0])))

    # YOLO segmentation on first frame
    yolo_model   = YOLO(YOLO_WEIGHTS)
    labeled_mask = segment_first_frame_yolo(
        yolo_model, os.path.join(FRAMES_DIR, all_frame_names[0])
    )
    obj_ids = np.unique(labeled_mask)
    obj_ids = obj_ids[obj_ids != 0]

    if len(obj_ids) == 0:
        raise RuntimeError("No objects detected by YOLO — cannot proceed with tracking.")

    # Tracking (chunked)
    outline_lines = {int(oid): [] for oid in obj_ids}
    spine_lines   = {int(oid): [] for oid in obj_ids}
    current_mask  = labeled_mask.copy()

    for chunk_start in range(0, total_frames, CHUNK_SIZE):
        chunk_end         = min(chunk_start + CHUNK_SIZE, total_frames)
        chunk_frame_names = all_frame_names[chunk_start:chunk_end]

        # Symlink chunk frames into CHUNK_DIR
        if os.path.exists(CHUNK_DIR):
            shutil.rmtree(CHUNK_DIR)
        os.makedirs(CHUNK_DIR)
        for fname in chunk_frame_names:
            os.symlink(
                os.path.abspath(os.path.join(FRAMES_DIR, fname)),
                os.path.join(CHUNK_DIR, fname)
            )

        # Build predictor for this chunk
        predictor = build_sam2_video_predictor(MODEL_CFG, CHECKPOINT)
        inference_state = predictor.init_state(
            video_path=CHUNK_DIR,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )
        predictor.reset_state(inference_state)

        # Seed with masks from end of previous chunk (or YOLO mask for first chunk)
        for obj_id in np.unique(current_mask):
            if obj_id == 0:
                continue
            mask = (current_mask == obj_id)
            if not mask.any():
                continue
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=int(obj_id),
                mask=mask,
            )

        last_frame_masks = {}

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state=inference_state
        ):
            real_frame_idx = chunk_start + out_frame_idx
            time_value     = real_frame_idx / fps

            obj_masks = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

            for obj_id, mask in obj_masks.items():
                binary = mask.squeeze().astype(np.uint8)
                if not binary.any():
                    continue

                padded_label = f"{int(obj_id):05d}"

                # Outline
                contours = measure.find_contours(binary.astype(float), 0.5)
                if contours:
                    contour = max(contours, key=len)
                    line = f"{date_str} {padded_label} {time_value:.3f}"
                    for point in contour:
                        x, y = point[1], point[0]
                        line += f" {x:.4f} {y:.4f}"
                    outline_lines[int(obj_id)].append(line)

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
                    spine_lines[int(obj_id)].append(spine_line)

            last_frame_masks = obj_masks.copy()

        # Hand off last frame masks to next chunk
        if last_frame_masks:
            h, w = list(last_frame_masks.values())[0].squeeze().shape
            current_mask = np.zeros((h, w), dtype=np.uint8)
            for obj_id, mask in last_frame_masks.items():
                if mask.any():
                    current_mask[mask.squeeze()] = obj_id

        del predictor, inference_state, last_frame_masks
        torch.cuda.empty_cache()
        gc.collect()

    # Write output files
    with open(outline_file, 'w') as f:
        for oid in sorted(outline_lines):
            for line in outline_lines[oid]:
                f.write(line + "\n")

    with open(spine_file, 'w') as f:
        for oid in sorted(spine_lines):
            for line in spine_lines[oid]:
                f.write(line + "\n")

    # Cleanup
    if os.path.exists(FRAMES_DIR):
        shutil.rmtree(FRAMES_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track objects in video and export outline/spine files."
    )
    parser.add_argument("video_path", help="Path to input video file")
    args = parser.parse_args()
    main(args.video_path)