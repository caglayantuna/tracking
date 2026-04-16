import numpy as np
import os
import cv2
import torch
import gc
import shutil
from sam2.build_sam import build_sam2_video_predictor

def get_mask_center(mask):
    mask_2d = mask.squeeze().astype(np.uint8)
    coords = np.where(mask_2d > 0)
    if len(coords[0]) == 0:
        return None
    cy = int(np.mean(coords[0]))
    cx = int(np.mean(coords[1]))
    return (cx, cy)

print(f"CUDA available: {torch.cuda.is_available()}")

# ========== CONFIG ==========
VIDEO_DIR = os.path.expanduser("/pasteur/appa/homes/ctuna/frames_white")
LABELED_MASK_PATH = os.path.expanduser("/pasteur/appa/homes/ctuna/labeled_mask_white.png")
CHECKPOINT = os.path.expanduser("/pasteur/appa/homes/ctuna/sam2_checkpoints/sam2_hiera_small.pt")
MODEL_CFG = "sam2_hiera_s.yaml"
OUTPUT_DIR = os.path.expanduser("/pasteur/appa/homes/ctuna/output_white")
TEMP_DIR = os.path.expanduser("/pasteur/appa/homes/ctuna/temp_chunk")
OVERLAY_COLOR = (0, 0, 255)

CHUNK_SIZE = 300

os.makedirs(os.path.join(OUTPUT_DIR, "masks"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "overlay"), exist_ok=True)

# ========== Load Labeled Mask ==========
labeled_mask = cv2.imread(LABELED_MASK_PATH, cv2.IMREAD_GRAYSCALE)
obj_ids = np.unique(labeled_mask)
obj_ids = obj_ids[obj_ids != 0]
print(f"✅ Loaded labeled mask: {labeled_mask.shape}")
print(f"   Found {len(obj_ids)} objects")

# ========== Get ALL frame names ==========
all_frame_names = sorted([
    f for f in os.listdir(VIDEO_DIR)
    if f.endswith(('.jpg', '.jpeg', '.png'))
])
total_frames = len(all_frame_names)
print(f"📹 Video: {total_frames} frames")


current_mask = labeled_mask.copy()
chunks = list(range(0, total_frames, CHUNK_SIZE))


#spline_file = 
#outline_file = 

for chunk_idx, chunk_start in enumerate(chunks):
    chunk_end = min(chunk_start + CHUNK_SIZE, total_frames)
    chunk_frame_names = all_frame_names[chunk_start:chunk_end]

    print(f"\n{'='*60}")
    print(f"CHUNK {chunk_idx+1}/{len(chunks)}: frames {chunk_start}-{chunk_end-1} ({len(chunk_frame_names)} frames)")
    print(f"{'='*60}")

    # Copy ONLY this chunk's frames to temp directory
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    for fname in chunk_frame_names:
        shutil.copy2(
            os.path.join(VIDEO_DIR, fname),
            os.path.join(TEMP_DIR, fname)
        )
    print(f"  📁 Copied {len(chunk_frame_names)} frames to temp dir")

    # Fresh predictor — loads ONLY chunk frames
    predictor = build_sam2_video_predictor(MODEL_CFG, CHECKPOINT)
    inference_state = predictor.init_state(
        video_path=TEMP_DIR,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )
    predictor.reset_state(inference_state)

    # Add current masks at frame 0 of this chunk
    active_obj_ids = np.unique(current_mask)
    active_obj_ids = active_obj_ids[active_obj_ids != 0]

    for obj_id in active_obj_ids:
        mask = (current_mask == obj_id)
        if not mask.any():
            continue
        _, _, _ = predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=int(obj_id),
            mask=mask,
        )
    print(f"  Added {len(active_obj_ids)} objects")

    # Track this chunk
    print(f"  🚀 Tracking...")
    last_frame_masks = {}

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state=inference_state
    ):
        real_frame_idx = chunk_start + out_frame_idx

        obj_masks = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

        sample_mask = list(obj_masks.values())[0].squeeze()
        h, w = sample_mask.shape

        # Labeled mask
        frame_labeled = np.zeros((h, w), dtype=np.uint8)
        for obj_id, mask in obj_masks.items():
            frame_labeled[mask.squeeze()] = obj_id

        cv2.imwrite(
            os.path.join(OUTPUT_DIR, "masks", f"{real_frame_idx:05d}.png"),
            frame_labeled
        )

        # Overlay
        frame = cv2.imread(
            os.path.join(VIDEO_DIR, all_frame_names[real_frame_idx])
        )
        overlay = frame.copy()
        merged = frame_labeled > 0
        overlay[merged] = (
            overlay[merged] * 0.5 + np.array(OVERLAY_COLOR) * 0.5
        ).astype(np.uint8)

        num_visible = 0
        for obj_id, mask in obj_masks.items():
            if not mask.any():
                continue
            num_visible += 1
            center = get_mask_center(mask)
            if center:
                cx, cy = center
                text = str(int(obj_id))
                (tw, th), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                )
                cv2.rectangle(overlay,
                             (cx-2, cy-th-2), (cx+tw+2, cy+2),
                             (0, 0, 0), -1)
                cv2.putText(overlay, text, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 255, 255), 1)

        cv2.putText(overlay,
                    f"Frame:{real_frame_idx} Objects:{num_visible}/{len(obj_ids)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, "overlay", f"{real_frame_idx:05d}.png"),
            overlay
        )

        last_frame_masks = obj_masks.copy()

        if (out_frame_idx + 1) % 50 == 0:
            print(f"    Frame {real_frame_idx}/{total_frames} | "
                  f"Visible: {num_visible}/{len(obj_ids)}")

    # Use last frame's masks for next chunk
    if last_frame_masks:
        h, w = list(last_frame_masks.values())[0].squeeze().shape
        current_mask = np.zeros((h, w), dtype=np.uint8)
        for obj_id, mask in last_frame_masks.items():
            if mask.any():
                current_mask[mask.squeeze()] = obj_id

    # Free memory
    del predictor, inference_state, last_frame_masks
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  ✅ Chunk done, memory freed")

# Clean up temp dir
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)

# ========== Save Video ==========
print(f"\n🎬 Creating video...")
first_overlay = os.path.join(OUTPUT_DIR, "overlay", "00000.png")
if os.path.exists(first_overlay):
    sample = cv2.imread(first_overlay)
    h, w = sample.shape[:2]
    writer = cv2.VideoWriter(
        os.path.join(OUTPUT_DIR, "tracked_result_maestro_small.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'), 60, (w, h)
    )

    for frame_idx in range(total_frames):
        fpath = os.path.join(OUTPUT_DIR, "overlay", f"{frame_idx:05d}.png")
        if os.path.exists(fpath):
            writer.write(cv2.imread(fpath))

    writer.release()

print(f"\n🎉 DONE!")
print(f"  Video: {OUTPUT_DIR}/tracked_result_maestro_small.mp4")