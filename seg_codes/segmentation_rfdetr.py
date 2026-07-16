import os

from rfdetr import RFDETRSegXLarge

DATASET_DIR = "/pasteur/helix/users/ctuna/rfdetr_training"
OUTPUT_DIR  = "/pasteur/helix/users/ctuna/runs/segmentation_rfdetr_xlarge"

os.makedirs(OUTPUT_DIR, exist_ok=True)
DONE_MARKER = os.path.join(OUTPUT_DIR, "TRAINING_COMPLETE")

# Auto-resume: if a previous (wall-time-killed) job left a checkpoint, continue
# from it — restores model weights AND optimizer state. On a genuinely fresh
# start (no checkpoint) we clear any stale completion marker.
last_ckpt = os.path.join(OUTPUT_DIR, "checkpoint.pth")
resume = last_ckpt if os.path.exists(last_ckpt) else None
if resume:
    print(f"[resume] Continuing from {resume}", flush=True)
else:
    print("[resume] No checkpoint found — starting from scratch.", flush=True)
    if os.path.exists(DONE_MARKER):
        os.remove(DONE_MARKER)

model = RFDETRSegXLarge()

model.train(
    dataset_dir=DATASET_DIR,
    epochs=50,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-5,
    early_stopping=True,
    early_stopping_patience=15,
    early_stopping_min_delta=0.005,
    output_dir=OUTPUT_DIR,
    resume=resume,
)

# train() returned normally (finished all epochs or early-stopped) → signal the
# SLURM wrapper to stop re-queuing this job.
open(DONE_MARKER, "w").close()
print("[done] Training complete — wrote", DONE_MARKER, flush=True)
