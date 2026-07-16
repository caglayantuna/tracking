#!/bin/bash
#SBATCH --partition=dedicatedgpu
#SBATCH --qos=fast
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/pasteur/helix/users/ctuna/apply_pipeline/track_efficienttam.out
#SBATCH --error=/pasteur/helix/users/ctuna/apply_pipeline/track_efficienttam.err

# ── Environment ───────────────────────────────────────────────────────────────
# The env must have: torch, ultralytics, opencv, scikit-image, hydra-core, and the
# EfficientTAM package (`efficient_track_anything`). Two ways to provide it:
#   1) pip install -e /pasteur/helix/users/ctuna/EfficientTAM   (recommended)
#   2) leave it un-installed and set EFFICIENTTAM_ROOT below to the cloned repo;
#      the python script adds it to sys.path automatically.
source /pasteur/helix/users/ctuna/sam2_env/bin/activate
export EFFICIENTTAM_ROOT=/pasteur/helix/users/ctuna/EfficientTAM

python /pasteur/helix/users/ctuna/apply_pipeline/track_larva_with_yolo_and_efficienttam.py /pasteur/helix/users/ctuna/data/sample_video.mp4
