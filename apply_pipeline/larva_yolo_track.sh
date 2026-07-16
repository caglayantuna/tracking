#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/pasteur/helix/users/ctuna/apply_pipeline/track_yolo_white.out
#SBATCH --error=/pasteur/helix/users/ctuna/apply_pipeline/track_yolo_white.err

source /pasteur/helix/users/ctuna/sam2_env/bin/activate
python /pasteur/helix/users/ctuna/apply_pipeline/track_larva_with_only_yolo.py  /pasteur/helix/users/ctuna/data/sample_video.mp4 
