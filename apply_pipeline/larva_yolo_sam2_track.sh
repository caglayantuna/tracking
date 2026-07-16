#!/bin/bash
#SBATCH --partition=dedicatedgpu
#SBATCH --qos=fast
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/pasteur/helix/users/ctuna/apply_pipeline/track_sam2_black.out
#SBATCH --error=/pasteur/helix/users/ctuna/apply_pipeline/track_sam2_black.err

source /pasteur/helix/users/ctuna/sam2_env/bin/activate
python /pasteur/helix/users/ctuna/apply_pipeline/track_larva_with_yolo_and_sam2.py /pasteur/helix/users/ctuna/data/sample_video.mp4 
