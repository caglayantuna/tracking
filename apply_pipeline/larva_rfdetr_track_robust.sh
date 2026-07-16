#!/bin/bash
#SBATCH --partition=dedicatedgpu
#SBATCH --qos=fast
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/pasteur/helix/users/ctuna/apply_pipeline/track_rfdetr_robust_A1_2025-04-11-162931-0000.out
#SBATCH --error=/pasteur/helix/users/ctuna/apply_pipeline/track_rfdetr_robust_A1_2025-04-11-162931-0000.err

source /pasteur/helix/users/ctuna/sam2_env/bin/activate
python /pasteur/helix/users/ctuna/apply_pipeline/track_larva_with_rfdetr_robust.py /pasteur/helix/users/ctuna/data/A1_2025-04-11-162931-0000.avi
