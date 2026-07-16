#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/pasteur/appa/homes/ctuna/larva_maskrcnn.out
#SBATCH --error=/pasteur/appa/homes/ctuna/larva_maskrcnn.err

# ── Hyperparameters (edit these) ──────────────────────────────────────────────
LR=0.001
EPOCHS=50
BATCH_SIZE=8
# ─────────────────────────────────────────────────────────────────────────────

BASE=/pasteur/helix/users/ctuna/maskrcnn_training
SCRIPT=/pasteur/helix/users/ctuna/seg_codes/segmentation_maskrcnn.py

source /pasteur/appa/homes/ctuna/yolo_env/bin/activate

python ${SCRIPT} \
    --train_json  ${BASE}/train_maskrcnn.json \
    --train_imgs  ${BASE}/images/train \
    --val_json    ${BASE}/val_maskrcnn.json \
    --val_imgs    ${BASE}/images/val \
    --num_classes 2 \
    --epochs      ${EPOCHS} \
    --batch_size  ${BATCH_SIZE} \
    --lr          ${LR} \
    --num_workers 8 \
    --output_dir  ${BASE}/runs/maskrcnn

