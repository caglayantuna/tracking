#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=/pasteur/appa/homes/ctuna/m2f_full.out
#SBATCH --error=/pasteur/appa/homes/ctuna/m2f_full.err

source /pasteur/appa/homes/ctuna/yolo_env/bin/activate
python segmentation_mask2former.py --train_json maskrcnn_training/train_maskrcnn.json --val_json maskrcnn_training/val_maskrcnn.json --train_imgs maskrcnn_training/images/train --val_imgs maskrcnn_training/images/val --num_classes 1 --lr 5e-6 --output_dir runs/m2f_full
