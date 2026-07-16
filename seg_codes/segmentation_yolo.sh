#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=/pasteur/appa/homes/ctuna/larva_%j.out
#SBATCH --error=/pasteur/appa/homes/ctuna/larva_%j.err

source /pasteur/appa/homes/ctuna/yolo_env/bin/activate
python /pasteur/appa/homes/ctuna/segmentation_YOLO_maestro.py
