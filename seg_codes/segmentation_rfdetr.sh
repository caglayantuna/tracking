#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --signal=B:USR1@600
#SBATCH --output=/pasteur/appa/homes/ctuna/larva_rfxlarge_%j.out
#SBATCH --error=/pasteur/appa/homes/ctuna/larva_rfxlarge_%j.err

# XLarge seg training does not finish 50 epochs within one 20h wall-time. This
# script re-queues itself ~10 min before wall-time (via the USR1 signal above);
# the Python side auto-resumes from checkpoint.pth, so the chain of jobs keeps
# going until training actually completes and writes TRAINING_COMPLETE.

SCRIPT=/pasteur/appa/homes/ctuna/seg_codes/segmentation_rfdetr.sh
PY=/pasteur/appa/homes/ctuna/seg_codes/segmentation_rfdetr.py
OUTDIR=/pasteur/helix/users/ctuna/runs/segmentation_rfdetr_xlarge

resubmit() {
    if [ ! -f "${OUTDIR}/TRAINING_COMPLETE" ]; then
        echo "[wrapper] Wall-time approaching — resubmitting to resume from checkpoint."
        sbatch "$SCRIPT"
    else
        echo "[wrapper] Training already complete — not resubmitting."
    fi
}
trap resubmit USR1

source /pasteur/appa/homes/ctuna/yolo_env/bin/activate

# Run in background + wait so the USR1 trap can fire while training is running.
python "$PY" &
wait
