#!/bin/bash
#SBATCH --job-name=prism3d_train
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
#SBATCH --cpus-per-task=192
#SBATCH --time=02:00:00
#SBATCH --mem=232G


set -e

CONTAINER="prism3d.sif"
OUTDIR="./train_output"
mkdir -p "$OUTDIR"

echo "Training ML accelerator with 200000 BDF samples (192 workers)"
echo "Container: $CONTAINER"
echo "Started at $(date)"

# Phase 1: Generate BDF training data (parallelized)
apptainer exec --bind "$OUTDIR":/output "$CONTAINER" \
    python -m prism3d.chemistry.train_hpc \
        --n-samples 200000 \
        --n-workers 192 \
        --output /output/training_data_200000.npz

if [ ! -f "$OUTDIR/training_data_200000.npz" ]; then
    echo "ERROR: Phase 1 failed — training data not generated"
    exit 1
fi

# Phase 2: Train GBRT accelerator from data
apptainer exec --bind "$OUTDIR":/output "$CONTAINER" \
    python -m prism3d.chemistry.train_hpc \
        --train \
        --data /output/training_data_200000.npz \
        --save /output/accelerator_200000.pkl

echo "Finished at $(date)"
echo "Output: $OUTDIR/accelerator_200000.pkl"
