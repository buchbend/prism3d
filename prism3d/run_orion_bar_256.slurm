#!/bin/bash
#SBATCH --job-name=prism3d_orion256
#SBATCH --output=orion256_%j.out
#SBATCH --error=orion256_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --exclusive

set -e

# ── Configuration ────────────────────────────────────────
CONTAINER="./prism3d.sif"
ACCELERATOR="./train_output/accelerator_200000.pkl"   # ← adjust path
OUTDIR="./orion_bar_256"
N=256
MAX_ITER=20

# ── Sanity checks ───────────────────────────────────────
if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found: $CONTAINER"
    exit 1
fi

if [ ! -f "$ACCELERATOR" ]; then
    echo "WARNING: Accelerator not found: $ACCELERATOR"
    echo "Running WITHOUT ML accelerator (will be much slower)"
    ACC_BIND=""
    ACC_FLAG=""
else
    echo "Using ML accelerator: $ACCELERATOR"
    ACC_BIND="--bind $(realpath $ACCELERATOR):/model/accelerator.pkl"
    ACC_FLAG="--accelerator /model/accelerator.pkl"
fi

mkdir -p "$OUTDIR"

# ── Threading: let NumPy/OpenBLAS use all cores ─────────
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ── Run ─────────────────────────────────────────────────
echo "============================================"
echo "  PRISM-3D: Orion Bar ${N}³ cube"
echo "  Job:    $SLURM_JOB_ID"
echo "  Node:   $(hostname)"
echo "  CPUs:   $SLURM_CPUS_PER_TASK"
echo "  Start:  $(date)"
echo "============================================"

apptainer exec \
    --bind "$(realpath $OUTDIR)":/output \
    $ACC_BIND \
    --env OMP_NUM_THREADS=$OMP_NUM_THREADS \
    --env OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS \
    --env MKL_NUM_THREADS=$MKL_NUM_THREADS \
    --env PYTHONUNBUFFERED=1 \
    "$CONTAINER" \
    python -u -m prism3d.run \
        --orion-bar \
        --n $N \
        --max-iter $MAX_ITER \
        --output /output \
        $ACC_FLAG

echo "============================================"
echo "  Finished: $(date)"
echo "  Output:   $OUTDIR/"
echo "============================================"
