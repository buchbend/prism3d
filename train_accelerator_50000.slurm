#!/bin/bash
#SBATCH --job-name=prism3d_train
#SBATCH --output=train_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --mem=10G

source $HOME/prism3d_env/bin/activate

echo "Training ML accelerator with 50000 BDF samples"
echo "Started at $(date)"

python -m prism3d.chemistry.train_hpc \
    --n-samples 50000 \
    --output training_data_50000.npz

python -m prism3d.chemistry.train_hpc \
    --train \
    --data training_data_50000.npz \
    --save accelerator_50000.pkl

echo "Finished at $(date)"
