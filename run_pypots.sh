#!/bin/bash -l



#SBATCH --output=/scratch/users/k23031260/PyPOTS/output_logs/Imputation.out
#SBATCH --job-name=pypots
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --signal=USR2
# Load required modules
module load anaconda3/2021.05-gcc-9.4.0
module load cuda/11.1.1-gcc-9.4.0
module load cudnn/8.0.5.39-11.1-gcc-9.4.0
nvidia-smi

# Activate the conda environment
source activate imputation

# Navigate to the directory containing the Python script
cd /scratch/users/k23031260/PyPOTS

python csai.py


