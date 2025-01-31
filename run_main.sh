#!/bin/bash
##ATHENA
#SBATCH --job-name=matrix_completion
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgeetraining-gpu-a100
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/slurm-%j.out

eval "$(conda shell.bash hook)"
conda activate clpi_env
#source src/configs/env_variables.sh

WANDB__SERVICE_WAIT=300 python -m main