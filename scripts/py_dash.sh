#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=100GB  # Requested Memory
#SBATCH -p gpu-preempt  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 8:00:00  # Job time limit
#SBATCH -o logs/slurm-%j-example0.out  # %j = job ID for output log
#SBATCH -e logs/slurm-%j-example0.err  # %j = job ID for error log
#SBATCH --constraint=a100  # Constraint to use A100 GPU

module load conda/latest
conda activate /work/pi_jensen_umass_edu/jnainani_umass_edu/mechinterp
python mask_finding/py_dashboard.py