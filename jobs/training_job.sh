#!/bin/sh
#
#SBATCH --job-name="AZ-TRAIN"
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --account=Research-EEMCS-INSY

module load 2024r1

module load python

source venv/bin/activate

srun python src/experiments/train_from_config.py --train_seed=9