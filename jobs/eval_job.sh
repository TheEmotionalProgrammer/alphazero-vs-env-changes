#!/bin/sh
#
#SBATCH --job-name="AZ-EVAL"
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=Research-EEMCS-INSY

module load 2024r1

module load python

source venv/bin/activate

srun python src/experiments/evaluate_from_config.py

