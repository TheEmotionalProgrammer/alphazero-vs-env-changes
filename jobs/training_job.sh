#!/bin/sh
#
#SBATCH --job-name="AZ-TRAIN"
#SBATCH --partition=compute
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=16G
#SBATCH --account=Research-EEMCS-INSY

module load 2024r1

module load python

source venv/bin/activate

srun python src/experiments/train_from_config.py --train_seed=37 --iterations=60 --planning_budget=128 --tree_evaluation_policy="visit" --selection_policy="PUCT" > run.log