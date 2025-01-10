#!/bin/sh
#
#SBATCH --job-name="AZ-TRAIN"
#SBATCH --partition=compute
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=16G
#SBATCH --account=Education-EEMCS-MSc-CS

module load 2024r1

module load python

source venv/bin/activate

srun python src/experiments/evaluate_from_config.py --selection_policy="PolicyPUCT" --tree_evaluation_policy="mvc" --test_env_desc="INVERSE_DEAD_END"

