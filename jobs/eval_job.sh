#!/bin/sh
#
#SBATCH --job-name="AZ-TRAIN"
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=16G
#SBATCH --account=Research-EEMCS-INSY

module load 2024r1

module load python

source venv/bin/activate

srun python src/experiments/evaluate_from_config.py --selection_policy="UCT" --tree_evaluation_policy="visit" --predictor="original_env" --unroll_budget="3" --threshold="0.001" --test_env_desc="NARROW_SIMPLIFIED"

