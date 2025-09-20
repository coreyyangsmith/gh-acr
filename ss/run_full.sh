#!/bin/bash
#SBATCH --job-name=gh-acr
#SBATCH --output=logs/gh-acr-%j.out
#SBATCH --error=logs/gh-acr-%j.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=corey.yangsmith@ucalgary.ca


# Abort on error
set -euo pipefail

# Load the necessary modules
module load gcc arrow/16.1.0 python/3.11

# Create a virtual environment
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"

echo "Step 0: Install dependencies"

# Install wheels from Compute-Canada's local wheelhouse only
pip install --no-index --upgrade pip
pip install -r "$SLURM_SUBMIT_DIR/src/requirements.txt"

echo "Step 1: Running Job..." > logs/job_$SLURM_JOB_ID.out

# Run the job
python -m src.cli.run_all --methods base_a base_b agent bypass7 --max-scenarios 50 --mode clone --model-name local:Qwen/Qwen3-8B