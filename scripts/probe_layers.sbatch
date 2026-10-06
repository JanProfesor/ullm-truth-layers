#!/bin/bash
#SBATCH --job-name=probe_acts
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"

uv run python scripts/probe_layers.py \
    --acts activations/llama3-8b-instruct \
    --out results/llama3-8b-instruct \
    --n-jobs 18

echo "finished $(date)"