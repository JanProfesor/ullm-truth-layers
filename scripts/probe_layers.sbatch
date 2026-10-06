#!/bin/bash
#SBATCH --job-name=probe_acts
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"

uv run python scripts/probe_layers.py \
    --acts activations/llama3-8b-instruct \
    --out results/llama3-8b-instruct \
    --n-jobs 128

echo "finished $(date)"