#!/bin/bash
#SBATCH --job-name=extract_acts
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#
# Extract last-token activations of Meta-Llama-3-8B-Instruct on Snellius.
# Submit from the repository root (create logs/ first: mkdir -p logs):
#
#   sbatch scripts/extract_activations.sbatch                                  # full run
#   sbatch scripts/extract_activations.sbatch --limit 200 --out activations/test   # quick test
#
# Extra arguments are forwarded to extract_activations.py and override the defaults below.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

uv run python scripts/extract_activations.py \
    --model "$HOME/models/Meta-Llama-3-8B-Instruct" \
    --out activations/llama3-8b-instruct \
    --batch-size 64 \
    "$@"

echo "finished $(date)"
