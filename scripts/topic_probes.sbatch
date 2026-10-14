#!/bin/bash
#SBATCH --job-name=topic_probes
#SBATCH --partition=genoa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#
# Experiments E1-E3 (scripts/topic_probes.py) on a CPU node of Snellius; no GPU needed.
# Submit from the repository root (mkdir -p logs first):
#
#   sbatch scripts/topic_probes.sbatch --acts activations/qwen2.5-7b-instruct --out results/qwen2.5-7b-instruct
#
# Arguments are forwarded to topic_probes.py.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"

uv run python scripts/topic_probes.py --n-jobs "$SLURM_CPUS_PER_TASK" "$@"

echo "finished $(date)"
