#!/bin/bash
#SBATCH --job-name=topic_probes
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#
# Experiments E1-E3 (scripts/topic_probes.py) on Snellius. The probes only use the CPUs,
# but the job runs on gpu_a100 because our budget covers no CPU partition; 18 cores is
# the CPU share that comes with one A100.
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
