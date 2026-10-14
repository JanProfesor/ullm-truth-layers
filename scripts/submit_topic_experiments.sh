#!/bin/bash
# Submit activation extraction (GPU) and, once it succeeds, experiments E1-E3 (CPU)
# for Mistral-7B-Instruct-v0.3 and Qwen2.5-7B-Instruct. Run from the repository root
# on Snellius after downloading the weights to ~/models (see README):
#
#   bash scripts/submit_topic_experiments.sh             # both models
#   bash scripts/submit_topic_experiments.sh qwen        # one model (mistral | qwen)
#
# Qwen2.5 overflows in float16, so both models run in bfloat16 (their native dtype)
# and Qwen activations are saved as float32.

set -euo pipefail
mkdir -p logs

submit() {  # name, model dir, extra extraction args...
    local name=$1 model=$2
    shift 2
    if [ ! -f "$model/config.json" ]; then
        echo "no model at $model; download it first (see README)" >&2
        exit 1
    fi
    local ext
    ext=$(sbatch --parsable --job-name="extract_$name" scripts/extract_activations.sbatch \
        --model "$model" --out "activations/$name" --dtype bfloat16 "$@")
    local probe
    probe=$(sbatch --parsable --dependency=afterok:"$ext" --job-name="probes_$name" scripts/topic_probes.sbatch \
        --acts "activations/$name" --out "results/$name")
    echo "$name: extraction job $ext, E1-E3 job $probe (starts after $ext succeeds)"
}

models=${1:-"mistral qwen"}
for m in $models; do
    case $m in
        mistral) submit mistral-7b-instruct-v0.3 "$HOME/models/Mistral-7B-Instruct-v0.3" ;;
        qwen) submit qwen2.5-7b-instruct "$HOME/models/Qwen2.5-7B-Instruct" --save-dtype float32 ;;
        *) echo "unknown model '$m' (use mistral or qwen)" >&2; exit 1 ;;
    esac
done
echo "check progress with: squeue -u \$USER   and   tail -f logs/<job name>_<job id>.out"
