#!/bin/bash
#SBATCH -J egovm-sweep
#SBATCH -p gpu-long
#SBATCH --gres=gpu:h100-47:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 20:00:00
#SBATCH -o ~/Files/EgoVM-data/logs/m6_%x_%j.out
#
# EGOVM_ARMS  which arms to run (default: the headline set)
# EGOVM_OUT   record file; concurrent jobs must use different files
#
# --order plan keeps the plan's per-question grouping, so a job that is cut short still
# leaves results balanced across Δt rather than concentrated in one bin.
set -uo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME=~/Files/EgoVM-data/hf_home
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd "$EGOVM_REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
~/miniconda3/envs/egovm/bin/python -m egovm.m6_score \
    --arms "${EGOVM_ARMS:-main,prior}" --order plan \
    --out "${EGOVM_OUT:-~/Files/EgoVM-data/artifacts/m6_records.jsonl}" \
    --time-budget-s "${EGOVM_BUDGET:-68400}"
echo "SWEEP_EXIT=$?"
