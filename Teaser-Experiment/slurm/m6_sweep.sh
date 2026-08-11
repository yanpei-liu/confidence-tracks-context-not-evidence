#!/bin/bash
#SBATCH -J egovm-sweep
#SBATCH -p gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 2:55:00
#SBATCH -o ~/Files/EgoVM-data/logs/m6_%x_%j.out
#
# EGOVM_ARMS  which arms to run (default: the headline set)
# EGOVM_OUT   record file; concurrent jobs must use different files
#
# --order plan keeps the plan's per-question grouping, so a job that is cut short still
# leaves results balanced across Δt rather than concentrated in one bin.
set -uo pipefail

# GPU race (see Misc/SoC GPU Info.md): the same job is submitted once per GPU type under one
# -J name and whichever starts first cancels the still-pending twins. The doc's warning about
# two jobs starting before the scancel lands is harmless here — M6 is idempotent per row and
# M7 de-duplicates on record_id, so the worst case is duplicated work, not wrong results.
if [ -n "${EGOVM_RACE:-}" ]; then
  scancel -n "$SLURM_JOB_NAME" -u "$USER" --state=PENDING 2>/dev/null || true
fi

export PYTHONUNBUFFERED=1
export HF_HOME=~/Files/EgoVM-data/hf_home
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd "$EGOVM_REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
~/miniconda3/envs/egovm/bin/python -m egovm.m6_score \
    --arms "${EGOVM_ARMS:-main,prior}" --order plan --shard "${EGOVM_SHARD:-0/1}" \
    --out "${EGOVM_OUT:-~/Files/EgoVM-data/artifacts/m6_records.jsonl}" \
    --time-budget-s "${EGOVM_BUDGET:-9600}"
echo "SWEEP_EXIT=$?"
