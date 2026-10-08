#!/bin/bash
#SBATCH -J egovm-pilot
#SBATCH -p gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 2:30:00
#SBATCH -o m6_pilot_%j.out
# Small main-arm run to measure real throughput and check that Δt=1 min accuracy clears the
# language-prior floor before committing the full sweep.
set -uo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME="${EGOVM_DATA:-$HOME/Files/EgoVM-data}/hf_home"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd "$EGOVM_REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"${EGOVM_PY:-$HOME/miniconda3/envs/egovm/bin/python}" -m egovm.m6_score \
    --arms main,prior --order plan --limit "${EGOVM_LIMIT:-250}" \
    --out "${EGOVM_DATA:-$HOME/Files/EgoVM-data}/artifacts/m6_pilot.jsonl" \
    --time-budget-s 7800
echo "PILOT_EXIT=$?"
