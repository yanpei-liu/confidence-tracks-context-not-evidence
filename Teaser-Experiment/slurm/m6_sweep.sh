#!/bin/bash
#SBATCH -J egovm-sweep
#SBATCH -p gpu-long
#SBATCH --gres=gpu:h100-96:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 24:00:00
#SBATCH -o ~/Files/EgoVM-data/logs/m6_sweep_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME=~/Files/EgoVM-data/hf_home
export TOKENIZERS_PARALLELISM=false
cd "$EGOVM_REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec ~/miniconda3/envs/egovm/bin/python -m egovm.m6_score \
    --arms "${EGOVM_ARMS:-main}" --dts "${EGOVM_DTS:-all}" --time-budget-s "${EGOVM_BUDGET:-84000}"
