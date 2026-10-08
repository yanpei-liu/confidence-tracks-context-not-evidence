#!/bin/bash
#SBATCH -J egovm-m0
#SBATCH -p gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0:40:00
#SBATCH -o m0_%j.out
set -uo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME="${EGOVM_DATA:-$HOME/Files/EgoVM-data}/hf_home"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd "$EGOVM_REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"${EGOVM_PY:-$HOME/miniconda3/envs/egovm/bin/python}" -m egovm.m0_smoke
echo "M0_EXIT=$?"
