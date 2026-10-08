#!/bin/bash
#SBATCH -J egovm-x1b
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0:50:00
#SBATCH -o x1b_%x_%j.out
set -uo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME="${EGOVM_DATA:-$HOME/Files/EgoVM-data}/hf_home"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EGOVM_OFFLOAD="${EGOVM_OFFLOAD:-/tmp/egovm_kv_$SLURM_JOB_ID}"
mkdir -p "$EGOVM_OFFLOAD"
trap 'rm -rf "$EGOVM_OFFLOAD"' EXIT
cd "$EGOVM_REPO"
nvidia-smi --query-gpu=name --format=csv,noheader
"${EGOVM_PY:-$HOME/miniconda3/envs/egovm/bin/python}" -m egovm.streaming.x1b_timeprobe
echo "X1B_EXIT=$?"
