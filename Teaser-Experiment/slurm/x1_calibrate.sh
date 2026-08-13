#!/bin/bash
#SBATCH -J egovm-x1
#SBATCH -p gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0:45:00
#SBATCH -o ~/Files/EgoVM-data/logs/x1_%x_%j.out
#
# X1 — calibrate the streaming backend. See EXTENDING.md §5.
set -uo pipefail

export PYTHONUNBUFFERED=1
export HF_HOME=~/Files/EgoVM-data/hf_home
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Disk offload writes every layer's evicted KV (~GB per question) and the ReKV retriever reads
# layer 0 back. It must land on node-local scratch, never the shared NFS /home.
export EGOVM_OFFLOAD="${EGOVM_OFFLOAD:-/tmp/egovm_kv_$SLURM_JOB_ID}"
mkdir -p "$EGOVM_OFFLOAD"
trap 'rm -rf "$EGOVM_OFFLOAD"' EXIT

cd "$EGOVM_REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /tmp | tail -1
~/miniconda3/envs/egovm/bin/python -m egovm.streaming.x1_calibrate
echo "X1_EXIT=$?"
