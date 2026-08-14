#!/bin/bash
#SBATCH -J egovm-x3
#SBATCH -p gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 2:55:00
#SBATCH -o ~/Files/EgoVM-data/logs/x3_%x_%j.out
#
# X3 — the streaming driver. See EXTENDING.md §5.
#
#   EGOVM_KINDS   main | budget | prior | tokmatch | nots   (comma-separated)
#   EGOVM_TAG     record file tag -> m6_records_stream_<tag>.jsonl
#   EGOVM_SHARD   i/n, split on qid
#   EGOVM_XTRA    extra flags passed verbatim (--max-rank, --budgets, --limit, --no-retrieve)
#
# Peak GPU measured at 19.1 GB, so h100-47 and a100-40 are in the pool too — unlike the
# full-reprocess sweep, which needed a single ~12 GB vision-tower allocation on top of a
# 37k-token context.
set -uo pipefail

export PYTHONUNBUFFERED=1
export HF_HOME=~/Files/EgoVM-data/hf_home
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Disk offload must be node-local: the shared NFS /home would be the bottleneck and the
# offload store writes one directory tree per cache.
export EGOVM_OFFLOAD="${EGOVM_OFFLOAD:-/tmp/egovm_kv_$SLURM_JOB_ID}"
mkdir -p "$EGOVM_OFFLOAD"
trap 'rm -rf "$EGOVM_OFFLOAD"' EXIT

cd "$EGOVM_REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# '+' is the comma-safe separator submit_stream.sh uses: Slurm's --export list is itself
# comma-separated and silently truncates a value containing one.
EGOVM_KINDS="${EGOVM_KINDS:-main}"
EGOVM_KINDS="${EGOVM_KINDS//+/,}"
echo "kinds=${EGOVM_KINDS} tag=${EGOVM_TAG:-main} shard=${EGOVM_SHARD:-0/1}"

~/miniconda3/envs/egovm/bin/python -m egovm.streaming.x3_stream_score \
    --kinds "${EGOVM_KINDS}" \
    --tag "${EGOVM_TAG:-main}" \
    --shard "${EGOVM_SHARD:-0/1}" \
    --time-budget-s "${EGOVM_BUDGET:-9300}" ${EGOVM_XTRA:-}
echo "X3_EXIT=$?"
