#!/bin/bash
#SBATCH -J egovm-frames
#SBATCH -p long
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH -t 6:00:00
#SBATCH -o m3_shard_%j.out
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$EGOVM_REPO"
"${EGOVM_PY:-$HOME/miniconda3/envs/egovm/bin/python}" -m egovm.m3_frames \
    --workers 32 --shard "$EGOVM_SHARD"
echo "M3_SHARD_EXIT=$?"
