#!/bin/bash
#SBATCH -J egovm-frames
#SBATCH -p long
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o m3_frames_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$EGOVM_REPO"
exec "${EGOVM_PY:-$HOME/miniconda3/envs/egovm/bin/python}" -m egovm.m3_frames --workers 32
