#!/bin/bash
# usage: submit_arm.sh <job-name> <arms> <out-file> [gpu-types...]
#
# sbatch --export splits its argument on commas, so --export=ALL,EGOVM_ARMS=main,prior
# silently set EGOVM_ARMS=main and created a stray variable named "prior" — the prior and
# tokmatch_evkeep arms never ran. Exporting in the caller's shell and passing --export=ALL
# keeps multi-value settings intact.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="${EGOVM_DATA:-$HOME/Files/EgoVM-data}/logs"   # sbatch -o: #SBATCH lines cannot expand variables
NAME=$1
ARMS=$2
OUT=$3
shift 3
TYPES=${*:-"h200-141 h100-96 a100-80"}

export EGOVM_REPO="$REPO"
export EGOVM_ARMS="$ARMS"
export EGOVM_OUT="$OUT"
export EGOVM_RACE=1

for G in $TYPES; do
  sbatch --parsable -o "$LOGS/m6_%x_%j.out" -J "$NAME" --gres=gpu:"$G":1 --export=ALL "$REPO/slurm/m6_sweep.sh"
done
