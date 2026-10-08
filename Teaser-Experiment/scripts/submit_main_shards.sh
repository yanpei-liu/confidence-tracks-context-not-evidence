#!/bin/bash
# Shard the heavy `main` arm across GPUs. At 8.5 s/row a single job needs ~4.3 h for the
# 1,849 remaining rows (each row scores the memory question and its control over windows of
# up to 769 frames); four shards bring that to ~1.1 h. Each shard writes its own record file
# and M7 merges them, de-duplicating on record_id.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="${EGOVM_DATA:-$HOME/Files/EgoVM-data}/logs"   # sbatch -o: #SBATCH lines cannot expand variables
A="${EGOVM_DATA:-$HOME/Files/EgoVM-data}/artifacts"
N=${N:-4}

export EGOVM_REPO="$REPO"
export EGOVM_ARMS=main

for i in $(seq 0 $((N - 1))); do
  export EGOVM_SHARD="$i/$N"
  export EGOVM_OUT="$A/m6_records_main_${i}of${N}.jsonl"
  for G in a100-80 h100-96 h200-141; do
    export EGOVM_RACE=1
    sbatch --parsable -o "$LOGS/m6_%x_%j.out" -J "mainS$i" --gres=gpu:"$G":1 --export=ALL "$REPO/slurm/m6_sweep.sh"
  done
done
