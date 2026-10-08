#!/bin/bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="${EGOVM_DATA:-$HOME/Files/EgoVM-data}/logs"   # sbatch -o: #SBATCH lines cannot expand variables
N=${1:-6}
for i in $(seq 0 $((N - 1))); do
  sbatch --parsable -o "$LOGS/m3_shard_%j.out" --export=ALL,EGOVM_REPO="$REPO",EGOVM_SHARD="$i/$N" "$REPO/slurm/m3_shard.sh"
done
