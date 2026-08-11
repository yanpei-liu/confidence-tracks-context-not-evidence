#!/bin/bash
REPO=~/Files/Egocentric-Vision_Metamemory/.claude/worktrees/teaser-experiment/Teaser-Experiment
N=${1:-6}
for i in $(seq 0 $((N - 1))); do
  sbatch --parsable --export=ALL,EGOVM_REPO="$REPO",EGOVM_SHARD="$i/$N" "$REPO/slurm/m3_shard.sh"
done
