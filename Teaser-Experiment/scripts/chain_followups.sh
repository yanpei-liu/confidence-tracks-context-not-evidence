#!/bin/bash
# The gpu partition caps at 3 h and the headline sweep needs ~4 h, so queue follow-up jobs
# that start after the current one ends and resume from m6_records_*.jsonl.
# Distinct -J names: the GPU race cancels same-name PENDING jobs at start, which would
# otherwise eat the chain.
REPO=~/Files/Egocentric-Vision_Metamemory/.claude/worktrees/teaser-experiment/Teaser-Experiment
A=~/Files/EgoVM-data/artifacts
PREV_A=$1
PREV_B=$2

for k in 2 3 4; do
  PREV_A=$(sbatch --parsable -J "sweepA$k" --dependency=afterany:"$PREV_A" --gres=gpu:a100-80:1 \
    --export=ALL,EGOVM_REPO="$REPO",EGOVM_ARMS=main,prior,EGOVM_OUT="$A/m6_records_A.jsonl" \
    "$REPO/slurm/m6_sweep.sh")
  echo "sweepA$k = $PREV_A"
done

for k in 2 3; do
  PREV_B=$(sbatch --parsable -J "sweepB$k" --dependency=afterany:"$PREV_B" --gres=gpu:a100-80:1 \
    --export=ALL,EGOVM_REPO="$REPO",EGOVM_ARMS=tokmatch_uniform,tokmatch_evkeep,EGOVM_OUT="$A/m6_records_B.jsonl" \
    "$REPO/slurm/m6_sweep.sh")
  echo "sweepB$k = $PREV_B"
done
