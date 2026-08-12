#!/bin/bash
# usage: submit_supplements.sh [s1|s2|s3|s4|all]
#
# The four supplementary runs described in PLAN.md. Each writes its own record file so the
# jobs can run concurrently and M7 merges them (it globs m6_records*.jsonl and de-duplicates
# on record_id). Every one of these is resumable: re-running a supplement picks up whatever
# the previous job did not finish.
#
# Cost, extrapolated from the measured s/row of the completed sweep:
#   S1  extended token-fixed bins (Δt = 2 h, 4 h)          ~0.8 GPU-h
#   S2  no-memory control on token-matched frames          ~1.1 GPU-h
#   S3  no-evidence floor in every Δt bin                  ~1.8 GPU-h
#   S4  explicit abstention option (five-way softmax)      ~2.4 GPU-h
#
# As with submit_arm.sh: export in this shell and pass --export=ALL, because sbatch splits
# --export on commas and would mangle a multi-arm value.
set -uo pipefail
REPO=~/Files/Egocentric-Vision_Metamemory/.claude/worktrees/teaser-experiment/Teaser-Experiment
ART=~/Files/EgoVM-data/artifacts
WHICH=${1:-all}
TYPES=${EGOVM_GPUS:-"h200-141 h100-96 a100-80"}

submit() {  # submit <name> <arms> <dts> <out> [extra flags]
  export EGOVM_REPO="$REPO" EGOVM_ARMS="$2" EGOVM_DTS="$3" EGOVM_OUT="$ART/$4"
  export EGOVM_EXTRA="${5:-}" EGOVM_RACE=1
  for G in $TYPES; do
    sbatch --parsable -J "$1" --gres=gpu:"$G":1 --export=ALL "$REPO/slurm/m6_sweep.sh"
  done
}

case "$WHICH" in s1|all)
  # S1 — elapsed time past the context budget. Tokens are pinned at the Δt=15 min count, so
  # these rows cost the same as the Δt=30 min matched ones however long the window spans.
  submit egovm-s1 tokmatch_evkeep,tokmatch_uniform 7200,14400 m6_records_s1.jsonl ;;
esac
case "$WHICH" in s2|all)
  # S2 — the no-memory control on token-matched frames. `tokmatch_ckptkeep` needs its memory
  # question too (it is a new arm); `tokmatch_uniform` at 30/60 min already has its memory
  # records, so those rows run the control alone.
  submit egovm-s2 tokmatch_ckptkeep,tokmatch_uniform 1800,3600 m6_records_s2.jsonl ;;
esac
case "$WHICH" in s3|all)
  # S3 — the no-evidence floor in every bin, token-matched to `main` bin by bin. The Δt=60
  # min bin dominates the cost at ~37k visual tokens per row.
  submit egovm-s3 prior 60,900,1800,3600 m6_records_s3.jsonl ;;
esac
case "$WHICH" in s4|all)
  # S4 — the abstention option. Separate record file *and* a separate record kind
  # (`memory_abstain`), because a five-way softmax is not comparable with the four-way one.
  submit egovm-s4 main,prior 300,3600 m6_records_s4_abstain.jsonl --abstain ;;
esac
