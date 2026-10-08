#!/bin/bash
# Submit an X3 streaming pass.
#
#   scripts/submit_stream.sh <tag> <kinds> <n_shards> <gres> [extra flags...]
#
# Shards split on qid, so one question's whole Δt series stays on one worker — a modulo split
# over rows would scatter it and each worker would re-ingest the stream.
#
# Peak GPU is 19.1 GB, so a100-40 / h100-47 are usable here even though the full-reprocess
# sweep needed 80 GB cards.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="${EGOVM_DATA:-$HOME/Files/EgoVM-data}/logs"   # sbatch -o: #SBATCH lines cannot expand variables
TAG="${1:?tag}"
KINDS="${2:?kinds}"
N="${3:-1}"
GRES="${4:-h100-96}"
shift 4 || true
EXTRA="$*"

# Slurm's --export list is ITSELF comma-separated, so a value containing a comma is silently
# truncated and its tail is read as another variable name. `EGOVM_KINDS=prior,tokmatch` set
# EGOVM_KINDS=prior and looked for an (absent) variable `tokmatch` — the tokmatch pass produced
# zero records and the job still exited 0. Commas are carried as '+' and restored in the job
# script; nothing here may contain a comma.
KINDS_SAFE="${KINDS//,/+}"
case "${EXTRA}" in
  *,*) echo "refusing: extra flags must not contain a comma (Slurm --export limitation)" >&2
       exit 1 ;;
esac

# xgpj0 is a different CPU architecture from the rest of the gpu partition, and the conda env
# at ~/miniconda3/envs/egovm is built for the others: python there dies with "cannot execute
# binary file: Exec format error", the job exits 126 after two seconds, and *Slurm reports it
# COMPLETED*. Nothing is written and nothing looks wrong. Excluded rather than diagnosed twice.
EXCLUDE_NODES="xgpj0"

for i in $(seq 0 $((N - 1))); do
  sbatch -o "${LOGS}/x3_%x_%j.out" --gres="gpu:${GRES}:1" --exclude="${EXCLUDE_NODES}" \
         -J "egovm-x3-${TAG}-${i}" \
         --export="ALL,EGOVM_REPO=${REPO},EGOVM_KINDS=${KINDS_SAFE},EGOVM_TAG=${TAG}_${i}of${N},EGOVM_SHARD=${i}/${N},EGOVM_XTRA=${EXTRA}" \
         "${REPO}/slurm/x3_stream.sh"
done
echo "submitted ${N} shard(s) for tag=${TAG} kinds=${KINDS} gres=${GRES} extra='${EXTRA}'"
