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
TAG="${1:?tag}"
KINDS="${2:?kinds}"
N="${3:-1}"
GRES="${4:-h100-96}"
shift 4 || true
EXTRA="$*"

for i in $(seq 0 $((N - 1))); do
  sbatch --gres="gpu:${GRES}:1" \
         -J "egovm-x3-${TAG}-${i}" \
         --export="ALL,EGOVM_REPO=${REPO},EGOVM_KINDS=${KINDS},EGOVM_TAG=${TAG}_${i}of${N},EGOVM_SHARD=${i}/${N},EGOVM_XTRA=${EXTRA}" \
         "${REPO}/slurm/x3_stream.sh"
done
echo "submitted ${N} shard(s) for tag=${TAG} kinds=${KINDS} gres=${GRES} extra='${EXTRA}'"
