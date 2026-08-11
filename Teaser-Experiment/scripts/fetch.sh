#!/bin/bash
# Robust parallel fetch of a HF repo file list via plain HTTPS.
#
# huggingface_hub's snapshot_download (with and without hf_xet) stalled repeatedly on this
# cluster: workers went to sleep with no socket timeout and never recovered. curl with hard
# --max-time / --connect-timeout and an outer completeness loop is slower per connection but
# actually terminates.
#
# usage: fetch.sh <manifest> <base_url> <dest_root>
#   manifest: one repo-relative path per line
set -uo pipefail
MANIFEST=$1
BASE=$2
DEST=$3
PAR=${PAR:-12}

fetch_one() {
  rel="$1"
  out="$DEST/$rel"
  [ -s "$out" ] && return 0
  mkdir -p "$(dirname "$out")"
  curl -sfL --connect-timeout 20 --max-time 600 --retry 4 --retry-delay 3 --retry-all-errors \
       -o "$out.part" "$BASE/$rel" && mv "$out.part" "$out" || { rm -f "$out.part"; return 1; }
}
export -f fetch_one
export DEST BASE

for round in $(seq 1 40); do
  missing=$(while read -r rel; do [ -s "$DEST/$rel" ] || echo "$rel"; done < "$MANIFEST")
  n=$(echo "$missing" | grep -c . || true)
  echo "round $round: $n files missing"
  [ "$n" -eq 0 ] && { echo "FETCH_COMPLETE"; exit 0; }
  echo "$missing" | grep . | xargs -P "$PAR" -I{} bash -c 'fetch_one "$@"' _ {}
  sleep 2
done
echo "FETCH_INCOMPLETE"
exit 1
