"""Pin the analysis inputs without committing them.

The m6 record files are the primary artifact of the sweep — ~6 GPU-hours of scoring — and they
live in $EGOVM_DATA/artifacts/, outside the repo, with no backup. This writes a checksum
manifest into the repo so a later session can tell whether the records it has are the ones the
committed m7_metrics.json was computed from.

    python scripts/records_manifest.py            # write artifacts/m6_records_manifest.json
    python scripts/records_manifest.py --check     # compare against the committed manifest

Deliberately records only hashes and counts: EgoLife is non-redistributable, and the records
contain gold answers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere

from egovm.config import ARTIFACTS, BIG_ARTIFACTS  # noqa: E402

MANIFEST = ARTIFACTS / "m6_records_manifest.json"


def scan() -> dict:
    files = []
    all_ids: set[str] = set()
    for p in sorted(BIG_ARTIFACTS.glob("m6_records*.jsonl")):
        h = hashlib.sha256()
        n_lines = 0
        ids: set[str] = set()
        with p.open("rb") as f:
            for raw in f:
                h.update(raw)
                n_lines += 1
                try:
                    ids.add(json.loads(raw)["record_id"])
                except Exception:  # noqa: BLE001 - tolerate a torn final line
                    pass
        all_ids |= ids
        files.append({
            "name": p.name,
            "bytes": p.stat().st_size,
            "lines": n_lines,
            "unique_record_ids": len(ids),
            "sha256": h.hexdigest(),
        })
    return {
        "note": "inputs to m7_analyze; the files themselves stay outside the repo",
        "dir": str(BIG_ARTIFACTS),
        "files": files,
        "total_lines": sum(f["lines"] for f in files),
        "unique_record_ids": len(all_ids),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="compare against the committed manifest instead of overwriting it")
    a = ap.parse_args()

    cur = scan()
    if a.check:
        if not MANIFEST.exists():
            raise SystemExit(f"no manifest at {MANIFEST}")
        old = json.loads(MANIFEST.read_text())
        old_f = {f["name"]: f for f in old["files"]}
        cur_f = {f["name"]: f for f in cur["files"]}
        bad = []
        for name in sorted(set(old_f) | set(cur_f)):
            if name not in cur_f:
                bad.append(f"MISSING  {name}")
            elif name not in old_f:
                bad.append(f"EXTRA    {name}")
            elif old_f[name]["sha256"] != cur_f[name]["sha256"]:
                bad.append(f"CHANGED  {name}")
        for line in bad:
            print(line)
        print(f"{len(cur['files'])} files, {cur['unique_record_ids']} unique record ids "
              f"(manifest: {old['unique_record_ids']})")
        raise SystemExit(1 if bad else 0)

    MANIFEST.write_text(json.dumps(cur, indent=1) + "\n")
    print(f"{len(cur['files'])} files, {cur['total_lines']} lines, "
          f"{cur['unique_record_ids']} unique record ids -> {MANIFEST}")


if __name__ == "__main__":
    main()
