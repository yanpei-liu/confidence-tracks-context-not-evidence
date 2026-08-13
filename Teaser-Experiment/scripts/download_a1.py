"""Resumable EgoLife A1 download into $EGOVM_DATA/raw/EgoLife.

snapshot_download stalled once with all workers idle and no socket timeout, so this wraps it
in a retry loop with an explicit HF_HUB_DOWNLOAD_TIMEOUT and a watchdog-friendly log line
after every attempt. Partial files under .cache/huggingface/download resume automatically.

If it still will not converge, fall back to the plain-HTTPS path:
    python scripts/make_manifest.py
    PAR=12 scripts/fetch.sh $EGOVM_DATA/manifest_egolife.txt \\
        https://huggingface.co/datasets/lmms-lab/EgoLife/resolve/main $EGOVM_DATA/raw/EgoLife

EgoLife is S-Lab License 1.0 (research use, non-redistributable) — nothing downloaded here
may be committed back into this repo.
"""
import os
import time

DATA = os.environ.get("EGOVM_DATA", "~/Files/EgoVM-data")
os.environ.setdefault("HF_HOME", f"{DATA}/hf_home")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from huggingface_hub import snapshot_download  # noqa: E402

LOCAL = f"{DATA}/raw/EgoLife"
PATTERNS = [
    "A1_JAKE/*",
    "EgoLifeQA/*",
    "EgoLifeCap/DenseCaption/A1_JAKE/*",
    "README.md",
]

t0 = time.time()
for attempt in range(1, 200):
    try:
        p = snapshot_download(
            repo_id="lmms-lab/EgoLife",
            repo_type="dataset",
            allow_patterns=PATTERNS,
            local_dir=LOCAL,
            max_workers=8,
        )
        print("DONE %s after %d attempts, %.1f min" % (p, attempt, (time.time() - t0) / 60),
              flush=True)
        break
    except Exception as e:  # noqa: BLE001
        n = sum(len(f) for _, _, f in os.walk(LOCAL + "/A1_JAKE"))
        print("attempt %d failed (%s: %s); have %d mp4; retrying"
              % (attempt, type(e).__name__, str(e)[:160], n), flush=True)
        time.sleep(10)
else:
    raise SystemExit("download did not converge")
