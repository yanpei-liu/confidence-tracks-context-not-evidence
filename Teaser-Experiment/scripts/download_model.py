"""Download Qwen3-VL-8B-Instruct into $EGOVM_DATA/model/Qwen3-VL-8B-Instruct.

That path is what `config.MODEL_ID` points at, and the Slurm scripts set HF_HUB_OFFLINE=1, so
the model must be a local snapshot on disk before any GPU job runs.
"""
import os
import time

DATA = os.environ.get("EGOVM_DATA", "~/Files/EgoVM-data")
os.environ.setdefault("HF_HOME", f"{DATA}/hf_home")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from huggingface_hub import snapshot_download  # noqa: E402

REPO = "Qwen/Qwen3-VL-8B-Instruct"
LOCAL = f"{DATA}/model/Qwen3-VL-8B-Instruct"

t0 = time.time()
for attempt in range(1, 50):
    try:
        p = snapshot_download(REPO, local_dir=LOCAL, max_workers=8)
        print("MODEL_DONE", p, "%.1f min" % ((time.time() - t0) / 60), flush=True)
        break
    except Exception as e:  # noqa: BLE001
        print("attempt %d failed: %s: %s" % (attempt, type(e).__name__, str(e)[:160]), flush=True)
        time.sleep(10)
else:
    raise SystemExit("model download did not converge")
