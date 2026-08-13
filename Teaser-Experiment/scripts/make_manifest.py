"""Write the file lists that scripts/fetch.sh consumes.

Only needed on the plain-HTTPS fallback path: huggingface_hub's snapshot_download stalled
repeatedly on this cluster (workers asleep, no socket timeout), and fetch.sh downloads a
flat manifest with curl instead. Writes $EGOVM_DATA/manifest_{egolife,model}.txt.
"""
import json
import os
import urllib.request

DATA = os.environ.get("EGOVM_DATA", "~/Files/EgoVM-data")

meta = json.load(urllib.request.urlopen(
    "https://huggingface.co/api/datasets/lmms-lab/EgoLife"))
sib = [s["rfilename"] for s in meta["siblings"]]
want = [
    f for f in sib
    if f.startswith("A1_JAKE/")
    or f.startswith("EgoLifeQA/")
    or f.startswith("EgoLifeCap/DenseCaption/A1_JAKE/")
]
want = [f for f in want if not f.endswith(".DS_Store")]
with open(f"{DATA}/manifest_egolife.txt", "w") as fh:
    fh.write("\n".join(want) + "\n")
print("egolife files:", len(want))

mmeta = json.load(urllib.request.urlopen(
    "https://huggingface.co/api/models/Qwen/Qwen3-VL-8B-Instruct"))
msib = [s["rfilename"] for s in mmeta["siblings"]]
msib = [f for f in msib if not f.startswith(".") and "/" not in f]
with open(f"{DATA}/manifest_model.txt", "w") as fh:
    fh.write("\n".join(msib) + "\n")
print("model files:", len(msib), msib)
