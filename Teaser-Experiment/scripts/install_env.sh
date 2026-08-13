#!/bin/bash
# Build the `egovm` conda env. Vendored from the data root so the repo can rebuild its own
# environment; see requirements.txt for the versions this experiment actually ran on.
#
# usage: EGOVM_PY=/path/to/env/bin/python scripts/install_env.sh
set -eux
PY=${EGOVM_PY:-~/miniconda3/envs/egovm/bin/python}
PIP="$PY -m pip"

$PIP install -q torch torchvision --index-url https://download.pytorch.org/whl/cu126
$PIP install -q "transformers>=4.57" accelerate "qwen-vl-utils[decord]" av pillow numpy \
                pandas scipy matplotlib tqdm huggingface_hub

# The pipeline imports only torch, transformers, PIL, av and matplotlib. The rest are here
# because install_env.sh predates knowing that, and because qwen-vl-utils pulls decord.
$PY - <<'PY'
import torch, transformers, PIL, av
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
from transformers import Qwen3VLForConditionalGeneration
print("Qwen3VL class OK")
PY
echo "INSTALL_DONE"
