"""Streaming-backend constants.

Kept separate from `egovm.config` so the Qwen3-VL full-reprocess sweep stays byte-for-byte
reproducible: nothing in this module is imported by M0–M9.
"""
from __future__ import annotations

import os
from pathlib import Path

from ..config import ARTIFACTS, BIG_ARTIFACTS, DATA_ROOT, INPUT_RES, PATCH_PX

# --- where the upstream EgoStream checkout lives ----------------------------------------
# It ships no LICENSE and is not a git checkout, so it is NOT vendored into this repo (see
# EXTENDING.md §1b). We import it from outside the tree and record a checksum manifest.
# Default: a git-ignored EgoStream/ checkout at the repository root.
_DEFAULT_EGOSTREAM = (
    Path(__file__).resolve().parents[3] / "EgoStream" / "Online_Cache_Optimization"
)
EGOSTREAM_ROOT = Path(os.environ.get("EGOVM_EGOSTREAM", str(_DEFAULT_EGOSTREAM)))

# The upstream files we actually depend on. X0 hashes exactly these, so a silent upstream
# edit is caught rather than absorbed.
UPSTREAM_DEPS = (
    "src/model_operations.py",
    "src/tensor_ops.py",
    "src/utils.py",
    "Caches/FlexibleCache.py",
    "Caches/FlexibleCacheLayer.py",
    "Caches/CacheFactory.py",
    "Caches/frame_state.py",
    "Caches/offload_store.py",
    "Caches/pruning_pipeline.py",
    "Caches/reducers.py",
    "Caches/retrievers.py",
    "Caches/similarity_ops.py",
    "prompts/template_questions.txt",
)

PRIMING_PROMPT = EGOSTREAM_ROOT / "prompts" / "template_questions.txt"
UPSTREAM_CACHE_CONF = EGOSTREAM_ROOT.parent / "Configs_files" / "conf_sing_vid.json"

# --- cost model -------------------------------------------------------------------------
# The streaming backend ingests through Qwen3-VL's **image** path, which has no temporal
# merge, so a frame costs (res//32)^2 rather than the video path's (res//32)^2/2. Verified
# against the processor: 320x320 -> image_grid_thw [1,20,20] -> 400/4 = 100 tokens.
# `m4_eval_plan.jsonl`'s visual_tokens column is therefore stale by exactly 2x here and is
# recomputed by X2 rather than quoted.
def visual_tokens_per_frame(res: int = None) -> int:
    r = res or INPUT_RES
    return (r // PATCH_PX) ** 2


TOKENS_PER_FRAME = visual_tokens_per_frame()  # 100 at INPUT_RES=320

# --- cache budgets ----------------------------------------------------------------------
# The context-length axis. Under a bounded cache we cannot manipulate context by choosing
# frames — the budget is what pins it — so the sweep's "4x more visual tokens" manipulation
# is recreated here. Live cap = sink(716) + mutable + hat(2 frames = 200 at 320x320).
BUDGETS = {
    "small": 1024,
    "default": 8192,
    "large": 32768,
}
DEFAULT_BUDGET = "default"
SINK_TOKENS = 716        # the priming prompt, measured; asserted at load time
HAT_NUM_FRAMES = 2


def live_cap(budget_label: str = DEFAULT_BUDGET, res: int = None) -> int:
    return SINK_TOKENS + BUDGETS[budget_label] + HAT_NUM_FRAMES * visual_tokens_per_frame(res)


# --- which passes the schedule emits ----------------------------------------------------
# `main` nests: all five Δt checkpoints come off one ingestion (asserted in X2). `prior` does
# not — M4 draws each bin's evidence-free window from an independently chosen start — so each
# prior bin costs its own pass. The first sweep planned only the three cheap bins for that
# reason; measured per-pass cost then put the two expensive ones at ~1.2 and ~3.9 GPU-h, which
# is affordable, and they are what turns the evidence contrast from three points into the
# whole Δt grid. X3 resumes on `record_id`, so re-running `--kinds prior` scores only the new
# bins.
PRIOR_DTS = (60, 300, 900, 1800, 3600)
TOKMATCH_ARMS = ("tokmatch_evkeep", "tokmatch_uniform")
TOKMATCH_DTS = (3600,)
BUDGET_ARM_BUDGETS = ("small", "large")

# --- ingestion --------------------------------------------------------------------------
INGEST_BATCH_FRAMES = 16     # the paper's sliding-window stride
GRID_S = 5.0                 # 0.2 fps frame bank

# Disk offload is written for every layer (~4 GB per 60-min question) and read back by the
# ReKV retriever. It must never land on the shared NFS /home. Slurm jobs set EGOVM_OFFLOAD
# to node-local scratch; the default is only sane for an interactive debug session.
OFFLOAD_ROOT = Path(os.environ.get("EGOVM_OFFLOAD", "/tmp/egovm_kv_offload"))

# --- artifacts --------------------------------------------------------------------------
X0_REPORT = ARTIFACTS / "x0_backend_report.json"
X1_REPORT = ARTIFACTS / "x1_calibration.json"
X2_SCHEDULE = ARTIFACTS / "x2_stream_schedule.jsonl"
X2_REPORT = ARTIFACTS / "x2_schedule_report.md"
X3_SCHEMA_CHECK = ARTIFACTS / "x3_schema_check.json"
X4_PILOT_REPORT = ARTIFACTS / "x4_pilot_report.md"

STREAM_RECORD_PREFIX = "m6_records_stream_"


def records_path(tag: str) -> Path:
    return BIG_ARTIFACTS / f"{STREAM_RECORD_PREFIX}{tag}.jsonl"


MODEL_ID = str(DATA_ROOT / "model" / "Qwen3-VL-8B-Instruct")
