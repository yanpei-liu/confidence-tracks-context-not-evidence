"""Shared paths and experiment constants.

Data lives outside the repo: EgoLife is S-Lab License 1.0 (research, non-redistributable),
so no dataset content is committed.
"""
from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("EGOVM_DATA", "~/Files/EgoVM-data"))
RAW = DATA_ROOT / "raw" / "EgoLife"
FRAMES = DATA_ROOT / "frames"
BIG_ARTIFACTS = DATA_ROOT / "artifacts"

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"
FIGURES = REPO / "figures"

PARTICIPANT = "A1_JAKE"
QA_JSON = RAW / "EgoLifeQA" / "EgoLifeQA_A1_JAKE.json"
VIDEO_DIR = RAW / PARTICIPANT
CAPTION_DIR = RAW / "EgoLifeCap" / "DenseCaption" / PARTICIPANT

# --- experiment design -------------------------------------------------------
# Mitigation 1: time-invariant categories only (see PLAN.md).
KEEP_TYPES = ("EntityLog", "EventRecall", "RelationMap")
SCREEN_TYPES = ("HabitInsight",)  # scored, reported separately, not in the headline set
DROP_TYPES = ("TaskMaster",)  # prospective / state-dependent

DT_GRID_S = (300, 900, 1800, 3600)  # 5 / 15 / 30 / 60 min
TOKMATCH_REF_DT_S = 900  # token-matched arm copies the Δt=15 min frame count
PRE_EVIDENCE_S = 60.0  # lead-in before the earliest evidence point
SAMPLE_FPS = 0.2  # one frame per 5 s of stream

CHUNK_NOMINAL_S = 30.0  # each A1 mp4 is exactly 30.000 s (verified by ffprobe)
# Source video is 1408x1408 @ 20 fps. EgoLife timestamps are HHMMSSFF with FF a frame
# index; the observed FF distribution has its mass on 0..19, i.e. a 20 fps base. A few
# annotations carry FF > 19 (up to 92) — parser clamps those to < 1 s. Sub-second error
# is irrelevant against a Δt grid measured in minutes.
SRC_FPS = 20.0

# Frame bank geometry. Source is square (1408x1408 fisheye), so the bank is square too.
# Qwen3-VL costs (T/2)*(H/32)*(W/32) visual tokens => 448/32 = 14 -> 196 tokens per
# 2-frame group = 98 tokens/frame. A 60 min window at 0.2 fps is ~770 frames ~ 38k
# visual tokens, comfortably inside the 262k context.
FRAME_W = 448
FRAME_H = 448
JPEG_QUALITY = 88

MODEL_HUB_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_ID = str(DATA_ROOT / "model" / "Qwen3-VL-8B-Instruct")  # local snapshot, no hub at job time
OPTION_LETTERS = ("A", "B", "C", "D")

# Resolution actually fed to the model, independent of the frame bank on disk. Qwen3-VL's
# video processor otherwise smart-resizes to a *total* pixel budget (25,165,824 px), which
# would silently shrink frames as the window grows — i.e. it would make resolution a function
# of Δt, which is precisely the confound the design exists to remove. So this is pinned and
# the processor's rescaling is disabled.
INPUT_RES = 320
PATCH_PX = 32  # patch_size 16 * merge_size 2
TEMPORAL_PATCH = 2


def visual_tokens_per_frame(res: int = None) -> float:
    r = res or INPUT_RES
    return (r // PATCH_PX) ** 2 / TEMPORAL_PATCH

for _d in (BIG_ARTIFACTS, FRAMES, ARTIFACTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)
