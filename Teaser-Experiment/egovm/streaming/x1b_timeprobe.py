"""X1b — can the streaming backend perceive elapsed time at all?

X1 check (6) failed in a way that matters. With the *frames held fixed* (120 real frames) and
only the injected clock varied over 60 / 600 / 3600 s, the model answered "about 10 minutes"
every time — identical to the markers-off control. The markers are demonstrably in the token
stream (decoded: `<0 seconds><|vision_start|><|image_pad|><|vision_end|><300 seconds>…`), so
the injection works mechanically and the model is not using it.

That is a design-critical result rather than a bug, because the teaser's headline finding is
that **confidence moves on elapsed time**. If elapsed time is not an input in this regime, the
streaming condition cannot be compared like-for-like with the full-reprocess sweep on that axis.

This module separates three explanations, which the failed check cannot:

  **(1) Eviction.** 120 frames × 100 tokens = 12,000 presented against a 9,108-token cap, so
  most of the clock is evicted before the query. Testing at 32 frames (3,916 live tokens,
  nothing evicted) isolates this. If readout works unevicted and fails evicted, "the cache
  evicts the clock" is a mechanism — and an interesting one for this experiment.

  **(2) Marker format.** `<600 seconds>` may simply not be a form this model reads. Three
  formats are tried, including the wall-clock form the video path emits.

  **(3) Integration.** "How much time has elapsed" needs the model to combine the first and
  last markers. A pure *readout* ("what is the largest timestamp shown") needs neither, so
  failing both localises the problem to reading rather than reasoning.

Artifact: artifacts/x1b_timeprobe.json
Run: python -m egovm.streaming.x1b_timeprobe
"""
from __future__ import annotations

import json
import subprocess
import time

import torch
from PIL import Image

from ..m3_frames import frame_path
from . import sconf
from .backend import StreamingScorer

# format name -> callable(seconds) -> marker text
FORMATS = {
    "angle_seconds": lambda t: f"<{t:.0f} seconds>",           # what X1 used
    "bracket_clock": lambda t: f"[t={int(t)//60:02d}:{int(t)%60:02d}]",
    "prose": lambda t: f"(time {t/60:.1f} min into the stream)",
}

READOUT_Q = ("What is the largest timestamp shown anywhere in the video stream you have "
             "observed?")
ELAPSED_Q = ("Based on the timestamps in the video stream you have observed, approximately how "
             "much video time has elapsed since the stream began?")
CHOICES = {"a": "About 1 minute", "b": "About 10 minutes",
           "c": "About 1 hour", "d": "About 4 hours"}
SPANS = [(60.0, "A"), (600.0, "B"), (3600.0, "C")]


def bank(start_gid: int, n: int):
    out = []
    for g in range(start_gid, start_gid + n):
        with Image.open(frame_path(g)) as im:
            out.append(im.convert("RGB").copy())
    return out


def run_condition(sc, n_frames, span, fmt, question, timestamps=True):
    sc.marker_fmt = fmt or (lambda t: f"<{t:.0f} seconds>")
    cache = sc.new_cache("default", offload_dir=sconf.OFFLOAD_ROOT / "x1b")
    try:
        for i in range(0, n_frames, 16):
            k = min(16, n_frames - i)
            times = [(i + j) * span / max(n_frames - 1, 1) for j in range(k)]
            sc.ingest(cache, bank(3000 + i, k), times, timestamps=timestamps)
        live = cache.get_seq_length()
        r = sc.query(cache, question, CHOICES)
        r["live_tokens"] = live
        return r
    finally:
        StreamingScorer.release(cache)
        del cache
        torch.cuda.empty_cache()


def main() -> None:
    sc = StreamingScorer.load()
    cap = sconf.live_cap("default")
    rep = {
        "cap": cap,
        "host": subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip(),
        "note": "markers verified present in the token stream; this asks whether they are used",
        "conditions": [],
    }

    t0 = time.time()
    # 32 frames -> 3,200 presented tokens, under the 9,108 cap: nothing evicted.
    # 120 frames -> 12,000 presented: the head of the clock is evicted.
    for n_frames in (32, 120):
        for q_name, question in (("readout", READOUT_Q), ("elapsed", ELAPSED_Q)):
            for fmt_name, fmt in FORMATS.items():
                preds = {}
                for span, gold in SPANS:
                    r = run_condition(sc, n_frames, span, fmt, question)
                    preds[int(span)] = {"gold": gold, "pred": r["pred"],
                                        "p_gold": r["letter_probs"][gold],
                                        "confidence": r["confidence"],
                                        "live_tokens": r["live_tokens"]}
                pl = [v["pred"] for v in preds.values()]
                n_ok = sum(v["pred"] == v["gold"] for v in preds.values())
                rep["conditions"].append({
                    "n_frames": n_frames, "question": q_name, "format": fmt_name,
                    "presented_tokens": n_frames * sconf.TOKENS_PER_FRAME,
                    "evicted": n_frames * sconf.TOKENS_PER_FRAME > cap,
                    "preds": pl, "n_distinct": len(set(pl)), "n_correct": n_ok,
                    "detail": preds,
                })
                print(f"  n={n_frames:3d} {q_name:8s} {fmt_name:14s} preds={pl} "
                      f"correct={n_ok}/3", flush=True)

    # markers-off control at both sizes
    for n_frames in (32, 120):
        preds = {}
        for span, gold in SPANS:
            r = run_condition(sc, n_frames, span, None, ELAPSED_Q, timestamps=False)
            preds[int(span)] = {"gold": gold, "pred": r["pred"],
                                "p_gold": r["letter_probs"][gold]}
        pl = [v["pred"] for v in preds.values()]
        rep["conditions"].append({
            "n_frames": n_frames, "question": "elapsed", "format": "MARKERS_OFF",
            "presented_tokens": n_frames * sconf.TOKENS_PER_FRAME,
            "evicted": n_frames * sconf.TOKENS_PER_FRAME > cap,
            "preds": pl, "n_distinct": len(set(pl)),
            "n_correct": sum(v["pred"] == v["gold"] for v in preds.values()),
            "detail": preds})
        print(f"  n={n_frames:3d} elapsed  MARKERS_OFF    preds={pl}", flush=True)

    best = max(rep["conditions"], key=lambda c: (c["n_distinct"], c["n_correct"]))
    rep["best_condition"] = {k: best[k] for k in
                             ("n_frames", "question", "format", "n_distinct", "n_correct")}
    rep["any_condition_tracks_clock"] = best["n_distinct"] >= 2 and best["n_correct"] >= 2
    rep["elapsed_s"] = round(time.time() - t0, 1)

    out = sconf.X1_REPORT.with_name("x1b_timeprobe.json")
    out.write_text(json.dumps(rep, indent=2))
    print(f"\nany condition tracks the injected clock: {rep['any_condition_tracks_clock']}")
    print(f"best: {rep['best_condition']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
