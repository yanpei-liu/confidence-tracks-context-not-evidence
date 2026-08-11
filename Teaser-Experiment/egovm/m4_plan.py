"""M4 — build the evaluation plan.

One row per (question, Δt, arm). Each row carries the exact global frame ids to feed, so M6
is a pure execution step and the plan can be audited without a GPU.

Window for question q at Δt:

    lo = stream(min evidence) - PRE_EVIDENCE_S
    ck = stream(max evidence) + Δt          <- the recall checkpoint
    frames = grid points in [lo, ck]

Arms:
  main               0.2 fps over the whole window. Tokens grow with Δt.
  tokmatch_uniform   window uniformly subsampled to the Δt=15 min frame count.
                     Holds tokens fixed; evidence gets sampled more sparsely.
  tokmatch_evkeep    same target count, but every frame up to (evidence_max + 60 s) is kept
                     and only the filler is thinned. Holds tokens *and* evidence resolution
                     fixed, so Δt is the only thing that moves.
  prior              the same question over a window that contains **no** evidence (drawn
                     >= 2 h away in stream time), matched to the Δt=5 min frame count. This
                     is the language-prior floor: what the model scores from the question and
                     options alone. Without it, "accuracy falls with Δt" is uninterpretable —
                     the drop only means forgetting if the Δt=5 min bin is meaningfully above
                     this floor, and reaching it at Δt=60 min means the evidence is fully gone.

The two token-matched arms bracket the confound from both sides: if accuracy at Δt=60 min
stays low in `tokmatch_evkeep` — same tokens, same evidence sampling — the drop cannot be
context length or evidence resolution.

Artifact: artifacts/m4_eval_plan.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import math

from .config import (
    ARTIFACTS, DT_GRID_S, INPUT_RES, MAX_EVIDENCE_SPAN_S, PRE_EVIDENCE_S, PRIOR_REF_DT_S, SAMPLE_FPS,
    TOKMATCH_REF_DT_S, visual_tokens_per_frame,
)
from .m2_stream_index import StreamIndex

GRID_S = 1.0 / SAMPLE_FPS
TOK_PER_FRAME = visual_tokens_per_frame()


def window_gids(lo_s: float, hi_s: float, n_frames_total: int) -> list[int]:
    g0 = max(0, math.ceil(lo_s / GRID_S))
    g1 = min(n_frames_total - 1, math.floor(hi_s / GRID_S))
    return list(range(g0, g1 + 1))


def uniform_subsample(gids: list[int], k: int) -> list[int]:
    """Evenly spaced k-subset, endpoints included."""
    if k >= len(gids):
        return list(gids)
    if k <= 1:
        return [gids[-1]]
    idx = [round(i * (len(gids) - 1) / (k - 1)) for i in range(k)]
    return [gids[i] for i in sorted(set(idx))]


def evkeep_subsample(gids: list[int], k: int, keep_until_g: int) -> list[int]:
    """Keep every frame with gid <= keep_until_g; thin the tail to reach k total."""
    head = [g for g in gids if g <= keep_until_g]
    tail = [g for g in gids if g > keep_until_g]
    if len(head) >= k:
        return uniform_subsample(head, k) if k < len(head) else head
    return head + uniform_subsample(tail, k - len(head))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-screen", action="store_true",
                    help="also plan HabitInsight (reported separately)")
    a = ap.parse_args()

    idx = StreamIndex.load()
    fi = json.loads((ARTIFACTS / "m3_frame_index.json").read_text())
    # gids are grid indices, not a dense 0..n-1 range (a few chunks are clipped short), so
    # the upper bound is the largest gid the bank actually produced
    n_frames_total = fi["max_gid"] + 1
    missing = set(fi.get("missing_gids", []))

    questions = [json.loads(l) for l in (ARTIFACTS / "m1_questions.jsonl").read_text().splitlines()]
    sel = [q for q in questions if q["keep"] or (a.include_screen and q["screen"])]

    rows, dropped = [], collections.Counter()
    frames_by_dt: dict[int, list[int]] = {}

    n_span_excluded = 0
    for q in sel:
        s_ev_min = idx.to_stream(q["evidence_min_abs_s"])
        s_ev_max = idx.to_stream(q["evidence_max_abs_s"])
        if s_ev_max - s_ev_min > MAX_EVIDENCE_SPAN_S:
            n_span_excluded += 1
            continue
        lo = max(0.0, s_ev_min - PRE_EVIDENCE_S)
        keep_until_g = math.floor((s_ev_max + 60.0) / GRID_S)

        per_dt = {}
        for dt in DT_GRID_S:
            ck = s_ev_max + dt
            if ck > idx.total_s:
                dropped[dt] += 1
                continue
            gids = [g for g in window_gids(lo, ck, n_frames_total) if g not in missing]
            if not gids:
                dropped[dt] += 1
                continue
            per_dt[dt] = (gids, ck)

        ref = per_dt.get(TOKMATCH_REF_DT_S)
        n_ref = len(ref[0]) if ref else None

        # language-prior floor: an evidence-free window of the same size as the Δt=5 min one
        prior_gids = None
        short = per_dt.get(PRIOR_REF_DT_S)
        if short:
            n_short = len(short[0])
            span_g = math.ceil(n_short * 1.0)
            ev_lo_g = math.floor((lo - 7200) / GRID_S)
            ev_hi_g = math.ceil((s_ev_max + 7200) / GRID_S)
            max_g = n_frames_total - 1
            starts = [g for g in range(0, max_g - span_g, max(span_g, 1))
                      if g + span_g < ev_lo_g or g > ev_hi_g]
            if starts:
                s0 = starts[(int(q["qid"]) * 7919) % len(starts)]
                cand = [g for g in range(s0, s0 + span_g) if g not in missing]
                if len(cand) >= max(8, n_short // 2):
                    prior_gids = cand[:n_short]

        for dt, (gids, ck) in per_dt.items():
            arms = {"main": gids}
            if n_ref is not None and dt > TOKMATCH_REF_DT_S and len(gids) > n_ref:
                arms["tokmatch_uniform"] = uniform_subsample(gids, n_ref)
                arms["tokmatch_evkeep"] = evkeep_subsample(gids, n_ref, keep_until_g)
            if dt == PRIOR_REF_DT_S and prior_gids:
                arms["prior"] = prior_gids
            for arm, fr in arms.items():
                rows.append({
                    "row_id": f"{q['qid']}|{dt}|{arm}",
                    "qid": q["qid"], "type": q["type"], "dt_s": dt, "arm": arm,
                    "answer": q["answer"], "need_audio": q["need_audio"],
                    "checkpoint_stream_s": round(ck, 3),
                    "window_lo_stream_s": round(lo, 3),
                    "evidence_max_stream_s": round(s_ev_max, 3),
                    "n_frames": len(fr),
                    "visual_tokens": int(round(len(fr) * TOK_PER_FRAME)),
                    "input_res": INPUT_RES,
                    "gids": fr,
                })
            frames_by_dt.setdefault(dt, []).append(len(gids))

    out = ARTIFACTS / "m4_eval_plan.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # ---- verification ----
    by = collections.Counter((r["dt_s"], r["arm"]) for r in rows)
    main_n = {dt: [r["n_frames"] for r in rows if r["dt_s"] == dt and r["arm"] == "main"]
              for dt in DT_GRID_S}
    for i in range(len(DT_GRID_S) - 1):
        lo_dt, hi_dt = DT_GRID_S[i], DT_GRID_S[i + 1]
        assert sum(main_n[hi_dt]) / len(main_n[hi_dt]) > sum(main_n[lo_dt]) / len(main_n[lo_dt]), \
            "main arm frame count must grow with Δt"
    for arm in ("tokmatch_uniform", "tokmatch_evkeep"):
        ns = {r["qid"]: r["n_frames"] for r in rows if r["arm"] == arm and r["dt_s"] == 3600}
        ref = {r["qid"]: r["n_frames"] for r in rows if r["arm"] == "main"
               and r["dt_s"] == TOKMATCH_REF_DT_S}
        bad = [k for k in ns if abs(ns[k] - ref[k]) > 1]
        assert not bad, f"{arm} not token-matched for {len(bad)} questions"
    assert all(len(set(r["gids"])) == len(r["gids"]) for r in rows), "duplicate gids"

    print(f"M4 OK: {len(rows)} rows over {len(sel)-n_span_excluded} questions "
          f"({n_span_excluded} excluded: evidence span > {MAX_EVIDENCE_SPAN_S:.0f}s) -> {out}")
    for dt in DT_GRID_S:
        ns = main_n[dt]
        print(f"  Δt={dt//60:3d}min  main n={len(ns):4d}  mean_frames={sum(ns)/len(ns):7.1f}  "
              f"mean_vis_tokens={sum(ns)/len(ns)*TOK_PER_FRAME:8.0f}  dropped={dropped[dt]}")
    for k, v in sorted(by.items()):
        if k[1] != "main":
            print(f"  Δt={k[0]//60:3d}min  {k[1]:18s} n={v}")
    tot = sum(r["n_frames"] for r in rows)
    print(f"  total frame-encodes: {tot:,}  (main only: "
          f"{sum(r['n_frames'] for r in rows if r['arm']=='main'):,})")


if __name__ == "__main__":
    main()
