"""X4 — the pilot gate. Nothing downstream runs until this passes.

Three pre-registered checks on a ~20-question streaming run. Each catches a distinct way the
driver can be wrong while still producing a well-formed records file:

  **(a) The confidences are not degenerate.** The letter softmax saturating at 1.0 on every row
  would make every calibration quantity meaningless. The X1 colour probe *does* hit 1.0, but on
  a trivially easy synthetic item; real EgoLifeQA items must spread.

  **(b) `live_tokens` saturates where X2 predicts.** The cache cap is the streaming regime's
  context-length variable. If it does not bind where the frame counts say it should, the budget
  arm (X6) is measuring nothing.

  **(c) Δt = 1 min agrees with the full-reprocess sweep, paired on question.** This is the
  sharpest of the three and the reason the pilot is cheap. At Δt = 1 min a window is ~13 frames
  ≈ 1.3k visual tokens against a 9,108-token cap: **nothing is evicted**, so the streaming cache
  holds the whole window and the two regimes see the same frames through the same model. They
  should agree closely. They will not agree exactly — the image path replaces the video path,
  the priming prompt replaces our system prompt — but a large gap at the bin where the cache is
  not binding is a driver bug, not a finding, and this is the cheapest place to find it.

Artifact: artifacts/x4_pilot_report.md (+ .json)
Run: python -m egovm.streaming.x4_pilot --records <pilot file>
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st

from ..config import BIG_ARTIFACTS
from . import sconf

# pre-registered thresholds — set before the pilot ran
GATE_MAX_MEAN_CONF = 0.99      # (a) mean max-prob must be below this
GATE_MIN_CONF_SPREAD = 0.02    # (a) and the s.d. must be above this
GATE_MAX_ACC_GAP = 0.15        # (c) |paired Δaccuracy| at Δt=1 min
GATE_MIN_ACC = 0.25            # (c) and streaming must clear 4-way chance


def load(paths) -> list[dict]:
    out, seen = [], set()
    for p in paths:
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if r["record_id"] in seen:
                continue
            seen.add(r["record_id"])
            out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(sconf.records_path("pilot")))
    ap.add_argument("--full-reprocess", default="",
                    help="glob for the committed Qwen3-VL records (default: $DATA/artifacts)")
    a = ap.parse_args()

    stream = [r for r in load([a.records]) if r["question_kind"] == "memory"]
    if not stream:
        raise SystemExit(f"no streaming memory records in {a.records}")

    fr_paths = sorted(glob.glob(a.full_reprocess)) if a.full_reprocess else [
        p for p in sorted(glob.glob(str(BIG_ARTIFACTS / "m6_records*.jsonl")))
        if sconf.STREAM_RECORD_PREFIX not in p
    ]
    full = [r for r in load(fr_paths) if r["question_kind"] == "memory"]

    rep: dict = {"records": a.records, "n_stream_records": len(stream),
                 "n_full_reprocess_records": len(full),
                 "questions": sorted({r["qid"] for r in stream}, key=int)}
    gates = []

    # ---- (a) confidences not degenerate ----
    confs = [r["confidence"] for r in stream]
    mean_c = st.mean(confs)
    sd_c = st.pstdev(confs)
    frac_1 = sum(c > 0.999 for c in confs) / len(confs)
    a_ok = mean_c < GATE_MAX_MEAN_CONF and sd_c > GATE_MIN_CONF_SPREAD
    gates.append({"gate": "confidence_not_degenerate", "ok": a_ok,
                  "mean_confidence": round(mean_c, 4), "sd": round(sd_c, 4),
                  "frac_above_0.999": round(frac_1, 4),
                  "threshold": f"mean < {GATE_MAX_MEAN_CONF}, sd > {GATE_MIN_CONF_SPREAD}"})

    # ---- (b) live_tokens saturates where predicted ----
    caps = {b: sconf.live_cap(b) for b in sconf.BUDGETS}
    by_dt: dict[int, list[dict]] = {}
    for r in stream:
        by_dt.setdefault(r["dt_s"], []).append(r)
    sat = {}
    for dt in sorted(by_dt):
        rs = by_dt[dt]
        cap = caps[rs[0]["budget_label"]]
        live = [r["live_tokens"] for r in rs]
        presented = [r["n_frames"] * sconf.TOKENS_PER_FRAME for r in rs]
        sat[dt] = {
            "n": len(rs),
            "mean_presented_tokens": round(st.mean(presented)),
            "mean_live_tokens": round(st.mean(live)),
            "cap": cap,
            "frac_at_cap": round(sum(x >= cap - 200 for x in live) / len(live), 3),
            "any_over_cap": max(live) > cap + 16 * sconf.TOKENS_PER_FRAME,
        }
    # below the cap live should track presented; above it, live should pin to the cap
    unsat = [d for d in sat if sat[d]["mean_presented_tokens"] < caps["default"] * 0.8]
    satd = [d for d in sat if sat[d]["mean_presented_tokens"] > caps["default"] * 1.5]
    b_ok = (not any(v["any_over_cap"] for v in sat.values())
            and all(sat[d]["frac_at_cap"] < 0.5 for d in unsat)
            and all(sat[d]["frac_at_cap"] > 0.5 for d in satd))
    gates.append({"gate": "cache_saturates_as_predicted", "ok": b_ok, "by_dt": sat,
                  "unsaturated_bins": unsat, "saturated_bins": satd})

    # ---- (c) Δt=1 min agrees with full-reprocess, paired ----
    DT1 = 60
    s1 = {r["qid"]: r for r in stream if r["dt_s"] == DT1 and r["arm"] == "main"}
    f1 = {r["qid"]: r for r in full if r["dt_s"] == DT1 and r["arm"] == "main"}
    shared = sorted(set(s1) & set(f1), key=int)
    if shared:
        acc_s = sum(s1[q]["correct"] for q in shared) / len(shared)
        acc_f = sum(f1[q]["correct"] for q in shared) / len(shared)
        conf_s = st.mean([s1[q]["confidence"] for q in shared])
        conf_f = st.mean([f1[q]["confidence"] for q in shared])
        agree = sum(s1[q]["pred"] == f1[q]["pred"] for q in shared) / len(shared)
        c_ok = abs(acc_s - acc_f) <= GATE_MAX_ACC_GAP and acc_s > GATE_MIN_ACC
        gates.append({
            "gate": "dt1_agrees_with_full_reprocess", "ok": c_ok, "n_paired": len(shared),
            "streaming_accuracy": round(acc_s, 4), "full_reprocess_accuracy": round(acc_f, 4),
            "delta_accuracy": round(acc_s - acc_f, 4),
            "streaming_confidence": round(conf_s, 4),
            "full_reprocess_confidence": round(conf_f, 4),
            "answer_agreement_rate": round(agree, 4),
            "threshold": f"|Δacc| <= {GATE_MAX_ACC_GAP} and acc > {GATE_MIN_ACC}",
            "note": "at Δt=1 min the window is far below the cache cap, so nothing is evicted "
                    "and the two regimes see the same frames",
        })
    else:
        gates.append({"gate": "dt1_agrees_with_full_reprocess", "ok": False,
                      "error": "no questions shared with the full-reprocess records"})

    rep["gates"] = gates
    rep["ok"] = all(g["ok"] for g in gates)

    # per-bin summary, useful even when a gate fails
    rep["by_dt"] = {}
    for dt in sorted(by_dt):
        rs = by_dt[dt]
        rep["by_dt"][dt] = {
            "n": len(rs),
            "accuracy": round(sum(r["correct"] for r in rs) / len(rs), 4),
            "confidence": round(st.mean([r["confidence"] for r in rs]), 4),
            "mean_live_tokens": round(st.mean([r["live_tokens"] for r in rs])),
            "mean_frames": round(st.mean([r["n_frames"] for r in rs])),
            "mean_offloaded_tokens": round(st.mean([r.get("offloaded_tokens", 0) for r in rs])),
        }

    (sconf.X4_PILOT_REPORT.with_suffix(".json")).write_text(json.dumps(rep, indent=2))

    L = ["# X4 — streaming pilot gate", "",
         f"**{'PASS' if rep['ok'] else 'FAIL'}** — {len(rep['questions'])} questions, "
         f"{len(stream)} memory records from `{a.records}`.", "",
         "| gate | result | key numbers |", "| --- | --- | --- |"]
    for g in gates:
        nums = ", ".join(f"{k}={v}" for k, v in g.items()
                         if k not in ("gate", "ok", "by_dt", "note", "threshold",
                                      "unsaturated_bins", "saturated_bins"))
        L.append(f"| `{g['gate']}` | {'PASS' if g['ok'] else 'FAIL'} | {nums[:300]} |")
    L += ["", "## Per-Δt", "",
          "| Δt | n | accuracy | confidence | frames | presented tokens | live tokens |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for dt, v in rep["by_dt"].items():
        L.append(f"| {dt//60} min | {v['n']} | {v['accuracy']:.3f} | {v['confidence']:.3f} | "
                 f"{v['mean_frames']} | {v['mean_frames']*sconf.TOKENS_PER_FRAME:,} | "
                 f"{v['mean_live_tokens']:,} |")
    L += ["", f"Cache cap at the default budget: **{caps['default']:,}** live tokens "
              f"(sink {sconf.SINK_TOKENS} + mutable {sconf.BUDGETS['default']:,} + hat "
              f"{sconf.HAT_NUM_FRAMES * sconf.TOKENS_PER_FRAME}).", ""]
    sconf.X4_PILOT_REPORT.write_text("\n".join(L))

    print("\n".join(L))
    print(f"\nX4 {'PASS' if rep['ok'] else 'FAIL'} -> {sconf.X4_PILOT_REPORT}")
    raise SystemExit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
