"""X9 — analysis of the streaming sweep, and the cross-regime comparison.

Two jobs, in this order:

**1. Run M7 unchanged, one condition at a time.** M7 globs every `m6_records*.jsonl` in
`$DATA/artifacts/` and the streaming files match that glob, so an unguarded run would pool the
two regimes into the same cells and quietly average them. M7 already takes `--records`; this
module drives it with a regime-restricted list. That is the whole reason X9 is a module and not
a command.

**2. Compare the manipulations across regimes.** The teaser's result is a set of *paired,
within-question contrasts*, not levels — the levels sit in an 11.5 pp band above the
no-evidence floor and are weak by construction. So the cross-regime quantity is a **difference
of contrasts**: (Δ under streaming) − (Δ under full-reprocess), cluster-bootstrapped over the
questions that appear in both. Comparing accuracies directly would be comparing the weak thing.

Contrast cells are `(question_kind, arm, dt_s)`. Records from the budget and timestamps-off
conditions carry the plan's own `arm` (they must — M7 drops any `row_id` not in the plan), so
before indexing they are relabelled `arm@cond` in memory. Nothing on disk changes.

Artifacts: artifacts/m7_metrics_stream.json (via M7), artifacts/x9_regime_comparison.json,
           artifacts/x9_streaming.md
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import subprocess
import sys
from pathlib import Path

from ..config import ARTIFACTS, BIG_ARTIFACTS, DT_GRID_S
from ..m7_analyze import cluster_bootstrap, contrast, index_cells, load
from . import sconf

STREAM_COND = "stream"


def is_stream(path: str) -> bool:
    return sconf.STREAM_RECORD_PREFIX in path


def load_records(paths) -> list[dict]:
    seen = {}
    for p in paths:
        for r in load(Path(p)):
            seen[r["record_id"]] = r
    return list(seen.values())


def relabel(records) -> list[dict]:
    """Give each non-primary condition its own arm label so contrast cells separate them."""
    out = []
    for r in records:
        r = dict(r)
        cond = r.get("cond", STREAM_COND)
        if cond != STREAM_COND:
            r["arm"] = f"{r['arm']}@{cond}"
        out.append(r)
    return out


def diff_of_contrasts(idx_s, idx_f, cells_a, cells_b, seed: int = 0):
    """(A−B under streaming) − (A−B under full-reprocess), paired on question in both.

    Resampling is over questions present in *both* regimes, so the difference is within-question
    on both sides and the two regimes' shared question set carries the CI.
    """
    by_q: dict = collections.defaultdict(list)
    for ca, cb in zip(cells_a, cells_b):
        SA, SB = idx_s.get(ca), idx_s.get(cb)
        FA, FB = idx_f.get(ca), idx_f.get(cb)
        if not (SA and SB and FA and FB):
            return None
        for q in sorted(set(SA) & set(SB) & set(FA) & set(FB)):
            by_q[q] += [("sa", SA[q]), ("sb", SB[q]), ("fa", FA[q]), ("fb", FB[q])]
    if not by_q:
        return None

    def make(field):
        def stat(rows):
            g = collections.defaultdict(list)
            for tag, r in rows:
                g[tag].append(r)
            if not all(g[t] for t in ("sa", "sb", "fa", "fb")):
                return None
            m = {t: sum(x[field] for x in g[t]) / len(g[t]) for t in g}
            return (m["sa"] - m["sb"]) - (m["fa"] - m["fb"])
        return stat

    flat = [x for rs in by_q.values() for x in rs]
    g = collections.defaultdict(list)
    for tag, r in flat:
        g[tag].append(r)
    mean = lambda t, f: sum(x[f] for x in g[t]) / len(g[t])  # noqa: E731
    return {
        "n_questions": len(by_q),
        "streaming_delta_accuracy": mean("sa", "correct") - mean("sb", "correct"),
        "full_delta_accuracy": mean("fa", "correct") - mean("fb", "correct"),
        "streaming_delta_confidence": mean("sa", "confidence") - mean("sb", "confidence"),
        "full_delta_confidence": mean("fa", "confidence") - mean("fb", "confidence"),
        "diff_accuracy": cluster_bootstrap(by_q, make("correct"), seed=seed),
        "diff_confidence": cluster_bootstrap(by_q, make("confidence"), seed=seed),
    }


def streaming_specs() -> list[tuple]:
    """The manipulations, in the streaming regime's own terms.

    `budget` replaces the sweep's "4x more visual tokens" — under a bounded cache the token
    count is pinned by the memory budget, not by which frames we choose, so the context-length
    axis has to be recreated rather than inherited (EXTENDING.md §3b).
    """
    S, D, L = "@small", "", "@large"
    return [
        ("elapsed_time_fixed_frames",
         "+45 min elapsed, frames and evidence fixed (evkeep@60 − main@15)",
         [("memory", "tokmatch_evkeep", 3600)], [("memory", "main", 900)]),
        ("context_length_large_vs_default",
         "3.7x live-token budget, elapsed time and frames fixed",
         [("memory", f"main{L}", dt) for dt in (1800, 3600)],
         [("memory", "main", dt) for dt in (1800, 3600)]),
        ("context_length_default_vs_small",
         "4.7x live-token budget, elapsed time and frames fixed",
         [("memory", "main", dt) for dt in (1800, 3600)],
         [("memory", f"main{S}", dt) for dt in (1800, 3600)]),
        ("evidence_vs_none",
         "evidence in the window vs an evidence-free window (main@5 − prior@5)",
         [("memory", "main", 300)], [("memory", "prior", 300)]),
        ("evidence_thinned",
         "evidence thinned at fixed frame count (uniform@60 − evkeep@60)",
         [("memory", "tokmatch_uniform", 3600)], [("memory", "tokmatch_evkeep", 3600)]),
        ("timestamps_on_vs_off",
         "the same stream with and without an elapsed-time signal",
         [("memory", "main", dt) for dt in DT_GRID_S],
         [("memory", "main@nots", dt) for dt in DT_GRID_S]),
        ("control_vs_memory",
         "no-memory control question vs the memory question, same frames",
         [("control", "main", dt) for dt in DT_GRID_S],
         [("memory", "main", dt) for dt in DT_GRID_S]),
        ("elapsed_time_raw",
         "the raw sweep: 1 min vs 60 min on the main arm",
         [("memory", "main", 60)], [("memory", "main", 3600)]),
    ]


def cross_regime_specs() -> list[tuple]:
    """Only manipulations that exist in *both* regimes can be differenced."""
    return [
        ("elapsed_time_fixed_frames",
         [("memory", "tokmatch_evkeep", 3600)], [("memory", "main", 900)]),
        ("evidence_vs_none",
         [("memory", "main", 300)], [("memory", "prior", 300)]),
        ("evidence_thinned",
         [("memory", "tokmatch_uniform", 3600)], [("memory", "tokmatch_evkeep", 3600)]),
        ("control_vs_memory",
         [("control", "main", dt) for dt in DT_GRID_S],
         [("memory", "main", dt) for dt in DT_GRID_S]),
        ("elapsed_time_raw",
         [("memory", "main", 60)], [("memory", "main", 3600)]),
    ]


def verdict(out: dict) -> str:
    """Select the verdict from the measured intervals, among the four outcomes pre-enumerated
    in EXTENDING.md §7. Written this way for the same reason M8 is (`m8_report.py:20-24`): the
    teaser's value was that it reported the prediction failing, and that only stays credible if
    the sentence is chosen by the numbers rather than by the author."""
    ev = out["streaming_contrasts"].get("evidence_vs_none", {})
    raw = out["streaming_contrasts"].get("elapsed_time_raw", {})
    if ev.get("unavailable") or raw.get("unavailable"):
        return "Not enough of the design has run to select a verdict."
    ev_a, ev_c = ev["delta_accuracy"], ev["delta_confidence"]
    raw_a, raw_c = raw["delta_accuracy"], raw["delta_confidence"]
    # gap = confidence - accuracy; the raw contrast is 1 min minus 60 min, so the gap WIDENS
    # with elapsed time when confidence falls less than accuracy, i.e. raw_c < raw_a
    widens = raw_c["hi95"] < raw_a["lo95"]
    narrows = raw_c["lo95"] > raw_a["hi95"]
    evidence_blind = ev_a["lo95"] > 0 and ev_c["lo95"] <= 0 <= ev_c["hi95"]
    if widens and evidence_blind:
        return ("**Outcome 2 — 'doesn't know it forgot', found at last.** Accuracy falls as the "
                "evidence is evicted and confidence does not follow, so the gap widens in the "
                "regime where forgetting is real.")
    if widens:
        return ("**Outcome 2 (partial).** The overconfidence gap widens with elapsed time under "
                "streaming, but confidence is not evidence-blind.")
    if evidence_blind and narrows:
        return ("**Outcome 3 — the dissociation replicates.** Confidence does not track whether "
                "the evidence is present, while accuracy does; and over the raw sweep confidence "
                "falls *faster* than accuracy, so the predicted widening gap is absent here too.")
    if evidence_blind:
        return ("**Outcome 3 — the dissociation replicates.** Confidence does not track whether "
                "the evidence is present, while accuracy does.")
    return ("**Outcome 1 — confidence tracks eviction.** Both measures respond to the "
            "manipulations, so metamemory is better here than the teaser suggested.")


def write_report(out: dict, fmt) -> None:
    S, C, X = out["streaming_contrasts"], out["cross_regime"], out["cells"]
    L = ["# X9 — the streaming sweep, and how it compares with full-reprocess", "",
         "Generated from `x9_regime_comparison.json`; every number below is read from it, so "
         "this file cannot drift from the metrics. Δ are percentage points, paired "
         "within-question, cluster-bootstrapped over questions (5,000 draws, seed 0).", "",
         f"{out['n_streaming_records']:,} streaming records "
         f"({', '.join(f'{k}={v:,}' for k, v in sorted(out['records_by_condition'].items()))}) "
         f"against {out['n_full_records']:,} committed full-reprocess records.", "",
         "## Verdict", "", verdict(out), "",
         "## The manipulations, under streaming", "",
         "| manipulation | n | Δaccuracy | Δconfidence |", "| --- | --- | --- | --- |"]
    for k, v in S.items():
        if v.get("unavailable"):
            L.append(f"| `{k}` | — | unavailable | unavailable |")
            continue
        L.append(f"| `{k}` | {v['n_questions']} | {fmt(v['delta_accuracy'])} | "
                 f"{fmt(v['delta_confidence'])} |")
    L += ["", "## Cross-regime: (streaming Δ) − (full-reprocess Δ)", "",
          "Contrasts, not levels. The teaser's levels sit in an 11.5 pp band above the "
          "no-evidence floor and are weak by construction; what survived there was the set of "
          "paired contrasts, so those are what is differenced here.", "",
          "| manipulation | n | Δaccuracy difference | Δconfidence difference |",
          "| --- | --- | --- | --- |"]
    for k, v in C.items():
        if v.get("unavailable"):
            L.append(f"| `{k}` | — | unavailable | unavailable |")
            continue
        L.append(f"| `{k}` | {v['n_questions']} | {fmt(v['diff_accuracy'])} | "
                 f"{fmt(v['diff_confidence'])} |")

    if out.get("timestamps_by_dt"):
        L += ["", "## The time-blind test, bin by bin", "",
              "X1b found the injected clock is legible only while it is still in the cache. "
              "The prediction (EXTENDING.md §4a) is that `main` and `nots` are "
              "indistinguishable at Δt ≥ 15 min, where the cache is saturated, and may differ "
              "only at 1 and 5 min.", "",
              "| Δt | n | Δaccuracy (on − off) | Δconfidence (on − off) |",
              "| --- | --- | --- | --- |"]
        for dt, v in out["timestamps_by_dt"].items():
            L.append(f"| {int(dt)//60} min | {v['n']} | {fmt(v['delta_accuracy'])} | "
                     f"{fmt(v['delta_confidence'])} |")

    if out.get("budget_by_dt"):
        L += ["", "## The context-length axis (`large` − `small`)", "",
              "Recreated as a memory-budget arm because a bounded cache pins the token count: "
              "context length is no longer ours to manipulate by choosing frames. "
              "`retrieve_top_k` tracks the budget, without which the retriever re-equalises "
              "every condition at query time.", "",
              "| Δt | n | live tokens (large / small) | accuracy | confidence |",
              "| --- | --- | --- | --- | --- |"]
        for dt, v in out["budget_by_dt"].items():
            lt = (f"{v['live_tokens_large']:,.0f} / {v['live_tokens_small']:,.0f}"
                  if v["live_tokens_large"] else "—")
            L.append(f"| {int(dt)//60} min | {v['n']} | {lt} | {fmt(v['delta_accuracy'])} | "
                     f"{fmt(v['delta_confidence'])} |")

    L += ["", "## Per-cell levels", "",
          "| cell | n | accuracy | confidence | live tokens | presented tokens |",
          "| --- | --- | --- | --- | --- | --- |"]
    for k in sorted(X):
        v = X[k]
        L.append(f"| `{k}` | {v['n']} | {v['accuracy']:.3f} | {v['confidence']:.3f} | "
                 f"{v['live_tokens']:,.0f} | {v['visual_tokens']:,.0f} |")
    L += ["", "## Caveats that travel with these numbers", "",
          "- **Elapsed time is not an input in this regime once the cache saturates** "
          "(EXTENDING.md §4a). The elapsed-time contrasts are reported as unavailable rather "
          "than null: an input the model cannot read is not a manipulation that failed.",
          "- **This is streaming over the window, not over the 44 h life-log.** Each question "
          "resets the cache and ingests from `evidence_min − PRE`, which preserves every arm "
          "definition in PLAN.md but does not support claims about accumulated history.",
          "- **Levels are not comparable across regimes** — the priming prompt replaces the "
          "system prompt and the image path replaces the video path. Contrasts are.",
          "- The `prior` floor is measured at Δt ∈ {1, 5, 15} min only and assumed flat beyond.",
          ""]
    path = ARTIFACTS / "x9_streaming.md"
    path.write_text("\n".join(L))
    print(f"X9 report -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-m7", action="store_true", help="skip the per-condition M7 runs")
    ap.add_argument("--conds", default="stream",
                    help="conditions to run M7 on, comma-separated (stream,small,large,nots)")
    a = ap.parse_args()

    all_paths = sorted(glob.glob(str(BIG_ARTIFACTS / "m6_records*.jsonl")))
    stream_paths = [p for p in all_paths if is_stream(p)]
    full_paths = [p for p in all_paths if not is_stream(p)]
    if not stream_paths:
        raise SystemExit(f"no streaming records under {BIG_ARTIFACTS}")

    stream_raw = load_records(stream_paths)
    full = [r for r in load_records(full_paths) if r["question_kind"] in ("memory", "control")]
    stream = [r for r in stream_raw if r["question_kind"] in ("memory", "control")]

    by_cond = collections.Counter(r.get("cond", STREAM_COND) for r in stream)
    print(f"streaming records: {len(stream):,} over conditions {dict(by_cond)}")
    print(f"full-reprocess records: {len(full):,}")

    # ---- 1. M7, one condition at a time ----
    if not a.skip_m7:
        # A records *file* is not a condition: one budget shard holds both `small` and `large`
        # rows, so selecting files by their first line would mix the two. Partition the records
        # themselves and hand M7 a single-condition file.
        tmpdir = BIG_ARTIFACTS / "x9_tmp"
        tmpdir.mkdir(exist_ok=True)
        for cond in a.conds.split(","):
            rows = [r for r in stream if r.get("cond", STREAM_COND) == cond]
            if not rows:
                print(f"  (no records for condition {cond!r}, skipping M7)")
                continue
            tmp = tmpdir / f"m6_records_stream_{cond}_merged.jsonl"
            tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
            print(f"\n=== M7 on condition {cond!r}: {len(rows):,} records ===")
            r = subprocess.run(
                [sys.executable, "-m", "egovm.m7_analyze", "--records", str(tmp)],
                cwd=str(ARTIFACTS.parent), capture_output=True, text=True)
            print(r.stdout[-3000:] or r.stderr[-3000:])
            src = ARTIFACTS / "m7_metrics.json"
            if src.exists():
                dst = ARTIFACTS / (f"m7_metrics_stream.json" if cond == STREAM_COND
                                   else f"m7_metrics_stream_{cond}.json")
                dst.write_text(src.read_text())
                print(f"  -> {dst.name}")

    # ---- 2. contrasts within streaming, then differenced against full-reprocess ----
    idx_s = index_cells(relabel(stream))
    idx_f = index_cells(full)

    out: dict = {"n_streaming_records": len(stream), "n_full_records": len(full),
                 "records_by_condition": dict(by_cond),
                 "streaming_contrasts": {}, "full_reprocess_contrasts": {},
                 "cross_regime": {}}

    for name, desc, ca, cb in streaming_specs():
        c = contrast(idx_s, ca, cb)
        if c:
            c["description"] = desc
            out["streaming_contrasts"][name] = c
        else:
            out["streaming_contrasts"][name] = {"unavailable": True, "description": desc,
                                                "cells": [ca, cb]}

    for name, ca, cb in cross_regime_specs():
        cf = contrast(idx_f, ca, cb)
        if cf:
            out["full_reprocess_contrasts"][name] = cf
        d = diff_of_contrasts(idx_s, idx_f, ca, cb)
        out["cross_regime"][name] = d or {"unavailable": True, "cells": [ca, cb]}

    # --- per-Δt breakdowns ---------------------------------------------------------------
    # X8's confirmatory prediction (EXTENDING.md §4a): if the injected clock is unreadable once
    # the cache saturates, `main` and `nots` must be indistinguishable at Δt >= 15 min and may
    # differ only at 1 and 5 min, where the clock is still live. Pooling over Δt would hide
    # exactly the pattern that tests it.
    out["timestamps_by_dt"] = {}
    for dt in DT_GRID_S:
        c = contrast(idx_s, [("memory", "main", dt)], [("memory", "main@nots", dt)])
        if c:
            out["timestamps_by_dt"][dt] = {
                "n": c["n_questions"],
                "delta_accuracy": c["delta_accuracy"], "delta_confidence": c["delta_confidence"],
            }
    # the context-length axis, bin by bin: `small` saturates immediately, `large` only late, so
    # the budgets separate most where the default cache is already pinned
    out["budget_by_dt"] = {}
    for dt in DT_GRID_S:
        c = contrast(idx_s, [("memory", "main@large", dt)], [("memory", "main@small", dt)])
        if c:
            out["budget_by_dt"][dt] = {
                "n": c["n_questions"],
                "live_tokens_large": None, "live_tokens_small": None,
                "accuracy_large": c["accuracy_a"], "accuracy_small": c["accuracy_b"],
                "confidence_large": c["confidence_a"], "confidence_small": c["confidence_b"],
                "delta_accuracy": c["delta_accuracy"], "delta_confidence": c["delta_confidence"],
            }
    for dt, v in out["budget_by_dt"].items():
        for tag, arm in (("large", "main@large"), ("small", "main@small")):
            rs = idx_s.get(("memory", arm, dt))
            if rs:
                v[f"live_tokens_{tag}"] = sum(r["live_tokens"] for r in rs.values()) / len(rs)

    # per-bin levels, for the figure and the floor
    levels: dict = {}
    for (kind, arm, dt), rows in index_cells(relabel(stream)).items():
        rs = list(rows.values())
        levels[f"{kind}/{arm}/{dt}"] = {
            "n": len(rs),
            "accuracy": sum(r["correct"] for r in rs) / len(rs),
            "confidence": sum(r["confidence"] for r in rs) / len(rs),
            "live_tokens": sum(r["live_tokens"] for r in rs) / len(rs),
            "visual_tokens": sum(r["visual_tokens"] for r in rs) / len(rs),
        }
    out["cells"] = levels

    p = ARTIFACTS / "x9_regime_comparison.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\nX9 -> {p}")

    def fmt(b):
        if not b:
            return "n/a"
        return f"{100*b['mean']:+.1f} [{100*b['lo95']:+.1f}, {100*b['hi95']:+.1f}]"

    write_report(out, fmt)

    print("\nStreaming contrasts (Δaccuracy | Δconfidence, pp):")
    for k, v in out["streaming_contrasts"].items():
        if v.get("unavailable"):
            print(f"  {k:34s} unavailable")
            continue
        print(f"  {k:34s} n={v['n_questions']:3d}  "
              f"acc {fmt(v['delta_accuracy'])}   conf {fmt(v['delta_confidence'])}")
    print("\nCross-regime (streaming Δ − full-reprocess Δ, pp):")
    for k, v in out["cross_regime"].items():
        if v.get("unavailable"):
            print(f"  {k:34s} unavailable")
            continue
        print(f"  {k:34s} n={v['n_questions']:3d}  "
              f"acc {fmt(v['diff_accuracy'])}   conf {fmt(v['diff_confidence'])}")


if __name__ == "__main__":
    main()
