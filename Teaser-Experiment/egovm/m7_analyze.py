"""M7 — analysis and the headline figure.

Everything here is derived from m6_records.jsonl; nothing re-touches the model.

Design note on CIs: each question is evaluated at all four Δt, so rows are not independent.
All intervals come from a *cluster* bootstrap that resamples questions (with replacement) and
keeps every row belonging to a resampled question. Resampling rows would understate the
uncertainty and, worse, would break the paired structure the headline claim depends on.

Primary quantity: the change in the overconfidence gap,

    Δgap = [conf(Δt_max) − acc(Δt_max)] − [conf(Δt_min) − acc(Δt_min)]

which is exactly the bin × {accuracy, confidence} interaction CLAUDE.md asks to power for.
A CI on Δgap that excludes 0 is the "doesn't know it forgot" result.

Artifacts: artifacts/m7_metrics.json, figures/headline.{pdf,png}
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random

from .config import ARTIFACTS, BIG_ARTIFACTS, DT_GRID_S, FIGURES

N_BOOT = 5000
N_ECE_BINS = 10


def ece(rows, n_bins: int = N_ECE_BINS) -> float:
    """Expected Calibration Error (Naeini et al. 2015) over max-softmax confidence."""
    if not rows:
        return float("nan")
    bins = [[] for _ in range(n_bins)]
    for r in rows:
        b = min(int(r["confidence"] * n_bins), n_bins - 1)
        bins[b].append(r)
    n = len(rows)
    return sum(
        len(b) / n * abs(sum(x["correct"] for x in b) / len(b) - sum(x["confidence"] for x in b) / len(b))
        for b in bins if b
    )


def summarize(rows) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    acc = sum(r["correct"] for r in rows) / n
    conf = sum(r["confidence"] for r in rows) / n
    return {
        "n": n,
        "accuracy": acc,
        "mean_confidence": conf,
        "overconfidence": conf - acc,
        "ece": ece(rows),
        "mean_frames": sum(r["n_frames"] for r in rows) / n,
        "mean_visual_tokens": sum(r["visual_tokens"] for r in rows) / n,
        "mean_entropy": sum(r["entropy"] for r in rows) / n,
    }


def cluster_bootstrap(by_q: dict[str, list], stat, n_boot: int = N_BOOT, seed: int = 0):
    """Resample questions with replacement; `stat` maps a flat row list to a float or None."""
    qids = list(by_q)
    rng = random.Random(seed)
    out = []
    for _ in range(n_boot):
        pick = [qids[rng.randrange(len(qids))] for _ in range(len(qids))]
        rows = [r for q in pick for r in by_q[q]]
        v = stat(rows)
        if v is not None and not math.isnan(v):
            out.append(v)
    out.sort()
    if not out:
        return None
    return {
        "mean": sum(out) / len(out),
        "lo95": out[int(0.025 * len(out))],
        "hi95": out[int(0.975 * len(out))],
        "p_two_sided_vs_0": 2 * min(
            sum(1 for v in out if v <= 0) / len(out),
            sum(1 for v in out if v >= 0) / len(out),
        ),
    }


def load(path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # torn final line from a killed job
    # de-duplicate on record_id, last write wins
    seen = {}
    for r in rows:
        seen[r["record_id"]] = r
    return list(seen.values())


def analyze(records, drop_audio: bool) -> dict:
    if drop_audio:
        records = [r for r in records if not r.get("need_audio")]

    out: dict = {"n_records": len(records), "excluded_need_audio": drop_audio}
    dts = sorted({r["dt_s"] for r in records})

    for kind in ("memory", "control"):
        for arm in sorted({r["arm"] for r in records}):
            sel = [r for r in records if r["question_kind"] == kind and r["arm"] == arm]
            if not sel:
                continue
            key = f"{kind}/{arm}"
            out.setdefault("bins", {})[key] = {}
            for dt in dts:
                rows = [r for r in sel if r["dt_s"] == dt]
                if not rows:
                    continue
                s = summarize(rows)
                by_q = collections.defaultdict(list)
                for r in rows:
                    by_q[r["qid"]].append(r)
                s["accuracy_ci"] = cluster_bootstrap(
                    by_q, lambda rs: sum(x["correct"] for x in rs) / len(rs) if rs else None)
                s["confidence_ci"] = cluster_bootstrap(
                    by_q, lambda rs: sum(x["confidence"] for x in rs) / len(rs) if rs else None)
                out["bins"][key][str(dt)] = s

    # ---- headline: change in the overconfidence gap, paired over questions ----
    mem = [r for r in records if r["question_kind"] == "memory" and r["arm"] == "main"]
    if mem:
        dt_lo, dt_hi = min(dts), max(dts)
        by_q = collections.defaultdict(list)
        for r in mem:
            by_q[r["qid"]].append(r)
        # keep only questions present in both extreme bins, so the contrast is truly paired
        paired = {q: rs for q, rs in by_q.items()
                  if any(r["dt_s"] == dt_lo for r in rs) and any(r["dt_s"] == dt_hi for r in rs)}

        def gap_change(rows):
            lo = [r for r in rows if r["dt_s"] == dt_lo]
            hi = [r for r in rows if r["dt_s"] == dt_hi]
            if not lo or not hi:
                return None
            g_lo = sum(r["confidence"] for r in lo) / len(lo) - sum(r["correct"] for r in lo) / len(lo)
            g_hi = sum(r["confidence"] for r in hi) / len(hi) - sum(r["correct"] for r in hi) / len(hi)
            return g_hi - g_lo

        def acc_change(rows):
            lo = [r for r in rows if r["dt_s"] == dt_lo]
            hi = [r for r in rows if r["dt_s"] == dt_hi]
            if not lo or not hi:
                return None
            return sum(r["correct"] for r in hi) / len(hi) - sum(r["correct"] for r in lo) / len(lo)

        def conf_change(rows):
            lo = [r for r in rows if r["dt_s"] == dt_lo]
            hi = [r for r in rows if r["dt_s"] == dt_hi]
            if not lo or not hi:
                return None
            return sum(r["confidence"] for r in hi) / len(hi) - sum(r["confidence"] for r in lo) / len(lo)

        out["headline"] = {
            "dt_lo_s": dt_lo, "dt_hi_s": dt_hi, "n_paired_questions": len(paired),
            "delta_gap": cluster_bootstrap(paired, gap_change),
            "delta_accuracy": cluster_bootstrap(paired, acc_change),
            "delta_confidence": cluster_bootstrap(paired, conf_change),
        }

    # ---- token-matched contrast: same tokens, different Δt ----
    for arm in ("tokmatch_uniform", "tokmatch_evkeep"):
        hi = [r for r in records if r["question_kind"] == "memory" and r["arm"] == arm
              and r["dt_s"] == max(dts)]
        ref = [r for r in records if r["question_kind"] == "memory" and r["arm"] == "main"
               and r["dt_s"] == 900]
        if not hi or not ref:
            continue
        hi_by_q = {r["qid"]: r for r in hi}
        ref_by_q = {r["qid"]: r for r in ref}
        common = sorted(set(hi_by_q) & set(ref_by_q))
        if not common:
            continue
        by_q = {q: [("hi", hi_by_q[q]), ("ref", ref_by_q[q])] for q in common}

        def d_acc(rows):
            h = [r for tag, r in rows if tag == "hi"]
            f = [r for tag, r in rows if tag == "ref"]
            if not h or not f:
                return None
            return sum(r["correct"] for r in h) / len(h) - sum(r["correct"] for r in f) / len(f)

        out.setdefault("token_matched", {})[arm] = {
            "n_questions": len(common),
            "mean_tokens_hi": sum(hi_by_q[q]["visual_tokens"] for q in common) / len(common),
            "mean_tokens_ref": sum(ref_by_q[q]["visual_tokens"] for q in common) / len(common),
            "accuracy_hi": sum(hi_by_q[q]["correct"] for q in common) / len(common),
            "accuracy_ref": sum(ref_by_q[q]["correct"] for q in common) / len(common),
            "delta_accuracy": cluster_bootstrap(by_q, d_acc),
        }

    # ---- per-category breakdown at the extremes ----
    for typ in sorted({r["type"] for r in mem}) if mem else []:
        for dt in (min(dts), max(dts)):
            rows = [r for r in mem if r["type"] == typ and r["dt_s"] == dt]
            if rows:
                out.setdefault("by_type", {}).setdefault(typ, {})[str(dt)] = summarize(rows)
    return out


def figure(metrics: dict, path_stem):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = metrics.get("bins", {}).get("memory/main", {})
    if not bins:
        print("no memory/main bins — skipping figure")
        return
    dts = sorted(int(k) for k in bins)
    x = [d / 60 for d in dts]
    acc = [bins[str(d)]["accuracy"] for d in dts]
    conf = [bins[str(d)]["mean_confidence"] for d in dts]
    acc_lo = [bins[str(d)]["accuracy_ci"]["lo95"] for d in dts]
    acc_hi = [bins[str(d)]["accuracy_ci"]["hi95"] for d in dts]
    cf_lo = [bins[str(d)]["confidence_ci"]["lo95"] for d in dts]
    cf_hi = [bins[str(d)]["confidence_ci"]["hi95"] for d in dts]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.fill_between(x, acc_lo, acc_hi, color="#1b6ca8", alpha=0.16, lw=0)
    ax.plot(x, acc, "o-", color="#1b6ca8", lw=2.2, ms=7, label="accuracy", zorder=3)
    ax.fill_between(x, cf_lo, cf_hi, color="#c1440e", alpha=0.16, lw=0)
    ax.plot(x, conf, "s--", color="#c1440e", lw=2.2, ms=7,
            label="mean confidence (option-letter softmax)", zorder=3)

    for xi, a, c in zip(x, acc, conf):
        ax.annotate("", xy=(xi, c), xytext=(xi, a),
                    arrowprops=dict(arrowstyle="<->", color="#666", lw=1.0, alpha=0.8))

    ctl = metrics.get("bins", {}).get("control/main", {})
    if ctl:
        cd = sorted(int(k) for k in ctl)
        ax.plot([d / 60 for d in cd], [ctl[str(d)]["accuracy"] for d in cd],
                "^:", color="#2e7d32", lw=1.8, ms=6,
                label="no-memory control accuracy", zorder=2)

    ax.axhline(0.25, color="#999", lw=1, ls=(0, (2, 3)))
    ax.text(x[0], 0.255, "chance (4-way)", fontsize=8, color="#777", va="bottom")

    # the language-prior floor: same questions, evidence-free window. Accuracy above this
    # is what the video evidence buys; reaching it means the evidence is effectively gone.
    prior = metrics.get("bins", {}).get("memory/prior", {})
    if prior:
        pk = sorted(prior, key=int)[0]
        pv = prior[pk]["accuracy"]
        ax.axhline(pv, color="#7b1fa2", lw=1.2, ls=(0, (5, 2)))
        ax.text(x[-1], pv + 0.008, "language-prior floor (no evidence in context)",
                fontsize=8, color="#7b1fa2", ha="right", va="bottom")

    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}" for v in x])
    ax.set_xlabel("Δt — stream time between evidence and recall checkpoint (min, log scale)")
    ax.set_ylabel("accuracy / confidence")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.18, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    for xi, d in zip(x, dts):
        b = bins[str(d)]
        ax.annotate(f"ECE {b['ece']:.2f}", (xi, min(b['accuracy'], b['mean_confidence']) - 0.055),
                    ha="center", fontsize=8, color="#444")

    h = metrics.get("headline", {})
    if h and h.get("delta_gap"):
        g = h["delta_gap"]
        ax.set_title(
            f"Confidence stays flat while accuracy falls\n"
            f"Δgap ({h['dt_lo_s']//60}→{h['dt_hi_s']//60} min) = "
            f"{g['mean']:+.3f} [{g['lo95']:+.3f}, {g['hi95']:+.3f}] 95% CI, "
            f"n={h['n_paired_questions']} questions",
            fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_stem}.{ext}", dpi=180)
    print(f"figure -> {path_stem}.pdf/.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", default=None,
                    help="record files; defaults to every m6_records*.jsonl in the data dir, "
                         "since the sweep runs as several sharded jobs writing separate files")
    ap.add_argument("--drop-audio", action="store_true",
                    help="exclude need_audio questions (model is vision-only)")
    a = ap.parse_args()

    from pathlib import Path

    paths = a.records or sorted(str(q) for q in BIG_ARTIFACTS.glob("m6_records*.jsonl"))
    if not paths:
        raise SystemExit(f"no record files under {BIG_ARTIFACTS}")
    print(f"merging {len(paths)} record file(s)")
    seen = {}
    for p in paths:
        if not Path(p).exists():
            print(f"  (skipping missing {p})")
            continue
        for r in load(Path(p)):
            seen[r["record_id"]] = r
    recs = list(seen.values())
    if not recs:
        raise SystemExit("no records")

    # Only rows in the current plan count. Superseded jobs may have written records for
    # questions the plan has since excluded (e.g. evidence span too wide), and those must
    # not leak into the bins.
    plan_rows = {json.loads(l)["row_id"]
                 for l in (ARTIFACTS / "m4_eval_plan.jsonl").read_text().splitlines()}
    before = len(recs)
    recs = [r for r in recs if r["row_id"] in plan_rows]
    if before != len(recs):
        print(f"  dropped {before - len(recs)} records not in the current plan")
    metrics = {
        "all_questions": analyze(recs, drop_audio=False),
        "vision_only_questions": analyze(recs, drop_audio=True),
    }
    (ARTIFACTS / "m7_metrics.json").write_text(json.dumps(metrics, indent=1))
    figure(metrics["all_questions"], str(FIGURES / "headline"))

    m = metrics["all_questions"]
    print(f"\nM7: {m['n_records']} records")
    for key, bins in m.get("bins", {}).items():
        print(f"  {key}")
        for dt in sorted(bins, key=int):
            b = bins[dt]
            print(f"    Δt={int(dt)//60:3d}min n={b['n']:4d} acc={b['accuracy']:.3f} "
                  f"conf={b['mean_confidence']:.3f} gap={b['overconfidence']:+.3f} "
                  f"ece={b['ece']:.3f} tok={b['mean_visual_tokens']:.0f}")
    h = m.get("headline")
    if h:
        for k in ("delta_accuracy", "delta_confidence", "delta_gap"):
            v = h[k]
            print(f"  {k:18s} {v['mean']:+.4f} [{v['lo95']:+.4f}, {v['hi95']:+.4f}] "
                  f"p={v['p_two_sided_vs_0']:.4f}")
    for arm, tm in m.get("token_matched", {}).items():
        v = tm["delta_accuracy"]
        print(f"  {arm}: acc {tm['accuracy_hi']:.3f} (Δt=60, {tm['mean_tokens_hi']:.0f} tok) vs "
              f"{tm['accuracy_ref']:.3f} (Δt=15, {tm['mean_tokens_ref']:.0f} tok)  "
              f"Δ={v['mean']:+.4f} [{v['lo95']:+.4f}, {v['hi95']:+.4f}]")


if __name__ == "__main__":
    main()
