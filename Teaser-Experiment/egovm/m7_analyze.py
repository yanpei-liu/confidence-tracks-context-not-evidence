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

    # ---- evidence awareness: does confidence know whether the evidence is even there? ----
    # `prior` and `main` at PRIOR_REF_DT_S are token-matched by construction (same frame
    # count), and differ only in whether the window contains the evidence. If accuracy rises
    # far more than confidence when the evidence is present, the model cannot tell whether it
    # has the information — a metamemory deficit that does not require any forgetting.
    pri = {r["qid"]: r for r in records
           if r["question_kind"] == "memory" and r["arm"] == "prior"}
    ref = {r["qid"]: r for r in records
           if r["question_kind"] == "memory" and r["arm"] == "main" and r["dt_s"] == 300}
    common = sorted(set(pri) & set(ref))
    if common:
        by_q = {q: [("evid", ref[q]), ("none", pri[q])] for q in common}

        def _delta(field):
            def f(rows):
                e = [r for t, r in rows if t == "evid"]
                n = [r for t, r in rows if t == "none"]
                if not e or not n:
                    return None
                return sum(r[field] for r in e) / len(e) - sum(r[field] for r in n) / len(n)
            return f

        out["evidence_awareness"] = {
            "n_questions": len(common),
            "mean_tokens_with_evidence": sum(ref[q]["visual_tokens"] for q in common) / len(common),
            "mean_tokens_without": sum(pri[q]["visual_tokens"] for q in common) / len(common),
            "accuracy_with_evidence": sum(ref[q]["correct"] for q in common) / len(common),
            "accuracy_without": sum(pri[q]["correct"] for q in common) / len(common),
            "confidence_with_evidence": sum(ref[q]["confidence"] for q in common) / len(common),
            "confidence_without": sum(pri[q]["confidence"] for q in common) / len(common),
            "delta_accuracy": cluster_bootstrap(by_q, _delta("correct")),
            "delta_confidence": cluster_bootstrap(by_q, _delta("confidence")),
        }

    # ---- per-category breakdown at the extremes ----
    for typ in sorted({r["type"] for r in mem}) if mem else []:
        for dt in (min(dts), max(dts)):
            rows = [r for r in mem if r["type"] == typ and r["dt_s"] == dt]
            if rows:
                out.setdefault("by_type", {}).setdefault(typ, {})[str(dt)] = summarize(rows)
    return out


# Validated categorical slots (scripts/validate_palette.js, light surface #fcfcfb):
# all four pass the lightness band, chroma floor, CVD separation and normal-vision floor.
# The aqua slot warns on contrast vs surface, which is discharged by the legend plus
# direct labels — identity is never carried by colour alone here.
C_ACC = "#2a78d6"     # accuracy
C_CONF = "#eb6834"    # confidence
C_CTRL = "#1baf7a"    # no-memory control
C_PRIOR = "#4a3aa7"   # language-prior floor
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"


def headline_title(metrics: dict) -> str:
    """State what the numbers show, not what was predicted."""
    h = metrics.get("headline") or {}
    da, dc, dg = h.get("delta_accuracy"), h.get("delta_confidence"), h.get("delta_gap")
    if not (da and dc and dg):
        return "Accuracy and confidence vs recall delay"
    lo, hi = h["dt_lo_s"] // 60, h["dt_hi_s"] // 60
    if dg["lo95"] > 0:
        lead = "Accuracy falls faster than confidence — the overconfidence gap widens"
    elif dg["hi95"] < 0:
        lead = "Confidence falls faster than accuracy — the gap narrows"
    else:
        lead = "Accuracy and confidence fall together — the overconfidence gap does not widen"
    return (f"{lead}\n"
            f"{lo}→{hi} min: accuracy {da['mean']*100:+.1f} pp "
            f"[{da['lo95']*100:+.1f}, {da['hi95']*100:+.1f}], "
            f"confidence {dc['mean']*100:+.1f} pp "
            f"[{dc['lo95']*100:+.1f}, {dc['hi95']*100:+.1f}], "
            f"gap {dg['mean']*100:+.1f} pp "
            f"[{dg['lo95']*100:+.1f}, {dg['hi95']*100:+.1f}]  (n={h['n_paired_questions']})")


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

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(13.4, 5.4), gridspec_kw={"width_ratios": [1.25, 1.0]})

    # ---------------- panel A: the Δt sweep ----------------
    ax.fill_between(x, [bins[str(d)]["accuracy_ci"]["lo95"] for d in dts],
                    [bins[str(d)]["accuracy_ci"]["hi95"] for d in dts],
                    color=C_ACC, alpha=0.15, lw=0)
    ax.plot(x, acc, "o-", color=C_ACC, lw=2, ms=8, label="memory accuracy", zorder=4)
    ax.fill_between(x, [bins[str(d)]["confidence_ci"]["lo95"] for d in dts],
                    [bins[str(d)]["confidence_ci"]["hi95"] for d in dts],
                    color=C_CONF, alpha=0.15, lw=0)
    ax.plot(x, conf, "s--", color=C_CONF, lw=2, ms=8,
            label="memory confidence (option-letter softmax)", zorder=4)

    for xi, a_, c_ in zip(x, acc, conf):
        ax.annotate("", xy=(xi, c_), xytext=(xi, a_),
                    arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.0))

    ctl = metrics.get("bins", {}).get("control/main", {})
    if ctl:
        cd = sorted(int(k) for k in ctl)
        ax.plot([d / 60 for d in cd], [ctl[str(d)]["accuracy"] for d in cd],
                "^:", color=C_CTRL, lw=2, ms=8,
                label="no-memory control accuracy", zorder=3)

    prior = metrics.get("bins", {}).get("memory/prior", {})
    if prior:
        pv = prior[sorted(prior, key=int)[0]]["accuracy"]
        ax.axhline(pv, color=C_PRIOR, lw=1.4, ls=(0, (5, 2)), zorder=2)
        ax.text(x[-1], pv - 0.022, "language-prior floor — no evidence in context",
                fontsize=8.5, color=C_PRIOR, ha="right", va="top")

    ax.axhline(0.25, color=MUTED, lw=1, ls=(0, (2, 3)), zorder=1)
    ax.text(x[0], 0.25 - 0.022, "chance (4-way)", fontsize=8.5, color=MUTED, va="top")

    # ECE on its own row so it cannot collide with the marks or the floor lines
    ax.text(x[0] * 0.66, 0.055, "ECE", fontsize=8.5, color=INK2, ha="left", va="center")
    for xi, d in zip(x, dts):
        ax.text(xi, 0.055, f"{bins[str(d)]['ece']:.2f}", ha="center", va="center",
                fontsize=8.5, color=INK2)

    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}" for v in x])
    ax.set_xlim(x[0] * 0.62, x[-1] * 1.45)
    ax.set_xlabel("Δt — stream time between evidence and recall checkpoint (min, log)")
    ax.set_ylabel("accuracy / confidence")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.15, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title(headline_title(metrics), fontsize=10, loc="left", color=INK)

    # ---------------- panel B: the decisive contrasts ----------------
    h = metrics.get("headline", {})
    ea = metrics.get("evidence_awareness", {})
    tm = metrics.get("token_matched", {})
    rows = []
    if ea:
        rows.append(("evidence in context vs none\n(token-matched)",
                     ea["delta_accuracy"], C_ACC, "accuracy"))
        rows.append(("evidence in context vs none\n(token-matched)",
                     ea["delta_confidence"], C_CONF, "confidence"))
    if h:
        rows.append((f"Δt {h['dt_hi_s']//60} vs {h['dt_lo_s']//60} min\n(tokens grow with Δt)",
                     h["delta_accuracy"], C_ACC, "accuracy"))
        rows.append((f"Δt {h['dt_hi_s']//60} vs {h['dt_lo_s']//60} min\n(tokens grow with Δt)",
                     h["delta_confidence"], C_CONF, "confidence"))
    for arm, lbl in (("tokmatch_uniform", "uniform thinning"),
                     ("tokmatch_evkeep", "evidence preserved")):
        if arm in tm:
            rows.append((f"Δt 60 vs 15 min, tokens FIXED\n({lbl})",
                         tm[arm]["delta_accuracy"], C_ACC, "accuracy"))

    ypos = list(range(len(rows)))[::-1]
    for y, (lbl, d, col, meas) in zip(ypos, rows):
        bx.plot([d["lo95"] * 100, d["hi95"] * 100], [y, y], color=col, lw=2.4,
                solid_capstyle="round", zorder=3)
        bx.plot([d["mean"] * 100], [y], "o", color=col, ms=9,
                markeredgecolor="#fcfcfb", markeredgewidth=2, zorder=4)
        sig = "" if d["lo95"] < 0 < d["hi95"] else "  ✱"
        bx.text(d["hi95"] * 100 + 0.7, y, f"{d['mean']*100:+.1f} pp{sig}",
                va="center", fontsize=8.5, color=INK2)
    bx.axvline(0, color=INK2, lw=1.1, zorder=2)
    bx.set_yticks(ypos)
    bx.set_yticklabels([r[0] for r in rows], fontsize=8.5, color=INK)
    bx.tick_params(axis="y", length=0)
    bx.set_ylim(-0.7, len(rows) - 0.3)
    span = max(abs(d["lo95"]) for _, d, _, _ in rows) * 100
    bx.set_xlim(-span * 1.25, max(d["hi95"] for _, d, _, _ in rows) * 100 + span * 0.65)
    bx.set_xlabel("change (percentage points, 95% cluster-bootstrap CI over questions)")
    bx.grid(axis="x", alpha=0.15, lw=0.7)
    for s in ("top", "right", "left"):
        bx.spines[s].set_visible(False)
    bx.set_title("What actually moves the model\n"
                 "✱ = interval excludes zero", fontsize=10, loc="left", color=INK)
    hd = [plt.Line2D([], [], color=C_ACC, lw=2.4, marker="o", ms=8, label="accuracy"),
          plt.Line2D([], [], color=C_CONF, lw=2.4, marker="o", ms=8, label="confidence")]
    bx.legend(handles=hd, frameon=False, fontsize=9, loc="lower right")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_stem}.{ext}", dpi=180, facecolor="#fcfcfb")
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
    ea = m.get("evidence_awareness")
    if ea:
        print(f"  evidence awareness (n={ea['n_questions']}, token-matched at "
              f"{ea['mean_tokens_with_evidence']:.0f} vs {ea['mean_tokens_without']:.0f} tok):")
        print(f"    accuracy   {ea['accuracy_without']:.3f} (no evidence) -> "
              f"{ea['accuracy_with_evidence']:.3f} (evidence)  "
              f"Δ={ea['delta_accuracy']['mean']:+.4f} "
              f"[{ea['delta_accuracy']['lo95']:+.4f}, {ea['delta_accuracy']['hi95']:+.4f}]")
        print(f"    confidence {ea['confidence_without']:.3f} (no evidence) -> "
              f"{ea['confidence_with_evidence']:.3f} (evidence)  "
              f"Δ={ea['delta_confidence']['mean']:+.4f} "
              f"[{ea['delta_confidence']['lo95']:+.4f}, {ea['delta_confidence']['hi95']:+.4f}]")
    for arm, tm in m.get("token_matched", {}).items():
        v = tm["delta_accuracy"]
        print(f"  {arm}: acc {tm['accuracy_hi']:.3f} (Δt=60, {tm['mean_tokens_hi']:.0f} tok) vs "
              f"{tm['accuracy_ref']:.3f} (Δt=15, {tm['mean_tokens_ref']:.0f} tok)  "
              f"Δ={v['mean']:+.4f} [{v['lo95']:+.4f}, {v['hi95']:+.4f}]")


if __name__ == "__main__":
    main()
