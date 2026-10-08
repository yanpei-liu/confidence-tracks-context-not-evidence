"""M7 — analysis and the headline figure.

Everything here is derived from m6_records.jsonl; nothing re-touches the model.

Design note on CIs: each question is evaluated at all four Δt, so rows are not independent.
All intervals come from a *cluster* bootstrap that resamples questions (with replacement) and
keeps every row belonging to a resampled question. Resampling rows would understate the
uncertainty and, worse, would break the paired structure the headline claim depends on.

Primary quantity: the change in the overconfidence gap,

    Δgap = [conf(Δt_max) − acc(Δt_max)] − [conf(Δt_min) − acc(Δt_min)]

which is exactly the bin × {accuracy, confidence} interaction the original design asks to power for.
A CI on Δgap that excludes 0 is the "doesn't know it forgot" result.

Artifacts: artifacts/m7_metrics.json, figures/headline.{pdf,png}
"""
from __future__ import annotations

import argparse
import bisect
import collections
import json
import math
import random

from .config import (
    ARTIFACTS, BIG_ARTIFACTS, DT_GRID_S, FIGURES, PRIOR_REF_DT_S, TOKMATCH_REF_DT_S,
)

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


def auroc(rows) -> float | None:
    """P(confidence of a correct item > confidence of a wrong one), ties counted as 0.5.

    This is the *discrimination* half of metamemory and is invariant to any monotone
    reshaping of the confidence scale — so it separates "the level slid" from "the
    ranking degraded", which mean-confidence alone cannot.
    """
    pos = [r["confidence"] for r in rows if r["correct"]]
    neg = sorted(r["confidence"] for r in rows if not r["correct"])
    if not pos or not neg:
        return None
    total = 0.0
    for p in pos:
        lo = bisect.bisect_left(neg, p)
        total += lo + 0.5 * (bisect.bisect_right(neg, p) - lo)
    return total / (len(pos) * len(neg))


def index_cells(records) -> dict:
    """(question_kind, arm, dt_s) -> {qid: record}. One record per cell per question."""
    idx: dict = collections.defaultdict(dict)
    for r in records:
        idx[(r["question_kind"], r["arm"], r["dt_s"])][r["qid"]] = r
    return idx


def _side(rows, tag):
    return [r for t, r in rows if t == tag]


def _mean(rows, field):
    return sum(r[field] for r in rows) / len(rows)


def _delta_field(field):
    def f(rows):
        a, b = _side(rows, "a"), _side(rows, "b")
        if not a or not b:
            return None
        return _mean(a, field) - _mean(b, field)
    return f


def _delta_gap(rows):
    a, b = _side(rows, "a"), _side(rows, "b")
    if not a or not b:
        return None
    gap = lambda rs: _mean(rs, "confidence") - _mean(rs, "correct")  # noqa: E731
    return gap(a) - gap(b)


def contrast(idx: dict, cells_a, cells_b, seed: int = 0) -> dict | None:
    """Paired A − B over questions, cluster-bootstrapped.

    `cells_a` / `cells_b` are equal-length lists of (question_kind, arm, dt_s) that are
    zipped and pooled: passing more than one pair pools the contrast over several Δt,
    which is how the token effect gets its power (it is only resolvable pooled over the
    30 and 60 min bins). Pairing is on `qid`, so every comparison is within-question and
    the cluster bootstrap resamples questions, never rows.
    """
    by_q: dict = collections.defaultdict(list)
    for ca, cb in zip(cells_a, cells_b):
        A, B = idx.get(ca), idx.get(cb)
        if not A or not B:
            return None
        # sorted, not raw set order: `by_q`'s insertion order becomes `qids` in
        # cluster_bootstrap, so an unordered set makes every resample depend on
        # PYTHONHASHSEED and the CIs move by ~0.3 pp between runs on identical data.
        for q in sorted(set(A) & set(B)):
            by_q[q].append(("a", A[q]))
            by_q[q].append(("b", B[q]))
    if not by_q:
        return None
    flat = [x for rs in by_q.values() for x in rs]
    a, b = _side(flat, "a"), _side(flat, "b")
    return {
        "n_questions": len(by_q),
        "n_pairs": len(a),
        "accuracy_a": _mean(a, "correct"), "accuracy_b": _mean(b, "correct"),
        "confidence_a": _mean(a, "confidence"), "confidence_b": _mean(b, "confidence"),
        "tokens_a": _mean(a, "visual_tokens"), "tokens_b": _mean(b, "visual_tokens"),
        "delta_accuracy": cluster_bootstrap(by_q, _delta_field("correct"), seed=seed),
        "delta_confidence": cluster_bootstrap(by_q, _delta_field("confidence"), seed=seed),
        "delta_gap": cluster_bootstrap(by_q, _delta_gap, seed=seed),
    }


def contrast_specs(idx: dict) -> list[tuple]:
    """(key, label, cells_a, cells_b) for every manipulation the headline rests on.

    Each is written so that `a` is the condition that should help — except the two
    Δt sweeps, which are written in the direction the sweep runs. Every entry that
    references a supplementary bin is emitted only if that bin has records, so this
    list grows on its own as S1–S4 land.
    """
    evk = sorted(dt for (k, arm, dt) in idx
                 if k == "memory" and arm == "tokmatch_evkeep" and dt > TOKMATCH_REF_DT_S)
    uni = sorted(dt for (k, arm, dt) in idx
                 if k == "memory" and arm == "tokmatch_uniform" and dt > TOKMATCH_REF_DT_S)
    ref = ("memory", "main", TOKMATCH_REF_DT_S)
    dts = sorted({dt for (k, arm, dt) in idx if k == "memory" and arm == "main"})
    lo, hi = (min(dts), max(dts)) if dts else (None, None)
    prior_dts = sorted(dt for (k, arm, dt) in idx if k == "memory" and arm == "prior")

    specs: list[tuple] = []
    for dt in prior_dts:
        suffix = "" if len(prior_dts) == 1 else f"_{dt // 60}min"
        specs.append((
            f"evidence_present{suffix}",
            f"evidence in context vs none (Δt={dt // 60} min)",
            [("memory", "main", dt)], [("memory", "prior", dt)]))
    if evk and uni:
        common = [dt for dt in evk if dt in uni]
        specs.append((
            "evidence_resolution", "evidence frames kept vs thinned (tokens, Δt fixed)",
            [("memory", "tokmatch_evkeep", dt) for dt in common],
            [("memory", "tokmatch_uniform", dt) for dt in common]))
    if evk:
        matched = [dt for dt in evk if ("memory", "main", dt) in idx]
        if matched:
            specs.append((
                "fewer_tokens", f"{4}× fewer visual tokens (Δt, evidence fixed)",
                [("memory", "tokmatch_evkeep", dt) for dt in matched],
                [("memory", "main", dt) for dt in matched]))
        for dt in evk:
            specs.append((
                f"more_elapsed_time_{dt // 60}min",
                f"+{(dt - TOKMATCH_REF_DT_S) // 60} min elapsed (tokens, evidence fixed)",
                [("memory", "tokmatch_evkeep", dt)], [ref]))
    if lo is not None and hi != lo:
        specs.append((
            "raw_sweep", f"Δt {lo // 60}→{hi // 60} min (context grows with Δt)",
            [("memory", "main", hi)], [("memory", "main", lo)]))
        if ("control", "main", hi) in idx and ("control", "main", lo) in idx:
            specs.append((
                "control_sweep", f"Δt {lo // 60}→{hi // 60} min, no-memory question",
                [("control", "main", hi)], [("control", "main", lo)]))
        for dt in (lo, hi):
            if ("control", "main", dt) in idx:
                specs.append((
                    f"question_kind_{dt // 60}min",
                    f"no-memory vs recall question, identical frames (Δt={dt // 60} min)",
                    [("control", "main", dt)], [("memory", "main", dt)]))
    # S2: the no-memory control on token-matched frames — does the control's own decline
    # survive holding context length fixed?
    for arm in ("tokmatch_ckptkeep", "tokmatch_uniform"):
        cells = sorted(dt for (k, a, dt) in idx if k == "control" and a == arm)
        if cells and ("control", "main", TOKMATCH_REF_DT_S) in idx:
            specs.append((
                f"control_{arm}", f"no-memory control, tokens fixed ({arm.split('_')[-1]})",
                [("control", arm, dt) for dt in cells],
                [("control", "main", TOKMATCH_REF_DT_S)] * len(cells)))
    return specs


def item_transitions(records, arm: str = "main", seed: int = 0) -> dict | None:
    """Is the confidence decline item-specific, or a flat discount on everything?

    Over adjacent Δt steps, restricted to items the model had *right* at the earlier
    bin: compare the confidence change on items it went on to lose against items it
    kept. A metamemory signal worth the name drops harder on the items actually lost.
    """
    idx = index_cells(records)
    dts = sorted(dt for (k, a, dt) in idx if k == "memory" and a == arm)
    if len(dts) < 2:
        return None
    by_q: dict = collections.defaultdict(list)
    for lo, hi in zip(dts, dts[1:]):
        A, B = idx[("memory", arm, lo)], idx[("memory", arm, hi)]
        for q in sorted(set(A) & set(B)):  # deterministic qid order — see contrast()
            if A[q]["correct"]:
                tag = "lost" if not B[q]["correct"] else "kept"
                by_q[q].append((tag, B[q]["confidence"] - A[q]["confidence"]))
    if not by_q:
        return None

    def group_mean(tag):
        def f(rows):
            v = [d for t, d in rows if t == tag]
            return sum(v) / len(v) if v else None
        return f

    def difference(rows):
        lost = [d for t, d in rows if t == "lost"]
        kept = [d for t, d in rows if t == "kept"]
        if not lost or not kept:
            return None
        return sum(lost) / len(lost) - sum(kept) / len(kept)

    flat = [x for rs in by_q.values() for x in rs]
    return {
        "n_transitions": len(flat),
        "n_lost": sum(1 for t, _ in flat if t == "lost"),
        "n_kept": sum(1 for t, _ in flat if t == "kept"),
        "delta_confidence_lost": cluster_bootstrap(by_q, group_mean("lost"), seed=seed),
        "delta_confidence_kept": cluster_bootstrap(by_q, group_mean("kept"), seed=seed),
        "difference": cluster_bootstrap(by_q, difference, seed=seed),
    }


def decomposition(contrasts: dict, bins: dict) -> dict | None:
    """Split the observed `main` 15→60 min change into elapsed time and context length.

    The token-matched arms make this identifiable: `evkeep@60 − main@15` is elapsed time
    at fixed tokens and fixed evidence sampling; `main@60 − evkeep@60` is the token count
    at fixed elapsed time. Their sum must land on the observed change — the residual is
    printed and asserted, because if it stops closing the arms are no longer matched.
    """
    elapsed = {int(k.split("_")[-1][:-3]) * 60: k
               for k in contrasts if k.startswith("more_elapsed_time_")}
    # decompose the segment the token-matched arms were actually run for: 15 → 60 min.
    dt_hi = max((dt for dt in elapsed if str(dt) in bins.get("memory/main", {})), default=None)
    if dt_hi is None or "fewer_tokens" not in contrasts:
        return None
    hi_key = elapsed[dt_hi]
    mem = bins.get("memory/main", {})
    if str(dt_hi) not in mem or str(TOKMATCH_REF_DT_S) not in mem:
        return None
    time_part = contrasts[hi_key]
    token_part = contrasts["fewer_tokens"]
    out = {"dt_ref_s": TOKMATCH_REF_DT_S, "dt_hi_s": dt_hi}
    for field, key in (("accuracy", "delta_accuracy"), ("confidence", "delta_confidence")):
        col = "accuracy" if field == "accuracy" else "mean_confidence"
        observed = mem[str(dt_hi)][col] - mem[str(TOKMATCH_REF_DT_S)][col]
        # `fewer_tokens` is evkeep − main, i.e. the *benefit* of the shorter context;
        # the sweep runs the other way, so the token term enters negated.
        modelled = time_part[key]["mean"] - token_part[key]["mean"]
        out[field] = {
            "observed": observed,
            "elapsed_time": time_part[key]["mean"],
            "context_length": -token_part[key]["mean"],
            "modelled": modelled,
            "residual": observed - modelled,
        }
    return out


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
    # pinned to one bin: with the prior arm running in every Δt bin (S3) an unfiltered
    # selection would silently mix bins and break the token match this contrast rests on
    pri = {r["qid"]: r for r in records
           if r["question_kind"] == "memory" and r["arm"] == "prior"
           and r["dt_s"] == PRIOR_REF_DT_S}
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

    # ---- the manipulation set: what does each intervention move? ----
    # The headline claim is not about any single arm, it is about which manipulations move
    # accuracy and which move confidence. Every contrast below is paired within-question.
    idx = index_cells(records)
    for key, label, cells_a, cells_b in contrast_specs(idx):
        c = contrast(idx, cells_a, cells_b)
        if c:
            out.setdefault("contrasts", {})[key] = dict(c, label=label)

    # ---- discrimination: does the confidence *ranking* survive what shifts its level? ----
    for (kind, arm, dt), cell in sorted(idx.items()):
        rows = list(cell.values())
        a = auroc(rows)
        if a is None:
            continue
        cc = [r["confidence"] for r in rows if r["correct"]]
        cw = [r["confidence"] for r in rows if not r["correct"]]
        out.setdefault("discrimination", {})[f"{kind}/{arm}/{dt}"] = {
            "n": len(rows), "auroc": a,
            "confidence_when_correct": sum(cc) / len(cc),
            "confidence_when_wrong": sum(cw) / len(cw),
        }
    if mem and len(dts) > 1:
        lo_c, hi_c = idx[("memory", "main", min(dts))], idx[("memory", "main", max(dts))]
        common = sorted(set(lo_c) & set(hi_c))
        if common:
            by_q = {q: [("a", hi_c[q]), ("b", lo_c[q])] for q in common}

            def d_auroc(rows):
                a, b = _side(rows, "a"), _side(rows, "b")
                x, y = auroc(a), auroc(b)
                return None if x is None or y is None else x - y

            out.setdefault("discrimination", {})["delta_auroc_sweep"] = \
                cluster_bootstrap(by_q, d_auroc)

    # ---- is the confidence decline item-specific or a flat discount? ----
    it = item_transitions(records)
    if it:
        out["item_transitions"] = it

    # ---- split the observed sweep into elapsed time vs context length ----
    dec = decomposition(out.get("contrasts", {}), out.get("bins", {}))
    if dec:
        out["decomposition"] = dec
        for field in ("accuracy", "confidence"):
            assert abs(dec[field]["residual"]) < 0.02, (
                f"{field} decomposition does not close (residual "
                f"{dec[field]['residual']:+.4f}) — the token-matched arms are no longer "
                f"matched to main at Δt={TOKMATCH_REF_DT_S}s")

    # ---- per-category breakdown at the extremes ----
    for typ in sorted({r["type"] for r in mem}) if mem else []:
        for dt in (min(dts), max(dts)):
            rows = [r for r in mem if r["type"] == typ and r["dt_s"] == dt]
            if rows:
                out.setdefault("by_type", {}).setdefault(typ, {})[str(dt)] = summarize(rows)
    return out


# Categorical slots checked against a light surface (#fcfcfb) for lightness band, chroma
# floor, CVD separation and normal-vision separation. The aqua slot is the marginal one on
# contrast against the surface, which is discharged by the legend plus direct labels —
# identity is never carried by colour alone in either figure.
C_ACC = "#2a78d6"     # accuracy
C_CONF = "#eb6834"    # confidence
C_CTRL = "#1baf7a"    # no-memory control
C_PRIOR = "#4a3aa7"   # language-prior floor
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
SURFACE = "#fcfcfb"

# Short forms for the sensitivity map, where the axes already say "accuracy" and
# "confidence" and the point labels only have to name the manipulation. Keys not listed
# fall back to the contrast's own label.
SHORT_LABEL = {
    "evidence_present": "evidence in context\nvs none at all",
    "evidence_resolution": "evidence frames kept\nvs thinned",
    "fewer_tokens": "4× fewer visual tokens\n(Δt, evidence fixed)",
    "raw_sweep": "Δt 1→60 min\n(raw sweep)",
    "control_sweep": "Δt 1→60 min\n(no-memory question)",
}
# Offsets in typographic points with a leader line back to the mark. The manipulations
# cluster near the origin — they are all a few points of accuracy apart — so the labels have
# to be pushed out and led back rather than nudged.
LABEL_OFFSET = {
    "evidence_present": (18, -30, "left"),
    "evidence_resolution": (-104, 22, "left"),
    "fewer_tokens": (-116, -26, "left"),
    "raw_sweep": (14, -30, "left"),
    "control_sweep": (14, 24, "left"),
    "_elapsed": (60, -8, "left"),
    "_question_kind": (-20, 26, "left"),
    "_default": (16, 18, "left"),
}


def sensitivity_points(metrics: dict) -> list[tuple]:
    """(key, label, contrast) for the manipulations the sensitivity map plots.

    Deliberately a curated subset: every contrast in `metrics["contrasts"]` is a valid
    measurement, but the map only earns its title with the manipulations that are
    mutually non-redundant — one evidence contrast, one token contrast, one elapsed-time
    contrast, the raw sweep they decompose, and the question-kind swap.
    """
    c = metrics.get("contrasts", {})
    elapsed = sorted((int(k.split("_")[-1][:-3]), k) for k in c
                     if k.startswith("more_elapsed_time_"))
    kinds = sorted((int(k.split("_")[-1][:-3]), k) for k in c
                   if k.startswith("question_kind_"))
    wanted = ["evidence_present", "evidence_resolution", "fewer_tokens"]
    if elapsed:
        wanted.append(elapsed[-1][1])
    wanted.append("raw_sweep")
    if kinds:
        wanted.append(kinds[-1][1])
    out = []
    for k in wanted:
        if k not in c:
            continue
        if k.startswith("more_elapsed_time_"):
            mins = (int(k.split("_")[-1][:-3]) * 60 - TOKMATCH_REF_DT_S) // 60
            label = f"+{mins} min elapsed\n(tokens, evidence fixed)"
        elif k.startswith("question_kind_"):
            label = "no-memory question\n(identical frames)"
        else:
            label = SHORT_LABEL.get(k, c[k]["label"])
        out.append((k, label, c[k]))
    return out


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


def figure_headline(metrics: dict, path_stem):
    """The headline: confidence responds to elapsed time, not to competence.

    Panel A  two Δt sweeps — one where context grows with Δt, one where it is pinned.
    Panel B  the sensitivity map: Δaccuracy against Δconfidence for every manipulation.
    Panel C  the confidence decline is a flat discount, and the ranking never degrades.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = metrics.get("bins", {})
    mem = bins.get("memory/main", {})
    if not mem:
        print("no memory/main bins — skipping headline figure")
        return
    fig = plt.figure(figsize=(14.6, 7.1))
    # explicit margins rather than tight_layout: the title block above the panels is a fixed
    # height and tight_layout cannot see it, so it eats into the panel titles
    gs = fig.add_gridspec(2, 3, width_ratios=[1.10, 1.24, 0.82], height_ratios=[1, 1],
                          wspace=0.30, hspace=0.78,
                          left=0.052, right=0.988, top=0.795, bottom=0.075)
    ax = fig.add_subplot(gs[:, 0])
    bx = fig.add_subplot(gs[:, 1])
    c1 = fig.add_subplot(gs[0, 2])
    c2 = fig.add_subplot(gs[1, 2])

    # ---------------- panel A: two sweeps, only one of them honest ----------------
    dts = sorted(int(k) for k in mem)
    x = [d / 60 for d in dts]
    for field, ci, col, lbl in ((("accuracy"), "accuracy_ci", C_ACC, "accuracy"),
                                ("mean_confidence", "confidence_ci", C_CONF, "confidence")):
        ax.fill_between(x, [mem[str(d)][ci]["lo95"] for d in dts],
                        [mem[str(d)][ci]["hi95"] for d in dts], color=col, alpha=0.13, lw=0)
        ax.plot(x, [mem[str(d)][field] for d in dts], "-", marker="o", color=col, lw=2,
                ms=8, label=f"{lbl} — context grows with Δt", zorder=4)

    # the token-fixed arm: anchored at main's Δt=15 min point, which *is* the budget
    evk = bins.get("memory/tokmatch_evkeep", {})
    tf_dts = [TOKMATCH_REF_DT_S] + sorted(int(k) for k in evk if int(k) > TOKMATCH_REF_DT_S)
    if len(tf_dts) > 1:
        tf_x = [d / 60 for d in tf_dts]
        cell = lambda d: mem[str(d)] if d == TOKMATCH_REF_DT_S else evk[str(d)]  # noqa: E731
        for field, col, lbl in (("accuracy", C_ACC, "accuracy"),
                                ("mean_confidence", C_CONF, "confidence")):
            ax.plot(tf_x, [cell(d)[field] for d in tf_dts], ls=(0, (1, 1.6)), marker="o",
                    color=col, lw=2, ms=8, mfc=SURFACE, mew=1.8,
                    label=f"{lbl} — context pinned at the 15 min budget", zorder=5)
        tok = evk[str(tf_dts[-1])]["mean_visual_tokens"]
        a_end = cell(tf_dts[-1])["accuracy"]
        ax.annotate(f"context pinned at {tok / 1000:.1f}k visual tokens:\n"
                    f"accuracy flat, confidence still falls",
                    xy=(tf_x[-1] * 0.97, a_end + 0.012), xytext=(x[0] * 0.88, 0.515),
                    fontsize=8.5, color=INK2, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9,
                                    connectionstyle="angle3,angleA=0,angleB=80"))

    prior = bins.get("memory/prior", {})
    if prior:
        pv = prior[sorted(prior, key=int)[0]]["accuracy"]
        ax.axhline(pv, color=C_PRIOR, lw=1.4, ls=(0, (5, 2)), zorder=2)
        ax.text(x[-1] * 1.4, pv - 0.016, "no-evidence floor", fontsize=8.5,
                color=C_PRIOR, ha="right", va="top")
    ax.axhline(0.25, color=MUTED, lw=1, ls=(0, (2, 3)), zorder=1)
    ax.text(x[0] * 0.68, 0.25 - 0.016, "chance (4-way)", fontsize=8.5, color=MUTED, va="top")

    ticks = sorted({*x, *(d / 60 for d in tf_dts)}) if len(tf_dts) > 1 else x
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(v)}" for v in ticks])
    ax.set_xlim(ticks[0] * 0.62, ticks[-1] * 1.45)
    ax.set_ylim(0.18, 0.95)
    ax.set_xlabel("Δt — stream time between evidence and recall (min, log)")
    ax.set_ylabel("accuracy / confidence")
    ax.grid(alpha=0.15, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=7.6, loc="upper left", handlelength=2.4,
              borderaxespad=0.2, labelspacing=0.35)
    ax.set_title("A  Pin the context length and the accuracy decline\n"
                 "     goes away — the confidence decline does not",
                 fontsize=9.5, loc="left", color=INK)

    # ---------------- panel B: the sensitivity map ----------------
    pts = sensitivity_points(metrics)
    xs = [100 * p[2]["delta_accuracy"]["mean"] for p in pts]
    ys = [100 * p[2]["delta_confidence"]["mean"] for p in pts]
    lox = min(100 * p[2]["delta_accuracy"]["lo95"] for p in pts)
    hix = max(100 * p[2]["delta_accuracy"]["hi95"] for p in pts)
    loy = min(100 * p[2]["delta_confidence"]["lo95"] for p in pts)
    hiy = max(100 * p[2]["delta_confidence"]["hi95"] for p in pts)
    xlim = (lox - 0.42 * (hix - lox), hix + 0.30 * (hix - lox))
    ylim = (loy - 0.30 * (hiy - loy), hiy + 0.26 * (hiy - loy))
    lim = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))

    bx.fill_between(lim, lim, ylim[1], color=C_CONF, alpha=0.05, lw=0, zorder=0)
    bx.fill_between(lim, ylim[0], lim, color=C_ACC, alpha=0.05, lw=0, zorder=0)
    bx.plot(lim, lim, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=1)
    bx.axhline(0, color=INK2, lw=0.9, zorder=1)
    bx.axvline(0, color=INK2, lw=0.9, zorder=1)
    diag = min(xlim[1], ylim[1]) - 0.30 * (lim[1] - lim[0])
    bx.text(diag, diag, "confidence tracks competence", fontsize=8, color=MUTED,
            rotation=45, rotation_mode="anchor", ha="left", va="bottom")
    bx.text(xlim[0] + 0.4, ylim[1] - 0.5, "confidence over-reacts", fontsize=8.5,
            color=C_CONF, ha="left", va="top")
    bx.text(xlim[1] - 0.4, ylim[0] + 0.5, "confidence under-reacts", fontsize=8.5,
            color=C_ACC, ha="right", va="bottom")

    for (key, label, c), px, py in zip(pts, xs, ys):
        bx.plot([100 * c["delta_accuracy"]["lo95"], 100 * c["delta_accuracy"]["hi95"]],
                [py, py], color=C_ACC, lw=2, alpha=0.75, solid_capstyle="round", zorder=3)
        bx.plot([px, px],
                [100 * c["delta_confidence"]["lo95"], 100 * c["delta_confidence"]["hi95"]],
                color=C_CONF, lw=2, alpha=0.75, solid_capstyle="round", zorder=3)
        bx.plot([px], [py], "o", color=INK, ms=8, mec=SURFACE, mew=2, zorder=5)
        slot = ("_elapsed" if key.startswith("more_elapsed_time_") else
                "_question_kind" if key.startswith("question_kind_") else key)
        dx, dy, ha = LABEL_OFFSET.get(slot, LABEL_OFFSET["_default"])
        bx.annotate(f"{label}\n{px:+.1f} acc / {py:+.1f} conf", (px, py),
                    textcoords="offset points", xytext=(dx, dy), fontsize=7.8, color=INK2,
                    ha=ha, va="bottom" if dy > 0 else "top", linespacing=1.4, zorder=6,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkA=1,
                                    shrinkB=7))

    bx.set_xlim(*xlim)
    bx.set_ylim(*ylim)
    bx.set_xlabel("change in accuracy (pp)", color=C_ACC)
    bx.set_ylabel("change in confidence (pp)", color=C_CONF)
    bx.tick_params(axis="x", colors=INK2)
    bx.tick_params(axis="y", colors=INK2)
    bx.grid(alpha=0.13, lw=0.7)
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    bx.set_title("B  What each manipulation moves — 95% CIs, horizontal\n"
                 "     for accuracy, vertical for confidence",
                 fontsize=9.5, loc="left", color=INK)

    # ---------------- panel C1: the decline is a flat discount ----------------
    it = metrics.get("item_transitions")
    if it:
        rows = [("items it kept", it["delta_confidence_kept"], it["n_kept"], C_ACC),
                ("items it lost", it["delta_confidence_lost"], it["n_lost"], C_CONF)]
        span = max(abs(100 * d["lo95"]) for _, d, _, _ in rows)
        left = -span * 1.75
        for y, (lbl, d, n, col) in zip((1, 0), rows):
            c1.plot([100 * d["lo95"], 100 * d["hi95"]], [y, y], color=col, lw=2.6,
                    solid_capstyle="round", zorder=3)
            c1.plot([100 * d["mean"]], [y], "o", color=col, ms=9, mec=SURFACE, mew=2,
                    zorder=4)
            c1.text(100 * d["hi95"] + 0.45, y, f"{100 * d['mean']:+.1f} pp  (n={n})",
                    va="center", fontsize=8.5, color=INK2)
            c1.text(left * 0.98, y, lbl, va="center", ha="left", fontsize=8.5, color=INK)
        c1.axvline(0, color=INK2, lw=0.9)
        c1.set_ylim(-0.85, 1.75)
        c1.set_xlim(left, span * 1.5)
        diff = it["difference"]
        c1.text(left * 0.98, -0.62,
                f"difference {100 * diff['mean']:+.1f} pp "
                f"[{100 * diff['lo95']:+.1f}, {100 * diff['hi95']:+.1f}], "
                f"p={diff['p_two_sided_vs_0']:.2f}",
                fontsize=7.8, color=MUTED, ha="left", va="center")
        c1.set_yticks([])
        c1.set_xlabel("confidence change over one Δt step (pp)", fontsize=8.5)
        c1.tick_params(labelsize=8)
        c1.grid(axis="x", alpha=0.13, lw=0.7)
        for s in ("top", "right", "left"):
            c1.spines[s].set_visible(False)
        c1.set_title("C  The drop lands on everything, not on\n"
                     "     the items actually lost", fontsize=9.5, loc="left", color=INK)

    # ---------------- panel C2: the ranking never degrades ----------------
    disc = metrics.get("discrimination", {})
    mem_auc = sorted((int(k.split("/")[-1]), v["auroc"]) for k, v in disc.items()
                     if k.startswith("memory/main/"))
    if mem_auc:
        ax2x = [d / 60 for d, _ in mem_auc]
        c2.plot(ax2x, [a for _, a in mem_auc], "-o", color=C_CONF, lw=2, ms=7,
                label="recall questions", zorder=4)
        pri = [v["auroc"] for k, v in disc.items() if k.startswith("memory/prior/")]
        if pri:
            c2.axhline(sum(pri) / len(pri), color=C_PRIOR, lw=1.4, ls=(0, (5, 2)), zorder=3)
            c2.text(ax2x[-1] * 1.3, sum(pri) / len(pri) - 0.008, "no evidence in context",
                    fontsize=8, color=C_PRIOR, ha="right", va="top")
        c2.axhline(0.5, color=MUTED, lw=1, ls=(0, (2, 3)), zorder=1)
        c2.text(ax2x[0] * 0.72, 0.505, "chance", fontsize=8, color=MUTED, va="bottom")
        c2.set_xscale("log")
        c2.set_xticks(ax2x)
        c2.set_xticklabels([f"{int(v)}" for v in ax2x])
        c2.set_xlim(ax2x[0] * 0.66, ax2x[-1] * 1.5)
        c2.set_ylim(0.46, 0.72)
        c2.set_xlabel("Δt (min, log)", fontsize=8.5)
        c2.set_ylabel("AUROC", fontsize=8.5)
        c2.tick_params(labelsize=8)
        c2.grid(alpha=0.13, lw=0.7)
        for s in ("top", "right"):
            c2.spines[s].set_visible(False)
        c2.set_title("     …while the ranking is untouched by Δt —\n"
                     "     it is only the level that slides",
                     fontsize=9.5, loc="left", color=INK)

    fig.suptitle("Confidence tracks the wrong variable", fontsize=15, x=0.007, ha="left",
                 y=0.982, color=INK, weight="bold")
    fig.text(0.007, 0.945, headline_subtitle(metrics), fontsize=8.6, color=INK2, ha="left",
             va="top", linespacing=1.5)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_stem}.{ext}", dpi=180, facecolor=SURFACE)
    plt.close(fig)
    print(f"headline figure -> {path_stem}.pdf/.png")


def headline_subtitle(metrics: dict) -> str:
    """One line naming the dissociation, generated from the numbers actually measured."""
    c = metrics.get("contrasts", {})
    elapsed = sorted((int(k.split("_")[-1][:-3]), k) for k in c
                     if k.startswith("more_elapsed_time_"))
    parts = []
    if elapsed:
        e = c[elapsed[-1][1]]
        mins = (elapsed[-1][0] * 60 - TOKMATCH_REF_DT_S) // 60
        parts.append(f"+{mins} min of elapsed time at fixed context costs "
                     f"{100 * e['delta_accuracy']['mean']:+.1f} pp accuracy but "
                     f"{100 * e['delta_confidence']['mean']:+.1f} pp confidence")
    if "fewer_tokens" in c:
        t = c["fewer_tokens"]
        parts.append(f"4× the context at fixed Δt costs "
                     f"{-100 * t['delta_accuracy']['mean']:+.1f} pp accuracy but "
                     f"{-100 * t['delta_confidence']['mean']:+.1f} pp confidence")
    if "evidence_present" in c:
        e = c["evidence_present"]
        parts.append(f"having the evidence at all is worth "
                     f"{100 * e['delta_accuracy']['mean']:+.1f} pp accuracy but "
                     f"{100 * e['delta_confidence']['mean']:+.1f} pp confidence")
    n = c.get("evidence_present", {}).get("n_questions")
    tail = f"  (n={n} questions, paired, cluster-bootstrapped)" if n else ""
    return ";\n".join(parts) + tail


def figure_sweep(metrics: dict, path_stem):
    """Supplementary: the Δt sweep as originally designed, plus the contrast forest.

    Superseded as the headline by `figure_headline` — the sweep's own decline is a
    composite of elapsed time and context length, and separating them is the result.
    """
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

    # S4's abstention rows share the record directory but not the softmax denominator: their
    # confidences run over five letters and cannot be averaged with the four-way ones. They
    # are set aside here and analysed on their own.
    n_abstain = sum(1 for r in recs if r["question_kind"] not in ("memory", "control"))
    if n_abstain:
        print(f"  set aside {n_abstain} five-way (abstention) records — not comparable with "
              f"the four-way confidences")
        recs = [r for r in recs if r["question_kind"] in ("memory", "control")]
    metrics = {
        "all_questions": analyze(recs, drop_audio=False),
        "vision_only_questions": analyze(recs, drop_audio=True),
    }
    (ARTIFACTS / "m7_metrics.json").write_text(json.dumps(metrics, indent=1))
    figure_headline(metrics["all_questions"], str(FIGURES / "headline"))
    figure_sweep(metrics["all_questions"], str(FIGURES / "supp_sweep"))

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

    print("\n  manipulations (paired, cluster-bootstrapped over questions):")
    for key, c in m.get("contrasts", {}).items():
        da, dc = c["delta_accuracy"], c["delta_confidence"]
        print(f"    {key:<26} n={c['n_questions']:<4} "
              f"acc {da['mean']*100:+6.2f} [{da['lo95']*100:+6.2f}, {da['hi95']*100:+6.2f}]  "
              f"conf {dc['mean']*100:+6.2f} [{dc['lo95']*100:+6.2f}, {dc['hi95']*100:+6.2f}]")
    dec = m.get("decomposition")
    if dec:
        print(f"\n  Δt {dec['dt_ref_s']//60}→{dec['dt_hi_s']//60} min decomposed "
              f"(elapsed time + context length = observed):")
        for f in ("accuracy", "confidence"):
            d = dec[f]
            print(f"    {f:<11} {d['elapsed_time']*100:+6.2f} pp (time) "
                  f"{d['context_length']*100:+6.2f} pp (context) = {d['modelled']*100:+6.2f} pp "
                  f"vs {d['observed']*100:+6.2f} pp observed "
                  f"(residual {d['residual']*100:+.2f})")
    it = m.get("item_transitions")
    if it:
        k, l_, d = it["delta_confidence_kept"], it["delta_confidence_lost"], it["difference"]
        print(f"\n  item-level: over one Δt step, among items correct at the earlier bin,")
        print(f"    confidence change  kept (n={it['n_kept']}) {k['mean']*100:+.2f} pp | "
              f"lost (n={it['n_lost']}) {l_['mean']*100:+.2f} pp | "
              f"difference {d['mean']*100:+.2f} [{d['lo95']*100:+.2f}, {d['hi95']*100:+.2f}] "
              f"p={d['p_two_sided_vs_0']:.3f}")
    da = m.get("discrimination", {}).get("delta_auroc_sweep")
    if da:
        print(f"  AUROC over the sweep: Δ={da['mean']:+.4f} [{da['lo95']:+.4f}, "
              f"{da['hi95']:+.4f}] p={da['p_two_sided_vs_0']:.3f}")


if __name__ == "__main__":
    main()
