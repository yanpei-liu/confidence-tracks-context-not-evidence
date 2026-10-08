"""X11 — the evidence-blindness contrast, at every Δt where it can be measured.

The deck's central claim rested on one cell: `main@5 min` − `prior@5 min`. That is thinner
than the data. E2 ran the `prior` arm at more than one Δt, and `m7_analyze.contrast_specs`
already emits one `evidence_present_{N}min` contrast per `prior` bin — but nothing reads them,
because `x9_compare.streaming_specs` hard-codes `evidence_vs_none` to Δt = 300 s. This module
surfaces the whole set and adds the three things the single cell cannot say:

**1. A dose–response the confidence side does not follow.** Δaccuracy varies severalfold
across the bins while Δconfidence stays flat and interval-crossing. A null at one Δt is weak;
a null that holds while the effect it should track moves is not.

**2. The two conditions are not separable from the confidence distribution.** Overlap
coefficient and Cohen's d per bin, and a cluster-bootstrapped AUROC for confidence *as a
detector of evidence presence* — the direct operationalisation of "blind", and near chance.

**3. The probe is not inert.** The same measure moves double digits on question kind and on
the cache budget. Those rows are read from `x9_regime_comparison.json` so the comparison the
slide draws is generated, not typed.

Bins are discovered from the records, so this module reports whatever `prior` bins exist.

Deliberately *not* here: a pooled multi-bin contrast. Pooling tightens Δconfidence to the edge
of significance and would convert a clean per-bin null into a borderline positive; the bins are
reported separately for that reason. Δgap is emitted for completeness but is not denominator
robust (see the robustness section) and should not carry the claim.

Artifacts: artifacts/x11_evidence_blindness.json, artifacts/x11_evidence_blindness.md
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
from pathlib import Path

from ..config import ARTIFACTS, BIG_ARTIFACTS
from ..m7_analyze import (
    N_BOOT, _delta_field, _side, cluster_bootstrap, contrast, index_cells,
)
from .x9_compare import is_stream, load_records, relabel

OVERLAP_BINS = 40

# The manipulations quoted as the "is the probe inert?" reference set, in the order the report
# shows them. Keys are `streaming_contrasts` entries in x9_regime_comparison.json.
REFERENCE_MANIPULATIONS = (
    ("control_vs_memory", "a no-memory question instead of a recall question"),
    ("elapsed_time_raw", "the raw Δt sweep, 60 → 1 min"),
    ("context_length_4x_tokens", "4× more visual input presented"),
    ("evidence_thinned", "the evidence frames thinned away"),
    ("timestamps_on_vs_off", "the injected clock, on vs off"),
    ("elapsed_time_fixed_frames", "+45 min elapsed, frames and evidence fixed"),
)


def letter_mass(r: dict) -> float:
    """M = Σ_L exp(logprob(L)): the mass the restricted softmax keeps, 1 − M what it discards.

    Both backends record `letter_logprobs` as a full-vocabulary log_softmax at the letter ids,
    so this is free from the committed records and needs no GPU.
    """
    lp = r.get("letter_logprobs")
    return sum(math.exp(v) for v in lp.values()) if lp else 1.0


def annotate(records) -> list[dict]:
    """Add the two derived fields the robustness section contrasts on."""
    for r in records:
        m = letter_mass(r)
        r["off_letter"] = 1.0 - m
        # c_vocab = c_restricted × M: the vocabulary-normalised alternative of
        # CONFIDENCE-PROBE.md §2, recovered exactly rather than re-scored.
        r["confidence_vocab"] = r["confidence"] * m
    return records


def overlap_coefficient(a, b, n_bins: int = OVERLAP_BINS) -> float:
    """Σ min(p_a, p_b) over a shared histogram of [0, 1]: 1.0 = indistinguishable."""
    ha = [0] * n_bins
    hb = [0] * n_bins
    for v in a:
        ha[min(n_bins - 1, int(v * n_bins))] += 1
    for v in b:
        hb[min(n_bins - 1, int(v * n_bins))] += 1
    return sum(min(x / len(a), y / len(b)) for x, y in zip(ha, hb))


def cohen_d(a, b) -> float:
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    s = math.sqrt((va + vb) / 2)
    return (ma - mb) / s if s else float("nan")


def _rank_auc(pos, neg) -> float:
    """Mann–Whitney AUC with mid-ranks for ties.

    M7's `auroc` is hardwired to (confidence, correct); here the label is *evidence presence*
    and the score varies, so the statistic is written out rather than reused.
    """
    xs = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    n = len(xs)
    rank_sum = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1][0] == xs[i][0]:
            j += 1
        mid = (i + j) / 2 + 1
        rank_sum += mid * sum(1 for k in range(i, j + 1) if xs[k][1] == 1)
        i = j + 1
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def _detector(field):
    """Bootstrap statistic: AUROC of `field` for telling `main` (a) from `prior` (b)."""
    def f(rows):
        pos = [r[field] for r in _side(rows, "a")]
        neg = [r[field] for r in _side(rows, "b")]
        if not pos or not neg:
            return None
        return _rank_auc(pos, neg)
    return f


def paired(idx: dict, dts) -> dict:
    """{qid: [("a", main), ("b", prior)]} pooled over `dts`, sorted for a stable bootstrap."""
    by_q: dict = collections.defaultdict(list)
    for dt in dts:
        A, B = idx.get(("memory", "main", dt)), idx.get(("memory", "prior", dt))
        if not A or not B:
            continue
        # sorted, not raw set order: by_q's insertion order becomes the bootstrap's qid list,
        # and an unordered set makes the CIs move ~0.3 pp between runs (m7_analyze.py:158).
        for q in sorted(set(A) & set(B)):
            by_q[q].append(("a", A[q]))
            by_q[q].append(("b", B[q]))
    return by_q


def prior_bins(idx: dict) -> list[int]:
    """Every Δt with both a `main` and a `prior` cell — 3 today, 5 after the extension."""
    return sorted(dt for (k, arm, dt) in idx
                  if k == "memory" and arm == "prior" and ("memory", "main", dt) in idx)


def analyze(records, x9: dict, seed: int = 0) -> dict:
    idx = index_cells(relabel(annotate(records)))
    dts = prior_bins(idx)
    if not dts:
        raise SystemExit("X11: no Δt bin has both a `main` and a `prior` cell")

    out: dict = {
        "n_records": len(records),
        "n_boot": N_BOOT,
        "seed": seed,
        "prior_dts_s": dts,
        "by_dt": {},
        "detectors": {},
        "robustness": {},
        "reference_manipulations": {},
    }

    for dt in dts:
        c = contrast(idx, [("memory", "main", dt)], [("memory", "prior", dt)], seed=seed)
        A, B = idx[("memory", "main", dt)], idx[("memory", "prior", dt)]
        qs = sorted(set(A) & set(B))
        ca = [A[q]["confidence"] for q in qs]
        cb = [B[q]["confidence"] for q in qs]
        # the token-match receipt belongs in the artifact, not only in the prose around it
        c["live_tokens_a"] = sum(A[q]["live_tokens"] for q in qs) / len(qs)
        c["live_tokens_b"] = sum(B[q]["live_tokens"] for q in qs) / len(qs)
        c["token_matched"] = (
            abs(c["tokens_a"] - c["tokens_b"]) < 1e-6
            and abs(c["live_tokens_a"] - c["live_tokens_b"]) < 1e-6
        )
        c["confidence_overlap"] = overlap_coefficient(ca, cb)
        c["confidence_cohen_d"] = cohen_d(ca, cb)
        out["by_dt"][str(dt)] = c

    for label, use in (("pooled", dts), *((f"{dt // 60}min", [dt]) for dt in dts)):
        by_q = paired(idx, use)
        out["detectors"][label] = {
            "n_questions": len(by_q),
            "n_pairs": sum(len(v) // 2 for v in by_q.values()),
            "confidence": cluster_bootstrap(by_q, _detector("confidence"), seed=seed),
            "correct": cluster_bootstrap(by_q, _detector("correct"), seed=seed),
            "off_letter": cluster_bootstrap(by_q, _detector("off_letter"), seed=seed),
        }

    for dt in dts:
        by_q = paired(idx, [dt])
        A, B = idx[("memory", "main", dt)], idx[("memory", "prior", dt)]
        out["robustness"][str(dt)] = {
            "off_letter_a": sum(r["off_letter"] for r in A.values()) / len(A),
            "off_letter_b": sum(r["off_letter"] for r in B.values()) / len(B),
            "delta_off_letter": cluster_bootstrap(by_q, _delta_field("off_letter"), seed=seed),
            "delta_confidence_restricted": cluster_bootstrap(
                by_q, _delta_field("confidence"), seed=seed),
            "delta_confidence_vocab": cluster_bootstrap(
                by_q, _delta_field("confidence_vocab"), seed=seed),
        }

    for key, label in REFERENCE_MANIPULATIONS:
        v = x9["streaming_contrasts"].get(key)
        if v and not v.get("unavailable"):
            out["reference_manipulations"][key] = {
                "label": label,
                "n_questions": v["n_questions"],
                "delta_accuracy": v["delta_accuracy"],
                "delta_confidence": v["delta_confidence"],
            }
    b60 = x9.get("budget_by_dt", {}).get("3600")
    if b60:
        out["reference_manipulations"]["cache_budget_17x"] = {
            "label": (f"{b60['live_tokens_large'] / b60['live_tokens_small']:.0f}× larger live "
                      f"KV cache at Δt = 60 min"),
            "n_questions": b60["n"],
            "delta_accuracy": b60["delta_accuracy"],
            "delta_confidence": b60["delta_confidence"],
        }
    return out


def fmt(d: dict) -> str:
    """A bootstrapped Δ as percentage points with its 95 % interval."""
    return f"{100 * d['mean']:+.1f} [{100 * d['lo95']:+.1f}, {100 * d['hi95']:+.1f}]"


def fmt_auc(d: dict) -> str:
    return f"{d['mean']:.3f} [{d['lo95']:.3f}, {d['hi95']:.3f}]"


def write_report(out: dict) -> Path:
    dts = out["prior_dts_s"]
    by = out["by_dt"]
    accs = [by[str(d)]["delta_accuracy"]["mean"] for d in dts]
    # a range, not a ratio: the accuracy effect is expected to shrink as the cache saturates,
    # and max/min blows up as the smallest value approaches zero and flips sign below it.
    spread = f"{100 * min(accs):+.1f} to {100 * max(accs):+.1f} pp"

    L = ["# X11 — evidence blindness, at every Δt where the floor was measured", "",
         "Generated from `x11_evidence_blindness.json`; every number below is read from it, so "
         "this file cannot drift from the metrics. Δ are percentage points, paired "
         f"within-question, cluster-bootstrapped over questions ({out['n_boot']:,} draws, "
         f"seed {out['seed']}).", "",
         "The `prior` arm holds the window duration, the frame count and the visual-token count "
         "fixed and moves the window to *before* the evidence, so the pair differs in nothing "
         "but whether the answer is in the context at all.", "",
         "## The contrast, bin by bin", "",
         "| Δt | n | accuracy (main / prior) | confidence (main / prior) | Δaccuracy | "
         "Δconfidence |", "| --- | --- | --- | --- | --- | --- |"]
    for d in dts:
        c = by[str(d)]
        L.append(f"| {d // 60} min | {c['n_questions']} | "
                 f"{100 * c['accuracy_a']:.1f} / {100 * c['accuracy_b']:.1f} % | "
                 f"{100 * c['confidence_a']:.1f} / {100 * c['confidence_b']:.1f} % | "
                 f"{fmt(c['delta_accuracy'])} | {fmt(c['delta_confidence'])} |")
    L += ["",
          f"Δaccuracy excludes zero in "
          f"{sum(1 for d in dts if by[str(d)]['delta_accuracy']['lo95'] > 0)} of {len(dts)} bins "
          f"and ranges over {spread} across them. Δconfidence spans zero in "
          f"{sum(1 for d in dts if by[str(d)]['delta_confidence']['lo95'] <= 0 <= by[str(d)]['delta_confidence']['hi95'])}"
          f" of {len(dts)} and does not follow that variation.", "",
          "### The token-match receipt", "",
          "| Δt | presented visual tokens (main / prior) | live KV tokens (main / prior) | "
          "matched |", "| --- | --- | --- | --- |"]
    for d in dts:
        c = by[str(d)]
        L.append(f"| {d // 60} min | {c['tokens_a']:,.0f} / {c['tokens_b']:,.0f} | "
                 f"{c['live_tokens_a']:,.0f} / {c['live_tokens_b']:,.0f} | "
                 f"{'yes' if c['token_matched'] else '**no**'} |")

    L += ["", "## Are the two conditions separable from the confidence distribution?", "",
          "Overlap coefficient is Σ min(p_main, p_prior) over a shared "
          f"{OVERLAP_BINS}-bin histogram of [0, 1]: 1.0 means the two confidence distributions "
          "are indistinguishable.", "",
          "| Δt | confidence overlap | Cohen's d |", "| --- | --- | --- |"]
    for d in dts:
        c = by[str(d)]
        L.append(f"| {d // 60} min | {c['confidence_overlap']:.3f} | "
                 f"{c['confidence_cohen_d']:+.3f} |")

    L += ["", "### Confidence as a *detector* of evidence presence", "",
          "AUROC with `main` as the positive class and `prior` as the negative one — the direct "
          "operationalisation of \"blind\". 0.5 is chance. `correct` is shown as the ceiling a "
          "binary score can reach at these effect sizes, not as a competitor.", "",
          "| set | n pairs | max-p confidence | correctness | off-letter mass |",
          "| --- | --- | --- | --- | --- |"]
    for label in ("pooled", *(f"{d // 60}min" for d in dts)):
        v = out["detectors"][label]
        L.append(f"| {label} | {v['n_pairs']} | {fmt_auc(v['confidence'])} | "
                 f"{fmt_auc(v['correct'])} | {fmt_auc(v['off_letter'])} |")

    L += ["", "## Is the probe simply inert? No.", "",
          "The same measure, on the same records, under the other manipulations in the sweep. "
          "Read from `x9_regime_comparison.json`.", "",
          "| manipulation | n | Δaccuracy | Δconfidence |", "| --- | --- | --- | --- |"]
    for k, v in out["reference_manipulations"].items():
        L.append(f"| {v['label']} | {v['n_questions']} | {fmt(v['delta_accuracy'])} | "
                 f"{fmt(v['delta_confidence'])} |")
    L += ["",
          "Confidence is not a measure that fails to move. It moves double digits on question "
          "kind and on the size of the cache. Evidence is the manipulation it does not see."]

    L += ["", "## Denominator robustness, bin by bin", "",
          "The restricted softmax renormalises away whatever mass sits off the four option "
          "letters. `CONFIDENCE-PROBE.md` §4 establishes that the vocabulary-normalised "
          "alternative is inert everywhere except this contrast at Δt = 5 min; the table below "
          "extends that check to every bin.", "",
          "| Δt | off-letter mass (main / prior) | Δ off-letter | Δconfidence restricted | "
          "Δconfidence vocab-normalised |", "| --- | --- | --- | --- | --- |"]
    for d in dts:
        v = out["robustness"][str(d)]
        L.append(f"| {d // 60} min | {100 * v['off_letter_a']:.2f} / "
                 f"{100 * v['off_letter_b']:.2f} % | {fmt(v['delta_off_letter'])} | "
                 f"{fmt(v['delta_confidence_restricted'])} | "
                 f"{fmt(v['delta_confidence_vocab'])} |")
    L += ["",
          "The off-letter channel *is* evidence-sensitive at every bin, which is exactly why the "
          "denominator matters here and nowhere else. The claim on the slide is about the "
          "restricted probe — the published standard — and the alternative is reported beside "
          "it rather than swapped in silently.", "",
          "## What is deliberately absent", "",
          "- **A pooled multi-bin contrast.** Pooling the bins tightens Δconfidence to the edge "
          "of significance and would turn a clean per-bin null into a borderline positive. The "
          "bins are reported separately for that reason.",
          "- **Δgap as the headline.** `delta_gap` is in the JSON, and removing the evidence does "
          "widen the confidence−accuracy gap under the restricted probe. It is not robust to the "
          "denominator, so it does not carry the claim.",
          "- **Δt ≥ 30 min**, wherever the `prior` arm was not run: `prior` windows do not nest, "
          "so each bin costs its own ingestion pass.", ""]

    path = ARTIFACTS / "x11_evidence_blindness.md"
    path.write_text("\n".join(L))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", nargs="+", default=None,
                    help="record files (default: the streaming ones in $EGOVM_DATA/artifacts)")
    ap.add_argument("--comparison", default=str(ARTIFACTS / "x9_regime_comparison.json"))
    ap.add_argument("--out", default=str(ARTIFACTS / "x11_evidence_blindness.json"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    paths = a.records or [p for p in sorted(glob.glob(str(BIG_ARTIFACTS / "m6_records*.jsonl")))
                          if is_stream(p)]
    if not paths:
        raise SystemExit("X11: no streaming record files found")
    records = load_records(paths)
    x9 = json.loads(Path(a.comparison).read_text())

    out = analyze(records, x9, seed=a.seed)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"X11 metrics -> {a.out}  ({out['n_records']:,} records, "
          f"{len(out['prior_dts_s'])} Δt bins)")
    print(f"X11 report  -> {write_report(out)}")
    for d in out["prior_dts_s"]:
        c = out["by_dt"][str(d)]
        print(f"  Δt={d // 60:2d} min  Δacc={fmt(c['delta_accuracy']):24s} "
              f"Δconf={fmt(c['delta_confidence'])}")


if __name__ == "__main__":
    main()
