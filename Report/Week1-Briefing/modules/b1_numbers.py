#!/usr/bin/env python3
"""B1 -- every number the deck quotes, generated from the authoritative artifacts.

Emits
    generated/numbers.tex   LaTeX macros, one per quoted quantity
    generated/numbers.json  the same values with their provenance
    generated/tab-*.tex     tabular bodies built from the same sources

Sources, in the precedence CLAUDE.md sets out:
    Teaser-Experiment/artifacts/m7_metrics.json          E1 (full-reprocess)
    Teaser-Experiment/artifacts/m7_metrics_stream.json   E2 per-bin / AUROC / decomposition
    Teaser-Experiment/artifacts/x9_regime_comparison.json E2 contrasts and cross-regime
    Report/Week1-Briefing/memo/*.md                      numbers that live only in a memo

Memo-sourced values cannot be recomputed here, so each one carries a literal string
that must still be present in its memo; --check fails if a memo is edited underneath
the deck.  Run --check in CI/`make verify`; run with no arguments to regenerate.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
BRIEF = HERE.parent
REPO = BRIEF.parent.parent
ART = REPO / "Teaser-Experiment" / "artifacts"
GEN = BRIEF / "generated"

# ---------------------------------------------------------------- formatting

MINUS = "$-$"
PLUS = "$+$"


def _signed(x: float, digits: int = 1) -> str:
    s = f"{abs(x):.{digits}f}"
    return (MINUS if x < 0 else PLUS) + s


def pp(d: dict, digits: int = 1) -> str:
    """A bootstrapped delta in percentage points: +9.7 [+4.8, +14.5]."""
    return (
        f"{_signed(d['mean'] * 100, digits)} "
        f"[{_signed(d['lo95'] * 100, digits)}, {_signed(d['hi95'] * 100, digits)}]"
    )


def raw(d: dict, digits: int = 3) -> str:
    """A bootstrapped delta on its own scale (AUROC), not in pp."""
    return (
        f"{_signed(d['mean'], digits)} "
        f"[{_signed(d['lo95'], digits)}, {_signed(d['hi95'], digits)}]"
    )


def ppmean(d: dict, digits: int = 1) -> str:
    return _signed(d["mean"] * 100, digits)


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}"


def num(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def pval(d: dict) -> str:
    p = d["p_two_sided_vs_0"] if isinstance(d, dict) else d
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def thousands(n: float) -> str:
    return f"{int(round(n)):,}"


# ---------------------------------------------------------------- the memos

MEMOS = {
    "confidence": BRIEF / "memo" / "CONFIDENCE-PROBE.md",
    "egosat": BRIEF / "memo" / "RELATED-EGOSAT.md",
    "cache": BRIEF / "memo" / "STREAMING-CACHE.md",
}

# macro -> (value, memo key, literal that must still appear in that memo)
MEMO_FACTS: list[tuple[str, str, str, str]] = [
    # -- CONFIDENCE-PROBE.md
    ("MassEone", "0.99988", "confidence", "| mean mass on {A,B,C,D} | 0.99988 | 0.99530 |"),
    ("MassEtwo", "0.99530", "confidence", "| mean mass on {A,B,C,D} | 0.99988 | 0.99530 |"),
    ("ArgmaxDisagree", "0 of 23,687", "confidence", "**0 / 23,687 records, both regimes combined**"),
    ("AurocRestrEone", "0.6276", "confidence", "| max-p, restricted (as shipped) | 0.6276 | 0.6199 |"),
    ("AurocVocabEone", "0.6275", "confidence", "| max-p, vocabulary-normalised | 0.6275 | 0.6203 |"),
    ("AurocRestrEtwo", "0.6199", "confidence", "| max-p, restricted (as shipped) | 0.6276 | 0.6199 |"),
    ("AurocVocabEtwo", "0.6203", "confidence", "| max-p, vocabulary-normalised | 0.6275 | 0.6203 |"),
    ("AurocEntropyEtwo", "0.6215", "confidence", "| negative entropy | 0.6228 | 0.6215 |"),
    ("AurocMarginEtwo", "0.6171", "confidence", "| top1 $-$ top2 margin | 0.6254 | 0.6171 |".replace("$-$", "−")),
    ("VocabEvidenceEtwo", f"{PLUS}2.61 [{PLUS}0.33, {PLUS}4.94]", "confidence", "**+2.61 [+0.33, +4.94] p = 0.028**"),
    ("RestrEvidenceEtwo", f"{PLUS}0.55 [{MINUS}1.52, {PLUS}2.65]", "confidence", "**+0.55 [−1.52, +2.65] p = 0.59**"),
    ("OffLetterEvidenceEtwo", f"{MINUS}3.01 [{MINUS}4.40, {MINUS}1.82]", "confidence", "**−3.01 [−4.40, −1.82]**"),
    ("HedgeRows", "129 rows (1.38\\%), over 79 of 373 questions", "confidence", "rows affected | 129 (1.38 %), spread over **79 of 373 questions**"),
    ("HedgePrior", "8.76\\%", "confidence", "**`prior` 8.76 %** vs `main` 0.42 %"),
    ("HedgeMain", "0.42\\%", "confidence", "**`prior` 8.76 %** vs `main` 0.42 %"),
    ("HedgeEoneMax", "0.0034", "confidence", "mean 0.0001, **max 0.0034**"),
    ("BiasPredEone", "A .124 \\quad B .280 \\quad \\textbf{C .435} \\quad D .161", "confidence", "A .124 B .280 **C .435** D .161"),
    ("BiasPredEtwo", "A .134 \\quad \\textbf{B .342} \\quad C .330 \\quad D .194", "confidence", "A .134 **B .342** C .330 D .194"),
    ("BiasGold", "A .222 \\quad B .256 \\quad C .261 \\quad D .261", "confidence", "A .222 B .256 C .261 D .261"),
    ("BiasAccEone", "21.4\\% on gold-A vs 58.6\\% on gold-C", "confidence", "A 21.4 % → **C 58.6 %**"),
    ("PriorArmConfEone", "0.659", "confidence", "mean confidence 0.659 (E1) and 0.803 (E2)"),
    ("PriorArmConfEtwo", "0.803", "confidence", "mean confidence 0.659 (E1) and 0.803 (E2)"),
    # -- STREAMING-CACHE.md
    ("CleanBudgetConf", f"{MINUS}6.0 [{MINUS}7.3, {MINUS}4.7]", "cache", "**−6.0 [−7.3, −4.7]**"),
    ("CleanBudgetRatio", "3.7$\\times$", "cache", "**−6.0 pp at 3.7× is the clean, single-mechanism number**"),
    ("OffloadDefault", "0 of 12,890", "cache", "0 of 12,890 default+large rows offloaded anything"),
    ("OffloadSmall", "3,675 of 3,675", "cache", "3,675 of 3,675 `small` rows"),
    ("SaturationPoint", "$\\approx$85 frames ($\\approx$7 min)", "cache", "The sweep crosses saturation at ≈85 frames (≈7 min)"),
    ("SurvivingShare", "$\\approx$11\\%", "cache", "~11 % of presented survives"),
    ("SmallBatchNote", "1,024 mutable tokens vs a 1,600-token ingestion batch", "cache", "**smaller than one 1,600-token ingestion batch**"),
    ("LiveTokensIdentical", "9,108", "cache", "**both sit at exactly 9,108 live tokens**"),
    ("EoneClockIsText", "$<$150.0 seconds$>$", "cache", "The clock is **ordinary text the model tokenizes and reads**"),
    ("TimestripCost", "$\\approx$1 GPU-h", "cache", "**8.1 Strip the timestamps from E1 (~1 GPU-h).**"),
    # -- RELATED-EGOSAT.md
    ("EgosatCite", "Lei et al., arXiv 2606.24422, 23 Jun 2026", "egosat", "# EgoSAT (Lei et al., arXiv 2606.24422, 23 Jun 2026)"),
    ("EgosatScale", "1,997 Ego4D sessions, 165 h, $\\sim$4,800 MCQ", "egosat", "1,997 Ego4D sessions (165 h, avg 296.8 s), ~4,800 MCQ"),
    ("EgosatLag", "8 / 16 / 24 s", "egosat", "τ/2τ/3τ = **8 / 16 / 24 s**"),
    ("EgosatUnderreact", "3--7$\\times$, or an outright sign flip", "egosat", "ratios of 3–7× or an outright sign"),
    ("EgosatAccBand", f"{PLUS}4.5 to {PLUS}11.6 pp", "egosat", "Theirs +4.5 to +11.6 pp"),
    ("EgosatConfBand", f"{MINUS}4.6 to {PLUS}3.3 pp", "egosat", "Theirs −4.6 to +3.3 pp"),
    ("EgosatStreamSlopes", f"{PLUS}0.99 and {PLUS}4.61", "egosat", "models give +0.99 and +4.61"),
    ("EgosatSixOfEight", "six of eight $\\geq$ 0", "egosat", "six of eight are ≥ 0"),
    ("EgosatMemHorizon", "3.6 pp and 2.4 pp, both non-monotone", "egosat", "increasing the memory horizon does not lead to uniform gains"),
    ("EgosatOnlineGap", "31.63 vs 34.46", "egosat", "TimeChat-Online-7B\n31.63 vs Qwen2.5-VL-7B 34.46"),
    ("WinsorNinety", f"{PLUS}1.18 [{MINUS}0.92, {PLUS}3.28], p = 0.259", "confidence", "| 0.90 | +1.18 [−0.92, +3.28] p = 0.259 |"),
    ("EoneEvidenceBoth", f"{PLUS}3.32 [{PLUS}1.24, {PLUS}5.40]", "confidence", "| Δconf **restricted** | +3.32 [+1.24, +5.40] p = 0.003 |"),
    ("OffLetterSweep", f"{PLUS}0.60 [{PLUS}0.21, {PLUS}1.11]", "confidence", "**+0.60 [+0.21, +1.11]**"),
    ("EgosatSeventyTwo", f"{PLUS}9.45 / {MINUS}4.57", "egosat", "| Qwen2.5-VL-72B | +9.45 | **−4.57** | sign-flipped |"),
    ("EgosatSevenB", f"{PLUS}10.05 / {PLUS}1.75", "egosat", "| Qwen2.5-VL-7B | +10.05 | +1.75 | 5.7× |"),
    ("EgosatTimeChat", f"{PLUS}4.51 / {PLUS}0.80", "egosat", "| TimeChat-Online-7B | +4.51 | +0.80 | 5.6× |"),
    ("EgosatRoi", f"{PLUS}5.42 / {PLUS}0.74", "egosat", "| ROI-TimeChat-Online | +5.42 | +0.74 | 7.3× |"),
    ("EgosatCcMinusCw", f"{PLUS}2.78 and {MINUS}0.35", "egosat", "separations of +2.78 and **−0.35**"),
    ("EgosatAccDecline", f"{MINUS}2.6 to {MINUS}7.4 pp", "egosat", "monotonically over 8 → 24 s (−2.6 to −7.4 pp)"),
]


# ---------------------------------------------------------------- generation

def build() -> tuple[dict, dict]:
    e1 = json.load(open(ART / "m7_metrics.json"))["all_questions"]
    e2m = json.load(open(ART / "m7_metrics_stream.json"))["all_questions"]
    x9 = json.load(open(ART / "x9_regime_comparison.json"))

    sc = x9["streaming_contrasts"]
    xr = x9["cross_regime"]
    bd = x9["budget_by_dt"]
    c1 = e1["contrasts"]
    d1 = e1["decomposition"]
    d2 = e2m["decomposition"]

    V: dict[str, tuple[str, str]] = {}

    def put(macro: str, value: str, source: str) -> None:
        if macro in V:
            raise SystemExit(f"duplicate macro {macro}")
        V[macro] = (value, source)

    M7 = "m7_metrics.json"
    M7S = "m7_metrics_stream.json"
    X9 = "x9_regime_comparison.json"

    # -- scale of the two experiments
    put("NrecEone", thousands(e1["n_records"]), f"{M7} all_questions.n_records")
    put("NrecEtwo", thousands(x9["n_streaming_records"]), f"{X9} n_streaming_records")
    put("Nquestions", str(e1["headline"]["n_paired_questions"]), f"{M7} headline.n_paired_questions")

    # -- E1: the original prediction
    put("EoneGap", pp(e1["headline"]["delta_gap"]), f"{M7} headline.delta_gap")
    put("EoneGapP", pval(e1["headline"]["delta_gap"]), f"{M7} headline.delta_gap")
    put("EtwoGap", pp(e2m["headline"]["delta_gap"]), f"{M7S} headline.delta_gap")
    put("EtwoGapP", pval(e2m["headline"]["delta_gap"]), f"{M7S} headline.delta_gap")

    # -- E1: the raw sweep and the contrasts that decompose it
    put("EoneSweepAcc", pp(e1["headline"]["delta_accuracy"]), f"{M7} headline.delta_accuracy")
    put("EoneSweepConf", pp(e1["headline"]["delta_confidence"]), f"{M7} headline.delta_confidence")
    for macro, key in [
        ("EoneElapsed", "more_elapsed_time_60min"),
        ("EoneTokens", "fewer_tokens"),
        ("EoneEvidence", "evidence_present"),
        ("EoneEvRes", "evidence_resolution"),
        ("EoneKind", "question_kind_1min"),
    ]:
        put(macro + "Acc", pp(c1[key]["delta_accuracy"]), f"{M7} contrasts.{key}.delta_accuracy")
        put(macro + "Conf", pp(c1[key]["delta_confidence"]), f"{M7} contrasts.{key}.delta_confidence")

    ev = e1["evidence_awareness"]
    ratio = ev["delta_accuracy"]["mean"] / ev["delta_confidence"]["mean"]
    put("EoneEvidenceRatio", f"{ratio:.1f}$\\times$", f"{M7} evidence_awareness (derived ratio)")
    put("EoneBinOneAcc", pct(e1["bins"]["memory/main"]["60"]["accuracy"]), f"{M7} bins.memory/main.60.accuracy")
    put("EoneBinOneConf", pct(e1["bins"]["memory/main"]["60"]["mean_confidence"]), f"{M7} bins.memory/main.60.mean_confidence")
    put("EoneFloorAcc", pct(ev["accuracy_without"]), f"{M7} evidence_awareness.accuracy_without")
    put("EoneFloorConf", pct(ev["confidence_without"]), f"{M7} evidence_awareness.confidence_without")
    # the usable band is bin-1 accuracy above the no-evidence floor, not the 5 min contrast
    put("EoneRange", f"{(e1['bins']['memory/main']['60']['accuracy'] - ev['accuracy_without']) * 100:.1f}",
        f"{M7} bins.memory/main.60.accuracy - evidence_awareness.accuracy_without (derived)")
    put("EtwoRange", f"{(e2m['bins']['memory/main']['60']['accuracy'] - e2m['bins']['memory/prior']['300']['accuracy']) * 100:.1f}",
        f"{M7S} bins.memory/main.60.accuracy - bins.memory/prior.300.accuracy (derived)")

    # -- E1: decomposition, discrimination, item level
    put("EoneDecompAccElapsed", ppmean({"mean": d1["accuracy"]["elapsed_time"]}, 1), f"{M7} decomposition.accuracy.elapsed_time")
    put("EoneDecompAccContext", ppmean({"mean": d1["accuracy"]["context_length"]}, 1), f"{M7} decomposition.accuracy.context_length")
    put("EoneDecompConfElapsed", ppmean({"mean": d1["confidence"]["elapsed_time"]}, 1), f"{M7} decomposition.confidence.elapsed_time")
    put("EoneDecompConfContext", ppmean({"mean": d1["confidence"]["context_length"]}, 1), f"{M7} decomposition.confidence.context_length")
    put("EoneAurocLo", num(e1["discrimination"]["memory/main/60"]["auroc"]), f"{M7} discrimination.memory/main/60.auroc")
    put("EoneAurocHi", num(e1["discrimination"]["memory/main/3600"]["auroc"]), f"{M7} discrimination.memory/main/3600.auroc")
    put("EoneAurocFloor", num(e1["discrimination"]["memory/prior/300"]["auroc"]), f"{M7} discrimination.memory/prior/300.auroc")
    put("EoneAurocDelta", raw(e1["discrimination"]["delta_auroc_sweep"]), f"{M7} discrimination.delta_auroc_sweep")
    put("EoneItemDiff", pp(e1["item_transitions"]["difference"]), f"{M7} item_transitions.difference")
    put("EoneItemDiffP", pval(e1["item_transitions"]["difference"]), f"{M7} item_transitions.difference")

    # -- E2: the contrasts
    for macro, key in [
        ("EtwoEvidence", "evidence_vs_none"),
        ("EtwoElapsed", "elapsed_time_fixed_frames"),
        ("EtwoFourx", "context_length_4x_tokens"),
        ("EtwoThinned", "evidence_thinned"),
        ("EtwoTimestamps", "timestamps_on_vs_off"),
        ("EtwoKind", "control_vs_memory"),
        ("EtwoRaw", "elapsed_time_raw"),
        ("EtwoLargeDefault", "context_length_large_vs_default"),
    ]:
        put(macro + "Acc", pp(sc[key]["delta_accuracy"]), f"{X9} streaming_contrasts.{key}.delta_accuracy")
        put(macro + "Conf", pp(sc[key]["delta_confidence"]), f"{X9} streaming_contrasts.{key}.delta_confidence")

    b60 = bd["3600"]
    put("EtwoBudgetAcc", pp(b60["delta_accuracy"]), f"{X9} budget_by_dt.3600.delta_accuracy")
    put("EtwoBudgetConf", pp(b60["delta_confidence"]), f"{X9} budget_by_dt.3600.delta_confidence")
    put("EtwoBudgetRatio", f"{b60['live_tokens_large'] / b60['live_tokens_small']:.0f}$\\times$", f"{X9} budget_by_dt.3600 (derived)")
    put("EtwoBudgetLive", f"{thousands(b60['live_tokens_large'])} vs {thousands(b60['live_tokens_small'])}", f"{X9} budget_by_dt.3600 live_tokens_*")

    # -- E2: decomposition, discrimination, item level
    put("EtwoDecompConfElapsed", ppmean({"mean": d2["confidence"]["elapsed_time"]}, 2), f"{M7S} decomposition.confidence.elapsed_time")
    put("EtwoDecompConfContext", ppmean({"mean": d2["confidence"]["context_length"]}, 2), f"{M7S} decomposition.confidence.context_length")
    put("EtwoDecompConfObs", ppmean({"mean": d2["confidence"]["observed"]}, 2), f"{M7S} decomposition.confidence.observed")
    put("EtwoDecompResid", ppmean({"mean": d2["confidence"]["residual"]}, 2), f"{M7S} decomposition.confidence.residual")
    put("EtwoAurocLo", num(e2m["discrimination"]["memory/main/60"]["auroc"]), f"{M7S} discrimination.memory/main/60.auroc")
    put("EtwoAurocHi", num(e2m["discrimination"]["memory/main/3600"]["auroc"]), f"{M7S} discrimination.memory/main/3600.auroc")
    put("EtwoAurocFloor", num(e2m["discrimination"]["memory/prior/300"]["auroc"]), f"{M7S} discrimination.memory/prior/300.auroc")
    put("EtwoAurocDelta", raw(e2m["discrimination"]["delta_auroc_sweep"]), f"{M7S} discrimination.delta_auroc_sweep")
    put("EtwoAurocP", pval(e2m["discrimination"]["delta_auroc_sweep"]), f"{M7S} discrimination.delta_auroc_sweep")
    put("EtwoItemDiff", pp(e2m["item_transitions"]["difference"]), f"{M7S} item_transitions.difference")
    put("EtwoItemDiffP", pval(e2m["item_transitions"]["difference"]), f"{M7S} item_transitions.difference")
    put("EtwoItemLost", str(e2m["item_transitions"]["n_lost"]), f"{M7S} item_transitions.n_lost")
    put("EtwoBinOneAcc", pct(e2m["bins"]["memory/main"]["60"]["accuracy"]), f"{M7S} bins.memory/main.60.accuracy")
    put("EtwoFloorAcc", pct(e2m["bins"]["memory/prior"]["300"]["accuracy"]), f"{M7S} bins.memory/prior.300.accuracy")

    # -- cross-regime differences of contrasts
    for macro, key in [
        ("XregElapsed", "elapsed_time_fixed_frames"),
        ("XregFourx", "context_length_4x_tokens"),
        ("XregEvidence", "evidence_vs_none"),
        ("XregKind", "control_vs_memory"),
    ]:
        put(macro + "Acc", pp(xr[key]["diff_accuracy"]), f"{X9} cross_regime.{key}.diff_accuracy")
        put(macro + "Conf", pp(xr[key]["diff_confidence"]), f"{X9} cross_regime.{key}.diff_confidence")

    # -- memo-sourced values
    memo_text = {k: p.read_text(encoding="utf-8") for k, p in MEMOS.items()}
    missing = []
    for macro, value, memo, literal in MEMO_FACTS:
        if literal not in memo_text[memo]:
            missing.append((macro, memo, literal))
        put(macro, value, f"{MEMOS[memo].name} (literal check)")
    if missing:
        for macro, memo, literal in missing:
            print(f"B1: {macro}: literal not found in {MEMOS[memo].name}: {literal!r}", file=sys.stderr)
        raise SystemExit(f"B1: {len(missing)} memo literal(s) no longer present -- deck would quote a stale number")

    # -- generated tabular bodies
    tables = {}

    rows = []
    for dt, label in [("60", "1"), ("300", "5"), ("900", "15"), ("1800", "30"), ("3600", "60")]:
        b = e1["bins"]["memory/main"][dt]
        rows.append(
            f"{label} & {pct(b['accuracy'])} & {pct(b['mean_confidence'])} & "
            f"{_signed(b['overconfidence'] * 100)} & {thousands(b['mean_visual_tokens'])} \\\\"
        )
    tables["TabEoneSweep"] = "\n".join(rows)

    rows = []
    for dt, label in [("60", "1"), ("300", "5"), ("900", "15"), ("1800", "30"), ("3600", "60")]:
        b = bd[dt]
        rows.append(
            f"{label} & {thousands(b['live_tokens_large'])} / {thousands(b['live_tokens_small'])} & "
            f"{pp(b['delta_accuracy'])} & {pp(b['delta_confidence'])} \\\\"
        )
    tables["TabEtwoBudget"] = "\n".join(rows)

    rows = []
    for dt, label in [("60", "1"), ("300", "5"), ("900", "15"), ("1800", "30"), ("3600", "60")]:
        b = bd[dt]
        rows.append(f"{label} & {pp(b['delta_accuracy'])} & {pp(b['delta_confidence'])} \\\\")
    tables["TabEtwoBudgetNarrow"] = "\n".join(rows)

    rows = []
    for label, key in [
        ("evidence in context vs none", "evidence_vs_none"),
        ("$+$45 min elapsed, frames + evidence fixed", "elapsed_time_fixed_frames"),
        ("3.8$\\times$ more presented tokens", "context_length_4x_tokens"),
        ("no-memory vs recall question", "control_vs_memory"),
    ]:
        r = xr[key]
        rows.append(
            f"{label} & {ppmean({'mean': r['full_delta_confidence']})} & "
            f"{ppmean({'mean': r['streaming_delta_confidence']})} & {pp(r['diff_confidence'])} \\\\"
        )
    tables["TabCrossRegime"] = "\n".join(rows)

    return V, tables


def render(V: dict[str, tuple[str, str]]) -> str:
    out = [
        "% Generated by modules/b1_numbers.py -- do not edit.",
        "% Every value below is derived from the artifacts named in generated/numbers.json.",
        "",
    ]
    for macro, (value, source) in V.items():
        out.append(f"\\newcommand{{\\{macro}}}{{{value}}}%  <- {source}")
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if the generated files are stale")
    args = ap.parse_args()

    V, tables = build()
    tex = render(V)
    meta = {m: {"value": v, "source": s} for m, (v, s) in V.items()}
    meta_txt = json.dumps(meta, indent=1, ensure_ascii=False) + "\n"

    tabtex = ["% Generated by modules/b1_numbers.py -- do not edit.", ""]
    for name, body in tables.items():
        tabtex.append(f"\\newcommand{{\\{name}}}{{%\n{body}}}")
        tabtex.append("")
    targets = {
        GEN / "numbers.tex": tex,
        GEN / "numbers.json": meta_txt,
        GEN / "tables.tex": "\n".join(tabtex),
    }

    if args.check:
        stale = [p.name for p, body in targets.items() if not p.exists() or p.read_text(encoding="utf-8") != body]
        if stale:
            print(f"B1 FAIL: stale or missing: {', '.join(stale)} -- run `make numbers`", file=sys.stderr)
            return 1
        print(f"B1 ok: {len(V)} macros, {len(tables)} tables, all current")
        return 0

    GEN.mkdir(exist_ok=True)
    for p, body in targets.items():
        p.write_text(body, encoding="utf-8")
    print(f"B1: wrote {len(V)} macros and {len(tables)} tables to {GEN.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
