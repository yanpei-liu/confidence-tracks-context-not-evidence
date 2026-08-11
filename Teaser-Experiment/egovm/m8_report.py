"""M8 — render the teaser write-up from m7_metrics.json.

The verdict sentences are chosen by the measured intervals, not written ahead of time: the
report states what the numbers show, including when they show nothing. Three outcomes are
distinguishable and all three are worth reporting:

  A. accuracy falls, confidence flat/rising  -> the predicted metamemory failure
  B. accuracy falls, confidence falls with it -> the model does track its own forgetting
  C. accuracy flat                            -> no forgetting at this Δt range to explain

Artifact: artifacts/m8_teaser.md
"""
from __future__ import annotations

import json

from .config import ARTIFACTS


def fmt_ci(d, pct=True, signed=True):
    if not d:
        return "n/a"
    k = 100 if pct else 1
    s = "+" if signed else ""
    u = " pp" if pct else ""
    return f"{d['mean']*k:{s}.1f} [{d['lo95']*k:{s}.1f}, {d['hi95']*k:{s}.1f}]{u}"


def verdict(h) -> tuple[str, str]:
    da, dc, dg = h.get("delta_accuracy"), h.get("delta_confidence"), h.get("delta_gap")
    if not (da and dc and dg):
        return "inconclusive", "Not enough paired data to evaluate the headline contrast."
    acc_fell = da["hi95"] < 0
    conf_fell = dc["hi95"] < 0
    gap_grew = dg["lo95"] > 0
    if acc_fell and gap_grew:
        return "A: metamemory failure", (
            "Accuracy declines with Δt while confidence does not follow it down, so the "
            "confidence-accuracy gap widens. This is the predicted 'doesn't know it forgot' "
            "signature and it is the result the recalibration argument needs."
        )
    if acc_fell and conf_fell and not gap_grew:
        return "B: confidence tracks forgetting", (
            "Accuracy declines with Δt, but confidence declines with it, so the gap does not "
            "widen. The model degrades gracefully and is, to this resolution, aware of it — "
            "which undercuts the metamemory framing and argues for re-scoping the claim."
        )
    if not acc_fell:
        return "C: no measurable forgetting", (
            "Accuracy does not fall significantly across the Δt range tested, so there is no "
            "forgetting here for a metamemory gap to be about. The Δt range, the frame rate, "
            "or the question set would need to change before the headline claim is testable."
        )
    return "mixed", "The intervals do not fall cleanly into any of the three outcomes."


def main() -> None:
    m = json.loads((ARTIFACTS / "m7_metrics.json").read_text())
    env = json.loads((ARTIFACTS / "m0_env_report.json").read_text())
    a = m["all_questions"]
    v = m["vision_only_questions"]
    h = a.get("headline", {})
    tag, para = verdict(h)

    bins = a.get("bins", {}).get("memory/main", {})
    dts = sorted(bins, key=int)
    rows = "\n".join(
        f"| {int(d)//60} | {bins[d]['n']} | {bins[d]['accuracy']*100:.1f} | "
        f"{bins[d]['mean_confidence']*100:.1f} | {bins[d]['overconfidence']*100:+.1f} | "
        f"{bins[d]['ece']:.3f} | {bins[d]['mean_frames']:.0f} | {bins[d]['mean_visual_tokens']:.0f} |"
        for d in dts
    )

    ctl = a.get("bins", {}).get("control/main", {})
    ctl_rows = "\n".join(
        f"| {int(d)//60} | {ctl[d]['n']} | {ctl[d]['accuracy']*100:.1f} | "
        f"{ctl[d]['mean_confidence']*100:.1f} |" for d in sorted(ctl, key=int)
    ) or "| — | — | — | — |"

    prior = a.get("bins", {}).get("memory/prior", {})
    prior_txt = "not run"
    if prior:
        k = sorted(prior, key=int)[0]
        prior_txt = (f"{prior[k]['accuracy']*100:.1f}% accuracy at "
                     f"{prior[k]['mean_confidence']*100:.1f}% confidence (n={prior[k]['n']})")

    tm = a.get("token_matched", {})
    tm_rows = "\n".join(
        f"| `{arm}` | {t['n_questions']} | {t['mean_tokens_ref']:.0f} | {t['accuracy_ref']*100:.1f} | "
        f"{t['accuracy_hi']*100:.1f} | {fmt_ci(t['delta_accuracy'])} |"
        for arm, t in tm.items()
    ) or "| — | — | — | — | — | — |"

    by_type = a.get("by_type", {})
    type_rows = "\n".join(
        f"| {t} | {d[min(d, key=int)]['accuracy']*100:.1f} | {d[max(d, key=int)]['accuracy']*100:.1f} | "
        f"{d[min(d, key=int)]['mean_confidence']*100:.1f} | {d[max(d, key=int)]['mean_confidence']*100:.1f} |"
        for t, d in by_type.items() if len(d) >= 2
    ) or "| — | — | — | — | — |"

    lo_min = h.get("dt_lo_s", 0) // 60
    hi_min = h.get("dt_hi_s", 0) // 60

    txt = f"""# Teaser result — does a streaming ego video-LLM know it forgot?

**Verdict: {tag}.** {para}

## Setup

{env.get('model_hub_id')} (Apache 2.0), BF16 on one {env.get('gpu')}. EgoLifeQA participant
A1/Jake — the only released participant — restricted by Mitigation 1 to the time-invariant
categories EntityLog + EventRecall + RelationMap. Evidence timestamps come from the released
`target_time` field, which is populated on all 500 items.

Δt is elapsed **stream** time (video the model has actually observed) between the evidence
and the recall checkpoint, not wall-clock: the A1 recording has 31 gaps including six
overnight breaks. Frames are sampled at 0.2 fps at a pinned {env.get('input_res')}x{env.get('input_res')}
({env.get('expected_tokens_per_frame')} visual tokens/frame, verified constant in window length).
Confidence is the softmax over the option-letter logprobs {{A,B,C,D}}; no text is generated.
It is a proxy, reported with ECE, not a calibrated probability.

## Headline

| Δt (min) | n | accuracy % | confidence % | gap (pp) | ECE | frames | visual tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
{rows}

Paired over the {h.get('n_paired_questions', 0)} questions present at both ends
({lo_min} -> {hi_min} min), cluster-bootstrapped over questions:

| quantity | change | p |
| --- | --- | --- |
| accuracy | {fmt_ci(h.get('delta_accuracy'))} | {h.get('delta_accuracy', {}).get('p_two_sided_vs_0', float('nan')):.4f} |
| confidence | {fmt_ci(h.get('delta_confidence'))} | {h.get('delta_confidence', {}).get('p_two_sided_vs_0', float('nan')):.4f} |
| **overconfidence gap** | **{fmt_ci(h.get('delta_gap'))}** | {h.get('delta_gap', {}).get('p_two_sided_vs_0', float('nan')):.4f} |

The gap row is the bin x {{accuracy, confidence}} interaction the design was powered for.

![headline](../figures/headline.png)

## Controls

**Language-prior floor** — the same questions over an evidence-free window: {prior_txt}.
Any accuracy above this floor is what the video evidence is buying; the sweep is only
informative in the range between this floor and the Δt={lo_min} min bin.

**No-memory control** — a current-instant question over the identical context:

| Δt (min) | n | accuracy % | confidence % |
| --- | --- | --- | --- |
{ctl_rows}

If this stays flat while memory accuracy falls, generic long-context degradation is ruled out.

**Token-matched** — Δt=60 min subsampled to the Δt=15 min visual-token count:

| arm | n | tokens (ref) | acc @15 min % | acc @60 min % | Δ accuracy |
| --- | --- | --- | --- | --- | --- |
{tm_rows}

`tokmatch_uniform` thins the whole window; `tokmatch_evkeep` keeps every frame near the
evidence and thins only the filler, so it holds token count *and* evidence resolution fixed
and leaves elapsed time as the only difference.

## By category

| category | acc @{lo_min} min % | acc @{hi_min} min % | conf @{lo_min} min % | conf @{hi_min} min % |
| --- | --- | --- | --- | --- |
{type_rows}

## Caveats

- Vision only: {a['n_records'] - v['n_records']} records come from questions flagged
  `need_audio`, which this model cannot answer in principle. The `vision_only_questions`
  block in `m7_metrics.json` repeats every number with those excluded.
- The model sees a *window* around the evidence, not the full 44 h history EgoLifeQA assumes,
  so absolute accuracy is not comparable to published EgoLifeQA numbers. The claim is about
  the slope and the gap, not the level.
- One participant, one model. N per bin is bounded by the single released EgoLife participant.
- Control questions are auto-generated from the DenseCaption track and posed in Chinese
  (the captions are Chinese); they are compared only against themselves across Δt.
"""
    out = ARTIFACTS / "m8_teaser.md"
    out.write_text(txt)
    print(f"M8 -> {out}\n\nVerdict: {tag}")


if __name__ == "__main__":
    main()
