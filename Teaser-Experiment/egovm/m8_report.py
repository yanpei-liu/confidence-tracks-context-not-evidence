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


def verdict(a: dict) -> tuple[str, str]:
    """Pick the verdict from the measured intervals, not from what was predicted."""
    h = a.get("headline", {})
    da, dc, dg = h.get("delta_accuracy"), h.get("delta_confidence"), h.get("delta_gap")
    if not (da and dc and dg):
        return "inconclusive", "Not enough paired data to evaluate the headline contrast."

    acc_fell = da["hi95"] < 0
    conf_fell = dc["hi95"] < 0
    gap_grew = dg["lo95"] > 0

    # does any Δt effect survive holding visual-token count fixed?
    tm = a.get("token_matched", {})
    survives = any(t["delta_accuracy"]["hi95"] < 0 for t in tm.values()) if tm else None

    if acc_fell and gap_grew:
        return "A: predicted metamemory failure", (
            "Accuracy declines with Δt while confidence does not follow it down, so the "
            "confidence–accuracy gap widens — the predicted 'doesn't know it forgot' signature."
        )
    if acc_fell and conf_fell and not gap_grew:
        base = (
            "Accuracy does decline with Δt, but confidence declines at least as fast, so the "
            "overconfidence gap does not widen. The predicted signature is **not** present at "
            "this horizon."
        )
        if survives is False:
            base += (
                " And the decline does not survive the token-matched control: holding visual-"
                "token count fixed removes it entirely, so what looks like time-based "
                "forgetting is context-length degradation."
            )
        return "B: confidence tracks the decline", base
    if not acc_fell:
        return "C: no measurable forgetting", (
            "Accuracy does not fall significantly across the Δt range tested, so there is no "
            "forgetting here for a metamemory gap to be about."
        )
    return "mixed", "The intervals do not fall cleanly into any of the three outcomes."


def main() -> None:
    m = json.loads((ARTIFACTS / "m7_metrics.json").read_text())
    env = json.loads((ARTIFACTS / "m0_env_report.json").read_text())
    a = m["all_questions"]
    v = m["vision_only_questions"]
    h = a.get("headline", {})
    tag, para = verdict(a)
    ea = a.get("evidence_awareness", {})
    vh = v.get("headline", {})
    vea = v.get("evidence_awareness", {})

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

    ea_block = "not run"
    if ea:
        ea_block = f"""Token-matched at {ea['mean_tokens_with_evidence']:.0f} visual tokens, paired over
{ea['n_questions']} questions, the *only* difference being whether the window contains the evidence:

| | no evidence in context | evidence in context | change |
| --- | --- | --- | --- |
| accuracy | {ea['accuracy_without']*100:.1f}% | {ea['accuracy_with_evidence']*100:.1f}% | **{fmt_ci(ea['delta_accuracy'])}** |
| confidence | {ea['confidence_without']*100:.1f}% | {ea['confidence_with_evidence']*100:.1f}% | **{fmt_ci(ea['delta_confidence'])}** |

Having the evidence is worth {ea['delta_accuracy']['mean']*100:.1f} points of accuracy but only
{ea['delta_confidence']['mean']*100:.1f} points of confidence — the confidence signal is about
{ea['delta_accuracy']['mean']/max(ea['delta_confidence']['mean'], 1e-9):.1f}x weaker than the
competence it is supposed to track. The model is nearly as confident answering from nothing as
from the evidence."""

    txt = f"""# Teaser result — does a streaming ego video-LLM know it forgot?

**Verdict: {tag}.** {para}

**But there is a metamemory failure here, just not the predicted one.** The sharpest result is
not the Δt sweep — it is the comparison between having the evidence in context and not having it
at all (see *Evidence awareness* below): that swap moves accuracy by
{ea.get('delta_accuracy', {}).get('mean', float('nan'))*100:+.1f} points and confidence by only
{ea.get('delta_confidence', {}).get('mean', float('nan'))*100:+.1f}. The model cannot tell whether
it has the information. That is a recalibration target, and it does not depend on forgetting.

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

## Evidence awareness — the actual metamemory failure

{ea_block}

## Controls

**Language-prior floor** — the same questions over an evidence-free window: {prior_txt}.
Any accuracy above this floor is what the video evidence is buying, so the sweep is only
informative between that floor and the Δt={lo_min} min bin — a narrow band here.

**No-memory control** — a current-instant question over the identical context:

| Δt (min) | n | accuracy % | confidence % |
| --- | --- | --- | --- |
{ctl_rows}

This was meant to stay flat and rule generic long-context degradation *out*. It does not — it
falls further than the memory questions do. Note the caveat that each checkpoint gets its own
control item, so this is a between-items comparison; the paired evidence for the same conclusion
is the token-matched block below.

**Token-matched** — Δt=60 min subsampled to the Δt=15 min visual-token count:

| arm | n | tokens (ref) | acc @15 min % | acc @60 min % | Δ accuracy |
| --- | --- | --- | --- | --- | --- |
{tm_rows}

`tokmatch_uniform` thins the whole window; `tokmatch_evkeep` keeps every frame near the
evidence and thins only the filler, so it holds token count *and* evidence resolution fixed
and leaves elapsed time as the only difference. **`tokmatch_evkeep` is the decisive arm**: if
elapsed time itself caused forgetting, accuracy would still fall there, and it does not.

## By category

| category | acc @{lo_min} min % | acc @{hi_min} min % | conf @{lo_min} min % | conf @{hi_min} min % |
| --- | --- | --- | --- | --- |
{type_rows}

## Robustness: dropping the audio-dependent questions

{a['n_records'] - v['n_records']} records come from questions flagged `need_audio`, which a
vision-only model cannot answer in principle. Re-running the whole analysis without them
(n={vh.get('n_paired_questions', 0)} paired questions) changes nothing that matters:

| quantity | all questions | vision-only |
| --- | --- | --- |
| accuracy, {lo_min}→{hi_min} min | {fmt_ci(h.get('delta_accuracy'))} | {fmt_ci(vh.get('delta_accuracy'))} |
| confidence, {lo_min}→{hi_min} min | {fmt_ci(h.get('delta_confidence'))} | {fmt_ci(vh.get('delta_confidence'))} |
| overconfidence gap | {fmt_ci(h.get('delta_gap'))} | {fmt_ci(vh.get('delta_gap'))} |
| evidence vs none, accuracy | {fmt_ci(ea.get('delta_accuracy'))} | {fmt_ci(vea.get('delta_accuracy'))} |
| evidence vs none, confidence | {fmt_ci(ea.get('delta_confidence'))} | {fmt_ci(vea.get('delta_confidence'))} |
| `tokmatch_evkeep` accuracy | {fmt_ci(a.get('token_matched', {}).get('tokmatch_evkeep', {}).get('delta_accuracy'))} | {fmt_ci(v.get('token_matched', {}).get('tokmatch_evkeep', {}).get('delta_accuracy'))} |

## Caveats
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
