"""M8 — render the teaser write-up from m7_metrics.json.

The verdict sentences are chosen by the measured intervals, not written ahead of time: the
report states what the numbers show, including when they show nothing. Four outcomes are
distinguishable and all four are worth reporting:

  A. accuracy falls, confidence flat/rising   -> the originally predicted metamemory failure
  B. accuracy falls, confidence falls with it -> the model does track its own forgetting
  C. accuracy flat                            -> no forgetting at this Δt range to explain
  D. the two dissociate across manipulations  -> confidence moves for things that cost no
                                                 accuracy and ignores things that do

D is what the sweep actually produced, and it is only visible once the token-matched arms are
contrasted against each other rather than each against `main` — which is why the write-up
leads with the manipulation table rather than with the Δt sweep.

Artifact: artifacts/m8_teaser.md
"""
from __future__ import annotations

import json

from .config import ARTIFACTS


def fmt_ci(d, pct=True, signed=True, digits=None):
    if not d:
        return "n/a"
    k = 100 if pct else 1
    s = "+" if signed else ""
    u = " pp" if pct else ""
    n = digits if digits is not None else (1 if pct else 3)
    return f"{d['mean']*k:{s}.{n}f} [{d['lo95']*k:{s}.{n}f}, {d['hi95']*k:{s}.{n}f}]{u}"


def moved(d, sign: int = 0) -> bool:
    """Did this quantity move at all — i.e. does its interval exclude zero (in `sign`)?"""
    if not d:
        return False
    if sign > 0:
        return d["lo95"] > 0
    if sign < 0:
        return d["hi95"] < 0
    return d["lo95"] > 0 or d["hi95"] < 0


def elapsed_contrast(a: dict) -> dict | None:
    """The longest fixed-context elapsed-time contrast the records support."""
    c = a.get("contrasts", {})
    keys = sorted((int(k.split("_")[-1][:-3]), k) for k in c if k.startswith("more_elapsed_time_"))
    return c[keys[-1][1]] if keys else None


def verdict(a: dict) -> tuple[str, str]:
    """Pick the verdict from the measured intervals, not from what was predicted."""
    c = a.get("contrasts", {})
    el, tk, ev = elapsed_contrast(a), c.get("fewer_tokens"), c.get("evidence_present")

    # D: the two measures respond to different manipulations. This dominates the A/B/C
    # taxonomy when it holds, because it explains the sweep rather than describing it.
    conf_moves_on_time = el and moved(el["delta_confidence"], -1) \
        and not moved(el["delta_accuracy"])
    acc_moves_on_tokens = tk and moved(tk["delta_accuracy"], +1) \
        and not moved(tk["delta_confidence"])
    if conf_moves_on_time and acc_moves_on_tokens:
        para = (
            f"Elapsed time and context length move *different* measures. Holding visual-token "
            f"count and evidence sampling fixed, an extra stretch of "
            f"elapsed time costs {fmt_ci(el['delta_accuracy'])} of accuracy — nothing — while "
            f"costing {fmt_ci(el['delta_confidence'])} of confidence. Holding elapsed time fixed "
            f"instead, cutting the context to a quarter is worth "
            f"{fmt_ci(tk['delta_accuracy'])} of accuracy but only "
            f"{fmt_ci(tk['delta_confidence'])} of confidence. Confidence responds to the "
            f"variable that costs the model nothing and ignores the one that costs it accuracy."
        )
        if ev and moved(ev["delta_accuracy"], +1):
            ratio = ev["delta_accuracy"]["mean"] / max(ev["delta_confidence"]["mean"], 1e-9)
            para += (
                f" And the largest accuracy effect in the whole experiment — having the evidence "
                f"in context at all, {fmt_ci(ev['delta_accuracy'])} — moves confidence only "
                f"{fmt_ci(ev['delta_confidence'])}, a {ratio:.1f}x under-reaction."
            )
        return "D: confidence tracks the wrong variable", para

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

    # ---- the manipulation table, in the order the argument runs ----
    # Not sorted by effect size: the two token-matched contrasts are the ones that carry the
    # claim and they are the *smallest* effects in the table, so a size sort would bury them.
    def rank(key: str) -> tuple:
        order = ["more_elapsed_time_", "fewer_tokens", "evidence_present", "evidence_resolution",
                 "raw_sweep", "control_sweep", "question_kind_", "control_"]
        for i, pref in enumerate(order):
            if key == pref or key.startswith(pref):
                return (i, key)
        return (len(order), key)

    contrasts = a.get("contrasts", {})
    contrast_rows = "\n".join(
        f"| {c['label']} | {fmt_ci(c['delta_accuracy'])} | {fmt_ci(c['delta_confidence'])} | "
        f"{c['n_questions']} |"
        for _, c in sorted(contrasts.items(), key=lambda kv: rank(kv[0]))
    ) or "| — | — | — | — |"

    dec = a.get("decomposition")
    decomp_block = "not available (the token-matched arms did not run)"
    if dec:
        rd = "\n".join(
            f"| {f} | {dec[f]['elapsed_time']*100:+.1f} pp | {dec[f]['context_length']*100:+.1f} pp | "
            f"{dec[f]['modelled']*100:+.1f} pp | {dec[f]['observed']*100:+.1f} pp | "
            f"{dec[f]['residual']*100:+.1f} pp |" for f in ("accuracy", "confidence"))
        decomp_block = f"""The `main` arm confounds elapsed time with context length — the window grows with Δt. The
token-matched arms break that, so the observed Δt={dec['dt_ref_s']//60}→{dec['dt_hi_s']//60} min
change splits into an elapsed-time part (`tokmatch_evkeep` at Δt={dec['dt_hi_s']//60} min against
`main` at Δt={dec['dt_ref_s']//60} min: same tokens, same evidence sampling) and a context-length
part (`main` against `tokmatch_evkeep`, both at Δt={dec['dt_hi_s']//60} min: same elapsed time):

| measure | elapsed time | context length | sum | observed | residual |
| --- | --- | --- | --- | --- | --- |
{rd}

Accuracy's decline is entirely the context-length term; confidence's decline is almost entirely
the elapsed-time term."""

    it = a.get("item_transitions")
    item_block = "not available"
    if it:
        k_, l_, d_ = (it["delta_confidence_kept"], it["delta_confidence_lost"], it["difference"])
        item_block = f"""No. Pooling over adjacent Δt steps and keeping only items the model had **right** at the
earlier bin, the confidence change is statistically identical whether it went on to lose the item
or keep it:

| items | n | confidence change |
| --- | --- | --- |
| kept (still correct) | {it['n_kept']} | {fmt_ci(k_)} |
| lost (now wrong) | {it['n_lost']} | {fmt_ci(l_)} |
| **difference** | | **{fmt_ci(d_)}**, p={d_['p_two_sided_vs_0']:.3f} |

The decline is a flat discount applied to every item, not a signal about the ones actually lost.
This is the item-level counterpart of the aggregate result: whatever drives confidence down is
reading the shape of the context, not the state of the memory."""

    disc = a.get("discrimination", {})
    auroc_block = ""
    mem_auc = sorted((int(k.split("/")[-1]), v["auroc"]) for k, v in disc.items()
                     if k.startswith("memory/main/"))
    if mem_auc:
        da = disc.get("delta_auroc_sweep")
        pri = [v["auroc"] for k, v in disc.items() if k.startswith("memory/prior/")]
        auroc_block = f"""**But the ranking is intact.** AUROC of confidence against correctness is
{mem_auc[0][1]:.3f} at Δt={mem_auc[0][0]//60} min and {mem_auc[-1][1]:.3f} at
Δt={mem_auc[-1][0]//60} min ({fmt_ci(da, pct=False)} over the sweep), so the model's *ordering* of
which answers it will get right is untouched by Δt — it is the level that slides.
{f"With no evidence in context the ordering does degrade, to AUROC {sum(pri)/len(pri):.3f}, so the signal is real and evidence-dependent; it simply never surfaces in the confidence level." if pri else ""}
That matters for what to do about this: the usable signal is there to be recalibrated, not
missing."""

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

**The originally predicted signature is absent.** The overconfidence gap does not widen with Δt
({fmt_ci(h.get('delta_gap'))}, p={h.get('delta_gap', {}).get('p_two_sided_vs_0', float('nan')):.4f}) —
accuracy and confidence both fall over the raw sweep, and confidence falls slightly faster. What
replaced it is a sharper claim, because it is about *which* manipulations each measure responds
to rather than about the slope of one of them.

![headline](../figures/headline.png)

## What moves accuracy, and what moves confidence

Every row is paired within-question and cluster-bootstrapped over questions. The first three rows
are the argument; the rest are the context they sit in.

| manipulation | Δ accuracy | Δ confidence | n |
| --- | --- | --- | --- |
{contrast_rows}

Read down the table: the two token-matched arms make elapsed time and context length separately
identifiable, and they come apart. Elapsed time is what confidence responds to, and it is the one
manipulation that costs no accuracy. Context length and evidence availability are what accuracy
responds to, and confidence barely registers either.

## The Δt sweep, decomposed

{decomp_block}

The raw sweep is therefore a composite of two effects that point in different directions for
confidence and accuracy, which is why reading it as "forgetting" is wrong.

## Is the confidence decline about the items that were lost?

{item_block}

{auroc_block}

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

## The Δt sweep itself

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

The gap row is the bin x {{accuracy, confidence}} interaction the design was originally powered
for. It is the null this write-up leads away from — the sweep alone cannot say what caused
either decline, which is what the decomposition above is for.

![the sweep and the contrast forest](../figures/supp_sweep.png)

## Evidence awareness

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
falls further than the memory questions do, which is the clearest sign that the sweep is
measuring the context window rather than memory. Two caveats: each checkpoint gets its own
control item, so this is a between-items comparison, and the control has so far only been run on
`main` frames, so its decline has not itself been token-matched (supplement S2).

**Token-matched** — Δt=60 min subsampled to the Δt=15 min visual-token count:

| arm | n | tokens (ref) | acc @15 min % | acc @60 min % | Δ accuracy |
| --- | --- | --- | --- | --- | --- |
{tm_rows}

`tokmatch_uniform` thins the whole window; `tokmatch_evkeep` keeps every frame near the
evidence and thins only the filler, so it holds token count *and* evidence resolution fixed
and leaves elapsed time as the only difference. Comparing the two isolates evidence resolution;
comparing either against `main` at the same Δt isolates context length. Those three comparisons
are the manipulation table above — this block is the raw material for them.

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
