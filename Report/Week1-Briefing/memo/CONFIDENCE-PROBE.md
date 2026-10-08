# The confidence probe — is `max softmax({A,B,C,D})` the right construct?

*Every number below was recomputed from the committed records (`$EGOVM_DATA/artifacts/m6_records*.jsonl`,
7,122 E1 rows and 17,311 E2 rows as loaded) using `egovm.m7_analyze.contrast` /
`cluster_bootstrap` and `egovm.streaming.x9_compare.streaming_specs` unchanged, so the contrast
machinery is identical to the one that produced [`artifacts/m8_teaser.md`](../../../Teaser-Experiment/artifacts/m8_teaser.md)
and [`artifacts/x9_streaming.md`](../../../Teaser-Experiment/artifacts/x9_streaming.md). The E1 row count
is larger than the sweep's 5,540 scored records because it is the raw file load, before M7 drops
`row_id`s absent from the plan; the contrasts below are computed on M7's own cells, so this does not
affect them. Δ are percentage points, paired within-question, cluster-bootstrapped over questions
(5,000 draws, seed 0).*

Prompted by the question: *should confidence be a softmax over the four option letters, or over the
whole vocabulary?* The short answer is that the choice is inert almost everywhere, that the one
place it is not is load-bearing, and that the interesting quantity turns out to be the thing the
restricted softmax throws away.

---

## 1. Verdict

**Keep `confidence = max softmax({A,B,C,D})` as the primary metric.** Three reasons, in order of
weight:

1. **It is the coherent denominator.** Correctness is `argmax` over the four letters *by
   construction*. The restricted softmax is therefore a probability of the event being scored
   ("this letter is the right one, given that I answer with one of them"); the vocabulary softmax
   is a probability of a different event ("this letter is the next token"), which folds in
   instruction-following. ECE against MCQ correctness is only coherent for the restricted version.
2. **It is field-standard, not our idiosyncrasy.** EgoSAT Eq. 6 is our definition verbatim, tracing
   to Tsvilodub et al. 2024 (arXiv 2403.00998) — see [`RELATED-EGOSAT.md`](RELATED-EGOSAT.md) §2.1.
   That memo already actions attributing the probe rather than presenting it as a design decision;
   this memo is the empirical half of the same conclusion.
3. **The alternative changes nothing we report, except one number where it changes it for the
   wrong reason** (§4).

The design choice that *does* deserve scrutiny is not the denominator. It is option-letter bias
(§6), and to a lesser extent the fact that mean confidence in pp is not invariant to monotone
rescaling.

---

## 2. The alternative is free — it is already on disk

Both backends record `letter_logprobs` — the full-vocabulary `log_softmax` read at the letter ids —
alongside `letter_probs` (`scoring.py:133-136`, `streaming/backend.py:388-389`). So with

```
M  = Σ_{L∈{A,B,C,D}} exp(letter_logprobs[L])      # mass on the option letters
c_vocab = c_restricted × M
```

the vocabulary-normalised confidence is recoverable for every committed record at **zero GPU cost**,
and `M` is itself a measurable quantity. No re-run is needed to answer the question, and none was
done.

---

## 3. Where the choice makes no difference

### 3.1 The model puts essentially all its next-token mass on the letters

| | E1 full-reprocess | E2 streaming |
| --- | ---: | ---: |
| mean mass on {A,B,C,D} | 0.99988 | 0.99546 |
| median | 0.99996 | 0.99997 |
| p5 / p1 | 0.9997 / 0.9994 | 0.9971 / 0.9242 |
| fraction with mass < 0.99 | 0.14 % | 2.50 % |
| fraction with mass < 0.50 | 0.00 % | 0.28 % |
| mean confidence, restricted → vocab | 0.7099 → 0.7098 | 0.8435 → 0.8400 |
| **`argmax` disagreements** | **0 / 24,433 records, both regimes combined** | |

Prediction and therefore accuracy are *identical* under the two definitions in every record we have.

### 3.2 Discrimination is unchanged — and so is the choice of scalar

AUROC of the score against correctness, memory rows only (E1 n = 3,746; E2 n = 10,071):

| discriminator | E1 | E2 |
| --- | ---: | ---: |
| max-p, restricted (as shipped) | 0.6276 | 0.6196 |
| max-p, vocabulary-normalised | 0.6275 | 0.6198 |
| negative entropy | 0.6228 | 0.6209 |
| top1 − top2 margin | 0.6254 | 0.6169 |
| mass on letters alone | 0.5761 | 0.5620 |

Two things worth carrying into the paper. First, the denominator is invisible to the ranking.
Second — the stronger robustness statement — **however the four-way distribution is collapsed to a
scalar, the answer is the same to 2–3 decimals**, so no conclusion rests on the choice of `max` over
entropy or margin. The last row is discussed in §5; it is not a rescaling of the others.

### 3.3 Every contrast but one is unchanged to 0.01 pp

Recomputing M7's `contrast_specs` (E1) and X9's `streaming_specs` (E2) under both definitions:

| contrast | Δconf restricted | Δconf vocab |
| --- | ---: | ---: |
| E1 `raw_sweep` | −9.63 [−11.35, −7.90] | −9.65 [−11.37, −7.92] |
| E1 `more_elapsed_time_60min` | −3.68 [−4.93, −2.43] | −3.68 [−4.93, −2.43] |
| E1 `question_kind_1min` | +13.80 [+10.76, +16.79] | +13.80 [+10.77, +16.79] |
| E2 `context_length_default_vs_small` | −7.43 [−8.93, −5.90] | −7.42 [−8.92, −5.89] |
| E2 `context_length_4x_tokens` | −6.71 [−8.42, −4.97] | −6.71 [−8.42, −4.96] |
| E2 `elapsed_time_fixed_frames` | −0.13 [−1.67, +1.40] | +0.08 [−1.49, +1.67] |
| E2 `timestamps_on_vs_off` | −0.72 [−1.22, −0.24] | −0.71 [−1.20, −0.23] |
| E2 `control_vs_memory` | +12.57 [+10.78, +14.37] | +12.73 [+10.94, +14.55] |

The two headline claims are untouched. E2's context-length effect — the rotation that
[`INTERPRETATION.md`](../../../Teaser-Experiment/INTERPRETATION.md) Part 1 is built on — is identical
to two decimal places, and so is E2's null on elapsed time at fixed frames.

---

## 4. The one place it does — and it is load-bearing

| `evidence_vs_none` (main@5 min − prior@5 min) | E1 | E2 |
| --- | ---: | ---: |
| Δaccuracy | +9.66 [+4.81, +14.44] | +8.06 [+2.95, +13.14] |
| Δconf **restricted** | +3.32 [+1.24, +5.40] p = 0.003 | **+0.55 [−1.52, +2.65] p = 0.59** |
| Δconf **vocabulary** | +3.32 [+1.25, +5.40] p = 0.003 | **+2.61 [+0.33, +4.94] p = 0.028** |
| Δ mass on letters | +0.003 pp, p = 0.34 | **+3.01 pp [+1.82, +4.40], p < 0.001** |

Under vocabulary normalisation, E2's confidence stops being evidence-blind. That flips things that
are actually written down:

- **The X9 verdict.** `x9_compare.verdict` selects on
  `evidence_blind = ev_a.lo95 > 0 and ev_c.lo95 ≤ 0 ≤ ev_c.hi95` (`x9_compare.py:192`). With
  `ev_c.lo95 = +0.33 > 0` the flag is false, `widens` and `narrows` are both false, and the module
  emits **Outcome 1 — "confidence tracks eviction"** instead of **Outcome 3 — "the dissociation
  replicates."** The verdict sentence is chosen by the numbers, which is the whole point of writing
  it that way, so a denominator change silently rewrites the paper's conclusion.
- **INTERPRETATION.md Part 2.** The E2 − E1 difference on evidence Δconfidence (−2.8 [−5.1, −0.4])
  would shrink to roughly −0.7 and lose significance, taking with it the claim that the confidence
  response to evidence is *weaker* under streaming.
- **[`RELATED-EGOSAT.md`](RELATED-EGOSAT.md) §3.2.** The table's `ours — E2 (streaming)` row
  (+8.1 / +0.5, "16× under-reaction, CI through zero") becomes +8.1 / +2.6, i.e. a 3.1×
  under-reaction — moving us out of the extreme band we currently place ourselves in alongside
  their Qwen2.5-VL-72B.

**So: do not switch the denominator quietly.** It is not a cosmetic normalisation.

---

## 5. What the flip is actually made of — implicit abstention

The E2 effect is not a smooth rescaling. Winsorising the mass from below removes it:

| floor on `M` | E2 `evidence_vs_none`, Δconf vocab |
| --- | ---: |
| none | +2.61 [+0.33, +4.94] p = 0.028 |
| 0.50 | +2.22 [+0.01, +4.44] p = 0.049 |
| 0.90 | +1.18 [−0.92, +3.28] p = 0.259 |
| 0.99 | +0.66 [−1.41, +2.76] p = 0.524 |

It rides on a small, heavy-tailed minority of rows. Characterising rows with off-letter mass > 10 %
(E2 memory rows, n = 10,071):

| | value |
| --- | --- |
| rows affected | 130 (1.29 %), spread over **79 of 373 questions** |
| by arm | **`prior` 5.31 %** vs `main` 0.42 % |
| by Δt | 1 min 4.24 %, 5 min 1.77 %, 15 min 0.91 %, **30 min 0.00 %, 60 min 0.04 %** |
| by condition | `stream` 2.35 %, `small` 0.54 %, `nots` 0.48 %, `large` 0.32 % |
| accuracy on those rows | 0.338, vs 0.367 elsewhere |
| E1 equivalent | `prior`@5 min off-mass: mean 0.0001, **max 0.0034** — the phenomenon does not exist |

> **Recomputed 2026-08-21 on 17,311 E2 records**, after the `prior` arm was extended to Δt ∈
> {30, 60} min (X11 / `sconf.PRIOR_DTS`). The two rates that moved are denominator effects, not
> a change in the phenomenon: `prior` went from 1,119 to 1,865 rows and the two new bins
> essentially never hedge, so the pooled `prior` rate fell 8.76 % → 5.31 % and the overall rate
> 1.38 % → 1.29 % while the *count* rose 129 → 130. The per-Δt row is the stable one, and the
> new bins confirm what it predicted: hedging is a short-Δt phenomenon and is gone by 30 min.
>
> Note what this episode shows about the tripwire. `b1_numbers.py --check` verifies that each
> literal below still *appears* in this memo; it cannot verify that the literal is still *true*
> of the records. Adding records is therefore a silent way to make the deck quote a stale
> number, and it did. Until these tables are generated (INTERPRETATION.md §2.8 item 3,
> `x13_confidence_robustness.py`), **re-run this memo's numbers whenever the record set grows.**

Read plainly: **with a bounded cache and no evidence in the window, the streaming model sometimes
stops wanting to emit a letter at all.** Paired within-question contrasts on *off-letter* mass
(signs are for off-letter mass, so positive means more hedging on the left-hand condition):

| contrast | E1 Δ off-letter mass | E2 Δ off-letter mass |
| --- | ---: | ---: |
| evidence − none (main@5 − prior@5) | −0.003 pp, p = 0.34 | **−3.01 [−4.40, −1.82]**, p < 0.001 |
| raw sweep (main@1 min − main@60 min) | −0.032 pp, p < 0.001 | **+0.60 [+0.21, +1.11]**, p < 0.001 |
| control − memory | −0.015 pp, p < 0.001 | **−0.20 [−0.38, −0.06]**, p < 0.001 |

Removing the evidence *raises* off-letter mass by 3.0 pp in E2; E1's equivalent effects are two to
three orders of magnitude smaller and of no practical size, however significant.

That is a metacognitive signal — an *implicit abstention* — living in exactly the regime where max-p
is silent. It should be reported as **its own quantity**, not folded into confidence, for two
reasons: the two are different constructs ("which of these four, given that I answer" vs "do I
answer in the required form at all"), and folding them lets a 1.4 % tail rewrite §4's verdict.

Note also that the two channels move in *opposite* directions on the Δt axis: at 1 min E2 confidence
is highest (0.844) and off-letter mass is also highest (mean `M` = 0.978), falling to `M` = 0.9998 by
30 min. More peaked given a letter, and less willing to produce one. Do not describe them as one
signal.

This is the cheapest positive result available to us: it costs no GPU time, it is recoverable from
committed records, and it converts one of E2's nulls into a measured effect — provided it is labelled
as abstention rather than confidence.

---

## 6. The construct threat that is real — option-letter bias

Not the denominator. This:

| | gold distribution | prediction distribution | accuracy by gold letter |
| --- | --- | --- | --- |
| E1 | A .222 B .256 C .261 D .261 | A .124 B .280 **C .435** D .161 | A 21.4 % → **C 58.6 %** |
| E2 | (same) | A .134 **B .342** C .330 D .194 | A 22.2 % → **B 46.2 %** |

Mean per-letter probability follows the same skew (E1: A .163 / B .270 / C .364 / D .204). In the
`prior` arm — no evidence in the window at all — the model still predicts the same skewed letters at
mean confidence 0.659 (E1) and 0.803 (E2).

Consequences, in order of importance:

1. **This is a mechanism for evidence-blind confidence, which we currently report only
   phenomenologically.** A substantial share of what we call confidence is a content-independent
   option prior, and a prior does not move when evidence is removed. Stating the mechanism is
   stronger than stating the phenomenon.
2. **The paired design largely survives it, and that defence should be written down.** The letter
   prior is a property of the question (its option ordering), held fixed across arms, so it
   differences out of every within-question contrast to first order. This is a further reason the
   level claims are weak (`INTERPRETATION.md` 0.2–0.3) and the contrasts are not.
3. **It is an obvious reviewer question, and EgoSAT already runs the check we do not** — their C10
   includes an option-order re-shuffle. The cheap strengthener is a cyclic option-permutation arm at
   one Δt bin on `main` and `prior` (4× cost on a subset), averaging per-option probability across
   permutations before the softmax.

---

## 7. Alternatives, ranked

| option | worth it? |
| --- | --- |
| **off-letter mass as a secondary metric** | **Yes — highest value.** Free from committed records; evidence-sensitive in E2 where max-p is not (§5). |
| **cyclic option-permutation debias** | **Yes — second.** Addresses the actual construct threat (§6); ~4× on one bin. |
| **S4 explicit abstention ({A,B,C,D,E})** | Yes, as already planned in PLAN.md — the principled version of what off-letter mass measures accidentally. Keep the existing incomparability caveat: a five-way softmax is not comparable with the four-way confidences the headline rests on. |
| whole-vocabulary softmax as *primary* | **No.** Inert in E1, wrong denominator in principle, and in E2 it flips a verdict on a 1.4 % tail (§4). Report it as a robustness row — it is free. |
| entropy / margin instead of max-p | No — measured, inert (§3.2). Cite as robustness. |
| length-normalised loglik over option *text* | Only as a deliberate second protocol. Immune to letter bias, but imports surface-form and length bias and a normalisation choice. |
| verbalised confidence ("0–100") | A different construct — a metacognitive *report*, arguably closer to "metamemory" than logits are. A strong addition, never a replacement; requires generation. |

One caveat to carry regardless of any of this: **mean confidence in pp is not invariant to monotone
rescaling**, so the magnitude of E2's −15.2 pp budget effect is scale-dependent even though its sign
and significance are not. AUROC is the rank-based backstop and is already reported.

---

## 8. What has to change on our side

*Status as of 2026-08-20 — all **must fix** items below are actioned; see the pointers.*

| # | landed in |
| --- | --- |
| 1 | `INTERPRETATION.md` §0.1 — denominator named, the one sensitive cell quoted, and §2.7 carries it on the ladder |
| 2 | `EXTENDING.md` §7 records the sensitivity; emitting both values from `x9_compare` is a code change, on the roadmap |
| 3 | `CLAUDE.md` *Design decisions that held → Confidence* |
| 4 | `INTERPRETATION.md` §2.6 (bias table + the pairing defence) |
| 5–7 | not actioned; carried on the roadmap, `INTERPRETATION.md` §2.8 |

**Must fix:**

1. **`INTERPRETATION.md` 0.1** — extend the ground rule from "not a calibrated probability" to state
   the denominator explicitly, that the vocabulary-normalised alternative was computed on the
   committed records and is inert except on E2 `evidence_vs_none`, and that the choice is therefore
   a *reported* one rather than an assumed one.
2. **Guard the verdict.** `x9_compare.verdict` currently reads `evidence_blind` off a quantity that
   a denominator change flips. Either assert the denominator in the module or emit both values
   alongside the verdict, so the sensitivity is visible in the artifact rather than in this memo.
3. **Attribute the probe** to Tsvilodub et al. 2024 / EgoSAT Eq. 6 — already actioned as
   `RELATED-EGOSAT.md` §4 item 2; this memo supplies the empirical justification for keeping it.
4. **Report the option-letter bias table (§6) and the pairing defence.** It will be asked, and the
   answer is good.

**Should add:**

5. **`x11_confidence_robustness.py`** — a module that regenerates §3–§5 from `m6_records*.jsonl` into
   `artifacts/x11_confidence_robustness.{json,md}`, so these tables are generated rather than
   transcribed, matching the standard M8/X9 already meet. Nothing in it needs a GPU.
6. **Off-letter mass on every record.** It is derivable from `letter_logprobs`, but promoting it to a
   named field (`off_letter_mass`) and a reported contrast makes §5 a result rather than a
   reanalysis.
7. **A top-k vocabulary dump on the `prior` arm at Δt ∈ {1, 5} min** (~1 GPU-h) to identify *which*
   tokens take the mass in the 8.8 % of evidence-free streaming rows that hedge. If they are
   refusal or hedging tokens, §5 becomes a much stronger claim; if they are formatting artefacts, it
   becomes a caveat instead. This is currently unknown and is the one open question in this memo.

**Newly de-risked:** the primary metric is not a soft spot. It is the published standard, it is
robust to the denominator, to the scalarisation, and to the discrimination measure, and the single
sensitivity has been located, quantified, and explained rather than left for a reviewer to find.
