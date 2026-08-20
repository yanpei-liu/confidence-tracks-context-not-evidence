# Interpreting the results

Two experiments have run on the same 373 EgoLifeQA questions, the same frame bank and the same
frozen Qwen3-VL-8B backbone, differing only in how the context reaches the model:

| | experiment | records | authoritative artifact |
| --- | --- | --- | --- |
| **E1** | full-reprocess per checkpoint — each row re-encodes its whole window | 5,540 | [`artifacts/m8_teaser.md`](artifacts/m8_teaser.md) |
| **E2** | streaming KV cache — one ingestion, queried at checkpoints against a bounded cache | 16,565 | [`artifacts/x9_streaming.md`](artifacts/x9_streaming.md) |

**This document is interpretation.** The two artifacts above are generated from
`m7_metrics.json` and `x9_regime_comparison.json` and are authoritative for every number; where
this file rounds or paraphrases, they win. Design lives in [PLAN.md](PLAN.md) (E1) and
[EXTENDING.md](EXTENDING.md) (E2).

Part 1 is how to read E2 on its own. Part 2 is how to read the pair, which is where the actual
claim lives.

---

## Part 0 — Ground rules that apply to both

**0.1 "Confidence" is a softmax over option-letter logits, not a calibrated probability.** It is
`max softmax({A,B,C,D})` at the first generated position, with no text generated. Always report
it as *model confidence (option-letter softmax)*. ECE is reported alongside it, but the quantity
was never calibrated and nothing here should be read as a claim about calibrated probabilities.
The probe follows Tsvilodub et al. 2024 and EgoSAT Eq. 6; it is the published standard, not an
invention here.

**The denominator is a reported choice, not an assumption.** The softmax is restricted to the four
option letters, so mass the model puts anywhere else is renormalised away. The
vocabulary-normalised alternative was computed on the committed records and is **inert almost
everywhere** — argmax agrees on 23,687 of 23,687 rows, and AUROC moves in the fourth decimal
(0.6276 → 0.6275 in E1, 0.6199 → 0.6203 in E2). It is *not* inert in one place: E2's
`evidence_vs_none` Δconfidence goes **+0.55 [−1.52, +2.65] → +2.61 [+0.33, +4.94]**, which is
enough to move `x9_compare`'s verdict from Outcome 3 to Outcome 1. The reason is that the
evidence-free `prior` arm hedges — 8.76 % of its streaming rows put real mass off the letters,
against 0.42 % for `main`. §2.7 carries this as a stated sensitivity.
See [`CONFIDENCE-PROBE.md`](../Report/Week1-Briefing/memo/CONFIDENCE-PROBE.md).

**0.2 Quote contrasts, not levels.** Every headline number in both experiments is a **paired,
within-question** difference between two conditions, cluster-bootstrapped over questions (5,000
draws, seed 0). Levels are weak by construction — see 0.3 — and are *not* comparable across E1
and E2, because E2 replaces the system prompt with EgoStream's priming prompt and the video path
with the image path. The regimes differ in more than the regime.

**0.3 The usable dynamic range is ~11 pp in both.** E1: bin-1 accuracy 42.9 % against a
no-evidence floor of 31.4 %. E2: 42.1 % against 30.8 %. Chance is 25 %. Any statement about an
accuracy *level* is being made inside an 11-point band, so accuracy nulls are low-precision by
construction and "no effect" always means *none resolvable at n = 373*.

**0.4 Distinguish three kinds of negative result.** They are not interchangeable and the
experiments produced all three:

| kind | meaning | example |
| --- | --- | --- |
| **null** | the manipulation happened; no effect resolvable | E2 accuracy vs budget, −1.9 pp [−6.4, +2.4] |
| **unavailable** | the manipulation never reached the model | E2 elapsed time — no readable clock (§1.3) |
| **not confirmed** | a specific prediction was tested and failed | E2's timestamps-off prediction (§1.8) |

Calling an *unavailable* axis a *null* is the single most damaging misreading available here, so
it gets its own section.

---

## Part 1 — Reading the streaming experiment (E2)

### 1.1 The one-sentence reading

**Accuracy responds to whether the evidence is in the window and to nothing else; confidence
responds to how much visual input the model has processed — whether or not it retained it — and
to nothing else.** The two measures are driven by disjoint variables.

The second clause has to be said that carefully, because *two different* manipulations move
confidence and it is tempting to call both "context length":

| | what varies | what is held | Δconfidence |
| --- | --- | --- | --- |
| **live context** | how much the model *holds* (`mutable_token_size`, 1,940 → 33,684) | frames presented | −15.2 pp at 17× |
| **presented volume** | how much it *processed and threw away* (`main@60 − evkeep@60`) | **live tokens, identical at 9,108** | −6.7 pp at 3.8× |

Both push confidence down, which is why the finding is unharmed — but they are not one axis, and
**the −6.71 pp context term in §1.4's decomposition is the second one**, where the live cache is
byte-for-byte the same size on both sides. The unifier above covers both. See
[`STREAMING-CACHE.md` §5](../Report/Week1-Briefing/memo/STREAMING-CACHE.md) for the measurement.
(`figures/streaming_headline.pdf` panel A groups these under one heading, *context volume in the
window*, and separates them by row; the prose here is the more precise reading of the same rows.)

The figure that says exactly this and nothing more is
[`figures/streaming_headline.pdf`](figures/streaming_headline.pdf) (X10): §1.2's rows are its
panel A, §1.4's dose–response and decomposition are panels C and D, §1.6's AUROC is panel E, and
§1.3's *unavailable* axis is drawn as such rather than as a null. (X9 also runs M7's own
template over these records, writing `figures/headline_stream.*`. Those are **not committed** and
are git-ignored: they carry the full-reprocess sweep's panel titles, two of which are false here.
See EXTENDING.md §0.)

### 1.2 Read the axes in this order

| axis | manipulation | Δaccuracy (pp) | Δconfidence (pp) | status |
| --- | --- | --- | --- | --- |
| evidence | evidence in context vs none (`main@5 − prior@5`) | **+8.1 [+2.9, +13.1]** | +0.5 [−1.5, +2.6] | real effect, accuracy only |
| live context | 17× live-token budget @60 min (`large − small`) | −1.9 [−6.4, +2.4] | **−15.2 [−17.5, −13.1]** | real effect, confidence only — but see §1.4's caveat |
| presented volume | 3.8× presented tokens @60 min (`main − evkeep`), live tokens fixed at 9,108 | −2.4 [−5.9, +1.1] | **−6.7 [−8.4, −5.0]** | same direction, different variable; the manipulation E1 shares, and the decomposition's context term |
| elapsed time | +45 min at fixed frames and evidence | −1.7 [−5.1, +1.6] | −0.1 [−1.7, +1.4] | **unavailable** (§1.3) |
| time signal | timestamps on vs off, pooled | −0.6 [−1.7, +0.5] | −0.7 [−1.2, −0.2] | present but ~1/20 the context term |

Everything else in E2 is a consequence of these four axes.

### 1.3 The elapsed-time axis is *unavailable*, not null

This is the interpretive crux of E2.

EgoStream ingests through Qwen3-VL's **image** path, which carries no `<t seconds>` markers —
those exist only on the video path E1 uses via `VideoMetadata`. We therefore injected the clock
as interleaved text, and verified by decoding the ingested token stream that it is physically
present. **X1b then established that the model does not use it**: holding the frames fixed at 120
and varying only the injected clock over 60 / 600 / 3600 s, the answer never moved
(`['B','B','B']`, identical to markers-off). Across 3 marker formats × {readout, integration} ×
{unevicted, evicted}, exactly one cell showed any tracking — 32 frames (under the cache cap),
`<t seconds>`, pure readout, 2/3 correct.

So at Δt ∈ {15, 30, 60} min — every bin where the cache is saturated — **elapsed time is not an
input.** The −0.1 pp confidence contrast is therefore not evidence that confidence ignores
elapsed time in streaming systems. It is evidence that *this* system cannot perceive elapsed
time once its cache saturates. Those are different claims and only the second is supported.

Three consequences that must travel with any use of E2:

1. **E2 cannot test E1's headline claim.** Do not report "the elapsed-time effect disappeared"
   as a failure to replicate a psychological finding. Report it as: the input that carried the
   effect in E1 is absent in E2, and the effect is absent with it.
2. **Eviction is one of two live explanations, and the design cannot yet separate them.** E1's
   clock is ordinary text on the video path — `<150.0 seconds>`, one marker per *merged frame
   pair*, valued at the pair mean — and Qwen3-VL carries no seconds in its RoPE at all
   (transformers `modeling_qwen3_vl.py:1483` removed Qwen2.5-VL's temporal scaling in favour of
   these markers). E2 injects the same channel by hand, and it differs from native in **three ways
   at once**: decimal vs integer format, position relative to the vision wrappers, and one marker
   per frame rather than per merged pair — plus it is prunable where E1's is re-encoded every
   query. X1b's single tracking cell was the format closest to native, *unevicted*. So "the cache
   evicted the clock" and "the clock was never in a form this model reads" are both consistent
   with the evidence. See [`STREAMING-CACHE.md` §7](../Report/Week1-Briefing/memo/STREAMING-CACHE.md);
   §2.8's first roadmap item is designed to separate them.
3. **This is a property of this cache and this clock, not of streaming per se.** A streaming
   architecture that
   preserved a readable clock — timestamps in the sink region, a positional encoding that
   survives eviction, a running clock token refreshed each batch — could restore the axis. That
   is a concrete follow-up, not a limitation to apologise for.
4. **X1b ran before the contrast was computed.** The mechanism is not a post-hoc explanation
   fitted to a null; it was established independently and predicted the null. That ordering is
   what makes the interpretation credible and is worth stating explicitly in any write-up.

### 1.4 The context-length dose–response is the positive result

Because a bounded cache pins token count, E2 recreates the context-length axis by varying the
memory budget rather than the frame selection. Across the Δt bins the live-token ratio between
`large` and `small` grows 1.7× → 17×, and confidence tracks it monotonically while accuracy does
not move at all:

| Δt | live tokens (large / small) | Δaccuracy (pp) | Δconfidence (pp) |
| --- | --- | --- | --- |
| 1 min | 3,273 / 1,900 | +3.0 [+0.0, +5.9] | −2.0 [−3.4, −0.6] |
| 5 min | 8,073 / 1,934 | −0.0 [−3.8, +3.8] | −4.2 [−5.9, −2.4] |
| 15 min | 20,072 / 1,939 | +1.6 [−2.1, +5.4] | −8.1 [−10.0, −6.2] |
| 30 min | 33,684 / 1,940 | −0.0 [−4.3, +4.3] | −10.5 [−12.4, −8.5] |
| 60 min | 33,684 / 1,940 | −1.9 [−6.4, +2.4] | **−15.2 [−17.5, −13.1]** |

A monotone dose–response on one measure with a flat response on the other is much stronger
evidence than either endpoint alone: it is hard to produce by accident and it rules out a
threshold artefact. Read it as **more context in the window buys no accuracy and costs a great
deal of confidence.**

> **The −15.2 pp endpoint spans two mechanisms; the single-mechanism number is −6.0 pp.**
> `small` sets `mutable_token_size` to 1,024, which is **below one 1,600-token ingestion batch**,
> so it is the only condition in the sweep that overflows to the offload store and the only one
> where ReKV retrieval is actually selecting anything: measured, `offloaded_tokens` is 0 on
> **12,890 of 12,890** default and large rows and non-zero on **3,675 of 3,675** `small` rows
> (§1.7). The `large − small` contrast therefore mixes a budget change with a
> retrieval-vs-no-retrieval change. The clean, single-mechanism contrast is
> **`large − default` at 60 min: −6.0 pp [−7.3, −4.7] for a 3.7× budget** — both sides
> offload-free. Quote −15.2 for the full span of the axis and −6.0 when the claim needs one
> mechanism. Neither disturbs the accuracy null, which is flat in both.
> See [`STREAMING-CACHE.md` §2.2 and §4](../Report/Week1-Briefing/memo/STREAMING-CACHE.md).

M7's decomposition, run unchanged on E2's records, says the same thing from the other direction
and closes almost exactly (residual +0.05 pp accuracy, +0.02 pp confidence, against a 2 pp
tolerance — tighter than E1's own −0.8 / −0.3):

```
accuracy    -1.68 (elapsed) + -2.39 (context) = -4.08  vs  -4.02 observed
confidence  -0.13 (elapsed) + -6.71 (context) = -6.84  vs  -6.82 observed
```

The confidence decline over the sweep is **entirely** the context term. The elapsed-time term is
not merely small; it is absent, exactly as §1.3 requires.

> **Caveat that must not be dropped.** The budget arm was neutralised on its first run: the ReKV
> retriever's `top_k_tokens` stayed at its shipped 8,192 while the budget varied, re-equalising
> every condition at query time (measured live tokens 9,104 / 9,108 / 9,099 for budgets meant to
> span 1,940 → 33,684). The numbers above come from the corrected run, in which `top_k` tracks
> the budget and `retrieve_top_k` is recorded on every row. Any future budget-like arm must
> check the *realised* `live_tokens`, not the configured budget.

### 1.5 Confidence is evidence-blind — here, almost perfectly

Token-matched, paired, with the *only* difference being whether the window contains the evidence:

| | E2 no evidence (`prior@5`) | E2 evidence (`main@5`) |
| --- | --- | --- |
| accuracy | 30.3 % | 38.3 % |
| confidence | 81.1 % | 81.6 % |

Having the evidence is worth **+8.1 pp [+2.9, +13.1]** of accuracy and **+0.5 pp [−1.5, +2.6]**
of confidence. The model is essentially exactly as confident answering from an evidence-free
window as from one containing what it needs. In E1 the same contrast bought +9.7 pp accuracy and
+3.3 pp confidence — a 2.9× under-reaction; in E2 the under-reaction is total.

### 1.6 The ranking degrades — the one genuinely new phenomenon

AUROC of confidence against correctness over the sweep: **Δ = −0.073 [−0.143, −0.003], p =
0.039**, against +0.012 [−0.046, +0.071] (flat) in E1.

This is the only place E2 shows something E1 could not produce, and it follows from the design:
under full-reprocess the evidence is always present at *some* resolution, so it can be diluted
but never removed; under a bounded cache it is genuinely evicted. When information is actually
gone, the *ordering* of confidences decays too, not just the level.

Treat this as a single test at p = 0.039 with a CI that nearly touches zero. It is the most
interesting result in E2 and the one most in need of replication.

Carry E1's honesty caveat with it: the absolute AUROC is 0.654 at Δt = 1 min falling to 0.581 at
60 min, against **0.574 for the no-evidence `prior` arm**. So most of the discriminative signal
is question difficulty rather than evidence grounding, and by 60 min what remains is barely
above what the model achieves with no evidence at all. The *change* is the finding; the level is
not evidence of good discrimination.

### 1.7 Eviction is permanent in the primary condition

Measured: `offloaded_tokens` is **0 at every Δt bin** in the default-budget condition, while the
`small` budget offloads heavily (up to 1.76 M token-layers at 60 min). So in the primary
condition the evicted KV is *dropped* rather than written to the offload store, and the ReKV
retriever — which is enabled — has only live blocks to select from. Eviction there is not
recoverable.

This matters because "the evidence is genuinely evicted" is doing real work in the argument
(§1.6). It is supported for the primary condition by this measurement, and the audit is now
exhaustive rather than a median: **0 of 12,890** default and large rows offloaded anything;
**3,675 of 3,675** `small` rows did.

The mechanism is also characterised. `small`'s 1,024-token mutable region is smaller than one
1,600-token ingestion batch, so a single batch overruns it and overflow is routed through the
storage step; the default (8,192) and large (32,768) regions absorb a batch without spilling.
That is why the claim does **not** generalise to the `small` arm, where retrieval may be
recovering content — and why §1.4's clean budget contrast excludes it.
See [`STREAMING-CACHE.md` §2.2](../Report/Week1-Briefing/memo/STREAMING-CACHE.md).

### 1.8 Three things not to claim from E2

- **Item-level metamemory.** Among items correct at the earlier bin, confidence fell −3.0 pp
  [−4.0, −2.0] on items kept and −6.8 pp [−11.2, −2.3] on items lost. The lost-item drop is
  individually significant (p = 0.005) and it is tempting to stop there — but the **difference**
  is −3.8 pp [−8.5, +1.1], **p = 0.118**, on 107 lost items. E1 found this flatly item-blind
  (+1.0 pp, p = 0.618), so E2 moves it in the direction of genuine metamemory without reaching
  significance. Report as *suggestive and underpowered*; do not quote the lost-item drop alone.
- **The timestamps-off prediction.** §4a of EXTENDING.md predicted `main` ≈ `nots` at saturated
  bins and a difference only at 1 and 5 min. Measured per bin, the confidence difference is
  −0.2 pp at 1 min (no effect, where one was predicted) and −1.6 / −1.2 / −2.2 pp at 15 / 30 /
  60 min with intervals excluding zero (an effect, where none was predicted) — the opposite
  shape. Accuracy is null everywhere. The effects are ~1/7 the context-length term and carry no
  accuracy difference, so §1.3's conclusion survives, but the prediction is **not confirmed** and
  the residual is most likely mechanical: markers cost ~5 % of tokens, so the two conditions hold
  slightly different amounts of visual content at a fixed cap, and confidence is strongly
  sensitive to exactly that.
- **Absolute overconfidence.** E2's gap is much larger than E1's (+42.2 pp vs +29.2 pp at bin 1;
  ECE 0.42 vs 0.29). This is a *level*, and E2 changes the prompt and the image path along with
  the regime, so it is not attributable to streaming. Do not present it as "streaming makes the
  model more overconfident."

### 1.9 Scope conditions

- **Streaming over the window, not the life-log.** Each question resets the cache and ingests
  from `evidence_min − PRE`. This preserves every arm definition from E1 and keeps the comparison
  clean, but supports no claim about 44 h of accumulated history.
- **The no-evidence floor is measured at Δt ∈ {1, 5, 15} min only** and assumed flat beyond,
  because `prior` windows do not nest and each bin costs its own ingestion.
- One participant (EgoLife A1), one model, one cache policy (EgoStream's shipped pruning pipeline
  plus ReKV retrieval).

---

## Part 2 — Reading the two experiments together

### 2.1 What the pair is for

E1 and E2 hold the backbone, the questions, the frames and the analysis fixed and vary **only how
context reaches the model**. That makes the regime comparison identifiable by construction — no
bridge condition is needed, because EgoStream is a plug-in memory manager over the same frozen
Qwen3-VL-8B. So a difference between E1 and E2 is attributable to the regime, subject to the two
prompt-path differences noted in 0.2.

The right cross-regime quantity is a **difference of contrasts**, not a difference of levels.

### 2.2 What is invariant

**Confidence does not track whether the evidence is present. Accuracy does.** This holds in both
regimes, and it is the finding.

| | E1 | E2 |
| --- | --- | --- |
| evidence vs none, Δaccuracy | +9.7 [+4.8, +14.5] | +8.1 [+2.9, +13.1] |
| evidence vs none, Δconfidence | +3.3 [+1.2, +5.4] | +0.5 [−1.5, +2.6] |
| difference (E2 − E1), Δconfidence | | **−2.8 [−5.1, −0.4]** |

The accuracy effect is statistically indistinguishable across regimes (−1.6 pp [−6.7, +3.5]);
the confidence response is *weaker* in E2 and the difference excludes zero. Changing the memory
architecture did not fix evidence-blindness — it deepened it.

**The predicted "doesn't know it forgot" signature is absent in both.** The overconfidence gap
does not widen with elapsed time: −3.2 pp [−7.5, +1.0] (p = 0.138) in E1, −4.0 pp [−9.0, +1.0]
(p = 0.105) in E2. Both point mildly *toward narrowing*. Two independent regimes, same answer.

### 2.3 What rotates

What confidence *does* respond to changes completely between regimes. Both rows below are
**differenced contrasts** — the same manipulation run in both regimes, paired on the questions
present in both:

| manipulation | E1 Δconfidence | E2 Δconfidence | difference (E2 − E1) |
| --- | --- | --- | --- |
| +45 min elapsed, frames + evidence fixed | **−3.7 [−4.9, −2.4]** | −0.1 [−1.7, +1.4] | **+3.5 [+1.7, +5.4]** |
| ~3.8× more presented tokens, elapsed time fixed (`main@60 − evkeep@60`) | −0.6 [−1.7, +0.5] | **−6.7 [−8.4, −5.0]** | **−6.1 [−8.1, −4.1]** |

In E1, confidence slid with elapsed time and ignored context length. In E2 it ignores elapsed
time and slides hard with context volume. **The two drivers swapped places, and both halves of
the swap have intervals excluding zero.**

The second row is the sharpest single piece of evidence in either experiment, because it is one
manipulation — literally the same frames, the same arms, the same token ratio (3.77× in E1,
3.79× in E2) — measured in both regimes:

| | E1 | E2 | difference |
| --- | --- | --- | --- |
| Δaccuracy | −3.2 [−6.1, −0.3] | −2.4 [−5.9, +1.1] | +0.8 [−4.0, +5.4] |
| Δconfidence | −0.6 [−1.7, +0.5] | −6.7 [−8.4, −5.0] | **−6.1 [−8.1, −4.1]** |

**The accuracy cost of extra context is the same in both regimes; the confidence response to it
is not.** That rules out the obvious deflationary reading — that E2's confidence–context
coupling is just a harder task or a noisier measure — because the task's accuracy response is
unchanged. Something about the bounded cache makes confidence sensitive to context volume that
full-reprocess does not.

This is why §1.3 matters so much for the joint reading: the rotation is not mysterious. In E1
the model *could* read a clock (the video path supplies `<t seconds>` markers) and elapsed time
was the variable that moved most while costing no accuracy. In E2 the clock is gone, and the only
surface property still varying across the sweep is how full the cache is.

### 2.4 The unified account

> **Confidence is computed from surface properties of the context — how much of it there is, and
> how long ago it says that was — not from whether the information needed to answer is present.**
> When several surface properties are available, confidence latches onto whichever varies; when
> one is removed, it moves to another. What it never latches onto is evidence.

This account explains, without extra assumptions:

- why the same manipulation (evidence presence) moves accuracy and not confidence in both
  regimes;
- why the confidence driver swaps when the clock is removed;
- why E1's confidence decline is item-blind — a surface property of the whole context cannot
  distinguish items;
- why the no-memory control swings confidence hard in both regimes (+13.9 pp E1, +12.6 pp E2):
  question kind is itself a surface property of the prompt;
- why the raw Δt sweep looks like forgetting in both and is not: in the `main` arm elapsed time
  and context volume grow together, so a surface-driven confidence and a genuinely
  context-limited accuracy decline in parallel for unrelated reasons.

It is a stronger and more useful claim than the original hypothesis. "Confidence stays flat while
accuracy falls" would be a *failure to update*. What we observe is confidence moving a great deal
— up to 15 pp — on variables that predict accuracy not at all. That is a failure of *what is
being tracked*, and it is the harder problem.

### 2.5 What the pair establishes about the project's original claim

The original claim had two halves. They came out differently.

**Half 1 — "a widening gap between confidence and accuracy as evidence ages" — is rejected.**
Twice, in two regimes, with the second run in the setting where evidence is genuinely evicted
rather than merely diluted. This was the setting the hypothesis most deserved, and it still did
not appear. The claim should be retired, not re-tested a third way.

**Half 2 — "a Δt/evidence-conditioned recalibration recovers most of the reliability at near-zero
cost" — needs revision on both variables.**

- *Conditioning on Δt is the wrong conditioner.* Under E1 it would work by accident, because
  confidence's slide was on elapsed time. Under E2, Δt has no purchase — the model cannot even
  read it. The variable that actually drives confidence in E2 is **`live_tokens`**, which is
  observable at inference time and free. A recalibration conditioned on realised context volume
  (and, in E1, on elapsed time) is what the data supports.
- *Conditioning on evidence is not available at inference.* "Evidence-conditioned" presumes
  knowing whether the evidence is in context, which is the very thing the model cannot report.
  That half is not implementable as stated.
- *The headroom differs between regimes.* Recalibration of a *level* only works if the *ranking*
  survives. In E1 it does (AUROC flat, Δ +0.012), so a monotone re-mapping could recover a lot.
  In E2 it does not (Δ −0.073, p = 0.039): part of the reliability loss is discriminative and no
  monotone recalibration can retrieve it. **The cheap fix gets cheaper-looking the less realistic
  the memory system is.**

That last point is the most consequential thing the pair says, and it only becomes visible
because both regimes were run.

### 2.6 What the pair does *not* establish

- **That streaming causes any of this.** E2 changes the prompt and the ingestion path along with
  the regime. The *contrasts* are comparable; a claim of the form "streaming causes X" needs the
  prompt held fixed too.
- **That the model has no metamemory.** Both experiments measure one proxy (option-letter
  softmax) under one elicitation. A model asked to *say* how sure it is, or offered an explicit
  abstention (PLAN.md S4, unrun), might behave differently. E2's item-level result is a hint in
  that direction and is underpowered. The elicitation also carries a position bias worth stating:
  against near-uniform gold (A .222 / B .256 / C .261 / D .261) the model predicts `C .435` in E1
  and `B .342` in E2, and E1's accuracy runs 21.4 % on gold-A against 58.6 % on gold-C. Every
  headline contrast is paired within-question, so the bias cancels — but it means no *level* here
  should be read as this model's competence.
  See [`CONFIDENCE-PROBE.md` §6](../Report/Week1-Briefing/memo/CONFIDENCE-PROBE.md).
- **Anything about elapsed time in streaming systems generally.** See §1.3. One cache, one clock
  representation.
- **Generalisation beyond one participant and one model.** N per bin is bounded by the single
  released EgoLife participant.

### 2.7 A claims ladder

**Safe to assert:**
- In both regimes, accuracy responds to whether the evidence is in context and confidence does
  not (E1 +9.7/+3.3, E2 +8.1/+0.5; confidence difference across regimes −2.8 [−5.1, −0.4]).
- The predicted widening overconfidence gap is absent in both (−3.2 pp p = 0.138; −4.0 pp
  p = 0.105).
- On the one context-length manipulation shared by both regimes, the accuracy cost is the same
  (difference +0.8 pp [−4.0, +5.4]) while the confidence response differs by −6.1 pp
  [−8.1, −4.1].
- Under a bounded cache, confidence falls monotonically with live-token budget while accuracy
  does not move: −15.2 pp over the full 17× span, −6.0 pp [−7.3, −4.7] on the offload-free 3.7×
  pair (§1.4).
- The confidence decline over E2's sweep decomposes entirely onto **presented volume** — not live
  context — at −6.71 of −6.82 pp, with the decomposition closing to +0.02 pp (§1.1).

**Defensible with the stated caveat:**
- Confidence tracks surface properties of the context rather than evidence state (§2.4) — a
  synthesis across two regimes, not a single measured quantity.
- Confidence ranking degrades when evidence is genuinely evicted (AUROC −0.073, p = 0.039) —
  one test, CI nearly touching zero, wants replication.
- Eviction is permanent in E2's default and large conditions — measured exhaustively,
  `offloaded_tokens` = 0 on 12,890 of 12,890 rows. It is *not* claimed for `small`, whose 1,024
  mutable tokens sit below one ingestion batch (§1.7).
- E2's evidence-blindness is **denominator-sensitive**: the restricted softmax gives Δconfidence
  +0.55 [−1.52, +2.65], the vocabulary-normalised one +2.61 [+0.33, +4.94]. The restricted probe
  is the published standard and is inert everywhere else, but this single cell must be reported
  with the claim rather than left for a reviewer to find (§0.1). Nothing about E1, and nothing
  about the accuracy side, moves.

**Not supported:**
- "Streaming models don't know they forgot" in the original sense — the gap does not widen.
- "Confidence tracks the items actually lost" — E2's difference is p = 0.118.
- "Streaming makes the model more overconfident" — a level, confounded with the prompt change.
- "Elapsed time does not affect confidence in streaming systems" — the axis was unavailable.

### 2.8 Where this goes next

**This list is canonical.** The week-1 briefing deck's *Ranked follow-ups* slide
(`Report/Week1-Briefing/slides/04-next.tex`) is a dated snapshot of it, and the three memos' *should
add* sections feed into it. Ranked by what each settles per GPU-hour, cheapest first.

#### Free — the records are already on disk, CPU only

1. **Recalibrate on `live_tokens`, separately per regime.** The corrected version of the project's
   Half-2 claim, and the sharpest cheap test available: the AUROC asymmetry predicts it recovers a
   lot in E1 (ranking intact, Δ +0.012) and only partly in E2 (ranking decays, Δ −0.073). If that
   asymmetry holds, "the cheap fix gets cheaper-looking the less realistic the memory system is"
   stops being a synthesis and becomes a measurement.
2. **Off-letter mass as a reported quantity.** Derivable from `letter_logprobs` already on every
   record. It turns §0.1's denominator sensitivity from a caveat into a result: the `prior` arm
   hedges on 8.76 % of streaming rows against `main`'s 0.42 %, contrast −3.01 pp [−4.40, −1.82].
   Promote it to a named field `off_letter_mass` and a contrast in X9.
3. **`x11_confidence_robustness.py` and `x12_cache_forensics.py`.** Regenerate the two memos'
   tables — the probe robustness sweep and the offload/cache audit — into
   `artifacts/x11_*.{json,md}` and `artifacts/x12_*.{json,md}`, so they meet the standard M8/X9/X10
   already meet. Today 52 of the deck's 130 macros are memo literals behind a substring tripwire;
   this moves them to derived.
4. **Three small integrity fixes.** Emit both denominators alongside `x9_compare.verdict` rather
   than leaving the sensitivity in a memo (§0.1); add `surviving_frame_tokens` to the record schema
   with an evidence-frame-vs-window-mean contrast; and footnote the `cache_policy` label, where
   `+rekv_retrieve` is nominal at the default and large budgets because nothing is offloaded and
   `top_k` ≥ live length.

#### Cheap — about 1 GPU-hour each

5. **Strip the timestamps from E1 and re-score the token-matched cells.** The highest value per
   GPU-hour on this list. E1's clock is *text* — `<150.0 seconds>` markers on the video path, with
   no seconds anywhere in RoPE — so the headline −3.7 pp elapsed-time slide may be a response to
   printed numbers rather than to elapsed time as such. It also separates §1.3's two live
   explanations for E2's failure: if E1's effect survives marker removal, eviction is the story; if
   it vanishes, the lexical channel is.
6. **Top-k vocabulary dump on the `prior` arm at Δt ∈ {1, 5} min.** Identifies *which* tokens take
   the mass in the 8.8 % of evidence-free streaming rows that hedge. If they are refusal or hedging
   tokens, item 2 becomes a much stronger claim; if they are formatting artefacts, it becomes a
   caveat. Currently unknown.

#### Larger

7. **Restore a readable clock under streaming** — timestamps in the sink region, or a clock token
   refreshed per batch and protected from eviction. The missing cell of the design: it would make
   E1 and E2 comparable on the elapsed-time axis and test directly whether §2.3's rotation is about
   the clock's *availability* rather than the regime. Best run after item 5, which tells you what
   form the clock has to take.
8. **Explicit abstention (PLAN.md S4, ~2.4 GPU-h, unrun).** The natural follow-up to evidence-blind
   confidence: given the vocabulary to say "not enough information", does the model use it when the
   evidence is absent? Attacks §2.6's elicitation caveat directly.
9. **Cyclic option-permutation arm, one Δt bin (4× the rows of one bin).** The option-letter bias
   check EgoSAT runs and this project does not: E1 predicts `C` on 43.5 % of items and E2 predicts
   `B` on 34.2 %, against near-uniform gold. Every headline contrast is paired within-question so
   the bias cancels by construction — but it will be asked, and the answer should be measured.
10. **Power the item-level test.** E2's −3.8 pp [−8.5, +1.1] on 107 lost items is the only hint of
    genuine metamemory anywhere in either experiment. More Δt steps, or the extended bins, would
    roughly double the transition count.
