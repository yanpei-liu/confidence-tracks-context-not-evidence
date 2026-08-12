# Teaser Experiment — Execution Plan

**Original goal (from CLAUDE.md):** the headline accuracy–confidence figure — accuracy vs Δt and
mean confidence vs Δt on one axis, per-bin ECE, bootstrap CIs. Predicted signature: accuracy
declines while confidence stays flat ("doesn't know it forgot").

**That signature is not there** (see *Findings* below), and the headline figure has been rebuilt
around what the arms actually show. The design is unchanged; only the claim it supports is.

The task is decomposed into 8 modules along **verifiable-artifact boundaries**: each module emits a
file that can be checked on its own, and downstream modules consume only that file. Any module can
be re-run without re-running its predecessors.

## Findings (sweep complete, 5,562 records)

The predicted signature is absent: the overconfidence gap does **not** widen with Δt
(−3.2 pp [−7.5, +1.0], p = 0.14). Accuracy and confidence both fall over the raw 1 → 60 min sweep,
and confidence falls slightly faster. What the arms show instead is a **double dissociation**: the
two measures respond to different manipulations. Every row below is paired within-question and
cluster-bootstrapped over questions.

| manipulation | Δ accuracy | Δ confidence |
| --- | --- | --- |
| +45 min elapsed, tokens **and** evidence fixed (`evkeep@60` − `main@15`) | +0.3 [−2.7, +3.2] | **−3.7 [−4.9, −2.4]** |
| 4× more visual tokens, elapsed time fixed (`main` − `evkeep`, 30 & 60 pooled) | **−2.4 [−4.7, −0.3]** | −0.3 [−1.1, +0.5] |
| evidence thinned at fixed tokens and Δt (`uniform` − `evkeep`) | **−2.9 [−5.0, −0.9]** | −1.2 [−2.0, −0.3] |
| evidence in context vs none (`main@5` − `prior@5`) | **+9.6 [+4.6, +14.5]** | +3.3 [+1.2, +5.4] |
| raw sweep 1 → 60 min (`main`) | −6.4 [−10.7, −2.4] | −9.7 [−11.4, −7.9] |
| raw sweep 1 → 60 min, no-memory control | −10.3 [−17.8, −2.9] | −10.7 [−13.4, −8.1] |

Read together: **confidence responds to elapsed time, which costs the model no accuracy, and
barely responds to context length or evidence availability, which are the only two things that
do.** The 15 → 60 min segment decomposes cleanly — accuracy's −2.9 pp is −2.4 pp of context length
plus +0.3 pp of elapsed time; confidence's −4.3 pp is −3.7 pp of elapsed time plus −0.3 pp of
context length. Two further facts:

- **The confidence decline is item-blind.** Pooled over adjacent Δt steps, among items correct at
  the earlier bin, confidence fell −2.9 pp on items kept and −1.9 pp on items lost — difference
  +1.0 pp [−2.8, +4.8], p = 0.63. It is a flat discount, not a signal about what was lost.
- **The ranking is intact.** AUROC of confidence against correctness is 0.643 at Δt=1 min and
  0.655 at 60 min (+0.012 [−0.046, +0.071]), falling to 0.568 only when the evidence is absent.
  The discriminative signal is real and Δt-invariant; it is the *level* that misbehaves.

The headline figure (`figures/headline.pdf`) is built on these: panel A contrasts the raw sweep
with the token-fixed one, panel B places every manipulation on a Δaccuracy × Δconfidence map
against the line where confidence would track competence, panel C shows the item-blindness and the
flat AUROC. The original two-panel sweep figure is kept as `figures/supp_sweep.pdf`.

## Paths

| Thing | Location |
| --- | --- |
| Code | `Teaser-Experiment/` (this repo) |
| Data root | `~/Files/EgoVM-data` (**outside the repo** — datasets are not redistributable) |
| Raw video / QA | `$DATA/raw/EgoLife/` |
| Frame bank | `$DATA/frames/` |
| Artifacts | `$DATA/artifacts/` (large) and `Teaser-Experiment/artifacts/` (small, committed) |

## Schema finding that changes the plan

The planning documents state that EgoLifeQA has *"no confirmed populated explicit evidence-interval
field"* and route precise Δt derivation to MultiHop-EgoQA. **This is wrong for the released file.**
`EgoLifeQA/EgoLifeQA_A1_JAKE.json` ships a populated `target_time` on all 500 items — the evidence
timestamp(s) for the question. Verified encodings (all 500 parse):

| Encoding | n | Example |
| --- | --- | --- |
| `{date, time}`, 8-digit `HHMMSSFF` | 451 | `{"date":"DAY1","time":"11152408"}` |
| `{date, time_list: [...]}` | 36 | `{"date":"DAY1","time_list":["13394908","13512718"]}` |
| `{date, time}` with concatenated `DAY<n>_HHMMSSFF` suffixes | 13 | `"11153417DAY1_11181201"` |

`FF` is a frame index at 30 fps. One item has lowercase `Day1`. Evidence-count distribution:
1 point ×451, 2 ×41, 3 ×6, 6 ×1, 7 ×1. Native `query_time − max(target_time)` ranges 4.9 min
(matching the paper's "at least 5 minutes prior") to 138 h, median ~6 h.

**Consequence:** Δt is derived exactly on EgoLifeQA itself. MultiHop-EgoQA is demoted from
"Mitigation-2 fallback needed for AVW" to an optional cross-dataset robustness check.

## Design

**Δt is elapsed *stream* time, not wall-clock.** The A1 stream is 6,266 × 30 s chunks = 52.22 h of
video spread over 7 days of ~11 h wall-clock spans, with 31 gaps (6 overnight, 10–14 h; several
intra-day, 2–3 h). A wall-clock Δt would silently jump across an overnight gap. So M2 builds a
monotone map wall-clock ↔ cumulative stream seconds, and every Δt is measured in that coordinate.

**Context window per evaluation.** For question *q* with evidence points `E`:

```
window = [ stream(min E) − PRE ,  stream(max E) + Δt ]        PRE = 60 s
```

The evidence sits at the head of the window; the model then observes Δt of intervening life before
being asked. Window length grows exactly with Δt. Frames are sampled at a fixed **0.2 fps** from the
frame bank (M3), so evidence is sampled at the same density at every Δt — the confound this creates
(token count grows with Δt) is what the controls exist to absorb.

**Δt grid:** 1 / 5 / 15 / 30 / 60 min. The 1 min bin is the near-ceiling anchor — the evidence is
still essentially in view, so it bounds how much of the later decline is forgetting rather than the
question simply being hard. Its window is short, which is exactly why the token-matched arms exist.
All 373 kept questions have ≥60 min of stream after their evidence, so N = 373 per bin — above the
≥150 target with margin.

**The context budget.** The model gets the whole window at 0.2 fps for as long as that fits
usefully. The Δt=15 min window — 194 frames, 9.7k visual tokens — is taken as the saturation point:
past it, the token-matched arms pin the count there and let only elapsed time keep growing. That is
why `main` stops at 60 min while the token-fixed arms carry the elapsed-time axis out to 4 h, and
it is also what makes those extended bins cheap — a pinned row costs the same at 4 h as at 30 min.

**Arms.** Five arms, plus a separate *control question kind* that rides on some of them. `main`
runs in every Δt bin; the other arms exist only where they are meaningful, which is why the plan is
7,460 rows and not 373 × 7 × 5:

| Arm | Δt bins (min) | Frames | Purpose | Rows |
| --- | --- | --- | --- | --- |
| `main` | 1/5/15/30/60 | 0.2 fps over the whole window | primary Δt sweep | 1,865 |
| `tokmatch_uniform` | 30/60/**120**/**240** | window uniformly subsampled to *that question's Δt=15 min frame count* | token-matched: isolates Δt from context length | 1,492 |
| `tokmatch_evkeep` | 30/60/**120**/**240** | same target count, but every frame up to `evidence_max + 60 s` is kept and only the filler is thinned | token-matched **and** evidence-resolution-matched: leaves elapsed time as the only difference | 1,492 |
| `tokmatch_ckptkeep` | **30/60** | same target count, but every frame from `checkpoint − 120 s` is kept and the head is thinned | the mirror of `evkeep`, for the *control* question — which asks about the checkpoint the other arms thin away | 746 |
| `prior` | **1**/5/**15**/**30**/**60** | an evidence-free window matched frame-for-frame to that question's window in the same bin, drawn ≥2 h away from the evidence | language-prior floor: what the question and options alone are worth, as a function of Δt | 1,865 |

Bold bins are supplementary (S1–S3 below); the rest are the completed sweep.

The no-memory `control` (M5) is **not** an arm — it is a different question asked over another
arm's frames, so it appears in the metrics as `control/<arm>`. It rides on `CONTROL_ARMS` =
`main`, `tokmatch_ckptkeep`, `tokmatch_uniform`. Control items are keyed `qid|dt_s`, so the same
control MCQ rides every arm at a given checkpoint and every control comparison is paired.

`main` at Δt=60 min vs the token-matched arms at Δt=60 min separates "more tokens" from "more
elapsed time", and the two evidence variants bracket the confound from both sides:
`tokmatch_uniform` holds token count fixed but thins the evidence along with everything else, while
`tokmatch_evkeep` holds token count *and* evidence sampling fixed. Contrasting the arms against
*each other* rather than each against `main` is what makes the manipulations in *Findings*
identifiable. `control` vs `main` separates generic long-context degradation from forgetting, and
`prior` bounds the sweep from below: accuracy above that floor is what the video evidence is
buying.

**Question set (Mitigation 1):** keep EntityLog (125) + EventRecall (126) + RelationMap (125) = 376.
Drop TaskMaster (63, prospective). HabitInsight (61) is scored but held out of the headline set and
reported separately. A further 3 are dropped by `MAX_EVIDENCE_SPAN_S`: a question whose evidence
points span more than 15 min has no single Δt, since the window must contain all of them. **N = 373.** `need_audio` items are flagged and reported both in and out (the model is
vision-only).

**Confidence:** softmax over the option-letter logprobs {A,B,C,D} at the first generated position;
confidence = max prob, correctness = argmax == GT. No generation. Reported as "model confidence
(option-letter softmax)", never as a calibrated probability.

## Modules

| # | Module | Artifact | Verification |
| --- | --- | --- | --- |
| M0 | env + model smoke test | `m0_env_report.json`, `m0_smoke.jsonl` | Qwen3-VL-8B loads on GPU; 4 letter logprobs extracted and sum-to-1 after softmax |
| M1 | QA parse + filter | `m1_questions.jsonl`, `m1_schema_report.md` | 500 in / 500 parsed, category counts match 125/126/125/61/63 |
| M2 | stream index | `m2_stream_index.json` | 6,266 chunks; sampled real durations match the 30 s nominal; monotone map round-trips |
| M3 | frame bank | `$DATA/frames/`, `m3_frame_index.json` | frame count ≈ 52.22 h × 0.2 fps ≈ 37.6k; every planned window resolves to ≥1 frame |
| M4 | eval plan | `m4_eval_plan.jsonl` | 7,460 rows over 373 questions × the arms defined for each bin; frame counts monotone in Δt for `main`, equal to the Δt=15 min count for every `tokmatch_*` row, equal to `main`'s for every `prior` row in the same bin, and the extended bins carry token-matched arms only |
| M5 | no-memory controls | `m5_control_questions.jsonl` | one control MCQ per distinct checkpoint; distractors drawn ≥30 min away |
| M6 | scoring sweep | `m6_records.jsonl` | one record per plan row per question kind, resumable on *either* record, logs n_frames/n_tokens/model revision |
| M7 | analysis + figures | `m7_metrics.json`, `figures/headline.pdf`, `figures/supp_sweep.pdf` | per-bin acc/conf/ECE + bootstrap CIs; the paired manipulation contrasts; the decomposition closes; AUROC per cell; item-level transitions |

## Statistics

Within-question design (each question appears in all 5 bins of the `main` arm), so:

- Bootstrap CIs by resampling **questions** (cluster bootstrap), not (question, Δt) rows.
- **Primary quantities: the paired manipulation contrasts** in *Findings*. Each is
  `contrast(idx, cells_a, cells_b)` in M7 — cells are `(question_kind, arm, dt_s)`, pairing is on
  `qid`, and passing several cell pairs pools over Δt (the token effect is only resolvable pooled
  over the 30 and 60 min bins). Reported as Δaccuracy, Δconfidence and Δgap for each.
- **The decomposition must close.** The observed `main` 15 → 60 min change has to equal the
  elapsed-time term plus the context-length term to within ~0.5 pp on both measures; M7 asserts
  this. If it stops closing, the token-matched arms are no longer matched.
- Originally-primary test, still reported: the **bin × measure interaction** — does the
  accuracy-vs-Δt slope differ from the confidence-vs-Δt slope? A cluster-bootstrap CI on
  `[conf(60) − acc(60)] − [conf(1) − acc(1)]`. This is the null the experiment returned.
- Evidence awareness: `prior` and `main` in the same bin are token-matched by construction and
  differ only in whether the window contains the evidence, so the pair measures whether confidence
  tracks having the information at all — no forgetting required.
- Discrimination vs level: AUROC of confidence against correctness per cell, which is invariant to
  any monotone reshaping of the confidence scale and so separates "the level slid" from "the
  ranking degraded".
- Item-level: over adjacent Δt steps, restricted to items correct at the earlier bin, the
  confidence change on items lost vs items kept. A signal worth the name drops harder on the lost.
- Secondary: mixed logistic `correct ~ log Δt + (1|question)` and `confidence ~ log Δt + (1|question)`.
- Confound separation: both `log Δt` and `log n_visual_tokens` entered as predictors, which is only
  identifiable because the `tokmatch_*` arms break their collinearity.

## Supplementary experiments

All 373 questions have ≥4 h of stream after their evidence, so **N = 373 with no attrition in every
bin below** (checked against `m2_stream_index.json`, 187,004 s of stream). Costs are extrapolated
from the measured `score_s + frame_load_s` per row of the completed sweep. Submit with
`scripts/submit_supplements.sh [s1|s2|s3|s4|all]`; each writes its own record file and M7 merges
every `m6_records*.jsonl` it finds. All four are resumable.

**S1 — extend the token-fixed elapsed-time axis to 2 h and 4 h** (~0.8 GPU-h). `tokmatch_evkeep`
and `tokmatch_uniform` at Δt ∈ {120, 240} min, still 194 frames / 9.7k tokens per row. Turns the
token-fixed line in panel A from three points into five and tests whether accuracy stays flat while
confidence keeps sliding across a 16× range of elapsed time. Caveat to carry into the write-up: 4 h
of *stream* time routinely crosses wall-clock day boundaries in the A1 recording.

**S2 — the no-memory control on token-matched frames** (~1.1 GPU-h). The control's own decline
(−10.3 pp) is currently un-token-matched, which is the weakest link in the "this is context length,
not forgetting" argument. Runs the control at Δt ∈ {30, 60} on `tokmatch_ckptkeep` (the correct
match for a present-tense question — `evkeep` thins exactly the frames the control asks about) and
on `tokmatch_uniform`. Because control ids are keyed `qid|dt_s`, these pair 1:1 with the existing
`control/main` records. If control accuracy recovers under `ckptkeep`, the entire Δt decline in
both question kinds is context length, stated with a paired test.

**S3 — the no-evidence floor in every Δt bin** (~1.8 GPU-h, dominated by the 60 min bin at 37k
tokens/row). `prior` token-matched to `main` bin by bin. Gives the *value of the evidence* as a
function of Δt: does `main@60` (36.5 %) decay to the floor — is the evidence worth anything at all
by 60 min — and does confidence ever register the difference?

**S4 — explicit abstention** (~2.4 GPU-h). A fifth option, "There is not enough information in the
video to answer", on `main` and `prior` at Δt ∈ {5, 60}; confidence becomes the softmax over
{A,B,C,D,E} and abstention rate is reported alongside accuracy on the four real options. Tests
whether the model *can* report absence when given the vocabulary for it, which is the natural
follow-up to evidence-blind confidence. Scored as `question_kind = "memory_abstain"` into its own
record file, and M7 sets those records aside — a five-way softmax is not comparable with the
four-way confidences the headline rests on.

Order: S1 and S2 first (cheap, and S1 feeds the headline figure), then S3, then S4.

## Known risks carried into execution

1. **Resolution vs accuracy.** Frames must be small enough that 60 min × 0.2 fps ≈ 770 frames fits
   comfortably. Qwen3-VL costs `(T/2)·(H/32)·(W/32)` visual tokens. A pilot (M0/M6-pilot) picks the
   frame size such that Δt=5 min accuracy is clearly above the 25 % chance floor; if no resolution
   clears it, the effect is unmeasurable and that is the finding to report.
2. **Absolute accuracy may be low.** EgoLifeQA is hard, and this setup gives the model a *window*,
   not the full 44 h history the benchmark assumes. Chance is 25 %. The claim is about the
   *slope and the gap*, not the level — but if bin-1 accuracy is at chance the sweep is vacuous.
3. **Auto-generated controls** are only as good as the DenseCaption SRT. They are a directional
   check, not a benchmark; distractor construction is documented in M5 and the control items are
   spot-checked by hand.

Risk 2 is the one that landed: bin-1 accuracy (42.9 %) sits only 11.5 pp above the no-evidence
floor (31.4 %), so the sweep's usable dynamic range is narrow and the *level* claims are weak. The
result that survives is about which manipulations move which measure, and those contrasts are
paired and within-question, so they do not depend on the level at all.
