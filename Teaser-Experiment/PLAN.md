# Teaser Experiment — Execution Plan

**Goal (from CLAUDE.md):** the headline accuracy–confidence figure — accuracy vs Δt and mean
confidence vs Δt on one axis, per-bin ECE, bootstrap CIs. Predicted signature: accuracy declines
while confidence stays flat ("doesn't know it forgot").

The task is decomposed into 8 modules along **verifiable-artifact boundaries**: each module emits a
file that can be checked on its own, and downstream modules consume only that file. Any module can
be re-run without re-running its predecessors.

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

**Δt grid:** 5 / 15 / 30 / 60 min. All 376 kept questions have ≥60 min of stream after their
evidence, so N = 376 per bin — above the ≥150 target with margin.

**Arms:**

| Arm | Frames | Purpose |
| --- | --- | --- |
| `main` | 0.2 fps over the window | primary Δt sweep |
| `tokmatch` | window uniformly subsampled to the *same frame count as that question's Δt=15 min window* | token-matched control: isolates Δt from context length |
| `control` | same frames as `main` | no-memory-required question about the checkpoint instant |

`main` at Δt=60 min vs `tokmatch` at Δt=60 min separates "more tokens" from "more elapsed time";
`control` vs `main` separates generic long-context degradation from forgetting.

**Question set (Mitigation 1):** keep EntityLog (125) + EventRecall (126) + RelationMap (125) = 376.
Drop TaskMaster (63, prospective). HabitInsight (61) is scored but held out of the headline set and
reported separately. `need_audio` items are flagged and reported both in and out (the model is
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
| M4 | eval plan | `m4_eval_plan.jsonl` | 376 × 4 Δt × arms rows; frame counts monotone in Δt for `main`, constant for `tokmatch` |
| M5 | no-memory controls | `m5_control_questions.jsonl` | one control MCQ per distinct checkpoint; distractors drawn ≥30 min away |
| M6 | scoring sweep | `m6_records.jsonl` | one record per plan row, resumable, logs n_frames/n_tokens/model revision |
| M7 | analysis + figure | `m7_metrics.json`, `figures/headline.pdf` | per-bin acc/conf/ECE + bootstrap CIs; bin×measure interaction test |

## Statistics

Within-question design (each question appears in all 4 bins), so:

- Bootstrap CIs by resampling **questions** (cluster bootstrap), not (question, Δt) rows.
- Primary test: the **bin × measure interaction** — does the accuracy-vs-Δt slope differ from the
  confidence-vs-Δt slope? Implemented as a cluster-bootstrap CI on
  `[conf(60) − acc(60)] − [conf(5) − acc(5)]` (the change in overconfidence gap).
- Secondary: mixed logistic `correct ~ log Δt + (1|question)` and `confidence ~ log Δt + (1|question)`.
- Confound separation: both `log Δt` and `log n_visual_tokens` entered as predictors, which is only
  identifiable because `tokmatch` breaks their collinearity.

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
