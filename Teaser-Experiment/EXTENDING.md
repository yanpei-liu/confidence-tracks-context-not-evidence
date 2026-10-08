# Extending the teaser to a streaming KV cache

The completed sweep runs **Qwen3-VL-8B-Instruct in full-reprocess mode**: every
(question × Δt × arm) row is scored independently, re-encoding its whole window from scratch.
This experiment runs the same design on a **streaming KV cache** over the same backbone — the
model ingests the window once and is queried at checkpoints against a retained, bounded cache.

Read [PLAN.md](PLAN.md) first for the design itself. This document is the execution plan for
the follow-up: the seam map, the facts measured while porting, and the module decomposition.

**Why it is worth running even though the teaser's prediction failed.** Under full-reprocess the
evidence is always present at some resolution — the sweep could dilute it but never remove it.
Under a bounded cache the evidence is *evicted*. That makes the streaming condition the first
setting in which "doesn't know it forgot" is testable against actual forgetting. The question
the teaser could not ask is the one this experiment exists to ask:

> **Does confidence still slide with elapsed time when the evidence has been evicted rather
> than merely diluted — and does accuracy finally move with it?**

---

## 0. Result — the sweep is complete

**17,311 records over 373 questions, five arms, 100 % of the planned schedule, zero skips**
(~9 GPU-h). [`artifacts/x9_streaming.md`](artifacts/x9_streaming.md) is generated from
`x9_regime_comparison.json` and is **authoritative for every number**; the prose here is
interpretation and may round.

**The figure to show is [`figures/streaming_headline.pdf`](figures/streaming_headline.pdf)**
(X10) — five panels drawn for this experiment alone, with every arm and every held-fixed
variable named on the page. `figures/headline_stream.*` is M7's own template applied to these
records; it is a useful mechanical cross-check but **its panel titles are the full-reprocess
sweep's claims, and two of them are false here** — it asserts the confidence ranking is
untouched by Δt (§0 shows it degrades) and treats elapsed time as a manipulation that returned
a null rather than an input the model cannot read (§4a). Do not put it in front of a reader.

> **How to read these results — including what not to claim from them — is
> [INTERPRETATION.md](INTERPRETATION.md).** Part 1 covers this experiment, Part 2 covers it
> jointly with the full-reprocess sweep. Read it before quoting anything below. Verdict selected by the measured intervals among the four
outcomes in §7: **Outcome 3 — the dissociation replicates.**

**The answer to the question above is: no, and the reason is instructive.** Confidence does
*not* slide with elapsed time under streaming, and accuracy does not move with it either —
because once the cache saturates the model has no elapsed-time input at all (§4a). What the
regime change did was not destroy the dissociation but **rotate** it.

| manipulation | Δaccuracy (pp) | Δconfidence (pp) |
| --- | --- | --- |
| +45 min elapsed, frames and evidence fixed (`evkeep@60 − main@15`) | −1.7 [−5.1, +1.6] | **−0.1 [−1.7, +1.4]** |
| 17× live-token budget, elapsed time and frames fixed (`large − small` @60 min) | −1.9 [−6.4, +2.4] | **−15.2 [−17.5, −13.1]** |
| evidence in context vs none (`main@5 − prior@5`) | **+8.1 [+2.9, +13.1]** | +0.5 [−1.5, +2.6] |
| timestamps on vs off, pooled | −0.6 [−1.7, +0.5] | −0.7 [−1.2, −0.2] |

Read together: **accuracy responds to whether the evidence is there and to nothing else;
confidence responds to how much context is in the window and to nothing else.** The two
measures are driven by disjoint variables. This is the teaser's finding with the confidence
term re-pointed: there confidence moved on elapsed time (an input it could read), here it moves
on context volume (the only input that still varies).

Five results that sharpen it:

- **The evidence row is a curve, not a cell.** The `prior` arm runs at every Δt in
  `sconf.PRIOR_DTS`, so the evidence contrast is measured at each of them (X11). Δaccuracy
  excludes zero in 3 of the 5 and runs from +11.3 pp at 1 min down to +1.6/+2.9 by 30–60 min,
  where the advantage has itself been evicted; Δconfidence spans zero in all five and tracks
  neither the effect nor its decay. That is a much stronger statement than the single 5 min
  cell in the table above, which is what X9 reports because `streaming_specs` pins the bin to
  keep its headline comparable across reruns. The per-bin table, the confidence-distribution
  overlap, the evidence-presence AUROC and both denominators are in
  `artifacts/x11_evidence_blindness.md`; quote that file for this row.
- **The teaser's headline contrast does not replicate, and the failure is significant.** The
  full-reprocess sweep's central claim — confidence slides with elapsed time at fixed tokens and
  evidence — gave −3.7 pp there and **−0.1 pp [−1.7, +1.4]** here. Cross-regime difference
  **+3.5 pp [+1.7, +5.4]**, excluding zero. X1b supplies the mechanism *independently and in
  advance*: the injected clock is illegible once the cache saturates.
- **A clean dose–response on confidence with a flat accuracy response.** As the live-token
  ratio grows 1.7× → 17× across the Δt bins, Δconfidence grows monotonically −2.0 → −4.2 → −8.1
  → −10.5 → **−15.2 pp**, while Δaccuracy sits at zero in every bin (widest point estimate
  −1.9 pp, every CI containing 0). More context buys no accuracy and costs a great deal of
  confidence. **Carry the mechanism caveat with the endpoint:** `small`'s 1,024 mutable tokens sit
  below one 1,600-token ingestion batch, so it is the only condition that offloads (3,675/3,675
  rows, against 0/13,636 for default and large) and the only one with retrieval in play. The
  clean single-mechanism contrast is `large − default` at 60 min, **−6.0 pp [−7.3, −4.7] at 3.7×**.
  Quote −15.2 for the span of the axis, −6.0 when the claim needs one mechanism.
- **M7's decomposition closes almost exactly** — tighter than on the teaser's own data
  (residual +0.05 pp accuracy, +0.02 pp confidence, against its 2 pp tolerance). The 15 → 60 min
  confidence decline is −6.71 pp context and **−0.13 pp elapsed time**: the time term is not
  merely small, it is absent.
- **Discrimination degrades, which it did not under full-reprocess.** AUROC over the sweep
  Δ = **−0.073 [−0.143, −0.003]**, p = 0.039, against +0.012 (flat) in the teaser. When evidence
  is genuinely evicted rather than diluted, the confidence *ranking* decays too — not just its
  level. This is the one place where the streaming regime shows something the teaser could not
  produce.

And one honest null, plus one prediction that failed:

- **Item-level metamemory: suggestive, underpowered, not significant.** Among items correct at
  the earlier bin, confidence fell −3.0 pp on items kept (465) and −6.8 pp on items lost, difference
  **−3.8 pp [−8.5, +1.1], p = 0.118**. The teaser found this flatly item-blind (+1.0, p = 0.62),
  so streaming moves it in the direction of genuine metamemory — but on 107 lost items the CI
  crosses zero and this must not be quoted as a positive finding.
- **X8's confirmatory prediction was wrong in shape.** §4a predicted `main` ≈ `nots` at the
  saturated bins and a difference only at Δt = 1 and 5 min. Observed: **no** difference at 1 min
  (−0.2 [−1.1, +0.7]) and small but interval-excluding-zero confidence differences at 15, 30 and
  60 min (−1.6, −1.2, −2.2 pp) — the opposite shape. The effects are an order of magnitude below
  the context-length term and carry no accuracy difference, so the §4a conclusion that the clock
  is not a usable input stands; but the specific prediction is recorded as **not confirmed**,
  and the residual effect is most likely the markers' own ~5 % token cost changing what the
  cache holds rather than any perception of time.

---

## 1. Status of the port

Everything in this section was measured, not inferred. The port smoke test
(`~/egostream-smoke/egostream_smoke.py`, 10/10 checks, zero failures) ran 2026-08-13 on
`h100-96` against the local Qwen3-VL-8B snapshot, ingesting our own frame bank. The token-cost
probe was re-run 2026-08-14 on CPU against the local processor.

| Fact | Value | How known |
| --- | --- | --- |
| Backbone | **Qwen3-VL-8B, frozen** — same family as the completed sweep | read `EgoStream/`; ran it |
| Tokens/frame at `INPUT_RES=320` | **100** = `(320//32)²` | processor probe: `image_grid_thw = [1,20,20]`, ÷4 merge |
| Tokens/frame at the paper's 896×512 | 448 | same probe; matches the smoke run's cache growth |
| Cache cap | `sink + mutable + hat`; **9,804** at the paper's default and 896×512 | smoke reproduced it exactly |
| Saturation | reached after ~20 frames at 448 tok/frame, then flat | smoke; 402,912 token-layers offloaded |
| Ingestion throughput | **0.0355 s/frame** at 320×320, full pruning pipeline (0.22 s at 896×512) | X1 on `h100-96`, steady state |
| Peak GPU | **19.1 GB** at 320×320 (22.15 GB at 896×512) | X1 — so `a100-40`/`h100-47` are eligible, unlike the full-reprocess sweep |
| Cache clone | `copy.deepcopy` = **0.016 s**, master's seq length unchanged after a query | X1 |
| Query cost | 0.31 s with ReKV retrieval, 0.06 s without | X1 |
| Option letters | `A`–`D` single-token, ids 32–35 | X1 |
| Live caps, measured exactly | small **1,940** / default **9,108** / large **33,684** | X1 — equals `sink+mutable+hat` for all three |

**Two structural facts about the plan, verified against `m4_eval_plan.jsonl` this session:**

- **The `main` arm's Δt windows nest exactly.** All five bins share `window_lo_stream_s`, and
  each bin's `gids` is a strict prefix of the next. Checked over all 373 questions × 4 adjacent
  pairs: **0 violations.** One 733-frame ingestion per question therefore yields *all five*
  `main` checkpoints **and** all five `control` checkpoints. This is the entire economic case
  for streaming, and it is now a machine-checked assertion in X2 rather than an assumption.
- **The `prior` arm's windows do not nest.** M4 draws each bin's evidence-free window from an
  independently chosen start (`m4_plan.py:148-159` — the span, and hence the start index,
  depends on that bin's frame count). Checked the same way: **1,491 of 1,491 adjacent pairs
  violate nesting.** Every `prior` bin costs its own ingestion. This single fact moves `prior`
  from "one extra pass" to the most expensive thing in the experiment, and it is why §5 runs
  `prior` on the three cheap bins only.

### 1a. The three traps, all of which fail silently

1. **Tokens per frame double.** EgoStream ingests through the **image** path
   (`content_list = [{"type": "image"} …]`, `src/model_operations.py:386`), not the video path.
   There is no temporal merge, so the cost is `(res//32)²`, not the sweep's `(res//32)²/2`.
   At `INPUT_RES=320` that is **100 tokens/frame, not 50**. Nothing errors; the numbers are
   just someone else's.
2. **The model cannot perceive elapsed time at all.** Qwen3-VL interleaves `<t>` markers
   between frames only on the *video* path; on the image path nothing carries a clock.
   *(Corrected after the fact: this section originally said the only remaining time signal is
   token position and that a bounded cache destroys it. That is not what the code does — keys are
   stored post-RoPE and `global_token_counter` advances by presented length, so positions survive
   eviction, and the token-matched contrast holds positional distance fixed anyway. The channel
   that fails is the lexical one. See `STREAMING-CACHE.md` §6, and §4a below for what actually
   happened.)* **Our headline finding is that confidence moves on elapsed time.** Unmodified,
   this backend has no elapsed-time input at all, so the experiment would be measuring
   something else entirely. §4 makes timestamp injection a first-class, validated condition.
3. **`hat_tokens_size` is hardcoded to 896×512.** `model_operations.py:173` computes
   `hat_num_frames * 448`. At 320×320 the correct value is `hat_num_frames * 100`; leave it and
   the protected recent-frame region is 4.5× oversized with no error.

Smaller ones: the repo pins torch 2.6 / transformers 4.57 and needs **three one-line shims**
under our torch 2.13 / transformers 5.15 (`FlexibleCacheLayer.get_max_length` now abstract;
`get_mask_sizes` receives an `int` not a tensor; `get_rope_index` requires `mm_token_type_ids`,
derivable from the `<|image_pad|>` mask). And `cv2` is absent from `egovm`, while
`src/video_sampler.py` imports it at module load — irrelevant to us, since frames come from our
bank rather than their sampler, but it must be stubbed before the import succeeds.

### 1b. The licence constraint shapes the code layout

The EgoStream checkout ships **no LICENSE file** and is not a git repository, so there is no
upstream commit to pin and no grant permitting redistribution. **Its source is therefore not
vendored into this repo and not committed.** X0 instead writes a thin adapter that imports
EgoStream from a path outside the packaged source (`$EGOVM_EGOSTREAM`, default
`<repo>/EgoStream/Online_Cache_Optimization` -- inside the working tree but gitignored, so it is
never committed), applies the shims at import time from our own code, and records a checksum
manifest of the upstream files it depends on. Only our adapter is committed. This mirrors the
existing rule that dataset content lives outside the repo, and it is why the plan has no
"vendor and pin" step of the kind the earlier sketch proposed.

## 2. What transfers untouched

**M1–M5 are model-independent** — they construct the question set, the stream-time coordinate,
the frame bank, the evaluation plan and the no-memory controls without loading a model. The
committed artifacts are re-used as-is: `m1_questions.jsonl` (373 questions),
`m2_stream_index.json`, `$DATA/frames/` + `m3_frame_index.json` (37,401 frames at 0.2 fps),
`m4_eval_plan.jsonl` (7,460 rows, each with an explicit `gids` list),
`m5_control_questions.jsonl` (1,825 control MCQs keyed `qid|dt_s`).

**M7–M9 are also model-independent.** They consume only the record schema, so a backend that
emits the same fields gets the whole analysis for free — per-bin accuracy/confidence/ECE, the
paired manipulation contrasts, the decomposition, AUROC, item transitions, both figures, the
generated write-up and the HTML page.

The contract M7 relies on, one JSON object per line in `$DATA/artifacts/m6_records*.jsonl`:

```
row_id  qid  type  dt_s  arm  n_frames  visual_tokens  checkpoint_stream_s
evidence_max_stream_s  need_audio  model  sample_fps
record_id            "<row_id>|memory" or "<row_id>|control"
question_kind        "memory" | "control"   (a new kind is set aside, not merged)
gold  correct  pred  confidence  entropy
letter_probs  letter_logprobs      {A,B,C,D} -> float
n_input_tokens  n_visual_tokens
frame_load_s  score_s
```

Two constraints on that schema are load-bearing, not cosmetic:

- **The softmax denominator must be constant across a records file.** This is why S4's
  five-letter abstention rows carry `question_kind = "memory_abstain"` and are set aside by M7
  (`m7_analyze.py:1037-1041`) rather than averaged in.
- **`record_id` is the resume and dedup key.** M6 resumes by scanning *every*
  `m6_records*.jsonl` in the directory (`m6_score.py:106-115`), and M7 dedups on it before
  filtering to rows in the current plan (`:1027-1035`). Streaming records are written to
  `m6_records_stream_*.jsonl`, which matches that glob, so both behaviours come along — and,
  importantly, **M7 run against `$DATA/artifacts/` would silently pool the two regimes.** X9
  runs M7 with a records-glob restricted to one regime at a time. That is a real footgun and
  the reason X9 is a module rather than a command.

### 2a. What must be re-derived, and what must not be quoted

| Thing | Status under the streaming backend |
| --- | --- |
| `visual_tokens_per_frame()` (`config.py:105`) | **replaced**: `(res//32)²`, no `/2`. 100 at `INPUT_RES=320`. |
| `m4_eval_plan.jsonl`'s `visual_tokens` column | **stale by 2×** for this backend. X2 recomputes it into the schedule; the plan file is left untouched so the Qwen3-VL sweep stays reproducible. Never quote the plan's column in a streaming context. |
| single-token option letters | unchanged, verified (ids 32–35) |
| `INPUT_RES` pinning | still required, and now enforced by our own resize before the processor sees the frame |
| `VideoMetadata` / `<t seconds>` markers | **not available** — image path. Replaced by explicit injection, §4. |
| `do_sample_frames` / `do_resize` overrides | not applicable; we call `processor.image_processor` directly on frames we have already sized |

> **The `tokmatch_*` arms are matched on frame count, not on token count.** Every row's
> `visual_tokens` is just `n_frames × 50` under the old cost model. The arms themselves transfer
> unchanged — the same frames are the same frames — but the column does not.

## 3. The structural break

The current design is full-reprocess per checkpoint, so **every plan row is independent**. That
independence buys modulo sharding (`m6_score.py:94-95`), longest-first ordering (`:99-102`),
row-level resume, and the OOM skip that keeps a multi-hour unattended job alive (`:182-193`).

Under a streaming cache none of it holds: the rows for one question at different Δt share a
cache and must be produced in ascending Δt order on one worker. Consequences, all of which the
X3 driver implements rather than inherits:

- **Shard on `qid`, not on plan index.** A modulo split over the plan scatters one question's Δt
  series across workers, each of which would re-ingest the stream — discarding exactly the
  saving that motivates streaming.
- **Order by `(qid, dt_s)` ascending.** `--order hard-first` must not be the default here.
- **Resume granularity is the question, not the row.** A half-ingested cache cannot be
  reconstructed from a records file, so a job killed at the 3 h wall limit loses the question in
  flight. At the measured throughput a whole question fits in ~1–3 min, so sizing the work unit
  is sufficient and cache checkpointing is unnecessary.
- **Querying mutates the cache and there is no rollback.** `set_QA(True)`
  (`Caches/FlexibleCacheLayer.py:429`) stops pruning, but `update()` still concatenates the
  question tokens onto the live KV (`:657`) and nothing truncates them afterwards. Multi-
  checkpoint querying therefore **requires the master/secondary clone** the paper describes in
  §B.6 and the released code does not ship. `copy.deepcopy` is the implementation and it works
  (0.016 s, master intact — the timing in §1's table). Two vestiges confirm this was the intended design: the unused
  `IncrementalSlidingWindowSampler` whose docstring names the master/question split, and
  `DiskOffloadStore.__deepcopy__`, which exists only because something deep-copies the cache.
- **The controls ride on the same cache**, which is a genuine advantage: both queries hit the
  same checkpoint, so a control costs one clone plus one forward pass, not a second ingestion.

### 3a. Ingest per question, not over the whole life-log

Each question's cache is reset and ingestion starts at `evidence_min − PRE`, exactly as the
full-reprocess window does. A true 44 h life-log stream would make Δt inseparable from
everything that preceded it and would cost days of GPU. Resetting per question preserves every
arm definition in PLAN.md and keeps the cross-regime comparison clean.

**This is a limitation to state in the write-up, not to hide:** this is streaming *over the
window*, not over the life-log. What it buys is a bounded cache and genuine eviction; what it
does not buy is a claim about 44 h of accumulated history.

### 3b. The arms change meaning under a bounded cache

This is the part with scientific consequences rather than engineering ones. The live cache is
capped, so **context length is no longer ours to manipulate by choosing frames** — it is pinned
by the memory budget.

| Arm | Full-reprocess meaning | Streaming meaning |
| --- | --- | --- |
| `main` | tokens grow with Δt | tokens grow with Δt only until saturation, then flat; what grows after that is *eviction pressure* |
| `tokmatch_uniform` / `tokmatch_evkeep` | hold token count fixed while Δt grows | **degenerate as a token match once saturated** — the cache already holds tokens fixed. They survive as a *frame-rate / evidence-density* manipulation at fixed elapsed time. Relabel them in the write-up; do not silently reuse the old gloss. |
| `prior` | evidence-free window, same frame count | unchanged in meaning; needs its own ingestion **per bin** (§1) |
| `control` | different question, same frames | unchanged; one extra clone at the same checkpoint |

At `INPUT_RES=320` (100 tok/frame) and the default budget (716 sink + 8,192 mutable + 200 hat =
**9,108** live tokens), saturation lands at roughly 91 frames of presented video — i.e. between
the Δt = 5 min bin (72 frames, unsaturated) and the Δt = 15 min bin (194 frames, saturated). The
sweep therefore spans the transition, which is itself worth reporting. (Measured after the fact
it is ≈85 frames, ≈7 min: the live cache also holds the prompt, 588 tokens at the 5 min bin, and
X10 derives the mark that way rather than from the 9,108 ÷ 100 arithmetic above.)

**Recreating the context-length axis: the `budget` arm.** Because the cache pins tokens, the
manipulation that carries the dissociation in the completed sweep — *4× more visual tokens at
fixed elapsed time* — has no analogue unless we create one. So we vary `mutable_token_size`
over three settings at fixed frames and fixed Δt:

| budget | `mutable_token_size` | live cap at 320×320 | vs default |
| --- | --- | --- | --- |
| `small` | 1,024 | 1,940 | 0.21× |
| `default` | 8,192 | 9,108 | 1× |
| `large` | 32,768 | 33,684 | 3.7× |

This is a within-question paired contrast in exactly the form M7 already consumes, and without
it the streaming condition has an elapsed-time axis and an evidence axis but **no context-length
axis**, so the headline contrast table in PLAN.md *Findings* cannot be reproduced. Note the
correction to the earlier sketch: a budget change alters *ingestion*, not just the query, so
each budget is a full re-ingestion. It is the second most expensive thing in the experiment,
not a ~3 GPU-h afterthought.

### 3c. Record schema additions

Streaming rows carry fields the full-reprocess schema has no analogue for. Existing fields keep
their existing meanings:

```
cache_policy          e.g. "prune_intra_vn10+inter_cos10+offload_int8"
mutable_token_size    the configured budget
budget_label          "small" | "default" | "large"
live_tokens           cache length at query time -- what the model actually attends to
offloaded_tokens      cumulative, summed over layers
frames_ingested       frames actually fed before this checkpoint
timestamps_injected   bool
ingest_s              cumulative ingestion cost to this checkpoint
clone_s               cost of the master/secondary deepcopy for this query
pass_id               which ingestion pass produced this row
```

`visual_tokens` keeps its meaning (tokens *presented*, now at 100/frame) so the column stays
comparable across conditions. **`live_tokens` is the new quantity and is the one to plot against
accuracy** — it is the streaming regime's context-length variable.

## 4. The timestamp condition

This is the one design call that changes what the experiment measures, so it is settled here
rather than left to the driver.

The full-reprocess sweep gives the model `<t seconds>` markers interleaved between frames via
`VideoMetadata`. The streaming backend's image path gives it nothing. Since the teaser's
headline finding is *confidence moves on elapsed time*, running the follow-up with elapsed time
removed from the input would not be a comparison — it would be a different experiment.

**Primary condition: timestamps on.** A short text marker is interleaved before each frame in
the ingestion batch, `<{t:.0f} seconds>`, with `t` measured in the same window-relative
coordinate the full-reprocess sweep uses (`frame_offsets[k] × 5 s`). Mechanically this is safe:
`compute_text_embeddings` divides the batch's visual tokens evenly across the `<|image_pad|>`
placeholders (`tensor_ops.py:128`), so interleaved text does not disturb the mapping, and
`prepare_frame_append` derives its frame boundaries from `visual_pos_mask`
(`FlexibleCacheLayer.py:600`), so it tolerates non-visual tokens in the appended span. Cost is
~5 tokens per 100, i.e. ~5 %.

### 4a. The validation failed, and that is a result

X1 and X1b tested the injection rather than assuming it, and the premise did not survive.

The first probe was not diagnostic — one 600 s window, "how long was this?", answered "about
10 minutes" at p = 0.999 both with markers and without, because that is also the model's prior.
Nothing was learned. The redesigned probe holds the **frames fixed** (120 real frames) and
varies **only the injected clock** over 60 / 600 / 3600 s. The answer did not move: `['B','B','B']`
with markers, `['B','B','B']` without.

The markers are provably present — decoding the ingested token stream gives
`<0 seconds><|vision_start|><|image_pad|><|vision_end|><300 seconds>…` — so this is not an
injection bug. X1b then separated the explanations over 3 marker formats × 2 question types ×
2 cache states:

| condition | 32 frames (3.2k tokens, **unevicted**) | 120 frames (12k tokens, **evicted**) |
| --- | --- | --- |
| readout, `<t seconds>` | `['C','B','C']` — 2/3 correct, **tracks** | `['C','C','C']` — no tracking |
| readout, other formats | 1/3, no tracking | 1/3, no tracking |
| elapsed (integration), any format | `['B','B','B']` | `['B','B','B']` |
| markers off | `['B','B','B']` | `['B','B','B']` |

Read across, with the caveat that each cell is three points and "tracks" means two distinct
answers rather than a measured effect:

- **The clock is at best weakly legible, and only while it is still in the cache.** The single
  condition showing any tracking is unevicted, `<t seconds>`, and a pure readout.
- **Once the cache saturates, the clock is gone.** No format, no question type recovers it.
- **Integration never works.** "How much time has elapsed" returns the prior in every cell
  including markers-off, so it measures nothing.

**Consequence for the design, stated plainly: at Δt ∈ {15, 30, 60} min — every bin where the
cache is saturated — the model has no usable elapsed-time input.** The streaming condition
therefore *cannot* test "does confidence move on elapsed time" the way the full-reprocess sweep
did. What it can test is what happens when the evidence is genuinely evicted, which is the
reason to run it (§0). The elapsed-time axis is reported as **unavailable in this regime**, not
as a null result — an input the model cannot read is not a manipulation that failed.

`<t seconds>` stays the primary format: it is the one with any legibility, and it matches what
the full-reprocess sweep receives via `VideoMetadata`.

> **Eviction is one of two live explanations, and this design cannot separate them.** Checked
> afterwards by running E1's own `scoring.py:build_inputs` on the local processor: E1's clock is
> ordinary text, `<150.0 seconds>`, emitted **once per merged frame pair** and valued at the
> pair's mean — and Qwen3-VL carries no seconds in RoPE at all, since transformers
> `modeling_qwen3_vl.py:1483` removed Qwen2.5-VL's temporal scaling in favour of these markers.
> E2's hand-injected channel differs from that in three ways at once — integer rather than
> decimal, placed before each frame's own vision block rather than between wrapper groups, and
> one marker per *frame* rather than per pair — and, unlike E1's, it is prunable. X1b's single
> tracking cell was the format closest to native, **unevicted**. So "the cache evicted the clock"
> and "the clock was never in a form this model reads" both fit the evidence, and §8.1's
> timestamps-off arm on E1 is what would separate them. See `STREAMING-CACHE.md` §7.

**X8 is promoted from ablation to confirmatory test.** If the above is right, `main` and `nots`
should be statistically indistinguishable at the saturated bins and may differ only at Δt = 1
and 5 min, where the clock is still live. That is a sharp, falsifiable prediction at the scale
of the real sweep, on 373 questions rather than three probe points — and it is a strong internal
validity check on the whole streaming condition. It runs on all questions, not a subset.

> **Outcome: the prediction failed, in shape rather than in magnitude.** Measured per bin, the
> confidence difference is −0.2 [−1.1, +0.7] at 1 min (no effect, where one was predicted) and
> −1.6 / −1.2 / −2.2 pp at 15 / 30 / 60 min with intervals excluding zero (an effect, where none
> was predicted). Accuracy is null in every bin. The effects are ~1/7 the size of the
> context-length term, so §4a's conclusion — the clock is not a usable input — survives; the
> sharp prediction built on top of it does not. The likeliest reading is mechanical rather than
> perceptual: markers cost ~5 % of tokens, so at a fixed cap the two conditions hold slightly
> different amounts of visual content, and confidence is strongly sensitive to exactly that.
> Recorded here rather than quietly dropped, because a confirmatory test that only counts when
> it confirms is not one.

## 5. Modules

Decomposed along **verifiable-artifact boundaries**, matching the M0–M9 convention: each module
emits a file that can be checked on its own, and downstream modules consume only that file. Any
module can be re-run without its predecessors. Verification is inline assertions, as in the
teaser — there is no test suite.

Code lives in `Teaser-Experiment/egovm/streaming/`; small artifacts in
`Teaser-Experiment/artifacts/`; records in `$DATA/artifacts/`.

| # | Module | Artifact | Verification |
| --- | --- | --- | --- |
| **X0** | EgoStream adapter + shims | `egovm/streaming/backend.py`, `artifacts/x0_backend_report.json` | imports under `egovm` with no `cv2` installed; all three shims applied; sha256 of every upstream file depended on; `hat` size parameterised on tokens/frame; licence status recorded |
| **X1** | backend calibration (GPU) | `artifacts/x1_calibration.json` | (1) **tokens/frame constant** at 100 across batch sizes 4/16/64 — the resolution-confound regression test; (2) cache saturates at `sink+mutable+hat` for each of the three budgets; (3) deepcopy leaves master seq length unchanged; (4) letters single-token; (5) **synthetic colour MCQ** — the model demonstrably reads frames through this path (4/4); (6) **timestamp probe**, frames held fixed and only the clock varied; (7) throughput s/frame and peak GB at 320×320 |
| **X1b** | elapsed-time diagnostic (GPU) | `artifacts/x1b_timeprobe.json` | runs only because X1 (6) failed. 3 marker formats × {readout, integration} × {unevicted, evicted}, separating "the markers do not reach the model" from "the cache evicts them" from "the model cannot integrate them". See §4a — the outcome changes what the experiment claims. |
| **X2** | streaming schedule | `artifacts/x2_stream_schedule.jsonl`, `x2_schedule_report.md` | one row per **ingestion pass**; every checkpoint's gid prefix is asserted **equal to the corresponding M4 row's `gids`** (the nesting proof, machine-checked); `visual_tokens` recomputed at 100/frame; pass `row_id` sets disjoint and covering; a control resolves for every `main` checkpoint |
| **X3** | streaming driver | `egovm/streaming/x3_stream_score.py`, `artifacts/x3_schema_check.json` | emits the §2 contract + §3c fields; a validator asserts M7 would accept the file; shards on `qid`; resumes at question granularity; `live_tokens ≤ cap` on every row |
| **X4** | pilot gate (~1 GPU-h) | `$DATA/artifacts/m6_records_stream_pilot*.jsonl`, `artifacts/x4_pilot_report.md` | three **pre-registered** gates: (a) letter probabilities not degenerate (mean max-prob < 0.99, sd > 0.02); (b) `live_tokens` saturates where X2 predicts; (c) Δt=1 min accuracy within ±15 pp of the committed full-reprocess records on the *same* questions, paired, and above chance. **Evaluate on ≥60 questions** — see below. |

> **The pilot must be a representative sample, and originally it was not.** `subset_rank` first
> ordered questions by `qid`, so `--max-rank 20` drew the twenty lowest ids — a set on which the
> *committed full-reprocess sweep* scores 0.200 at Δt = 1 min against 0.427 over all 373. Gate
> (c) failed on its "above chance" clause against that anomalous baseline, even though streaming
> beat full-reprocess in all five bins on those same questions and answer agreement was 0.58
> against a 0.25 floor. The fix is in the sampling, not the threshold: `subset_rank` is now a
> deterministic sha256 shuffle, so any prefix is representative. Re-evaluated at n = 86 the gate
> **passes as originally written** — streaming 0.454 vs full-reprocess 0.395, Δ = +5.8 pp,
> agreement 0.733. Do not weaken a pre-registered threshold to accommodate a bad sample.
| **X5** | main pass | `m6_records_stream_main_*.jsonl` | 373 questions × 5 checkpoints × {memory, control}; ≤ 3,730 records; `frames_ingested` monotone in `dt_s` within a question |
| **X6** | budget arm (context-length axis) | `m6_records_stream_budget_*.jsonl` | same questions at `small` and `large`; `live_tokens` separates the three budgets by ≥2× at Δt ≥ 15 min; pairs 1:1 with X5 on `qid` |
| **X7** | prior + tokmatch passes | `m6_records_stream_prior_*.jsonl`, `m6_records_stream_tokmatch_*.jsonl` | `prior` at every Δt on the grid (per-bin ingestion, §1 — `sconf.PRIOR_DTS`); `tokmatch_evkeep`/`tokmatch_uniform` at Δt=60 min; frame counts match the M4 rows exactly. The 30 and 60 min bins were added after the first sweep, at a measured ~1.2 and ~3.9 GPU-h; X3 resumes on `record_id`, so re-running `--kinds prior` scores only what is missing |
| **X8** | timestamps-off ablation | `m6_records_stream_nots_*.jsonl` | subset re-run with `timestamps_injected=false`; pairs 1:1 with X5 |
| **X9** | analysis + cross-regime comparison | `artifacts/m7_metrics_stream.json`, `x9_regime_comparison.json`, `x9_streaming.md` | M7 run **with a regime-restricted records glob** (§2); contrasts computed within streaming, then differenced against the committed full-reprocess contrasts with cluster-bootstrap CIs on the difference |
| **X10** | the standalone headline figure | `egovm/streaming/x10_figure.py`, `figures/streaming_headline.{pdf,png}` | CPU-only, ~2 s: reads `x9_regime_comparison.json` (contrasts, budget arm, cells) and `m7_metrics_stream.json` (per-bin levels, decomposition, AUROC, item transitions) and **computes nothing**, so the figure cannot drift from the metrics. Every panel names the cells it differences and what is held fixed; the elapsed-time rows are drawn as an *unavailable* axis, not a null (§4a) |
| **X11** | evidence blindness, per Δt | `egovm/streaming/x11_evidence_blindness.py`, `artifacts/x11_evidence_blindness.{json,md}` | CPU-only, ~1 min. The deck's central claim rested on one cell (`main@5 − prior@5`); this reports the same contrast at **every** Δt with a `prior` bin, discovering the bins from the records rather than hardcoding them. Adds confidence-distribution overlap, a cluster-bootstrapped AUROC for confidence *as a detector of evidence presence*, and the off-letter / vocabulary-normalised robustness rows at every bin. **Its per-bin contrasts must reproduce `m7_metrics_stream.json`'s `evidence_present_{N}min` exactly** — same `contrast`, same seed — which is the check that it is reading the same data the same way |
| **X12** | the evidence-blindness figure | `egovm/streaming/x12_evidence_figure.py`, `figures/evidence_blind.{pdf,png}`, `figures/evidence_blind_panelA.{pdf,png}` | CPU-only. Reads `x11_evidence_blindness.json` and **computes nothing**, as X10 does. Deliberately a separate file from `streaming_headline.pdf`: adding a panel there would change that figure's sha256 and invalidate every crop the briefing decks already take from it. The `_panelA` variant is authored at the size a slide shows it, because a crop of the 6.4 in figure displayed at 2.7 in renders every label at a third of its intended size |

### Order and gating

X0 → X1 → X2 → X3 → **X4 is a gate**. Nothing beyond X4 runs until its three checks pass; a
failure there means the driver is wrong and every GPU-hour after it would be wasted.

Then X5 (the headline sweep) → X6 (restores the context-length axis) → X7 → X8. X5 alone
supports a publishable result — the Δt sweep with controls, in a regime where evidence is
actually evicted. Each later pass adds an axis, so a partial run degrades gracefully instead of
collapsing. **Run in this order and stop wherever the budget stops.**

## 6. Cost

Frame counts are exact, read from `m4_eval_plan.jsonl` and re-derived by X2. The rate is
**measured, not extrapolated**: X1 on `h100-96` gives **0.0355 s/frame** steady state at
320×320 with the full pruning pipeline — 6× faster than the 896×512 figure, because a frame
costs 100 tokens here rather than 448. Query cost is 0.31 s per checkpoint with ReKV retrieval
(0.06 s without) plus a 0.016 s clone, i.e. ~3 s per question against ~26 s of ingestion.

| Pass | Ingestions | Frames | GPU-h @0.0355 s/frame | What it buys |
| --- | --- | --- | --- | --- |
| X5 main | 1 × 373 | **273,622** | **2.7** | `main` + `control` at all 5 Δt bins |
| X6 budget | 2 × 373 | **547,244** | **5.4** | the context-length axis at all 5 bins |
| X7 prior @{1,5,15} | 3 × 373 | **109,182** | **1.1** | the no-evidence floor at three bins |
| X7 prior @{30,60} | 2 × 373 | **412,964** | **5.1** | the floor at the remaining two bins (added later — see below) |
| X7 tokmatch @60 | 2 × 373 | **144,404** | **1.4** | elapsed time at fixed frame count |
| X8 timestamps-off | 1 × 373 | **273,622** | **2.7** | the time-blind ablation |

Full programme = **1,348,074 frames ≈ 13.3 GPU-h**, against ~6 GPU-h for the entire completed
sweep. That is cheap enough that **no subsetting is needed and every pass runs on all 373
questions.** Two levers stay available if the cluster is contended rather than for cost:

- X6 and X8 accept `--max-rank k` (first *k* by sorted `qid`), a deterministic subset; both are
  paired within-question contrasts and lose precision, not validity, at smaller *n*.
- `prior` at Δt ∈ {30, 60} min was deferred in the original programme and **has since been
  run**. The deferral cost more than it saved: the floor being measured at 1, 5 and 15 min only,
  and *assumed flat* beyond, was a scope condition on every statement about the evidence
  contrast. Measured per-pass cost put the two bins at ~1.2 and ~3.9 GPU-h — 412,964 frames,
  a little above the 413,000-frame estimate here but at a slower realised rate than 0.0355 s/frame,
  because ingestion cost per frame rises with the length of the window (0.008 s/frame at 1 min
  against 0.050 s/frame at 60 min). Flip `sconf.PRIOR_DTS`, re-run X2, and re-run X3 with
  `--kinds prior`: it resumes on `record_id` and scores only the missing bins.

**Scheduling.** `gpu` caps at 3 h and backfills well; `gpu-long` did not schedule in practice.
Every pass is sharded on `qid`: at 733 frames × 0.0355 s ≈ 26 s of ingestion plus ~3 s of
queries per question, one 3 h job covers ~370 questions — so the largest pass fits in a
**single shard**, and sharding is for parallelism and wall-clock latency rather than necessity.
Peak GPU is **19.1 GB**, so `a100-40` and `h100-47` are in the pool alongside the 80 GB cards;
the `h100-47` exclusion in the teaser's README does not apply to this experiment.

## 7. What the result will and will not be able to claim

**Identifiable, because the backbone is held fixed.** EgoStream is a plug-in memory manager over
a frozen Qwen3-VL — the same model family as the completed sweep — so streaming-vs-full-reprocess
is a within-backbone comparison by construction. A flatter (or steeper) confidence curve under
streaming is attributable to the regime rather than to the model. The "bridge run" the earlier
sketch required is unnecessary; its second purpose, validating the new scorer against a
known-good pipeline, is served by the X4 pilot gate.

**Compare contrasts, not levels.** The teaser's level claims are weak by construction — bin-1
accuracy 42.9 % against a 31.4 % no-evidence floor, an 11.5 pp band — which is exactly why the
result that survived is a set of paired, within-question contrasts. Expect the level to move
between regimes; it is not the quantity of interest. X9 differences *contrasts* across regimes,
not accuracies.

**Pin the model size before quoting the paper.** EgoStream's text-only baseline row is labelled
"Qwen 8B" but the shipped `conf_sing_vid.json` names `Qwen3-VL-2B-Instruct`. We run 8B to match
the completed sweep; note in the write-up that the published EgoStream numbers may not be at
that size.

**Four pre-enumerated outcomes**, so the write-up reports what the intervals show rather than
asserting a hypothesis — the standard M8 already holds:

1. **Confidence tracks eviction.** Accuracy falls with Δt (evidence evicted) *and* confidence
   falls with it, gap flat. Metamemory is better than the teaser suggested; the teaser's null
   was a ceiling artefact of never removing the evidence.
2. **"Doesn't know it forgot", found at last.** Accuracy falls with Δt and confidence does not,
   so the gap widens — the original prediction, recovered in the regime where forgetting is
   real. The strongest possible result and the reason to run this.
3. **The dissociation replicates.** Confidence keeps moving on elapsed time while accuracy moves
   on `live_tokens` and on evidence presence — the teaser's finding, now with a genuine
   context-length axis and genuine eviction.
4. **Bounded-cache floor effect.** Accuracy collapses to the `prior` floor by 15 min, so the
   later bins carry no signal. Then the finding is about where the cache stops supporting recall
   at all, and the Δt axis must be re-cut below saturation.

Outcome 4 is the live risk, and it is the reason the `prior` floor (X7) is not optional and the
pilot gate (X4) checks the Δt=1 min level before anything scales.

**The verdict is denominator-sensitive and the module does not say so.** `x9_compare.verdict`
selects between these outcomes off E2's `evidence_vs_none` Δconfidence, which is
+0.55 [−1.52, +2.65] under the restricted option-letter softmax (→ Outcome 3, as reported) and
+2.61 [+0.33, +4.94] under a vocabulary-normalised one (→ Outcome 1). The restricted probe is the
published standard and is inert on every other quantity measured — argmax agrees on 24,433 of
24,433 rows — so Outcome 3 stands; but the module should emit both values alongside the verdict
rather than leaving the sensitivity in a memo. Logged in INTERPRETATION.md §2.8.
See `CONFIDENCE-PROBE.md` §4.

## 8. Unaffected

The S1–S4 supplements in [PLAN.md](PLAN.md#supplementary-experiments) are independent of all of
this and remain available on the Qwen3-VL full-reprocess sweep. S1 is still the cheapest
(~0.8 GPU-h, rows already in the committed plan) and still extends the token-fixed elapsed-time
axis to 2 h and 4 h.
