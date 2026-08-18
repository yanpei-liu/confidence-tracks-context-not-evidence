# The streaming KV cache — what E2 actually does, and where its design is exposed

*Three classes of statement below, kept separate on purpose. **(a) Mechanism** is read from source:
`Teaser-Experiment/egovm/streaming/{backend,sconf,x3_stream_score}.py` (ours) and the upstream
EgoStream checkout at `$EGOVM_EGOSTREAM` (`Caches/FlexibleCacheLayer.py`, `Caches/pruning_pipeline.py`,
`Caches/similarity_ops.py`, `Configs_files/conf_sing_vid.json`), pinned by the sha256 manifest in
[`artifacts/x0_backend_report.json`](../../../Teaser-Experiment/artifacts/x0_backend_report.json).
**(b) Recomputed numbers** — §4's budget contrasts and §3's offload audit — come from the committed
records (`$EGOVM_DATA/artifacts/m6_records_stream_*.jsonl`) using `egovm.m7_analyze.cluster_bootstrap`
unchanged, 5,000 draws, seed 0, paired within-question, Δ in percentage points; they are new to this
memo and are not in any generated artifact. **(c) Quoted numbers** come from
[`artifacts/x9_streaming.md`](../../../Teaser-Experiment/artifacts/x9_streaming.md) and
[`artifacts/x1b_timeprobe.json`](../../../Teaser-Experiment/artifacts/x1b_timeprobe.json), which
remain authoritative. §7's prompt decode was run this session on the local Qwen3-VL-8B processor,
CPU only, through E1's own `scoring.py:build_inputs` path.*

Prompted by two questions: *what is the streaming KV cache doing, concretely?* and *are the design
choices built around it justified?* The mechanism turns out to be better engineered than what
upstream ships, and four of the interpretive labels attached to it are looser than the measurements
underneath them.

---

## 1. Verdict

**The infrastructure is sound and several choices are better than the upstream code or paper.** The
master/secondary clone, `qid`-sharding with ascending-Δt ordering, tying the retriever's `top_k` to
the budget, running X1b *before* the headline contrast, and refusing to weaken the X4 gate after a
bad sample — all correct, and the last three each caught or prevented a silent failure.

**The exposure is in arm semantics, not in engineering.** Four items, ranked by how much a reviewer
could do with them:

| # | issue | severity |
| --- | --- | --- |
| §4 | the headline budget contrast (`large − small`, −15.2 pp) spans a **change of memory mechanism**, not just a change of size | highest — it is a headline number |
| §5 | "context length" labels **two different variables**, and the decomposition uses the one where live tokens are *identical* | high — it is the one-sentence reading |
| §6 | the stated reason the clock is illegible ("a bounded cache destroys token position") is **not what the code does** | medium — conclusion survives, mechanism doesn't |
| §3 | "the evidence is genuinely evicted" is **inferred, not measured**, and eviction is content-scored rather than recency-ordered | medium — it underpins the AUROC result, and it is cheap to measure |

Two cheap experiments settle most of it (§8): **strip the timestamps from E1** (~1 GPU-h) and
**log per-frame surviving token counts** (minutes). Neither needs new infrastructure.

---

## 2. What the cache actually does

For readers without the background: a transformer caches a key and a value vector per token per
layer so later tokens can attend without recomputation — the **KV cache**. E1 (full-reprocess)
throws it away and re-encodes the whole window on every query, so the evidence frame is always
present at some resolution: it can be diluted, never removed. E2 keeps the cache between
checkpoints, which means it must be **bounded**, which means KV entries are **destroyed**. That is
the entire scientific reason E2 exists.

### 2.1 Layout

EgoStream's `FlexibleCache` splits each layer into three regions:

```
[  sink  ][              mutable               ][  hat  ]
   716        1,024 / 8,192 / 32,768              200
 priming     the evictable region — the        last 2 frames,
 prompt,     experiment's context-length       protected from
 pinned      knob (sconf.BUDGETS)              pruning
```

Live cap = `sink + mutable + hat` = **1,940 / 9,108 / 33,684**, reproduced exactly by X1. `sink` is
716 tokens of EgoStream's own `prompts/template_questions.txt` — one of the two prompt-path
differences from E1 that `INTERPRETATION.md` 0.2 already flags. `hat` is sized upstream as
`hat_num_frames × 448`, valid only at their 896×512; `backend.py:243-244` re-derives it at
`× 100` for `INPUT_RES=320`.

### 2.2 Ingestion and eviction

Frames arrive in batches of 16 (`sconf.INGEST_BATCH_FRAMES`). Each batch is embedded and appended;
`update()` (`FlexibleCacheLayer.py:643`) then compares length against `mutable + sink + hat` and, on
overflow, runs a pruning pipeline over **the mutable region only**:

1. `intra_value_norm` — drop the ~10 % of tokens with the smallest value-vector L2 norm;
2. `inter_adjacent_cosine` — drop the ~10 % most redundant against the adjacent frame;
3. `storage_offload_left_excess` — write the oldest still-excess KV to disk as int8;
4. fallback left trim — if still over, cut the oldest.

Two properties of this are not visible from the config and matter for how we describe E2:

**(i) Eviction is content-scored and global, not recency-ordered.** Candidates are collected over
*every* frame in the mutable region (`similarity_ops.py:968`, chunked only for memory), and
selection is a global bottom-k (`similarity_ops.py:1060`) with **no per-frame floor** in the
`value_norm` branch — `min_keep_per_frame` is honoured only in the sink-attention branch we do not
use. So a frame can go to zero tokens, but equally a *distinctive old* frame can outlive a
*redundant recent* one. "Older evidence is pushed out first" is not this policy.

**(ii) In the primary condition, steps 3 and 4 never fire.** A 16-frame batch adds 1,600 tokens;
steps 1+2 can remove ~10 % + ~10 % of ~9,800 ≈ 1,860 > 1,600, so overflow is always absorbed by
deletion before the pipeline reaches the storage step, which early-returns anyway
(`pruning_pipeline.py:91-99`). Audited over **every** streaming record on disk:

| budget | rows | rows with `offloaded_tokens > 0` |
| --- | ---: | ---: |
| `default` (`main`, `nots`, `prior`, `tokmatch`) | 9,215 | **0** |
| `large` | 3,675 | **0** |
| `small` | 3,675 | **3,675** |

So in the primary condition eviction is deletion and is unrecoverable — which is what `INTERPRETATION.md`
§1.7 claims, now audited exhaustively rather than by median. The mechanism it calls
"not fully characterised" is the batch-size arithmetic above.

### 2.3 Querying

Querying is destructive — `set_QA(True)` stops pruning but `update()` still concatenates the
question's KV and nothing removes it — so every query runs against a `copy.deepcopy` clone
(`backend.py:347`, 0.016 s) and the master is left intact. This is the master/secondary split
EgoStream's paper describes in §B.6 and the released code does not ship.

Before scoring, the **ReKV retriever** rebuilds the mutable region from the top-`k` question-scored
frame-blocks, drawing on offloaded chunks *and* live blocks. Note what that means at the default
budget given the table above: nothing was offloaded, and `top_k` now equals the budget, which is ≥
the live length. **The retriever is inert in the primary condition** and active only in `small`. Our
`cache_policy` string ends in `+rekv_retrieve` on every record regardless, which overstates it.

Then the prompt is appended and confidence is read as the softmax over `{A,B,C,D}` logits at the
first generated position — the same construct as E1, defended separately in
[`CONFIDENCE-PROBE.md`](CONFIDENCE-PROBE.md).

### 2.4 What the model is holding, per bin

Median over 373 questions, `main` arm at the default budget:

| Δt | frames ingested | tokens presented | live tokens | state |
| --- | ---: | ---: | ---: | --- |
| 1 min | 24 | 2,400 | 3,116 | unsaturated |
| 5 min | 72 | 7,200 | 7,916 | unsaturated |
| 15 min | 192 | 19,200 | **9,108** | saturated |
| 30 min | 372 | 37,200 | **9,108** | saturated |
| 60 min | 732 | 73,200 | **9,108** | saturated, ~11 % of presented survives |

The sweep crosses saturation at ≈85 frames (≈7 min), between bins 2 and 3. One 733-frame ingestion
yields all five `main` checkpoints plus all five controls, because the windows nest exactly (X2,
0 violations over 373 × 4 pairs); `prior` windows do not nest, which is why the floor is measured at
three bins only.

---

## 3. Finding 1 — "genuinely evicted" is inferred, and it is cheap to measure

§2.2's audit establishes that whatever was dropped is *unrecoverable*. It does **not** establish
that the evidence frames' tokens were among the dropped — and given that pruning is content-scored
rather than recency-ordered, a distinctive evidence frame could plausibly survive better than the
redundant frames around it. If so, E2 is "very aggressive compression" rather than "eviction", and
§1.6's AUROC result — the one genuinely new phenomenon in E2 — reads differently.

This is directly measurable and nearly free: `FlexibleCacheLayer` tracks per-frame surviving token
counts (`_split_frame_lengths_by_hat`, `get_frame_boundaries`). Logging them at each checkpoint and
reporting *surviving tokens on the evidence frames vs the window mean* converts the premise from an
inference into a measurement, for one extra record field and a few GPU-minutes.

Secondary consequence, worth a sentence in the write-up: because eviction is not recency-ordered,
**a flat accuracy-vs-Δt response is what this cache policy predicts**, not a surprise. Age per se
does not determine what is discarded here.

---

## 4. Finding 2 — the budget arm's headline contrast crosses a mechanism boundary

`large − small` at 60 min is quoted as "17× live-token budget, Δconfidence −15.2 pp". Those
endpoints differ in more than size:

- `small` = 1,024 mutable is **smaller than one 1,600-token ingestion batch**, so every batch
  overflows the whole mutable region;
- `small` is consequently the **only** condition where disk offload fires and the **only** one where
  the ReKV retriever has anything to retrieve (§2.2 table). It is a retrieval-augmented condition;
  `default` and `large` are pure lossy pruning.

`INTERPRETATION.md` §1.7 flags this for the *permanence* claim, but the headline number still spans
it. Recomputed within-question, cluster-bootstrapped, n = 353 questions present in both conditions:

| Δt | `large − default` (offload in neither) | | `default − small` (offload in `small` only) | |
| --- | ---: | ---: | ---: | ---: |
| | live ratio | Δconfidence | live ratio | Δconfidence |
| 1 min | 1.0× | −0.0 [−0.1, +0.1] | 1.6× | −2.0 [−3.4, −0.6] |
| 5 min | 1.0× | −0.1 [−0.2, −0.0] | 4.1× | −4.0 [−5.8, −2.3] |
| 15 min | 2.2× | −3.0 [−4.2, −1.8] | 4.7× | −4.7 [−6.6, −2.9] |
| 30 min | 3.7× | −4.4 [−5.8, −3.0] | 4.7× | −6.1 [−8.1, −4.2] |
| 60 min | 3.7× | **−6.0 [−7.3, −4.7]** | 4.7× | −9.2 [−11.1, −7.2] |

Accuracy is null throughout on the clean pair (`large − default`: +0.0 / −0.0 / +1.7 / +2.0 / +2.3,
every CI containing zero).

Three things to take from this:

1. **The dose–response survives inside the offload-free pair.** The confound does not manufacture
   the effect, and the monotone growth tracks the live ratio as it opens from 1.0× to 3.7×.
2. **−6.0 pp at 3.7× is the clean, single-mechanism number** and should lead; −15.2 pp at 17× is
   the full-span figure and should carry the mechanism caveat.
3. The 1 min and 5 min rows of `large − default` are a free determinism check: both conditions are
   below cap there, so the computation is literally identical, and the measured Δ is −0.0 [−0.1,
   +0.1].

A budget grid of 2,048 / 8,192 / 32,768 — all ≥ one ingestion batch — would have avoided this
entirely, and is the right shape for any future budget-like arm.

---

## 5. Finding 3 — "context length" names two different variables

`INTERPRETATION.md` §1.1's one-sentence reading is *confidence responds to how much context is in
the window*. That is true of the budget arm, where live tokens genuinely vary (1,940 vs 33,684 at a
fixed 732 frames presented). It is **not** true of the contrast the decomposition actually consumes.

`m7_analyze.decomposition` builds its context term from `fewer_tokens` = `evkeep@60 − main@60`
(`m7_analyze.py:294-328`). Checked on the records: `main@60` is 732 frames and `tokmatch_evkeep@60`
is 192 frames, and **both sit at exactly 9,108 live tokens**. The live context is identical; what
differs is how much was discarded to reach it.

| contrast | presented | live | what varies |
| --- | --- | --- | --- |
| budget arm (`large − small`, `large − default`) | fixed | **varies** | how much the model *holds* |
| `main@60 − evkeep@60` (the decomposition's context term) | **varies** | fixed at 9,108 | how much it *threw away* |

Both push confidence down, so the finding is unharmed — but they are different manipulations and
the current wording implies one axis. The honest unifier is broader, and stronger:

> **Confidence falls with the amount of visual input the model has processed, whether or not it
> retained it.**

That covers both rows, and it sharpens §2.4's "surface properties" account rather than weakening
it. Concretely: relabel §1.2's two "context length" rows as *live context* and *presented volume /
eviction pressure*, and say which one the −6.71 pp decomposition term is.

---

## 6. Finding 4 — the stated reason the clock is illegible is not what the code does

`EXTENDING.md` §1a claim 2 reads: *"the only time signal is token position — and a bounded cache is
precisely what destroys token position."* As implemented, it does not:

- keys are stored **post-RoPE**, so surviving entries keep the positional phase they were written
  with; pruning deletes rows, it does not re-encode them;
- `backend.py:315,327` advances `global_token_counter` by the **presented** sequence length at
  ingest, and `backend.py:374` positions the query at that counter.

So positional distance from the query back to the evidence is preserved through eviction and grows
with Δt exactly as it does under full reprocess. What fails is the **text** channel, which is what
X1b actually tested.

The conclusion survives untouched — and in fact survives for a better reason than the one stated:
the `evkeep@60 − main@15` contrast holds the presented token count fixed at 192 frames, so it holds
positional distance fixed too, leaving the injected clock as the only elapsed-time signal that
varies. That is a cleaner argument than the one in the document, and it should replace it.

---

## 7. Finding 5 — E1's clock is *text*, and E2's is a non-native imitation of it

Confirmed by running E1's own `scoring.py:build_inputs` on the local processor (4 frames, 0.2 fps,
`frames_indices = [0, 60, 120, 180]` → real times 0 / 300 / 600 / 900 s):

```
<|im_start|>user
<|vision_start|><150.0 seconds><|vision_start|><|video_pad|>×200<|vision_end|>
                <750.0 seconds><|vision_start|><|video_pad|>×200<|vision_end|><|vision_end|>Q?
<|im_end|>
```

- The clock is **ordinary text the model tokenizes and reads**, one marker per *merged frame pair*
  (`video_grid_thw = [2,20,20]` for 4 frames — that 2:1 temporal merge is exactly E1's `/2` cost
  model), valued at the **mean time of the pair**: mean(0, 300) = 150, mean(600, 900) = 750.
- **It is not RoPE.** Qwen3-VL's MRoPE temporal index carries no seconds; the transformers source is
  explicit — `models/qwen3_vl/modeling_qwen3_vl.py:1483`, *"Qwen3VL use timestamps and remove
  second_per_grid_ts"* — i.e. Qwen2.5-VL's temporal RoPE scaling was **removed** in favour of these
  text markers. E1 has exactly one time channel and it is lexical.

That makes E1's headline manipulation very concrete: `tokmatch_evkeep@60` vs `main@15` is *the same
192 frames' worth of visual tokens, with the numbers printed between them running to ~3600 instead
of ~900*, and the −3.7 pp confidence slide is a response to those printed numbers.

E2 injects the same channel by hand (`backend.py:184,278`), and it is verifiably in the token stream
— but it differs from native in three ways at once, none of which X1b's three formats reproduce:

| | E1 (video path, native) | E2 (image path, hand-injected) |
| --- | --- | --- |
| format | `<150.0 seconds>` (decimal) | `<300 seconds>` (`f"<{t:.0f} seconds>"`) |
| layout | between `<\|vision_start\|>…<\|vision_end\|>` groups, inside an outer vision wrapper | before each frame's own vision block, no outer wrapper |
| granularity | one per merged **pair**, value = pair mean | one per **frame** |
| survival | re-encoded every query, never evicted | ordinary mutable-region tokens, pruned like any other |

X1b's one cell showing any tracking was `angle_seconds` (the closest of the three to native),
**unevicted**, pure readout, 2/3 on three points. So format and eviction are both live explanations
for the failure and the design cannot currently separate them.

---

## 8. The two experiments that settle this

Ordered by what they change per GPU-hour.

**8.1 Strip the timestamps from E1 (~1 GPU-h).** Drop `video_metadata` — or zero the timestamps —
in `scoring.py:109`, and re-score the two token-matched cells (`main@15`, `evkeep@60`) on all 373
questions. This tests the shared premise of every account of the E1→E2 rotation: *is E1's
elapsed-time effect a response to those text tokens at all?* If it is, the story sharpens into
something mechanistically precise — *confidence responds to a printed number it can read; remove the
number, in either regime, and it responds to context volume instead* — and it stops depending on the
memory architecture, which `INTERPRETATION.md` §2.6 already concedes is confounded with the prompt
and ingestion path. This is currently **not** in §2.8's list and it is cheaper than everything that
is.

**8.2 Re-probe E2 with the exact native rendering (minutes).** Re-run X1b's probe with decimal
seconds, per-pair means and the matched vision wrapper, at an unevicted frame count. If legibility
recovers, §1.3's *unavailable* verdict is a property of the format we injected rather than of the
cache — a materially different limitation to write up, and one that changes what §2.8's "restore a
readable clock" follow-up should actually build.

**8.3 Log per-frame survival (minutes, §3).** Converts "the evidence is genuinely evicted" from an
inference into a measurement.

---

## 9. What has to change on our side

**Must fix:**

1. **`INTERPRETATION.md` §1.2 and §1.1** — split the two "context length" rows into *live context*
   and *presented volume / eviction pressure*, state that the decomposition's −6.71 pp term is the
   latter and that live tokens are identical (9,108) on both sides of it, and replace the
   one-sentence reading with §5's unifier.
2. **`EXTENDING.md` §0 and `INTERPRETATION.md` §1.4** — attach the mechanism caveat to the −15.2 pp
   headline and lead with **−6.0 pp [−7.3, −4.7] at 3.7×**, the offload-free contrast. Record that
   `small` (1,024) is below one ingestion batch (1,600) and is the only condition with retrieval in
   play.
3. **`EXTENDING.md` §1a claim 2** — the "bounded cache destroys token position" sentence is wrong;
   replace with §6's argument (positions survive; the text channel is what fails; the token-matched
   contrast holds positional distance fixed anyway).
4. **`EXTENDING.md` §4 / `INTERPRETATION.md` §1.3** — record the three-way format mismatch in §7 as
   a live alternative explanation for the injection failure, alongside eviction.

**Should add:**

5. **`x12_cache_forensics.py`** — regenerates §2.2's offload audit, §2.4's per-bin state table and
   §4's budget contrasts from `m6_records_stream_*.jsonl` into
   `artifacts/x12_cache_forensics.{json,md}`, so these become generated rather than transcribed,
   matching the standard X9/X10 already meet. CPU only.
6. **`surviving_frame_tokens` as a record field** (§3), plus an evidence-frame-vs-window-mean
   contrast in X9.
7. **Fix the `cache_policy` label** or footnote it: `+rekv_retrieve` is nominal at the default and
   large budgets, where nothing is offloaded and `top_k` ≥ live length.
8. **The E1 timestamps-off arm (§8.1)** into `PLAN.md`'s supplementary list, above S1 on
   cost-effectiveness grounds.

**Newly de-risked:** the eviction-permanence claim is now audited over all 16,565 streaming records
rather than sampled (0 of 12,890 default+large rows offloaded anything, 3,675 of 3,675 `small` rows
did), and the dose–response that carries the E2 result is confirmed to survive inside a single
memory mechanism, which was the most plausible way for the headline to have been an artefact.
