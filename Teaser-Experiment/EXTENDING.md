# Extending the teaser to a second backend

The completed sweep runs **Qwen3-VL-8B-Instruct in full-reprocess mode**: every
(question × Δt × arm) row is scored independently, re-encoding its whole window from scratch.
The next experiment runs the same design on a **streaming KV-cached VLM**, where the model
ingests the stream once and is queried at checkpoints against a retained cache.

This document is the seam map: what transfers untouched, what is Qwen3-VL-specific, what
breaks structurally, and what the result will and will not be able to claim. Read
[PLAN.md](PLAN.md) first for the design itself.

---

## 1. What transfers untouched

**M1–M5 are model-independent.** They construct the question set, the stream-time coordinate,
the frame bank, the evaluation plan and the no-memory controls without loading a model at all.
Re-use the committed artifacts as-is:

| Artifact | Why it is backend-agnostic |
| --- | --- |
| `artifacts/m1_questions.jsonl` | EgoLifeQA parse + Mitigation-1 filter. 373 kept questions. |
| `artifacts/m2_stream_index.json` | wall-clock ↔ cumulative stream seconds over the 6,266 chunks. |
| `$DATA/frames/` + `artifacts/m3_frame_index.json` | 37,401 frames at 0.2 fps, 448×448 on disk. Decoded once; every backend re-reads them. |
| `artifacts/m4_eval_plan.jsonl` | the 7,460 rows, each with an **explicit frame-id list** (`gids`). See the caveat in §3 about its `visual_tokens` column. |
| `artifacts/m5_control_questions.jsonl` | 1,825 DenseCaption-derived control MCQs, keyed `qid|dt_s`. |

**M7–M9 are also model-independent.** They consume only the record schema, so a backend that
emits the same fields gets the entire analysis — per-bin accuracy/confidence/ECE, the paired
manipulation contrasts, the decomposition, AUROC, item transitions, both figures, the generated
write-up and the HTML page — for free.

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
  (`m7_analyze.py:1037-1041`) rather than averaged in. A new backend that scores a different letter
  set must do the same.
- **`record_id` is the resume and dedup key.** M6 resumes by scanning *every* `m6_records*.jsonl`
  in the directory (`m6_score.py:106-115`), and M7 dedups on it before filtering to rows in the
  current plan (`:1027-1035`). Write a new backend's records to their own file(s) matching that
  glob and both behaviours come along.

## 2. The seam

`m6_score.py:143` hardcodes `from .scoring import LetterScorer`. That is the only place the
pipeline names a model. A second backend implements the same two-method contract:

```python
LetterScorer.load(model_id=..., letters=("A","B","C","D")) -> scorer
scorer.score(images, question, choices, frame_offsets=None) -> dict
```

returning `letter_probs`, `letter_logprobs`, `pred`, `confidence`, `entropy`,
`n_input_tokens`, `n_visual_tokens`. Add a `--backend` flag (or `EGOVM_BACKEND`) to M6 that
selects the import; leave `scoring.py` alone so the Qwen3-VL results stay reproducible.

`frame_offsets` are positions on the 0.2 fps grid relative to the window start, so frame *k*
sits at `frame_offsets[k] * 5 s`. They must be honoured — the non-uniformly sampled arms
(`tokmatch_evkeep`, `tokmatch_ckptkeep`) are *defined* by uneven spacing, and a backend that
assumes uniform frame timing will silently mislabel them.

## 3. What must be re-derived per backend

| Thing | Why it is Qwen3-VL-specific |
| --- | --- |
| `visual_tokens_per_frame()` (`config.py:105`) | `(res//32)²/2` is Qwen3-VL's cost model — at `INPUT_RES=320` that is 50 tokens/frame, verified constant in M0. Another backbone has a different figure. |
| single-token option letters (`scoring.py:61-68`) | `A`–`D` each encode to one token under this tokenizer (ids 32–35). Another tokenizer may not. The load path already raises on this; keep the check. |
| `INPUT_RES` pinning + `do_resize=False` (`config.py:100`, `scoring.py:115`) | exists to defeat Qwen3-VL's *total*-pixel-budget resize, which would make per-frame resolution a decreasing function of Δt — precisely the confound the design removes. |
| `VideoMetadata` / `<t seconds>` markers (`scoring.py:109-112`) | without it Qwen3-VL assumes 24 fps and labels a 60-minute window as ~16 seconds. Every backend needs *some* answer to "does the model know how long this window is?", and it will not be this API. |
| `do_sample_frames=False` (`scoring.py:115`) | the processor default resamples to 2 fps capped at 768 frames, delivering 6 frames where 64 were passed. |

> **The `tokmatch_*` arms are matched on frame count, not on token count.** Every row's
> `visual_tokens` is just `n_frames × 50`, derived from the Qwen3-VL cost model. So the arms
> themselves transfer to any backend unchanged — the same frames are the same frames — but the
> `visual_tokens` column in `m4_eval_plan.jsonl` is stale for a different backbone and must be
> recomputed before it is quoted. This is the easiest thing in the whole port to get wrong,
> because nothing will fail; the numbers will just be someone else's.

**Re-run M0 before anything else.** Its check (3) — visual tokens per frame constant at 8, 64
and 256 frames — is the regression test for the resolution confound, and check (5) is a
synthetic colour MCQ that verifies the model reads frames at all. Both are the cheapest way to
find out that a new processor is quietly rewriting the inputs.

## 4. The structural break

Everything above is configuration. This part is not.

The current design is *full-reprocess per checkpoint*, so **every plan row is independent**.
That independence is what buys:

- modulo sharding across concurrent jobs (`--shard i/n`, `m6_score.py:94-95`);
- longest-window-first ordering, so a truncated job leaves a Δt-balanced result set
  (`:99-102`);
- row-level resume against any record file in the directory;
- the OOM skip that drops one row and keeps a multi-hour unattended job alive (`:182-193`).

Under a streaming KV cache none of that holds, because the rows for one question at different
Δt **share a cache and must be produced in ascending Δt order on one worker**. Consequences to
design for up front:

- **Shard on `qid`, not on plan index.** A modulo split over the plan scatters one question's
  Δt series across workers, each of which would then have to re-ingest the stream — throwing
  away exactly the saving that motivates streaming.
- **Order rows by `(qid, dt_s)` ascending**, not by frame count. `--order hard-first` must not
  be the default for this backend.
- **Resume granularity becomes the question.** A half-ingested cache cannot be reconstructed
  from a records file, so a job killed at the 3 h `gpu` wall limit loses the question in
  flight, not just the row. Either checkpoint the cache to disk or size the work unit so a
  whole question fits inside the wall limit with margin.
- **The `prior` arm needs its own pass.** It draws an evidence-free window ≥2 h away from the
  evidence, matched frame-for-frame to the same question's `main` window. Under full-reprocess
  that is just a different frame list; under streaming it is a different cache state, so it
  needs a cache reset or a second ingestion.
- **The controls ride on the same frames as the memory question** (`m6_score.py:203-216`),
  which is a genuine advantage here: both queries hit the same cache at the same checkpoint,
  so the control costs one extra query rather than a second ingestion.

## 5. The confound, and the bridge condition

ReKV/StreamKV ship **no Qwen3-VL integration** — their backbones are LLaVA-OneVision, LongVA
and Video-LLaVA. So moving to a streaming KV cache moves the *backbone* at the same time, and
any difference in the dissociation is confounded between two changes at once:

|  | full-reprocess | streaming KV cache |
| --- | --- | --- |
| **Qwen3-VL-8B** | ✅ the completed sweep | not available |
| **streaming model's backbone** | ⬅ **run this first** | the target experiment |

**Run the existing full-reprocess pipeline on the streaming model's own backbone before running
the streaming condition.** It is the same code path with a new `scoring.py`, it costs roughly
what the original sweep cost, and it buys two things:

1. **Identification.** With the bottom-left cell filled, streaming-vs-full-reprocess is a
   within-backbone comparison and backbone effects are differenced out. Without it, a flatter
   confidence curve under streaming is unattributable.
2. **Validation.** It scores the new backend against a known-good pipeline on known-good
   inputs. If the new scorer disagrees with the committed Qwen3-VL numbers in ways the model
   change does not explain — chance-level accuracy at Δt=1 min, degenerate confidences, token
   counts that move with Δt — the bug is in the scorer, and you find that out before spending
   GPU-hours on the streaming arm.

Expect the *level* to move between backbones. The teaser's level claims are already weak
(bin-1 accuracy 42.9 % against a 31.4 % no-evidence floor — an 11.5 pp band), which is why the
result that survived is a set of **paired, within-question contrasts** between manipulations.
Those are the quantities to compare across backends; the raw accuracy numbers are not.

## 6. Suggested order

1. Re-run **M0** with the new backend. Do not proceed while `failures` is non-empty.
2. Recompute `visual_tokens_per_frame` for the new backbone; regenerate the plan's token column
   if you intend to quote it.
3. **Full-reprocess on the new backbone** (bridge condition), reusing M1–M5 artifacts and the
   existing M6 loop. Analyse with M7–M9 as-is.
4. Restructure M6 for shared-cache execution (§4) and run the **streaming** condition.
5. Compare the manipulation contrasts across all three cells.

The S1–S4 supplements in [PLAN.md](PLAN.md#supplementary-experiments) are unaffected by any of
this and remain available on the Qwen3-VL sweep; S1 in particular (~0.8 GPU-h, rows already in
the committed plan) still extends the token-fixed elapsed-time axis to 2 h and 4 h.
