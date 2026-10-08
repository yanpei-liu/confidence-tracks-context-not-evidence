# Teaser-Experiment

Accuracy–confidence-vs-Δt sweeps for egocentric metamemory, on EgoLifeQA A1 with
Qwen3-VL-8B-Instruct. **Two experiments, both complete**, over the same 373 questions and the same
frozen backbone — they differ only in how the context reaches the model.

| | regime | records | modules | design doc | authoritative artifact |
| --- | --- | --- | --- | --- | --- |
| **E1** | full-reprocess per checkpoint | 5,540 | M0–M9 | [PLAN.md](PLAN.md) | [artifacts/m8_teaser.md](artifacts/m8_teaser.md) |
| **E2** | streaming KV cache (EgoStream) | 17,311 | X0–X12 | [EXTENDING.md](EXTENDING.md) | [artifacts/x9_streaming.md](artifacts/x9_streaming.md) |

Both artifacts are generated from the metrics JSON and are **authoritative for every number**.

- [INTERPRETATION.md](INTERPRETATION.md) — **how to read both, and what not to claim.** Read it
  before quoting anything. §2.8 is the canonical roadmap.
- [../README.md](../README.md) — where the project stands overall.

Directory name: this began as a 2-day teaser and kept the name. Both experiments are still a
teaser for the wider project.

## Layout

```
egovm/config.py           paths + all experiment constants
egovm/timeparse.py        EgoLife DAY{n}/HHMMSSFF parsing (3 target_time encodings)
egovm/scoring.py          Qwen3-VL option-letter scorer  (the only model-specific module)
egovm/m0_smoke.py         env + model smoke test  (GPU)
egovm/m1_questions.py     parse + Mitigation-1 filter of EgoLifeQA A1
egovm/m2_stream_index.py  wall-clock <-> cumulative-stream-time index over 6,266 mp4 chunks
egovm/m3_frames.py        global 0.2 fps frame bank (decode once, re-read many)
egovm/m4_plan.py          (question x Δt x arm) plan with explicit frame lists
egovm/m5_controls.py      no-memory control MCQ from DenseCaption
egovm/m6_score.py         option-letter logprob sweep  (GPU, resumable)
egovm/m7_analyze.py       per-bin acc/conf/ECE, paired manipulation contrasts, figures
egovm/m8_report.py        the write-up, rendered from m7_metrics.json
egovm/m9_page.py          standalone shareable HTML, rendered from metrics + figure  (E1 only)

egovm/streaming/sconf.py       streaming config: budgets, cache policy, marker formats
egovm/streaming/backend.py     EgoStream FlexibleCache + ReKV over the frozen Qwen3-VL-8B
egovm/streaming/x0_report.py   backend provenance + sha256 of the imported EgoStream source
egovm/streaming/x1_calibrate.py   measured token cost, throughput, live caps, clone cost
egovm/streaming/x1b_timeprobe.py  can the model read the injected clock?  (it cannot -- see 4a)
egovm/streaming/x2_schedule.py    per-question ingestion schedule; asserts main nests, prior does not
egovm/streaming/x3_stream_score.py  the sweep  (GPU, resumable, sharded on qid)
egovm/streaming/x4_pilot.py       the gate: Δt=1 min level before anything scales
egovm/streaming/x9_compare.py     M7 per-condition + cross-regime differenced contrasts
egovm/streaming/x10_figure.py     the standalone headline figure; computes nothing
egovm/streaming/x11_evidence_blindness.py  the evidence contrast at every Δt, + robustness
egovm/streaming/x12_evidence_figure.py     the evidence-blindness figure; computes nothing

scripts/                  provisioning, submission helpers, incl. submit_supplements.sh (S1-S4)
slurm/                    batch scripts
requirements.txt          the versions the committed artifacts were produced on
artifacts/                small, committed artifacts (no dataset content)
figures/                  headline.pdf            E1, M7's 3-panel template
                          streaming_headline.pdf  E2, purpose-built by X10 -- the one to show
                          evidence_blind.pdf      E2 evidence blindness, per Δt (X12)
                          evidence_blind_panelA.pdf  the same panel A at slide size
                          supp_sweep*.pdf         the raw 2-panel Δt sweeps, supplementary
```

`figures/headline_stream*` is M7's E1 template applied to E2's records. It is **git-ignored and
not committed**: two of its panel titles are false on those records (see EXTENDING.md §0). X9
regenerates it as a mechanical cross-check; do not put it in front of a reader.

Data (video, frames, raw records) lives at `$EGOVM_DATA`
(default `~/Files/EgoVM-data`) and is **not** committed: EgoLife is
S-Lab License 1.0, research-use, non-redistributable.

## Environment

```bash
EGOVM_PY=/path/to/env/bin/python scripts/install_env.sh
```

The pipeline imports only `torch`, `transformers`, `PIL`, `av` and `matplotlib`.
`requirements.txt` pins the versions that actually produced the committed artifacts, read off
`artifacts/m0_env_report.json`: Python 3.11.15, torch 2.13.0+cu126, transformers 5.15.0,
accelerate 1.14.0, pillow 12.2.0, numpy 2.4.4, on an `NVIDIA H100 NVL MIG 3g.47gb`.

Data and model:

```bash
export EGOVM_DATA=$HOME/Files/EgoVM-data
python scripts/download_a1.py       # EgoLife A1 video + QA + DenseCaption -> $DATA/raw
python scripts/download_model.py    # Qwen3-VL-8B-Instruct -> $DATA/model  (config.MODEL_ID)
```

Both are retry loops around `snapshot_download`, which stalled on this cluster with all workers
idle and no socket timeout. If it will not converge, fall back to plain HTTPS:
`python scripts/make_manifest.py` then `scripts/fetch.sh` (see its header).

## Running

Paths default to `~/Files/EgoVM-data` and `~/miniconda3/envs/egovm`; override with
`EGOVM_DATA` and `EGOVM_PY`. The `scripts/submit_*.sh` wrappers send Slurm logs to
`$EGOVM_DATA/logs/`; a bare `sbatch slurm/<job>.sh` writes its log to the submit directory.

```bash
export EGOVM_REPO=$PWD
PY=$HOME/miniconda3/envs/egovm/bin/python

# CPU stages
$PY -m egovm.m1_questions
$PY -m egovm.m2_stream_index
sbatch --export=ALL,EGOVM_REPO="$EGOVM_REPO" slurm/m3_frames.sh    # ~14 CPU-hours / 32 cores
$PY -m egovm.m4_plan
$PY -m egovm.m5_controls

# GPU stages
sbatch --export=ALL,EGOVM_REPO="$EGOVM_REPO" slurm/m0_smoke.sh
sbatch --export=ALL,EGOVM_REPO="$EGOVM_REPO" slurm/m6_sweep.sh     # resumable; re-submit to continue

# analysis and write-up (CPU)
$PY -m egovm.m7_analyze     # -> artifacts/m7_metrics.json, figures/*.pdf, figures/*.png
$PY -m egovm.m8_report      # -> artifacts/m8_teaser.md
$PY -m egovm.m9_page        # -> artifacts/m9_teaser.html
```

### Running E2 (streaming)

```bash
$PY -m egovm.streaming.x0_report        # backend provenance    (CPU)
sbatch slurm/x1_calibrate.sh            # measured token cost, caps, throughput
sbatch slurm/x1b_timeprobe.sh           # is the injected clock readable?
$PY -m egovm.streaming.x2_schedule      # ingestion schedule    (CPU)
scripts/submit_stream.sh                # the sweep -- note: commas in EGOVM_KINDS are '+'
```

`m6_score.py` skips any `record_id` already present in **any** `m6_records*.jsonl` in the output
directory, so re-submitting after the partition wall limit continues where it stopped and
sharded jobs do not redo each other's rows. `--shard i/n` splits the plan across concurrent
jobs writing to separate output files.

### Reproducing the analysis without a GPU

Neither experiment's analysis needs the model or the video — only the record files.

```bash
# E1: M7 -> M8 -> M9   (needs $DATA/artifacts/m6_records_*.jsonl)
$PY -m egovm.m7_analyze && $PY -m egovm.m8_report && $PY -m egovm.m9_page

# E2: X9 -> X10        (needs $DATA/artifacts/m6_records_stream_*.jsonl)
$PY -m egovm.streaming.x9_compare    # -> m7_metrics_stream*.json, x9_regime_comparison.json,
                                     #    x9_streaming.md
$PY -m egovm.streaming.x10_figure    # -> figures/streaming_headline.{pdf,png}, ~2 s
$PY -m egovm.streaming.x11_evidence_blindness  # -> x11_evidence_blindness.{json,md}, ~1 min
$PY -m egovm.streaming.x12_evidence_figure     # -> figures/evidence_blind*.{pdf,png}, ~2 s
```

X11 discovers its Δt bins from the records, so it reports whatever `prior` bins exist. Its
per-bin contrasts must reproduce `m7_metrics_stream.json`'s `evidence_present_{N}min` exactly;
if they ever do not, the pairing or the bootstrap seeding has drifted and the rest of the file
should not be trusted either.

The cluster bootstrap is seeded (`m7_analyze.py:69`), so the metrics are deterministic — a rerun
that changes a number means the inputs changed.

**Never run M7 directly against `$DATA/artifacts/`.** It globs `m6_records*.jsonl` and writes
`m7_metrics.json` and `figures/headline.*` to fixed paths, so unguarded it pools the two regimes
and overwrites E1's committed figures. X9 drives it per-condition and restores those paths.

### The raw records

`$EGOVM_DATA/artifacts/m6_records*.jsonl` are the primary artifact — one record per
(question, checkpoint, arm, question-kind), ~6 GPU-hours of scoring, **no backup**. They are
not committed (they carry gold answers over a non-redistributable dataset), so the repo pins
them by checksum instead:

```bash
python scripts/records_manifest.py --check    # against artifacts/m6_records_manifest.json
```

## Cluster gotchas that cost time here

- **`sbatch --export` splits on commas.** `--export=ALL,EGOVM_ARMS=main,prior` sets
  `EGOVM_ARMS=main` and creates a stray variable `prior`; two arms were silently never
  scheduled and the jobs looked healthy. Export in the calling shell and pass `--export=ALL`
  (`scripts/submit_arm.sh`). Always check the job's `N rows to run` against the plan.
- **Long jobs do not schedule.** 20 h jobs on `gpu-long` sat at `PD (Priority)` indefinitely;
  the same work at `2:55:00` on `gpu` backfilled within a minute. M6 is resumable, so it runs
  as a chain of short jobs (`scripts/chain_followups.sh`) with distinct `-J` names — the GPU
  race cancels same-name pending jobs and would otherwise eat its own chain.
- **Reads from the login node are stale.** NFS attribute caching made record files and job
  logs look frozen for minutes while the jobs were writing normally. This looks exactly like
  a hang. Trust the job's own progress lines and file mtimes, not `wc -l` from elsewhere.
- **`/home` fsync is expensive.** M6 flushes every row but fsyncs every 100.
- **Not every node runs the conda python.** Two `long`-partition nodes (`xcng0`, `xcng1`)
  failed with `cannot execute binary file: Exec format error` — and so does **`xgpj0`, which is
  in the `gpu` partition** and advertises an ordinary `a100-80`. It is a different CPU
  architecture from the rest of the pool. The job exits 126 after two seconds, writes nothing,
  and **Slurm reports it `COMPLETED`**, so a shard that landed there looks like a shard that
  had no work to do. `scripts/submit_stream.sh` now passes `--exclude=xgpj0`; if you sbatch by
  hand, do the same, and check `X3_EXIT=` at the end of the log rather than the Slurm state.
- **`h100-47` really is ~47 GB visible** even though `nvidia-smi` names a 95 GB card. A 60 min
  window needs a single ~12 GB vision-tower allocation and thrashes there; the headline arms
  run on `a100-80` / `h100-96` / `h200-141`.

## Streaming gotchas that cost time here

All of these fail silently.

- **The image path costs `(res//32)²` tokens/frame, not `/2`** — 100 at `INPUT_RES=320`, double
  E1's. `m4_eval_plan.jsonl`'s `visual_tokens` column is stale by 2× for this backend; X2
  recomputes it.
- **The ReKV retriever's `top_k_tokens` must track `mutable_token_size`.** Left at its shipped
  8,192 while the budget varied, it re-equalised every condition at query time — measured live
  tokens 9,104 / 9,108 / 9,099 for budgets meant to span 1,940 → 33,684. A 17× axis became 1.0×
  with no error raised. Always check the *realised* `live_tokens`, not the configured budget.
- **`--export` splits on commas here too**, and `EGOVM_KINDS=prior,tokmatch` silently runs only
  `prior` and exits 0. `scripts/submit_stream.sh` carries commas as `+`.
- **Querying mutates the cache and there is no rollback**, so every query runs against a
  `copy.deepcopy` clone (0.016 s). The paper describes this master/secondary split; the released
  code does not ship it.
- **`hat_tokens_size` is hardcoded to 896×512.** At 320×320 the protected recent-frame region is
  4.5× oversized unless corrected, with no error.
- **EgoStream has no LICENSE and is not a git checkout**, so its source is imported from
  `$EGOVM_EGOSTREAM` and checksummed by X0 — never vendored, never committed.

## Notes for whoever picks this up

- **Δt is stream time, not wall-clock.** The A1 recording has 31 gaps including six overnight
  breaks of 10–14 h. See `m2_stream_index.json`.
- **Resolution is pinned** (`INPUT_RES`, `do_resize=False`). Qwen3-VL's video processor
  otherwise fits frames to a *total* pixel budget, which would make per-frame resolution a
  decreasing function of Δt. M0 check (3) exists specifically to catch that regression.
- **`target_time` in EgoLifeQA is the evidence timestamp** and it is populated on all 500
  items, contrary to what the pre-execution plan assumed. See `artifacts/m1_schema_report.md`.
- Confidence is the option-letter softmax max — a proxy, reported with ECE, never as a
  calibrated probability.
- **`scoring.py` is the only module that names a model** in E1; `streaming/backend.py` is its E2
  counterpart. M1–M5 and M7–M9 transfer untouched to any backend emitting the documented record
  schema — that is what made E2 cost ~14 GPU-h rather than a rebuild.
- **Streaming breaks row independence.** Rows for one question at different Δt share a cache and
  must run in ascending Δt order on one worker, so shard on `qid`. `main`'s five Δt windows nest
  exactly (one 733-frame ingestion yields all five checkpoints); `prior`'s do not, so each bin
  costs its own pass. X2 asserts both.
- A further ~20 ad-hoc `watch_*.sh` / `submit_*.sh` scripts from the original run remain at the
  data root and are deliberately not vendored — they are one-off job babysitters, superseded by
  the helpers in `scripts/`.
