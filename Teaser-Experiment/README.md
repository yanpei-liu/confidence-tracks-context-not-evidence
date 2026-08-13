# Teaser-Experiment

Accuracy–confidence-vs-Δt sweep for streaming egocentric metamemory, on EgoLifeQA A1 with
Qwen3-VL-8B-Instruct. **Complete**: 5,540 records over 373 questions, ten modules M0–M9.

- [PLAN.md](PLAN.md) — the design, the module decomposition, and what the sweep found.
- [artifacts/m8_teaser.md](artifacts/m8_teaser.md) — the write-up, generated from the metrics.
  **Authoritative for every number.**
- [EXTENDING.md](EXTENDING.md) — porting this design to a second backend (the next experiment
  is a streaming KV-cached VLM).

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
egovm/m9_page.py          standalone shareable HTML, rendered from metrics + figure
scripts/                  provisioning, submission helpers, incl. submit_supplements.sh (S1-S4)
slurm/                    batch scripts
requirements.txt          the versions the committed artifacts were produced on
artifacts/                small, committed artifacts (no dataset content)
figures/                  headline.pdf (3 panels: pinned-context sweep / dissociation map /
                          item transitions + AUROC) and supp_sweep.pdf (the original 2-panel
                          Δt sweep, kept as supplementary)
```

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
export EGOVM_DATA=~/Files/EgoVM-data
python scripts/download_a1.py       # EgoLife A1 video + QA + DenseCaption -> $DATA/raw
python scripts/download_model.py    # Qwen3-VL-8B-Instruct -> $DATA/model  (config.MODEL_ID)
```

Both are retry loops around `snapshot_download`, which stalled on this cluster with all workers
idle and no socket timeout. If it will not converge, fall back to plain HTTPS:
`python scripts/make_manifest.py` then `scripts/fetch.sh` (see its header).

## Running

```bash
export EGOVM_REPO=$PWD
PY=~/miniconda3/envs/egovm/bin/python

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

`m6_score.py` skips any `record_id` already present in **any** `m6_records*.jsonl` in the output
directory, so re-submitting after the partition wall limit continues where it stopped and
sharded jobs do not redo each other's rows. `--shard i/n` splits the plan across concurrent
jobs writing to separate output files.

### Reproducing the analysis without a GPU

M7→M8→M9 need only the record files, not the model or the video. With
`$EGOVM_DATA/artifacts/m6_records*.jsonl` in place, the three commands above regenerate every
committed artifact and both figures. The cluster bootstrap is seeded (`m7_analyze.py:69`), so
the metrics are deterministic — a rerun that changes a number means the inputs changed.

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
  failed with `cannot execute binary file: Exec format error`.
- **`h100-47` really is ~47 GB visible** even though `nvidia-smi` names a 95 GB card. A 60 min
  window needs a single ~12 GB vision-tower allocation and thrashes there; the headline arms
  run on `a100-80` / `h100-96` / `h200-141`.

## Notes for whoever picks this up

- **Δt is stream time, not wall-clock.** The A1 recording has 31 gaps including six overnight
  breaks of 10–14 h. See `m2_stream_index.json`.
- **Resolution is pinned** (`INPUT_RES`, `do_resize=False`). Qwen3-VL's video processor
  otherwise fits frames to a *total* pixel budget, which would make per-frame resolution a
  decreasing function of Δt. M0 check (3) exists specifically to catch that regression.
- **`target_time` in EgoLifeQA is the evidence timestamp** and it is populated on all 500
  items, contrary to `Planning/[v2-a]`. See `artifacts/m1_schema_report.md`.
- Confidence is the option-letter softmax max — a proxy, reported with ECE, never as a
  calibrated probability.
- **`scoring.py` is the only module that names a model.** See [EXTENDING.md](EXTENDING.md)
  before swapping it.
- A further ~20 ad-hoc `watch_*.sh` / `submit_*.sh` scripts from the original run remain at the
  data root and are deliberately not vendored — they are one-off job babysitters, superseded by
  the helpers in `scripts/`.
