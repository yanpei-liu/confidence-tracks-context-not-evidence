# Teaser-Experiment

Accuracy–confidence-vs-Δt sweep for streaming egocentric metamemory. See [PLAN.md](PLAN.md)
for the design and the module decomposition.

## Layout

```
egovm/config.py           paths + all experiment constants
egovm/timeparse.py        EgoLife DAY{n}/HHMMSSFF parsing (3 target_time encodings)
egovm/m0_smoke.py         env + model smoke test  (GPU)
egovm/m1_questions.py     parse + Mitigation-1 filter of EgoLifeQA A1
egovm/m2_stream_index.py  wall-clock <-> cumulative-stream-time index over 6,266 mp4 chunks
egovm/m3_frames.py        global 0.2 fps frame bank (decode once, re-read many)
egovm/m4_plan.py          (question x Δt x arm) plan with explicit frame lists
egovm/m5_controls.py      no-memory control MCQ from DenseCaption
egovm/m6_score.py         option-letter logprob sweep  (GPU, resumable)
egovm/m7_analyze.py       per-bin acc/conf/ECE, cluster bootstrap, headline figure
egovm/scoring.py          Qwen3-VL option-letter scorer
slurm/                    batch scripts
artifacts/                small, committed artifacts (no dataset content)
figures/                  headline figure
```

Data (video, frames, raw records) lives at `$EGOVM_DATA`
(default `~/Files/EgoVM-data`) and is **not** committed: EgoLife is
S-Lab License 1.0, research-use, non-redistributable.

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
$PY -m egovm.m7_analyze
```

`m6_score.py` skips any `row_id` already in `m6_records.jsonl`, so re-submitting after the
partition wall limit continues where it stopped. `--shard i/n` splits the plan across
concurrent jobs writing to separate output files.

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
