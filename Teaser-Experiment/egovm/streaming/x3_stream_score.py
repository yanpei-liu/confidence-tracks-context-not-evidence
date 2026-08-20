"""X3 — the streaming driver. The M6 analogue, restructured around a cache that has state.

M6's work unit is a plan row and every row is independent. Here the work unit is an
**ingestion pass** (X2): one cache, ingested once in ascending stream order, queried at several
checkpoints. That difference propagates:

  * **shard on `qid`**, not on plan index — a modulo split over rows would scatter one
    question's Δt series across workers, each re-ingesting the stream and throwing away the
    only saving streaming buys;
  * **checkpoints are consumed in ascending `dt_s`** — the cache only moves forward;
  * **resume granularity is the pass**, not the row. A half-ingested cache cannot be rebuilt
    from a records file, so a job killed at the wall limit loses the pass in flight. Passes are
    minutes, so sizing the work unit is enough and cache checkpointing is unnecessary;
  * **every query runs against a deepcopy.** `set_QA(True)` stops pruning but `update()` still
    concatenates the question tokens onto the live KV and nothing removes them, so querying the
    master would contaminate every later checkpoint.

Record ids embed the **condition** (`stream`, `small`, `large`, `nots`) because the budget and
timestamp conditions reuse the plan's `row_id` — they have to, since M7 drops any record whose
`row_id` is not in `m4_eval_plan.jsonl` (`m7_analyze.py:1033`). Keeping the plan's `row_id` and
separating conditions by record *file* is what lets M7 run unchanged on each condition, one at
a time. Running it over all of them at once would pool conditions into the same cells, which is
the footgun X9 exists to avoid.

Artifacts: $DATA/artifacts/m6_records_stream_<tag>.jsonl, artifacts/x3_schema_check.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from PIL import Image

from ..config import ARTIFACTS, BIG_ARTIFACTS, QA_JSON, SAMPLE_FPS
from ..m3_frames import frame_path
from . import sconf

# the fields M7 and M8 read; a records file missing any of these is not analysable
M7_REQUIRED = (
    "row_id", "qid", "type", "dt_s", "arm", "n_frames", "visual_tokens",
    "checkpoint_stream_s", "evidence_max_stream_s", "need_audio", "model", "sample_fps",
    "record_id", "question_kind", "gold", "correct", "pred", "confidence", "entropy",
    "letter_probs", "letter_logprobs", "n_input_tokens", "n_visual_tokens",
    "frame_load_s", "score_s",
)
STREAM_FIELDS = (
    "cache_policy", "mutable_token_size", "budget_label", "live_tokens", "offloaded_tokens",
    "frames_ingested", "timestamps_injected", "ingest_s", "clone_s", "pass_id", "cond",
)
# Added with the retriever fix, so the `main`/`nots`/`prior`/`tokmatch` passes predate it. Its
# value there is not ambiguous — those all run the default budget, where the shipped
# top_k_tokens (8,192) already equals mutable_token_size, which is exactly why they did not
# need re-running. Reported as a provenance gap rather than a schema failure, and never
# back-filled: the records are the primary artifact and are not rewritten after the fact.
LATE_FIELDS = {"retrieve_top_k": 8192}


def cond_tag(p: dict) -> str:
    """The condition a pass belongs to. Distinguishes records that share a plan `row_id`."""
    if not p["timestamps"]:
        return "nots"
    if p["budget_label"] != sconf.DEFAULT_BUDGET:
        return p["budget_label"]
    return "stream"


def expected_records(p: dict) -> list[tuple[str, dict, str]]:
    out = []
    c = cond_tag(p)
    for cp in p["checkpoints"]:
        out.append((f"{cp['row_id']}|{c}|memory", cp, "memory"))
        if cp.get("control_id"):
            out.append((f"{cp['row_id']}|{c}|control", cp, "control"))
    return out


def load_frames(gids: list[int]) -> list[Image.Image]:
    out = []
    for g in gids:
        with Image.open(frame_path(g)) as im:
            out.append(im.convert("RGB").copy())
    return out


def validate_file(path: Path) -> dict:
    """Assert M7 would accept this records file. Emitted as x3_schema_check.json."""
    plan_rows = {json.loads(l)["row_id"]
                 for l in (ARTIFACTS / "m4_eval_plan.jsonl").read_text().splitlines()}
    n, missing, bad_kind, off_plan, over_cap, ids = 0, set(), 0, 0, 0, set()
    late_missing, budgets_seen = set(), set()
    caps = {b: sconf.live_cap(b) for b in sconf.BUDGETS}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        n += 1
        missing |= {k for k in M7_REQUIRED + STREAM_FIELDS if k not in r}
        late_missing |= {k for k in LATE_FIELDS if k not in r}
        budgets_seen.add(r.get("budget_label"))
        if r["question_kind"] not in ("memory", "control"):
            bad_kind += 1
        if r["row_id"] not in plan_rows:
            off_plan += 1
        if r["live_tokens"] > caps[r["budget_label"]] + 16 * sconf.TOKENS_PER_FRAME:
            over_cap += 1
        ids.add(r["record_id"])
    rep = {
        "file": str(path), "n_records": n, "n_unique_record_ids": len(ids),
        "missing_fields": sorted(missing), "records_with_bad_question_kind": bad_kind,
        "records_not_in_plan": off_plan, "records_over_cache_cap": over_cap,
        "budgets": sorted(b for b in budgets_seen if b),
    }
    if late_missing:
        # only a real gap if a non-default budget is present, where the value could differ
        non_default = [b for b in budgets_seen if b and b != sconf.DEFAULT_BUDGET]
        rep["provenance_gap"] = {
            "fields": sorted(late_missing),
            "implied_values": {k: LATE_FIELDS[k] for k in sorted(late_missing)},
            "note": "field postdates this pass; value is unambiguous at the default budget",
            "blocking": bool(non_default),
        }
    blocking_gap = bool(late_missing) and rep.get("provenance_gap", {}).get("blocking")
    rep["ok"] = (n > 0 and not missing and bad_kind == 0 and off_plan == 0
                 and over_cap == 0 and len(ids) == n and not blocking_gap)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="main", help="comma-separated: main,budget,prior,tokmatch,nots")
    ap.add_argument("--budgets", default="", help="for --kinds budget: small,large (default both)")
    ap.add_argument("--shard", default="0/1", help="i/n — split on qid")
    ap.add_argument("--max-rank", type=int, default=0,
                    help="only questions with subset_rank < K (deterministic subset for X6/X8)")
    ap.add_argument("--limit", type=int, default=0, help="cap on passes this job runs")
    ap.add_argument("--tag", default="main", help="record file tag -> m6_records_stream_<tag>.jsonl")
    ap.add_argument("--time-budget-s", type=int, default=0)
    ap.add_argument("--no-retrieve", action="store_true",
                    help="disable the ReKV retrieval step: evicted evidence is then genuinely "
                         "gone rather than recoverable from disk")
    ap.add_argument("--validate", default="", help="validate an existing records file and exit")
    a = ap.parse_args()

    if a.validate:
        rep = validate_file(Path(a.validate))
        sconf.X3_SCHEMA_CHECK.write_text(json.dumps(rep, indent=2))
        print(json.dumps(rep, indent=2))
        raise SystemExit(0 if rep["ok"] else 1)

    kinds = set(a.kinds.split(","))
    budgets = set(a.budgets.split(",")) if a.budgets else set(sconf.BUDGET_ARM_BUDGETS)
    passes = [json.loads(l) for l in sconf.X2_SCHEDULE.read_text().splitlines()]
    passes = [p for p in passes if p["kind"] in kinds]
    if "budget" in kinds:
        passes = [p for p in passes if p["kind"] != "budget" or p["budget_label"] in budgets]
    if a.max_rank:
        passes = [p for p in passes if p["subset_rank"] < a.max_rank]

    i, n = (int(x) for x in a.shard.split("/"))
    passes = [p for p in passes if int(p["qid"]) % n == i]
    # ascending (qid, dt) — the cache only moves forward, so hard-first ordering is meaningless
    passes.sort(key=lambda p: (int(p["qid"]), p["checkpoints"][0]["dt_s"], p["pass_id"]))

    out_path = sconf.records_path(a.tag)
    done = set()
    for q in sorted(BIG_ARTIFACTS.glob(f"{sconf.STREAM_RECORD_PREFIX}*.jsonl")):
        with q.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["record_id"])
                except Exception:  # noqa: BLE001 — tolerate a torn final line
                    pass

    todo = [p for p in passes if any(rid not in done for rid, _, _ in expected_records(p))]
    if a.limit:
        todo = todo[: a.limit]
    n_frames_todo = sum(p["n_frames"] for p in todo)
    print(f"X3: {len(todo)} passes ({n_frames_todo:,} frames to ingest), "
          f"{len(done)} records already present", flush=True)
    if not todo:
        return

    raw = {str(x["ID"]): x for x in json.loads(QA_JSON.read_text())}
    controls = {}
    for line in (ARTIFACTS / "m5_control_questions.jsonl").read_text().splitlines():
        c = json.loads(line)
        controls[c["control_id"]] = c

    import torch

    from .backend import StreamingScorer, policy_string

    t0 = time.time()
    sc = StreamingScorer.load()
    print(f"model loaded in {time.time()-t0:.1f}s", flush=True)

    offload_root = sconf.OFFLOAD_ROOT / f"job{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    t_start = time.time()
    n_pass = n_rec = n_skip = 0
    frames_seen = 0

    with out_path.open("a") as fout:
        for p in todo:
            if a.time_budget_s and time.time() - t_start > a.time_budget_s:
                print("time budget reached, stopping cleanly", flush=True)
                break
            want = {rid for rid, _, _ in expected_records(p) if rid not in done}
            if not want:
                continue
            cond = cond_tag(p)
            gids = p["gids"]
            g0 = gids[0]
            cache = None
            try:
                cache = sc.new_cache(p["budget_label"],
                                     offload_dir=offload_root / p["pass_id"].replace("|", "_"),
                                     retrieve=not a.no_retrieve)
                policy = policy_string(cache.egovm_cfg)
                pos = 0
                t_frames = 0.0
                for cp in sorted(p["checkpoints"], key=lambda c: c["n_frames"]):
                    while pos < cp["n_frames"]:
                        k = min(sconf.INGEST_BATCH_FRAMES, cp["n_frames"] - pos)
                        tf = time.time()
                        batch = load_frames(gids[pos:pos + k])
                        t_frames += time.time() - tf
                        # window-relative seconds, the same clock the full-reprocess sweep's
                        # "<t seconds>" markers carry
                        times = [(g - g0) * sconf.GRID_S for g in gids[pos:pos + k]]
                        sc.ingest(cache, batch, times, timestamps=p["timestamps"])
                        pos += k
                    frames_seen = pos

                    base = {
                        "row_id": cp["row_id"], "qid": p["qid"], "type": p["type"],
                        "dt_s": cp["dt_s"], "arm": cp["arm"], "n_frames": cp["n_frames"],
                        "visual_tokens": cp["visual_tokens"],
                        "checkpoint_stream_s": cp["checkpoint_stream_s"],
                        "evidence_max_stream_s": p["evidence_max_stream_s"],
                        "need_audio": p["need_audio"],
                        "model": sconf.MODEL_ID, "sample_fps": SAMPLE_FPS,
                        "cache_policy": policy,
                        "mutable_token_size": sconf.BUDGETS[p["budget_label"]],
                        "budget_label": p["budget_label"],
                        # recorded so the budget arm is auditable: a top_k that does not track
                        # the budget silently re-equalises the conditions at query time
                        "retrieve_top_k": (cache.egovm_cfg.get("retrieve", {}) or {})
                        .get("top_k_tokens"),
                        "offloaded_tokens": int(cache.get_offloaded_token_count())
                        if hasattr(cache, "get_offloaded_token_count") else 0,
                        "frames_ingested": int(cache.egovm_frames),
                        "timestamps_injected": bool(p["timestamps"]),
                        "ingest_s": round(cache.egovm_ingest_s, 2),
                        "pass_id": p["pass_id"], "cond": cond,
                    }

                    q = raw[p["qid"]]
                    jobs = [("memory", q["question"], {c: q[f"choice_{c}"] for c in "abcd"},
                             cp["gold"], None)]
                    if cp.get("control_id") and cp["control_id"] in controls:
                        ctl = controls[cp["control_id"]]
                        jobs.append(("control", ctl["question"], ctl["choices"],
                                     ctl["answer"], ctl["lag_s"]))
                    for kind, question, choices, gold, lag in jobs:
                        rid = f"{cp['row_id']}|{cond}|{kind}"
                        if rid not in want:
                            continue
                        tq = time.time()
                        res = sc.query(cache, question, choices, retrieve=not a.no_retrieve)
                        rec = dict(base, record_id=rid, question_kind=kind, gold=gold,
                                   correct=res["pred"] == gold,
                                   frame_load_s=round(t_frames, 2),
                                   score_s=round(time.time() - tq, 2), **res)
                        if lag is not None:
                            rec["control_lag_s"] = lag
                        fout.write(json.dumps(rec) + "\n")
                        n_rec += 1
            except Exception as e:  # noqa: BLE001
                # Drop the pass, not the job. Unlike M6 this loses a whole question, which is
                # why the pass is the unit: a 3 h job holds ~120 of them.
                print(f"  SKIP {p['pass_id']} ({p['n_frames']} frames) at frame {frames_seen}: "
                      f"{type(e).__name__}: {str(e)[:140]}", flush=True)
                n_skip += 1
            finally:
                StreamingScorer.release(cache)
                del cache
                torch.cuda.empty_cache()

            fout.flush()
            n_pass += 1
            if n_pass % 20 == 0:
                os.fsync(fout.fileno())
            if n_pass % 5 == 0:
                el = time.time() - t_start
                seen = sum(x["n_frames"] for x in todo[:n_pass])
                print(f"  {n_pass}/{len(todo)} passes  {n_rec} records  "
                      f"{el/max(seen,1):.3f} s/frame  "
                      f"eta {(n_frames_todo-seen)*el/max(seen,1)/60:.0f} min", flush=True)

    import shutil

    shutil.rmtree(offload_root, ignore_errors=True)
    el = time.time() - t_start
    print(f"X3: {n_pass} passes, {n_rec} records, {n_skip} skipped in {el/60:.1f} min "
          f"-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
