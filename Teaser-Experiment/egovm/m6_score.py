"""M6 — run the Δt sweep.

Resumable by construction: every completed row appends a line to m6_records.jsonl keyed by
`row_id`, and a restart skips ids already present. The `gpu` partition caps at 3 h, so the
sweep is meant to be run as several jobs (or sharded) against the same output file.

Each plan row produces up to two records: the memory question, and — over the *same frames* —
the no-memory control for that checkpoint.

Raw per-(question, checkpoint) records are the primary artifact; the figure is derived.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from PIL import Image

from .config import ARTIFACTS, BIG_ARTIFACTS, DT_GRID_S, MODEL_ID, QA_JSON, SAMPLE_FPS
from .m3_frames import frame_path

RECORDS = BIG_ARTIFACTS / "m6_records.jsonl"


def load_frames(gids: list[int]) -> list[Image.Image]:
    out = []
    for g in gids:
        p = frame_path(g)
        with Image.open(p) as im:
            out.append(im.convert("RGB").copy())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="main", help="comma-separated, or 'all'")
    ap.add_argument("--dts", default="all", help="comma-separated Δt seconds, or 'all'")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n — split the plan across jobs")
    ap.add_argument("--out", default=str(RECORDS))
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument("--time-budget-s", type=int, default=0,
                    help="stop cleanly before the Slurm wall limit")
    ap.add_argument("--order", default="hard-first", choices=["hard-first", "plan", "easy-first"])
    a = ap.parse_args()

    plan = [json.loads(l) for l in (ARTIFACTS / "m4_eval_plan.jsonl").read_text().splitlines()]
    if a.arms != "all":
        keep = set(a.arms.split(","))
        plan = [r for r in plan if r["arm"] in keep]
    if a.dts != "all":
        keep_dt = {int(x) for x in a.dts.split(",")}
        plan = [r for r in plan if r["dt_s"] in keep_dt]

    i, n = (int(x) for x in a.shard.split("/"))
    plan = [r for idx, r in enumerate(plan) if idx % n == i]

    # Longest windows first: they dominate runtime, and finishing them early means a
    # truncated job still leaves a balanced (not Δt-biased) set of results.
    if a.order == "hard-first":
        plan.sort(key=lambda r: -r["n_frames"])
    elif a.order == "easy-first":
        plan.sort(key=lambda r: r["n_frames"])

    out_path = Path(a.out)
    done = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["record_id"])
                except Exception:  # noqa: BLE001 - tolerate a torn final line
                    pass
    todo = [r for r in plan if f"{r['row_id']}|memory" not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"M6: {len(todo)} rows to run ({len(done)} records already present)", flush=True)
    if not todo:
        return

    raw = {str(x["ID"]): x for x in json.loads(QA_JSON.read_text())}
    controls = {}
    cpath = ARTIFACTS / "m5_control_questions.jsonl"
    if cpath.exists() and not a.no_control:
        for line in cpath.read_text().splitlines():
            c = json.loads(line)
            controls[c["control_id"]] = c

    from .scoring import LetterScorer

    t_load = time.time()
    sc = LetterScorer.load()
    print(f"model loaded in {time.time()-t_load:.1f}s", flush=True)

    t_start = time.time()
    n_done = 0
    with out_path.open("a") as f:
        for r in todo:
            if a.time_budget_s and time.time() - t_start > a.time_budget_s:
                print("time budget reached, stopping cleanly", flush=True)
                break
            t0 = time.time()
            frames = load_frames(r["gids"])
            t_load_f = time.time() - t0

            q = raw[r["qid"]]
            base = {k: r[k] for k in
                    ("row_id", "qid", "type", "dt_s", "arm", "n_frames", "visual_tokens",
                     "checkpoint_stream_s", "evidence_max_stream_s", "need_audio")}
            base.update(model=MODEL_ID, sample_fps=SAMPLE_FPS)

            # positions on the 0.2 fps grid relative to the window start, so the "<t seconds>"
            # markers stay truthful for the non-uniformly sampled arms too
            offsets = [g - r["gids"][0] for g in r["gids"]]

            t0 = time.time()
            res = sc.score(frames, q["question"],
                           {c: q[f"choice_{c}"] for c in "abcd"}, frame_offsets=offsets)
            res_t = time.time() - t0
            rec = dict(base, record_id=f"{r['row_id']}|memory", question_kind="memory",
                       gold=r["answer"], correct=res["pred"] == r["answer"],
                       frame_load_s=round(t_load_f, 2), score_s=round(res_t, 2), **res)
            f.write(json.dumps(rec) + "\n")

            ctl = controls.get(f"{r['qid']}|{r['dt_s']}")
            if ctl is not None:
                t0 = time.time()
                cres = sc.score(frames, ctl["question"], ctl["choices"],
                                frame_offsets=offsets)
                rec = dict(base, record_id=f"{r['row_id']}|control", question_kind="control",
                           gold=ctl["answer"], correct=cres["pred"] == ctl["answer"],
                           control_lag_s=ctl["lag_s"], frame_load_s=0.0,
                           score_s=round(time.time() - t0, 2), **cres)
                f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())

            n_done += 1
            if n_done % 20 == 0:
                el = time.time() - t_start
                print(f"  {n_done}/{len(todo)} rows  {el/n_done:.2f}s/row  "
                      f"eta {(len(todo)-n_done)*el/n_done/60:.1f} min", flush=True)

    print(f"M6: wrote {n_done} rows in {(time.time()-t_start)/60:.1f} min -> {out_path}",
          flush=True)


if __name__ == "__main__":
    main()
