"""M6 — run the Δt sweep.

Resumable by construction: every completed row appends a line to m6_records.jsonl keyed by
`row_id`, and a restart skips ids already present. The `gpu` partition caps at 3 h, so the
sweep is meant to be run as several jobs (or sharded) against the same output file.

Each plan row produces up to two records: the memory question, and — over the *same frames* —
the no-memory control for that checkpoint (on the arms listed in CONTROL_ARMS). A row is
re-visited if *either* record is missing, so widening CONTROL_ARMS backfills controls onto arms
whose memory question already ran.

Raw per-(question, checkpoint) records are the primary artifact; the figure is derived.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from PIL import Image

from .config import (
    ABSTAIN_LETTER, ABSTAIN_TEXT, ARTIFACTS, BIG_ARTIFACTS, CONTROL_ARMS, DT_GRID_S, MODEL_ID,
    OPTION_LETTERS, QA_JSON, SAMPLE_FPS,
)
from .m3_frames import frame_path

RECORDS = BIG_ARTIFACTS / "m6_records.jsonl"


def score_resilient(sc, frames, question, choices, offsets, retries: int = 2):
    """Score, freeing the CUDA cache and retrying once on an out-of-memory failure."""
    import torch

    oom = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError
    for attempt in range(retries):
        try:
            return sc.score(frames, question, choices, frame_offsets=offsets)
        except oom:
            # torch.OutOfMemoryError does not subclass RuntimeError in torch 2.13, so it has
            # to be named explicitly — catching RuntimeError let it escape and killed two
            # shard jobs outright.
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if attempt == retries - 1:
                raise
            time.sleep(3.0)
    raise RuntimeError("unreachable")


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
    ap.add_argument("--abstain", action="store_true",
                    help="S4: offer a fifth option ('not enough information in the video') and "
                         "score the softmax over five letters. Implies --no-control and must "
                         "write to its own --out: a five-way softmax is not comparable with the "
                         "four-way confidences the headline is built on.")
    ap.add_argument("--time-budget-s", type=int, default=0,
                    help="stop cleanly before the Slurm wall limit")
    ap.add_argument("--order", default="hard-first", choices=["hard-first", "plan", "easy-first"])
    a = ap.parse_args()

    # abstention rows are a separate record kind so they never merge into the four-way
    # confidences: the softmax denominator is different and the numbers are not comparable
    mem_kind = "memory_abstain" if a.abstain else "memory"
    if a.abstain and Path(a.out) == RECORDS:
        raise SystemExit("--abstain needs its own --out (see the flag's help)")

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
    # Resume against *every* record file in the directory, not just this job's own output:
    # the sweep runs as many sharded jobs writing separate files, and a job that only reads
    # its own file re-runs rows another shard already finished.
    done = set()
    for p in sorted(out_path.parent.glob("m6_records*.jsonl")):
        with p.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["record_id"])
                except Exception:  # noqa: BLE001 - tolerate a torn final line
                    pass
    raw = {str(x["ID"]): x for x in json.loads(QA_JSON.read_text())}
    controls = {}
    cpath = ARTIFACTS / "m5_control_questions.jsonl"
    if cpath.exists() and not (a.no_control or a.abstain):
        for line in cpath.read_text().splitlines():
            c = json.loads(line)
            controls[c["control_id"]] = c

    def control_for(r):
        return controls.get(f"{r['qid']}|{r['dt_s']}") if r["arm"] in CONTROL_ARMS else None

    # A row is outstanding if *either* of its records is missing. Checking only the memory
    # record would silently skip rows whose memory question ran under an earlier config but
    # whose control did not — which is exactly the state S2 leaves the token-matched arms in.
    def wanted(r) -> tuple[bool, bool]:
        return (f"{r['row_id']}|{mem_kind}" not in done,
                control_for(r) is not None and f"{r['row_id']}|control" not in done)

    todo = [r for r in plan if any(wanted(r))]
    if a.limit:
        todo = todo[: a.limit]
    n_ctl_only = sum(1 for r in todo if not wanted(r)[0])
    print(f"M6: {len(todo)} rows to run ({len(done)} records already present"
          f"{f', {n_ctl_only} needing only the control' if n_ctl_only else ''})", flush=True)
    if not todo:
        return

    from .scoring import LetterScorer

    t_load = time.time()
    letters = (list(OPTION_LETTERS) + [ABSTAIN_LETTER]) if a.abstain else list(OPTION_LETTERS)
    sc = LetterScorer.load(letters=letters)
    print(f"model loaded in {time.time()-t_load:.1f}s", flush=True)

    t_start = time.time()
    n_done = 0
    n_skipped = 0
    t_write = 0.0
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

            want_mem, want_ctl = wanted(r)
            choices = {c: q[f"choice_{c}"] for c in "abcd"}
            if a.abstain:
                choices[ABSTAIN_LETTER.lower()] = ABSTAIN_TEXT

            t0 = time.time()
            try:
                res = score_resilient(sc, frames, q["question"], choices, offsets) \
                    if want_mem else None
            except Exception as e:  # noqa: BLE001
                # A 60 min window is ~37k visual tokens and the vision tower allocates it in
                # one block; GPUs here are shared, so the memory actually available varies
                # between rows. Skip the row rather than killing a multi-hour unattended job —
                # it stays in the plan and a later resubmission picks it up.
                print(f"  SKIP {r['row_id']} ({r['n_frames']} frames): "
                      f"{type(e).__name__}: {str(e)[:110]}", flush=True)
                n_skipped += 1
                import torch

                torch.cuda.empty_cache()
                continue
            res_t = time.time() - t0
            if res is not None:
                rec = dict(base, record_id=f"{r['row_id']}|{mem_kind}", question_kind=mem_kind,
                           gold=r["answer"], correct=res["pred"] == r["answer"],
                           frame_load_s=round(t_load_f, 2), score_s=round(res_t, 2), **res)
                if a.abstain:
                    rec["abstained"] = res["pred"] == ABSTAIN_LETTER
                f.write(json.dumps(rec) + "\n")

            # The control rides on `main` (the sweep it was designed for) and on the two
            # token-matched arms that suit a present-tense question, which is what shows
            # whether the control's *own* decline along Δt is context length rather than
            # anything about the checkpoint. Control ids are keyed `qid|dt_s`, so the same
            # control MCQ rides every arm at a given checkpoint and the comparison is paired.
            ctl = control_for(r) if want_ctl else None
            if ctl is not None:
                t0 = time.time()
                cres = score_resilient(sc, frames, ctl["question"], ctl["choices"], offsets)
                rec = dict(base, record_id=f"{r['row_id']}|control", question_kind="control",
                           gold=ctl["answer"], correct=cres["pred"] == ctl["answer"],
                           control_lag_s=ctl["lag_s"], frame_load_s=0.0,
                           score_s=round(time.time() - t0, 2), **cres)
                f.write(json.dumps(rec) + "\n")
            # flush every row so partial results are readable, but fsync only occasionally:
            # /home is a shared NFS mount and a per-row fsync was costing ~25 s against 2.3 s
            # of actual GPU work. flush() alone survives job kill and wall-clock timeout;
            # only a node crash could lose the last few rows, which are simply re-run.
            t0 = time.time()
            f.flush()
            n_done += 1
            if n_done % 100 == 0:
                os.fsync(f.fileno())
            t_write += time.time() - t0

            if n_done % 20 == 0:
                el = time.time() - t_start
                print(f"  {n_done}/{len(todo)} rows  {el/n_done:.2f}s/row  "
                      f"(write {t_write/n_done:.2f}s/row)  "
                      f"eta {(len(todo)-n_done)*el/n_done/60:.1f} min", flush=True)

    print(f"M6: wrote {n_done} rows ({n_skipped} skipped on OOM) in "
          f"{(time.time()-t_start)/60:.1f} min -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
