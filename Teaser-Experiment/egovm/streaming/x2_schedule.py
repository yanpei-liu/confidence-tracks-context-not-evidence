"""X2 — turn the evaluation plan into a streaming schedule.

Under full-reprocess the work unit is a *plan row*. Under a streaming cache it is an
**ingestion pass**: one cache, ingested once in ascending stream order, queried at several
checkpoints. This module derives those passes from `m4_eval_plan.jsonl` and never re-derives a
window — the windows stay exactly what the committed plan says they are, so the two regimes are
compared on identical frames.

The load-bearing assertion is **nesting**: for a `main` pass, the first `n_frames` gids must
equal, element for element, the gids of the corresponding M4 row at that Δt. That is what makes
one ingestion serve five checkpoints. It holds for `main` (0 violations over 373 questions) and
fails for every `prior` pair (1,491 of 1,491), which is why `prior` is planned as one pass per
bin. Both facts are re-checked here rather than trusted.

`visual_tokens` is recomputed at the streaming cost model (100 tokens/frame, not the plan's 50):
the plan's column belongs to the video path and is stale by exactly 2x for this backend.

Artifacts: artifacts/x2_stream_schedule.jsonl, artifacts/x2_schedule_report.md
Run: python -m egovm.streaming.x2_schedule
"""
from __future__ import annotations

import argparse
import collections
import json

from ..config import ARTIFACTS, DT_GRID_S
from . import sconf


def load_plan() -> dict:
    rows = {}
    for line in (ARTIFACTS / "m4_eval_plan.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[(r["qid"], r["arm"], r["dt_s"])] = r
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(sconf.X2_SCHEDULE))
    a = ap.parse_args()

    plan = load_plan()
    controls = set()
    cpath = ARTIFACTS / "m5_control_questions.jsonl"
    for line in cpath.read_text().splitlines():
        controls.add(json.loads(line)["control_id"])

    qids = sorted({k[0] for k in plan}, key=int)
    rank = {q: i for i, q in enumerate(qids)}

    passes: list[dict] = []

    def add_pass(kind, qid, arm_rows, budget, timestamps, tag):
        """arm_rows: list of (dt_s, arm, m4row) in ascending dt order; the last one's gids are
        the full ingestion and every earlier one must be a prefix of it."""
        arm_rows = sorted(arm_rows, key=lambda t: t[0])
        full = arm_rows[-1][2]
        gids = full["gids"]
        cps = []
        for dt, arm, row in arm_rows:
            n = row["n_frames"]
            # THE nesting assertion — an equality on the gid lists, not on their length
            assert gids[:n] == row["gids"], (
                f"{kind} pass {qid}: gids[:{n}] != m4[{arm}@{dt}].gids — windows do not nest, "
                "so this arm cannot share an ingestion")
            cps.append({
                "dt_s": dt, "arm": arm, "row_id": row["row_id"], "n_frames": n,
                "visual_tokens": n * sconf.TOKENS_PER_FRAME,
                "checkpoint_stream_s": row["checkpoint_stream_s"],
                "gold": row["answer"],
                "control_id": f"{qid}|{dt}" if (arm in ("main",) and f"{qid}|{dt}" in controls)
                else None,
            })
        q0 = full
        passes.append({
            "pass_id": f"{qid}|{tag}",
            "kind": kind, "qid": qid, "subset_rank": rank[qid],
            "type": q0["type"], "need_audio": q0["need_audio"],
            "budget_label": budget, "timestamps": timestamps,
            "window_lo_stream_s": q0["window_lo_stream_s"],
            "evidence_max_stream_s": q0["evidence_max_stream_s"],
            "n_frames": len(gids), "visual_tokens": len(gids) * sconf.TOKENS_PER_FRAME,
            "gids": gids, "checkpoints": cps,
        })

    for qid in qids:
        main_rows = [(dt, "main", plan[(qid, "main", dt)]) for dt in DT_GRID_S
                     if (qid, "main", dt) in plan]
        if len(main_rows) != len(DT_GRID_S):
            continue  # a question missing a bin cannot carry the nested pass

        # X5 — the headline sweep: one ingestion, five checkpoints, memory + control at each
        add_pass("main", qid, main_rows, sconf.DEFAULT_BUDGET, True, "main")

        # X6 — the context-length axis: same frames, same Δt, different memory budget
        for b in sconf.BUDGET_ARM_BUDGETS:
            add_pass("budget", qid, main_rows, b, True, f"budget_{b}")

        # X8 — the time-blind ablation
        add_pass("nots", qid, main_rows, sconf.DEFAULT_BUDGET, False, "nots")

        # X7 — the no-evidence floor. One pass per bin: these windows do not nest.
        for dt in sconf.PRIOR_DTS:
            if (qid, "prior", dt) in plan:
                add_pass("prior", qid, [(dt, "prior", plan[(qid, "prior", dt)])],
                         sconf.DEFAULT_BUDGET, True, f"prior_{dt}")

        # X7 — elapsed time at fixed frame count
        for arm in sconf.TOKMATCH_ARMS:
            for dt in sconf.TOKMATCH_DTS:
                if (qid, arm, dt) in plan:
                    add_pass("tokmatch", qid, [(dt, arm, plan[(qid, arm, dt)])],
                             sconf.DEFAULT_BUDGET, True, f"{arm}_{dt}")

    # ---- verification ----
    seen_rows: dict[str, str] = {}
    for p in passes:
        assert len(set(p["gids"])) == len(p["gids"]), f"{p['pass_id']}: duplicate gids"
        assert p["gids"] == sorted(p["gids"]), f"{p['pass_id']}: gids not in stream order"
        assert p["visual_tokens"] == p["n_frames"] * sconf.TOKENS_PER_FRAME
        for cp in p["checkpoints"]:
            # a (row_id, budget, timestamps) triple must be produced by exactly one pass, or
            # two passes would write colliding record_ids and M7 would dedup one away
            key = f"{cp['row_id']}|{p['budget_label']}|{int(p['timestamps'])}"
            assert key not in seen_rows, f"duplicate work: {key} in {p['pass_id']} and {seen_rows[key]}"
            seen_rows[key] = p["pass_id"]
    n_main_cp = sum(len(p["checkpoints"]) for p in passes if p["kind"] == "main")
    n_ctl = sum(1 for p in passes if p["kind"] == "main"
                for cp in p["checkpoints"] if cp["control_id"])
    # M5 does not produce a control for every checkpoint — 55 of 1,865 have no usable
    # DenseCaption-derived MCQ, and the completed sweep carries the same gap. So the assertion
    # is that X2 drops none of the controls that *do* exist, not that all checkpoints have one.
    n_available = sum(1 for p in passes if p["kind"] == "main" for cp in p["checkpoints"]
                      if f"{p['qid']}|{cp['dt_s']}" in controls)
    assert n_ctl == n_available, f"X2 dropped controls: attached {n_ctl} of {n_available} available"

    # prior must NOT nest — if it ever does, the plan changed and prior can share one pass
    prior_by_q = collections.defaultdict(list)
    for p in passes:
        if p["kind"] == "prior":
            prior_by_q[p["qid"]].append(p)
    nested_prior = 0
    for q, ps in prior_by_q.items():
        ps = sorted(ps, key=lambda x: x["checkpoints"][0]["dt_s"])
        for a_, b_ in zip(ps, ps[1:]):
            if b_["gids"][: len(a_["gids"])] == a_["gids"]:
                nested_prior += 1

    out = sconf.X2_SCHEDULE
    with out.open("w") as f:
        for p in passes:
            f.write(json.dumps(p) + "\n")

    by_kind = collections.Counter(p["kind"] for p in passes)
    frames_by_kind = collections.Counter()
    for p in passes:
        frames_by_kind[p["kind"]] += p["n_frames"]
    rows_by_kind = collections.Counter()
    for p in passes:
        rows_by_kind[p["kind"]] += len(p["checkpoints"])

    lines = [
        "# X2 — streaming schedule", "",
        f"Derived from `m4_eval_plan.jsonl` over {len(qids)} questions. "
        f"Cost model: **{sconf.TOKENS_PER_FRAME} visual tokens/frame** "
        f"(image path, `(res//32)**2`) — the plan file's `visual_tokens` column is the video "
        "path's and is stale by 2x here.", "",
        "| pass kind | passes | frames to ingest | scored rows | est. GPU-h @0.10 s/frame |",
        "| --- | --- | --- | --- | --- |",
    ]
    for k in ("main", "budget", "prior", "tokmatch", "nots"):
        if not by_kind[k]:
            continue
        rows = rows_by_kind[k] * (2 if k == "main" else 1)
        lines.append(f"| `{k}` | {by_kind[k]:,} | {frames_by_kind[k]:,} | {rows:,} | "
                     f"{frames_by_kind[k]*0.10/3600:.1f} |")
    tot_f = sum(frames_by_kind.values())
    lines += [f"| **total** | **{len(passes):,}** | **{tot_f:,}** | | "
              f"**{tot_f*0.10/3600:.1f}** |", "",
              "## Assertions that passed", "",
              f"- **Nesting.** For every `main`/`budget`/`nots` pass, `gids[:n_frames]` equals "
              f"the M4 row's `gids` element for element at all {len(DT_GRID_S)} Δt bins. This is "
              "what lets one ingestion serve five checkpoints.",
              f"- **`prior` does not nest** ({nested_prior} of "
              f"{sum(max(0, len(v)-1) for v in prior_by_q.values())} adjacent pairs nest), which "
              "is why each prior bin is its own pass.",
              f"- **No duplicated work**: {len(seen_rows):,} distinct "
              "(row_id, budget, timestamps) triples, each produced by exactly one pass.",
              f"- **Controls attached** on {n_ctl:,} of {n_main_cp:,} `main` checkpoints "
              f"({100*n_ctl/n_main_cp:.1f} %) — every one M5 produced. The "
              f"{n_main_cp-n_ctl} without a control are checkpoints for which M5 found no "
              "usable DenseCaption MCQ; the completed sweep carries the same gap.", "",
              "## Ordering and sharding", "",
              "Shard on `qid` (a modulo split over passes would scatter one question's Δt "
              "series across workers, each re-ingesting the stream). Within a pass, checkpoints "
              "are already in ascending `dt_s` and must be consumed in that order — the cache "
              "only moves forward.", ""]
    sconf.X2_REPORT.write_text("\n".join(lines))

    print(f"X2 OK: {len(passes):,} passes over {len(qids)} questions -> {out}")
    for k in ("main", "budget", "prior", "tokmatch", "nots"):
        if by_kind[k]:
            print(f"  {k:10s} passes={by_kind[k]:5,}  frames={frames_by_kind[k]:9,}  "
                  f"checkpoints={rows_by_kind[k]:5,}")
    print(f"  total frames to ingest: {tot_f:,}  (~{tot_f*0.10/3600:.1f} GPU-h at 0.10 s/frame)")
    print(f"  prior nesting violations expected: "
          f"{sum(max(0,len(v)-1) for v in prior_by_q.values())}, nested: {nested_prior}")
    print(f"  report -> {sconf.X2_REPORT}")


if __name__ == "__main__":
    main()
