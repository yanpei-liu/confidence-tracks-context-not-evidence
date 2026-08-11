"""M1 — parse and filter the EgoLifeQA A1 questions.

Artifacts:
  artifacts/m1_questions.jsonl    one row per question (all 500, with a keep flag)
  artifacts/m1_schema_report.md   what the released file actually contains

Question text is *not* written to the artifact (EgoLife is non-redistributable); only IDs,
timestamps, category and answer-letter metadata. The scorer re-reads the raw file for text.
"""
from __future__ import annotations

import collections
import json
import statistics

from .config import ARTIFACTS, DROP_TYPES, KEEP_TYPES, QA_JSON, SCREEN_TYPES
from .timeparse import encoding_of, parse_time_field


def build() -> list[dict]:
    raw = json.loads(QA_JSON.read_text())
    rows = []
    for x in raw:
        ev = parse_time_field(x["target_time"])
        q = parse_time_field(x["query_time"])
        assert len(q) == 1, f"query_time with multiple points: {x['ID']}"
        typ = x["type"]
        rows.append(
            {
                "qid": str(x["ID"]),
                "type": typ,
                "keep": typ in KEEP_TYPES,
                "screen": typ in SCREEN_TYPES,
                "drop": typ in DROP_TYPES,
                "answer": x["answer"].strip().upper(),
                "need_audio": bool(x["need_audio"]),
                "need_name": bool(x["need_name"]),
                "evidence_abs_s": ev,
                "evidence_min_abs_s": ev[0],
                "evidence_max_abs_s": ev[-1],
                "n_evidence": len(ev),
                "query_abs_s": q[0],
                "native_gap_s": q[0] - ev[-1],
                "target_time_encoding": encoding_of(x["target_time"]),
                "target_day": x["target_time"]["date"],
                "query_day": x["query_time"]["date"],
            }
        )
    return rows


def report(rows: list[dict]) -> str:
    n = len(rows)
    types = collections.Counter(r["type"] for r in rows)
    encs = collections.Counter(r["target_time_encoding"] for r in rows)
    nev = collections.Counter(r["n_evidence"] for r in rows)
    kept = [r for r in rows if r["keep"]]
    gaps = sorted(r["native_gap_s"] for r in rows)
    ans = collections.Counter(r["answer"] for r in rows)

    def tbl(counter, head):
        lines = [f"| {head} | n |", "| --- | --- |"]
        lines += [f"| {k} | {v} |" for k, v in sorted(counter.items(), key=lambda kv: -kv[1])]
        return "\n".join(lines)

    return f"""# M1 — EgoLifeQA A1 schema report

Source: `EgoLifeQA/EgoLifeQA_A1_JAKE.json` from HF `lmms-lab/EgoLife`.
Items: **{n}** (matches the 500 released MCQ for participant A1/Jake).

## Correction to the planning documents

`[v2-a]` states EgoLifeQA has *"no confirmed populated explicit evidence-interval field"* and
routes exact Δt derivation to MultiHop-EgoQA. The released file **does** carry a populated
`target_time` on every item — the evidence timestamp(s). All {n}/{n} parse. Δt can therefore be
derived exactly on EgoLifeQA, and MultiHop-EgoQA is demoted to an optional robustness check.

`target_time` is a *set of instants*, not an interval, so the "no precise evidence-interval"
caveat still holds in the strict sense — but instants are what the Δt sweep needs.

## Top-level keys

`ID, query_time, type, type_chinese, need_audio, need_name, last_time, trigger, trigger_chinese,
question, question_chinese, choice_a..choice_d (+_chinese), answer, target_time, keywords,
reason, reason_chinese` — present on all {n} items.

## Categories

{tbl(types, "type")}

Matches the `[v2-a]` split exactly (EntityLog 125, EventRecall 126, RelationMap 125,
HabitInsight 61, TaskMaster 63). Mitigation 1 keeps {len(kept)} questions
({", ".join(KEEP_TYPES)}); {SCREEN_TYPES[0]} is scored but held out of the headline set;
{DROP_TYPES[0]} is dropped.

## `target_time` encodings

{tbl(encs, "encoding")}

Evidence points per question:

{tbl(nev, "n_evidence")}

One item encodes its date as lowercase `Day1`; the parser is case-insensitive.

## Native evidence→query gap

min **{gaps[0]/60:.1f} min** (consistent with the paper's "at least 5 minutes prior"),
p10 {gaps[int(.1*n)]/60:.1f} min, median {statistics.median(gaps)/3600:.2f} h,
p90 {gaps[int(.9*n)]/3600:.2f} h, max {gaps[-1]/3600:.2f} h. No item has a non-positive gap.

The experiment does **not** use these native gaps — it places its own checkpoints at controlled
Δt after `max(target_time)` — but they confirm the field is semantically the evidence time.

## Other flags

- `need_audio` true on {sum(r['need_audio'] for r in rows)} items ({sum(r['need_audio'] for r in kept)} of the {len(kept)} kept). The
  model is vision-only, so these are flagged and results reported both with and without them.
- `need_name` true on {sum(r['need_name'] for r in rows)} items.
- Answer-letter balance: {dict(sorted(ans.items()))} — near-uniform, so the 25 % chance floor is clean.
"""


def main() -> None:
    rows = build()
    out = ARTIFACTS / "m1_questions.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (ARTIFACTS / "m1_schema_report.md").write_text(report(rows))

    # verification
    assert len(rows) == 500, len(rows)
    counts = collections.Counter(r["type"] for r in rows)
    assert counts == {
        "EventRecall": 126, "EntityLog": 125, "RelationMap": 125,
        "TaskMaster": 63, "HabitInsight": 61,
    }, counts
    assert sum(r["keep"] for r in rows) == 376
    assert all(r["native_gap_s"] > 0 for r in rows)
    assert all(r["answer"] in "ABCD" for r in rows)
    print(f"M1 OK: {len(rows)} questions, {sum(r['keep'] for r in rows)} kept -> {out}")


if __name__ == "__main__":
    main()
