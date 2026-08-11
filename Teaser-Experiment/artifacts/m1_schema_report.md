# M1 — EgoLifeQA A1 schema report

Source: `EgoLifeQA/EgoLifeQA_A1_JAKE.json` from HF `lmms-lab/EgoLife`.
Items: **500** (matches the 500 released MCQ for participant A1/Jake).

## Correction to the planning documents

`[v2-a]` states EgoLifeQA has *"no confirmed populated explicit evidence-interval field"* and
routes exact Δt derivation to MultiHop-EgoQA. The released file **does** carry a populated
`target_time` on every item — the evidence timestamp(s). All 500/500 parse. Δt can therefore be
derived exactly on EgoLifeQA, and MultiHop-EgoQA is demoted to an optional robustness check.

`target_time` is a *set of instants*, not an interval, so the "no precise evidence-interval"
caveat still holds in the strict sense — but instants are what the Δt sweep needs.

## Top-level keys

`ID, query_time, type, type_chinese, need_audio, need_name, last_time, trigger, trigger_chinese,
question, question_chinese, choice_a..choice_d (+_chinese), answer, target_time, keywords,
reason, reason_chinese` — present on all 500 items.

## Categories

| type | n |
| --- | --- |
| EventRecall | 126 |
| EntityLog | 125 |
| RelationMap | 125 |
| TaskMaster | 63 |
| HabitInsight | 61 |

Matches the `[v2-a]` split exactly (EntityLog 125, EventRecall 126, RelationMap 125,
HabitInsight 61, TaskMaster 63). Mitigation 1 keeps 376 questions
(EntityLog, EventRecall, RelationMap); HabitInsight is scored but held out of the headline set;
TaskMaster is dropped.

## `target_time` encodings

| encoding | n |
| --- | --- |
| time_scalar | 451 |
| time_list | 36 |
| time_concat_2 | 10 |
| time_concat_3 | 3 |

Evidence points per question:

| n_evidence | n |
| --- | --- |
| 1 | 451 |
| 2 | 41 |
| 3 | 6 |
| 6 | 1 |
| 7 | 1 |

One item encodes its date as lowercase `Day1`; the parser is case-insensitive.

## Native evidence→query gap

min **4.9 min** (consistent with the paper's "at least 5 minutes prior"),
p10 14.5 min, median 5.79 h,
p90 47.45 h, max 138.54 h. No item has a non-positive gap.

The experiment does **not** use these native gaps — it places its own checkpoints at controlled
Δt after `max(target_time)` — but they confirm the field is semantically the evidence time.

## Other flags

- `need_audio` true on 202 items (110 of the 376 kept). The
  model is vision-only, so these are flagged and results reported both with and without them.
- `need_name` true on 318 items.
- Answer-letter balance: {'A': 118, 'B': 126, 'C': 128, 'D': 128} — near-uniform, so the 25 % chance floor is clean.
