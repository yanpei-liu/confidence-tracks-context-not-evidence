# X2 — streaming schedule

Derived from `m4_eval_plan.jsonl` over 373 questions. Cost model: **100 visual tokens/frame** (image path, `(res//32)**2`) — the plan file's `visual_tokens` column is the video path's and is stale by 2x here.

| pass kind | passes | frames to ingest | scored rows | est. GPU-h @0.10 s/frame |
| --- | --- | --- | --- | --- |
| `main` | 373 | 273,622 | 3,730 | 7.6 |
| `budget` | 746 | 547,244 | 3,730 | 15.2 |
| `prior` | 1,865 | 522,146 | 1,865 | 14.5 |
| `tokmatch` | 746 | 144,404 | 746 | 4.0 |
| `nots` | 373 | 273,622 | 1,865 | 7.6 |
| **total** | **4,103** | **1,761,038** | | **48.9** |

## Assertions that passed

- **Nesting.** For every `main`/`budget`/`nots` pass, `gids[:n_frames]` equals the M4 row's `gids` element for element at all 5 Δt bins. This is what lets one ingestion serve five checkpoints.
- **`prior` does not nest** (1 of 1492 adjacent pairs nest), which is why each prior bin is its own pass.
- **No duplicated work**: 10,071 distinct (row_id, budget, timestamps) triples, each produced by exactly one pass.
- **Controls attached** on 1,810 of 1,865 `main` checkpoints (97.1 %) — every one M5 produced. The 55 without a control are checkpoints for which M5 found no usable DenseCaption MCQ; the completed sweep carries the same gap.

## Ordering and sharding

Shard on `qid` (a modulo split over passes would scatter one question's Δt series across workers, each re-ingesting the stream). Within a pass, checkpoints are already in ascending `dt_s` and must be consumed in that order — the cache only moves forward.
