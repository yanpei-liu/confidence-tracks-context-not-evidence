# X9 — the streaming sweep, and how it compares with full-reprocess

Generated from `x9_regime_comparison.json`; every number below is read from it, so this file cannot drift from the metrics. Δ are percentage points, paired within-question, cluster-bootstrapped over questions (5,000 draws, seed 0).

16,565 streaming records (large=3,675, nots=3,675, small=3,675, stream=5,540) against 5,562 committed full-reprocess records.

## Verdict

**Outcome 3 — the dissociation replicates.** Confidence does not track whether the evidence is present, while accuracy does.

## The manipulations, under streaming

| manipulation | n | Δaccuracy | Δconfidence |
| --- | --- | --- | --- |
| `elapsed_time_fixed_frames` | 373 | -1.7 [-5.1, +1.6] | -0.1 [-1.7, +1.4] |
| `context_length_large_vs_default` | 373 | +1.8 [-0.4, +4.0] | -5.4 [-6.5, -4.4] |
| `context_length_default_vs_small` | 373 | -2.8 [-5.9, +0.1] | -7.4 [-8.9, -5.9] |
| `evidence_vs_none` | 373 | +8.1 [+2.9, +13.1] | +0.5 [-1.5, +2.6] |
| `evidence_thinned` | 373 | -1.9 [-4.6, +0.8] | -1.3 [-2.5, -0.1] |
| `context_length_4x_tokens` | 373 | -2.4 [-5.9, +1.1] | -6.7 [-8.4, -5.0] |
| `timestamps_on_vs_off` | 373 | -0.6 [-1.7, +0.5] | -0.7 [-1.2, -0.2] |
| `control_vs_memory` | 373 | +18.4 [+13.5, +23.4] | +12.6 [+10.8, +14.4] |
| `elapsed_time_raw` | 373 | +8.1 [+3.5, +12.6] | +12.1 [+10.0, +14.2] |

## Cross-regime: (streaming Δ) − (full-reprocess Δ)

Contrasts, not levels. The teaser's levels sit in an 11.5 pp band above the no-evidence floor and are weak by construction; what survived there was the set of paired contrasts, so those are what is differenced here.

| manipulation | n | Δaccuracy difference | Δconfidence difference |
| --- | --- | --- | --- |
| `elapsed_time_fixed_frames` | 373 | -1.9 [-5.9, +1.9] | +3.5 [+1.7, +5.4] |
| `context_length_4x_tokens` | 373 | +0.8 [-4.0, +5.4] | -6.1 [-8.1, -4.1] |
| `evidence_vs_none` | 373 | -1.6 [-6.7, +3.5] | -2.8 [-5.1, -0.4] |
| `evidence_thinned` | 373 | +0.8 [-3.2, +4.8] | -0.3 [-2.0, +1.3] |
| `control_vs_memory` | 373 | +7.5 [+4.1, +11.0] | -1.7 [-3.3, -0.1] |
| `elapsed_time_raw` | 373 | +1.7 [-3.8, +7.2] | +2.5 [+0.0, +4.9] |

## The time-blind test, bin by bin

X1b found the injected clock is legible only while it is still in the cache. The prediction (EXTENDING.md §4a) is that `main` and `nots` are indistinguishable at Δt ≥ 15 min, where the cache is saturated, and may differ only at 1 and 5 min.

| Δt | n | Δaccuracy (on − off) | Δconfidence (on − off) |
| --- | --- | --- | --- |
| 1 min | 373 | +0.5 [-1.9, +2.9] | -0.2 [-1.1, +0.7] |
| 5 min | 373 | -1.3 [-3.5, +0.5] | +1.5 [+0.6, +2.5] |
| 15 min | 373 | +0.5 [-1.9, +2.9] | -1.6 [-2.7, -0.5] |
| 30 min | 373 | -1.8 [-4.6, +0.8] | -1.2 [-2.1, -0.2] |
| 60 min | 373 | -0.8 [-3.2, +1.6] | -2.2 [-3.1, -1.3] |

## The context-length axis (`large` − `small`)

Recreated as a memory-budget arm because a bounded cache pins the token count: context length is no longer ours to manipulate by choosing frames. `retrieve_top_k` tracks the budget, without which the retriever re-equalises every condition at query time.

| Δt | n | live tokens (large / small) | accuracy | confidence |
| --- | --- | --- | --- | --- |
| 1 min | 373 | 3,273 / 1,900 | +3.0 [+0.0, +5.9] | -2.0 [-3.4, -0.6] |
| 5 min | 373 | 8,073 / 1,934 | -0.0 [-3.8, +3.8] | -4.2 [-5.9, -2.4] |
| 15 min | 373 | 20,072 / 1,939 | +1.6 [-2.1, +5.4] | -8.1 [-10.0, -6.2] |
| 30 min | 373 | 33,684 / 1,940 | -0.0 [-4.3, +4.3] | -10.5 [-12.4, -8.5] |
| 60 min | 373 | 33,684 / 1,940 | -1.9 [-6.4, +2.4] | -15.2 [-17.5, -13.1] |

## Per-cell levels

| cell | n | accuracy | confidence | live tokens | presented tokens |
| --- | --- | --- | --- | --- | --- |
| `control/main/1800` | 361 | 0.537 | 0.888 | 9,108 | 37,362 |
| `control/main/300` | 366 | 0.574 | 0.929 | 7,945 | 7,360 |
| `control/main/3600` | 355 | 0.504 | 0.866 | 9,108 | 73,365 |
| `control/main/60` | 366 | 0.604 | 0.938 | 3,228 | 2,560 |
| `control/main/900` | 362 | 0.566 | 0.918 | 9,108 | 19,362 |
| `control/main@large/1800` | 361 | 0.524 | 0.874 | 33,684 | 37,362 |
| `control/main@large/300` | 366 | 0.577 | 0.928 | 8,076 | 7,360 |
| `control/main@large/3600` | 355 | 0.501 | 0.834 | 33,684 | 73,365 |
| `control/main@large/60` | 366 | 0.604 | 0.939 | 3,276 | 2,560 |
| `control/main@large/900` | 362 | 0.550 | 0.912 | 20,077 | 19,362 |
| `control/main@nots/1800` | 361 | 0.543 | 0.900 | 9,108 | 37,362 |
| `control/main@nots/300` | 366 | 0.566 | 0.929 | 7,945 | 7,360 |
| `control/main@nots/3600` | 355 | 0.510 | 0.888 | 9,108 | 73,365 |
| `control/main@nots/60` | 366 | 0.609 | 0.935 | 3,228 | 2,560 |
| `control/main@nots/900` | 362 | 0.550 | 0.929 | 9,108 | 19,362 |
| `control/main@small/1800` | 361 | 0.535 | 0.909 | 1,940 | 37,362 |
| `control/main@small/300` | 366 | 0.508 | 0.937 | 1,933 | 7,360 |
| `control/main@small/3600` | 355 | 0.454 | 0.902 | 1,940 | 73,365 |
| `control/main@small/60` | 366 | 0.577 | 0.948 | 1,899 | 2,560 |
| `control/main@small/900` | 362 | 0.481 | 0.923 | 1,938 | 19,362 |
| `memory/main/1800` | 373 | 0.349 | 0.739 | 9,108 | 37,357 |
| `memory/main/300` | 373 | 0.383 | 0.816 | 7,945 | 7,357 |
| `memory/main/3600` | 373 | 0.340 | 0.722 | 9,108 | 73,357 |
| `memory/main/60` | 373 | 0.421 | 0.843 | 3,227 | 2,557 |
| `memory/main/900` | 373 | 0.381 | 0.790 | 9,108 | 19,357 |
| `memory/main@large/1800` | 373 | 0.367 | 0.691 | 33,684 | 37,357 |
| `memory/main@large/300` | 373 | 0.383 | 0.815 | 8,073 | 7,357 |
| `memory/main@large/3600` | 373 | 0.359 | 0.661 | 33,684 | 73,357 |
| `memory/main@large/60` | 373 | 0.421 | 0.843 | 3,273 | 2,557 |
| `memory/main@large/900` | 373 | 0.397 | 0.758 | 20,072 | 19,357 |
| `memory/main@nots/1800` | 373 | 0.367 | 0.750 | 9,108 | 37,357 |
| `memory/main@nots/300` | 373 | 0.397 | 0.801 | 7,945 | 7,357 |
| `memory/main@nots/3600` | 373 | 0.349 | 0.744 | 9,108 | 73,357 |
| `memory/main@nots/60` | 373 | 0.416 | 0.845 | 3,227 | 2,557 |
| `memory/main@nots/900` | 373 | 0.375 | 0.806 | 9,108 | 19,357 |
| `memory/main@small/1800` | 373 | 0.367 | 0.796 | 1,940 | 37,357 |
| `memory/main@small/300` | 373 | 0.383 | 0.858 | 1,934 | 7,357 |
| `memory/main@small/3600` | 373 | 0.378 | 0.814 | 1,940 | 73,357 |
| `memory/main@small/60` | 373 | 0.391 | 0.863 | 1,900 | 2,557 |
| `memory/main@small/900` | 373 | 0.381 | 0.840 | 1,939 | 19,357 |
| `memory/prior/300` | 373 | 0.303 | 0.811 | 7,945 | 7,357 |
| `memory/prior/60` | 373 | 0.308 | 0.824 | 3,227 | 2,557 |
| `memory/prior/900` | 373 | 0.332 | 0.774 | 9,108 | 19,357 |
| `memory/tokmatch_evkeep/3600` | 373 | 0.365 | 0.789 | 9,108 | 19,357 |
| `memory/tokmatch_uniform/3600` | 373 | 0.346 | 0.776 | 9,108 | 19,357 |

## Caveats that travel with these numbers

- **Elapsed time is not an input in this regime once the cache saturates** (EXTENDING.md §4a). The elapsed-time contrasts are reported as unavailable rather than null: an input the model cannot read is not a manipulation that failed.
- **This is streaming over the window, not over the 44 h life-log.** Each question resets the cache and ingests from `evidence_min − PRE`, which preserves every arm definition in PLAN.md but does not support claims about accumulated history.
- **Levels are not comparable across regimes** — the priming prompt replaces the system prompt and the image path replaces the video path. Contrasts are.
- The `prior` floor is measured at Δt ∈ {1, 5, 15} min only and assumed flat beyond.
