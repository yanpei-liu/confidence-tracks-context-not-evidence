# X4 — streaming pilot gate

**FAIL** — 20 questions, 100 memory records from `~/Files/EgoVM-data/artifacts/m6_records_stream_pilot_0of1.jsonl`.

| gate | result | key numbers |
| --- | --- | --- |
| `confidence_not_degenerate` | PASS | mean_confidence=0.7087, sd=0.1821, frac_above_0.999=0.01 |
| `cache_saturates_as_predicted` | PASS |  |
| `dt1_agrees_with_full_reprocess` | FAIL | n_paired=20, streaming_accuracy=0.25, full_reprocess_accuracy=0.2, delta_accuracy=0.05, streaming_confidence=0.7656, full_reprocess_confidence=0.4828, answer_agreement_rate=0.55 |

## Per-Δt

| Δt | n | accuracy | confidence | frames | presented tokens | live tokens |
| --- | --- | --- | --- | --- | --- | --- |
| 1 min | 20 | 0.250 | 0.766 | 26 | 2,600 | 3,296 |
| 5 min | 20 | 0.200 | 0.733 | 74 | 7,400 | 8,001 |
| 15 min | 20 | 0.250 | 0.719 | 194 | 19,400 | 9,108 |
| 30 min | 20 | 0.300 | 0.664 | 374 | 37,400 | 9,108 |
| 60 min | 20 | 0.300 | 0.662 | 734 | 73,400 | 9,108 |

Cache cap at the default budget: **9,108** live tokens (sink 716 + mutable 8,192 + hat 200).
