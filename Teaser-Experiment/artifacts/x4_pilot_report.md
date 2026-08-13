# X4 — streaming pilot gate

**PASS** — 86 questions, 430 memory records from `~/Files/EgoVM-data/artifacts/m6_records_stream_main_0of4.jsonl`.

| gate | result | key numbers |
| --- | --- | --- |
| `confidence_not_degenerate` | PASS | mean_confidence=0.7826, sd=0.2038, frac_above_0.999=0.1419 |
| `cache_saturates_as_predicted` | PASS |  |
| `dt1_agrees_with_full_reprocess` | PASS | n_paired=86, streaming_accuracy=0.4535, full_reprocess_accuracy=0.3953, delta_accuracy=0.0581, streaming_confidence=0.846, full_reprocess_confidence=0.7413, answer_agreement_rate=0.7326 |

## Per-Δt

| Δt | n | accuracy | confidence | frames | presented tokens | live tokens |
| --- | --- | --- | --- | --- | --- | --- |
| 1 min | 86 | 0.454 | 0.846 | 25 | 2,500 | 3,174 |
| 5 min | 86 | 0.407 | 0.813 | 73 | 7,300 | 7,930 |
| 15 min | 86 | 0.430 | 0.792 | 193 | 19,300 | 9,108 |
| 30 min | 86 | 0.349 | 0.740 | 373 | 37,300 | 9,108 |
| 60 min | 86 | 0.349 | 0.722 | 733 | 73,300 | 9,108 |

Cache cap at the default budget: **9,108** live tokens (sink 716 + mutable 8,192 + hat 200).
