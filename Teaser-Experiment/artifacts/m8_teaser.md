# Teaser result — does a streaming ego video-LLM know it forgot?

**Verdict: B: confidence tracks the decline.** Accuracy does decline with Δt, but confidence declines at least as fast, so the overconfidence gap does not widen. The predicted signature is **not** present at this horizon. And the decline does not survive the token-matched control: holding visual-token count fixed removes it entirely, so what looks like time-based forgetting is context-length degradation.

**But there is a metamemory failure here, just not the predicted one.** The sharpest result is
not the Δt sweep — it is the comparison between having the evidence in context and not having it
at all (see *Evidence awareness* below): that swap moves accuracy by
+9.6 points and confidence by only
+2.7. The model cannot tell whether
it has the information. That is a recalibration target, and it does not depend on forgetting.

## Setup

Qwen/Qwen3-VL-8B-Instruct (Apache 2.0), BF16 on one NVIDIA H100 NVL MIG 3g.47gb. EgoLifeQA participant
A1/Jake — the only released participant — restricted by Mitigation 1 to the time-invariant
categories EntityLog + EventRecall + RelationMap. Evidence timestamps come from the released
`target_time` field, which is populated on all 500 items.

Δt is elapsed **stream** time (video the model has actually observed) between the evidence
and the recall checkpoint, not wall-clock: the A1 recording has 31 gaps including six
overnight breaks. Frames are sampled at 0.2 fps at a pinned 320x320
(50.0 visual tokens/frame, verified constant in window length).
Confidence is the softmax over the option-letter logprobs {A,B,C,D}; no text is generated.
It is a proxy, reported with ECE, not a calibrated probability.

## Headline

| Δt (min) | n | accuracy % | confidence % | gap (pp) | ECE | frames | visual tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 314 | 40.8 | 72.0 | +31.3 | 0.313 | 26 | 1276 |
| 5 | 315 | 41.0 | 68.5 | +27.5 | 0.275 | 74 | 3693 |
| 15 | 315 | 40.3 | 66.7 | +26.4 | 0.265 | 194 | 9693 |
| 30 | 314 | 39.2 | 64.1 | +24.9 | 0.249 | 374 | 18690 |
| 60 | 314 | 34.7 | 62.4 | +27.7 | 0.277 | 734 | 36676 |

Paired over the 314 questions present at both ends
(1 -> 60 min), cluster-bootstrapped over questions:

| quantity | change | p |
| --- | --- | --- |
| accuracy | -6.0 [-10.5, -1.6] pp | 0.0120 |
| confidence | -9.7 [-11.6, -7.7] pp | 0.0000 |
| **overconfidence gap** | **-3.7 [-8.3, +1.0] pp** | 0.1216 |

The gap row is the bin x {accuracy, confidence} interaction the design was powered for.

![headline](../figures/headline.png)

## Evidence awareness — the actual metamemory failure

Token-matched at 3693 visual tokens, paired over
315 questions, the *only* difference being whether the window contains the evidence:

| | no evidence in context | evidence in context | change |
| --- | --- | --- | --- |
| accuracy | 31.4% | 41.0% | **+9.6 [+4.4, +14.6] pp** |
| confidence | 65.8% | 68.5% | **+2.7 [+0.4, +4.9] pp** |

Having the evidence is worth 9.6 points of accuracy but only
2.7 points of confidence — the confidence signal is about
3.6x weaker than the
competence it is supposed to track. The model is nearly as confident answering from nothing as
from the evidence.

## Controls

**Language-prior floor** — the same questions over an evidence-free window: 31.4% accuracy at 66.0% confidence (n=373).
Any accuracy above this floor is what the video evidence is buying, so the sweep is only
informative between that floor and the Δt=1 min bin — a narrow band here.

**No-memory control** — a current-instant question over the identical context:

| Δt (min) | n | accuracy % | confidence % |
| --- | --- | --- | --- |
| 1 | 309 | 54.4 | 85.6 |
| 5 | 311 | 59.2 | 85.3 |
| 15 | 306 | 50.0 | 81.4 |
| 30 | 302 | 48.3 | 76.9 |
| 60 | 299 | 42.5 | 75.0 |

This was meant to stay flat and rule generic long-context degradation *out*. It does not — it
falls further than the memory questions do. Note the caveat that each checkpoint gets its own
control item, so this is a between-items comparison; the paired evidence for the same conclusion
is the token-matched block below.

**Token-matched** — Δt=60 min subsampled to the Δt=15 min visual-token count:

| arm | n | tokens (ref) | acc @15 min % | acc @60 min % | Δ accuracy |
| --- | --- | --- | --- | --- | --- |
| `tokmatch_uniform` | 315 | 9693 | 40.3 | 37.8 | -2.6 [-6.7, +1.3] pp |
| `tokmatch_evkeep` | 315 | 9693 | 40.3 | 40.6 | +0.3 [-3.2, +3.5] pp |

`tokmatch_uniform` thins the whole window; `tokmatch_evkeep` keeps every frame near the
evidence and thins only the filler, so it holds token count *and* evidence resolution fixed
and leaves elapsed time as the only difference. **`tokmatch_evkeep` is the decisive arm**: if
elapsed time itself caused forgetting, accuracy would still fall there, and it does not.

## By category

| category | acc @1 min % | acc @60 min % | conf @1 min % | conf @60 min % |
| --- | --- | --- | --- | --- |
| EntityLog | 36.2 | 31.4 | 73.3 | 66.3 |
| EventRecall | 52.8 | 41.5 | 80.8 | 71.9 |
| RelationMap | 33.0 | 31.1 | 61.7 | 48.6 |

## Caveats

- Vision only: 1472 records come from questions flagged
  `need_audio`, which this model cannot answer in principle. The `vision_only_questions`
  block in `m7_metrics.json` repeats every number with those excluded.
- The model sees a *window* around the evidence, not the full 44 h history EgoLifeQA assumes,
  so absolute accuracy is not comparable to published EgoLifeQA numbers. The claim is about
  the slope and the gap, not the level.
- One participant, one model. N per bin is bounded by the single released EgoLife participant.
- Control questions are auto-generated from the DenseCaption track and posed in Chinese
  (the captions are Chinese); they are compared only against themselves across Δt.
