# Teaser result — does a streaming ego video-LLM know it forgot?

**Verdict: B: confidence tracks the decline.** Accuracy does decline with Δt, but confidence declines at least as fast, so the overconfidence gap does not widen. The predicted signature is **not** present at this horizon. And the decline does not survive the token-matched control: holding visual-token count fixed removes it entirely, so what looks like time-based forgetting is context-length degradation.

**But there is a metamemory failure here, just not the predicted one.** The sharpest result is
not the Δt sweep — it is the comparison between having the evidence in context and not having it
at all (see *Evidence awareness* below): that swap moves accuracy by
+9.7 points and confidence by only
+3.3. The model cannot tell whether
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
| 1 | 373 | 42.9 | 72.1 | +29.2 | 0.292 | 26 | 1279 |
| 5 | 373 | 41.0 | 69.3 | +28.3 | 0.283 | 74 | 3679 |
| 15 | 373 | 39.4 | 66.7 | +27.3 | 0.274 | 194 | 9679 |
| 30 | 373 | 38.9 | 64.7 | +25.8 | 0.258 | 374 | 18679 |
| 60 | 373 | 36.5 | 62.4 | +26.0 | 0.260 | 734 | 36679 |

Paired over the 373 questions present at both ends
(1 -> 60 min), cluster-bootstrapped over questions:

| quantity | change | p |
| --- | --- | --- |
| accuracy | -6.4 [-10.5, -2.4] pp | 0.0024 |
| confidence | -9.7 [-11.5, -7.9] pp | 0.0000 |
| **overconfidence gap** | **-3.2 [-7.5, +1.0] pp** | 0.1376 |

The gap row is the bin x {accuracy, confidence} interaction the design was powered for.

![headline](../figures/headline.png)

## Evidence awareness — the actual metamemory failure

Token-matched at 3679 visual tokens, paired over
373 questions, the *only* difference being whether the window contains the evidence:

| | no evidence in context | evidence in context | change |
| --- | --- | --- | --- |
| accuracy | 31.4% | 41.0% | **+9.7 [+4.8, +14.5] pp** |
| confidence | 66.0% | 69.3% | **+3.3 [+1.2, +5.4] pp** |

Having the evidence is worth 9.7 points of accuracy but only
3.3 points of confidence — the confidence signal is about
2.9x weaker than the
competence it is supposed to track. The model is nearly as confident answering from nothing as
from the evidence.

## Controls

**Language-prior floor** — the same questions over an evidence-free window: 31.4% accuracy at 66.0% confidence (n=373).
Any accuracy above this floor is what the video evidence is buying, so the sweep is only
informative between that floor and the Δt=1 min bin — a narrow band here.

**No-memory control** — a current-instant question over the identical context:

| Δt (min) | n | accuracy % | confidence % |
| --- | --- | --- | --- |
| 1 | 366 | 53.6 | 86.0 |
| 5 | 366 | 57.9 | 85.4 |
| 15 | 362 | 50.8 | 81.0 |
| 30 | 361 | 47.4 | 78.1 |
| 60 | 355 | 42.5 | 75.2 |

This was meant to stay flat and rule generic long-context degradation *out*. It does not — it
falls further than the memory questions do. Note the caveat that each checkpoint gets its own
control item, so this is a between-items comparison; the paired evidence for the same conclusion
is the token-matched block below.

**Token-matched** — Δt=60 min subsampled to the Δt=15 min visual-token count:

| arm | n | tokens (ref) | acc @15 min % | acc @60 min % | Δ accuracy |
| --- | --- | --- | --- | --- | --- |
| `tokmatch_uniform` | 373 | 9679 | 39.4 | 37.0 | -2.4 [-5.9, +1.1] pp |
| `tokmatch_evkeep` | 373 | 9679 | 39.4 | 39.7 | +0.3 [-2.9, +3.2] pp |

`tokmatch_uniform` thins the whole window; `tokmatch_evkeep` keeps every frame near the
evidence and thins only the filler, so it holds token count *and* evidence resolution fixed
and leaves elapsed time as the only difference. **`tokmatch_evkeep` is the decisive arm**: if
elapsed time itself caused forgetting, accuracy would still fall there, and it does not.

## By category

| category | acc @1 min % | acc @60 min % | conf @1 min % | conf @60 min % |
| --- | --- | --- | --- | --- |
| EntityLog | 37.6 | 33.6 | 74.4 | 66.5 |
| EventRecall | 52.4 | 43.5 | 81.8 | 73.0 |
| RelationMap | 38.7 | 32.3 | 60.1 | 47.7 |

## Robustness: dropping the audio-dependent questions

1639 records come from questions flagged `need_audio`, which a
vision-only model cannot answer in principle. Re-running the whole analysis without them
(n=263 paired questions) changes nothing that matters:

| quantity | all questions | vision-only |
| --- | --- | --- |
| accuracy, 1→60 min | -6.4 [-10.5, -2.4] pp | -6.4 [-11.4, -1.5] pp |
| confidence, 1→60 min | -9.7 [-11.5, -7.9] pp | -9.4 [-11.5, -7.2] pp |
| overconfidence gap | -3.2 [-7.5, +1.0] pp | -3.0 [-8.1, +2.3] pp |
| evidence vs none, accuracy | +9.7 [+4.8, +14.5] pp | +11.4 [+5.3, +17.5] pp |
| evidence vs none, confidence | +3.3 [+1.2, +5.4] pp | +4.0 [+1.6, +6.5] pp |
| `tokmatch_evkeep` accuracy | +0.3 [-2.9, +3.2] pp | +0.8 [-3.0, +4.6] pp |

## Caveats
- The model sees a *window* around the evidence, not the full 44 h history EgoLifeQA assumes,
  so absolute accuracy is not comparable to published EgoLifeQA numbers. The claim is about
  the slope and the gap, not the level.
- One participant, one model. N per bin is bounded by the single released EgoLife participant.
- Control questions are auto-generated from the DenseCaption track and posed in Chinese
  (the captions are Chinese); they are compared only against themselves across Δt.
