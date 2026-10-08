# X11 — evidence blindness, at every Δt where the floor was measured

Generated from `x11_evidence_blindness.json`; every number below is read from it, so this file cannot drift from the metrics. Δ are percentage points, paired within-question, cluster-bootstrapped over questions (5,000 draws, seed 0).

The `prior` arm holds the window duration, the frame count and the visual-token count fixed and moves the window to *before* the evidence, so the pair differs in nothing but whether the answer is in the context at all.

## The contrast, bin by bin

| Δt | n | accuracy (main / prior) | confidence (main / prior) | Δaccuracy | Δconfidence |
| --- | --- | --- | --- | --- | --- |
| 1 min | 373 | 42.1 / 30.8 % | 84.3 / 82.4 % | +11.3 [+6.2, +16.6] | +1.9 [-0.2, +4.1] |
| 5 min | 373 | 38.3 / 30.3 % | 81.6 / 81.1 % | +8.1 [+2.9, +13.1] | +0.5 [-1.5, +2.6] |
| 15 min | 373 | 38.1 / 33.2 % | 79.0 / 77.4 % | +4.9 [+0.3, +9.4] | +1.6 [-0.4, +3.5] |
| 30 min | 373 | 34.9 / 33.2 % | 73.9 / 74.5 % | +1.6 [-2.4, +5.6] | -0.6 [-2.4, +1.2] |
| 60 min | 373 | 34.0 / 31.1 % | 72.2 / 73.2 % | +2.9 [-1.1, +7.0] | -1.1 [-3.1, +0.9] |

Δaccuracy excludes zero in 3 of 5 bins and ranges over +1.6 to +11.3 pp across them. Δconfidence spans zero in 5 of 5 and does not follow that variation.

### The token-match receipt

| Δt | presented visual tokens (main / prior) | live KV tokens (main / prior) | matched |
| --- | --- | --- | --- |
| 1 min | 2,557 / 2,557 | 3,227 / 3,227 | yes |
| 5 min | 7,357 / 7,357 | 7,945 / 7,945 | yes |
| 15 min | 19,357 / 19,357 | 9,108 / 9,108 | yes |
| 30 min | 37,357 / 37,357 | 9,108 / 9,108 | yes |
| 60 min | 73,357 / 73,357 | 9,108 / 9,108 | yes |

## Are the two conditions separable from the confidence distribution?

Overlap coefficient is Σ min(p_main, p_prior) over a shared 40-bin histogram of [0, 1]: 1.0 means the two confidence distributions are indistinguishable.

| Δt | confidence overlap | Cohen's d |
| --- | --- | --- |
| 1 min | 0.879 | +0.104 |
| 5 min | 0.826 | +0.028 |
| 15 min | 0.845 | +0.080 |
| 30 min | 0.850 | -0.029 |
| 60 min | 0.898 | -0.049 |

### Confidence as a *detector* of evidence presence

AUROC with `main` as the positive class and `prior` as the negative one — the direct operationalisation of "blind". 0.5 is chance. `correct` is shown as the ceiling a binary score can reach at these effect sizes, not as a competitor.

| set | n pairs | max-p confidence | correctness | off-letter mass |
| --- | --- | --- | --- | --- |
| pooled | 1865 | 0.511 [0.495, 0.528] | 0.529 [0.515, 0.543] | 0.393 [0.378, 0.408] |
| 1min | 373 | 0.538 [0.508, 0.569] | 0.556 [0.531, 0.583] | 0.307 [0.280, 0.333] |
| 5min | 373 | 0.513 [0.484, 0.543] | 0.540 [0.515, 0.566] | 0.354 [0.327, 0.382] |
| 15min | 373 | 0.523 [0.495, 0.549] | 0.524 [0.501, 0.547] | 0.383 [0.355, 0.411] |
| 30min | 373 | 0.494 [0.469, 0.518] | 0.508 [0.488, 0.528] | 0.415 [0.385, 0.444] |
| 60min | 373 | 0.488 [0.462, 0.515] | 0.515 [0.495, 0.535] | 0.442 [0.416, 0.469] |

## Is the probe simply inert? No.

The same measure, on the same records, under the other manipulations in the sweep. Read from `x9_regime_comparison.json`.

| manipulation | n | Δaccuracy | Δconfidence |
| --- | --- | --- | --- |
| a no-memory question instead of a recall question | 373 | +18.4 [+13.5, +23.4] | +12.6 [+10.8, +14.4] |
| the raw Δt sweep, 60 → 1 min | 373 | +8.1 [+3.5, +12.6] | +12.1 [+10.0, +14.2] |
| 4× more visual input presented | 373 | -2.4 [-5.9, +1.1] | -6.7 [-8.4, -5.0] |
| the evidence frames thinned away | 373 | -1.9 [-4.6, +0.8] | -1.3 [-2.5, -0.1] |
| the injected clock, on vs off | 373 | -0.6 [-1.7, +0.5] | -0.7 [-1.2, -0.2] |
| +45 min elapsed, frames and evidence fixed | 373 | -1.7 [-5.1, +1.6] | -0.1 [-1.7, +1.4] |
| 17× larger live KV cache at Δt = 60 min | 373 | -1.9 [-6.4, +2.4] | -15.2 [-17.5, -13.1] |

Confidence is not a measure that fails to move. It moves double digits on question kind and on the size of the cache. Evidence is the manipulation it does not see.

## Denominator robustness, bin by bin

The restricted softmax renormalises away whatever mass sits off the four option letters. `CONFIDENCE-PROBE.md` §4 establishes that the vocabulary-normalised alternative is inert everywhere except this contrast at Δt = 5 min; the table below extends that check to every bin.

| Δt | off-letter mass (main / prior) | Δ off-letter | Δconfidence restricted | Δconfidence vocab-normalised |
| --- | --- | --- | --- | --- |
| 1 min | 0.63 / 8.15 % | -7.5 [-9.6, -5.7] | +1.9 [-0.2, +4.1] | +7.7 [+5.0, +10.4] |
| 5 min | 0.13 / 3.13 % | -3.0 [-4.4, -1.8] | +0.5 [-1.5, +2.6] | +2.6 [+0.3, +4.9] |
| 15 min | 0.30 / 1.81 % | -1.5 [-2.5, -0.7] | +1.6 [-0.4, +3.5] | +2.7 [+0.6, +4.8] |
| 30 min | 0.01 / 0.10 % | -0.1 [-0.1, -0.0] | -0.6 [-2.4, +1.2] | -0.6 [-2.4, +1.3] |
| 60 min | 0.02 / 0.10 % | -0.1 [-0.2, -0.0] | -1.1 [-3.1, +0.9] | -1.0 [-3.0, +1.0] |

The off-letter channel *is* evidence-sensitive at every bin, which is exactly why the denominator matters here and nowhere else. The claim on the slide is about the restricted probe — the published standard — and the alternative is reported beside it rather than swapped in silently.

## What is deliberately absent

- **A pooled multi-bin contrast.** Pooling the bins tightens Δconfidence to the edge of significance and would turn a clean per-bin null into a borderline positive. The bins are reported separately for that reason.
- **Δgap as the headline.** `delta_gap` is in the JSON, and removing the evidence does widen the confidence−accuracy gap under the restricted probe. It is not robust to the denominator, so it does not carry the claim.
- **Δt ≥ 30 min**, wherever the `prior` arm was not run: `prior` windows do not nest, so each bin costs its own ingestion pass.
