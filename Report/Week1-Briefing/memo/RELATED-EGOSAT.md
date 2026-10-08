# EgoSAT (Lei et al., arXiv 2606.24422, 23 Jun 2026) — collision surface and cross-validation

*Paper: [arXiv 2606.24422](https://arxiv.org/abs/2606.24422).
Our numbers throughout come from [`artifacts/m8_teaser.md`](../../../Teaser-Experiment/artifacts/m8_teaser.md) (E1, full-reprocess)
and [`artifacts/x9_streaming.md`](../../../Teaser-Experiment/artifacts/x9_streaming.md) (E2, streaming KV cache); their numbers
are read off Tables 1, 2, 3, A1, A3, A7, A9. Derived quantities (Δ, ratios, `conf_correct − conf_wrong`)
are computed here and marked as such — they are not in the paper.*

This is prior art we must cite, not a scoop. It overlaps our confidence probe and one of our
headline contrasts, and it **independently replicates our central dissociation** on a different
dataset with different backbones. It does not run the control that our whole design is built
around, and where it comes closest to contradicting us, the contradiction dissolves into that
missing control.

---

## 1. What EgoSAT actually contributes

| # | Contribution | Collides with us? |
| --- | --- | --- |
| C1 | **The benchmark.** 1,997 Ego4D sessions (165 h, avg 296.8 s), ~4,800 MCQ over 56 scenarios, six tasks spanning retrospective / present / prospective, under strict online prefix truncation. | No — different dataset, different time scale. |
| C2 | **Unified streaming formulation.** Past / present / future as one prefix-constrained protocol over `X_{1:t}`. | No — framing only. |
| C3 | **Answerability quantification for future events.** *Surprise* (CLIP visual + semantic shift, probit-aligned, Eq. 1–2) and *branchiness* (successor entropy × Rao's quadratic entropy, Eq. 3–5). Human agreement 78 % / 84 %. | No — we have no prospective arm. This is their real technical novelty. |
| C4 | **Confidence probe.** `Conf = max_{k∈C} p(k)`, `Ent = −Σ p(k) log p(k)`, over the letter-normalised next-token distribution (Eq. 6, citing Tsvilodub et al. 2024, arXiv 2403.00998). | **Yes — identical to ours.** |
| C5 | **Confidence-vs-temporal-distance slope.** Least-squares slope over three probes at lag/lead τ, 2τ, 3τ (τ = 8 s), fitted only on items where **all three probes are correct** (A.2). | **Yes — same estimand as our Δt confidence slope.** |
| C6 | **Answerability-conditioned confidence diagnostics.** Table 3: accuracy and confidence on predictable vs unpredictable queries; `conf_correct` / `conf_wrong` (Table A3). | **Yes — structurally our `evidence_vs_none` contrast.** |
| C7 | **Online-vs-offline comparison.** TimeChat-Online-7B / Flash-VStream / LLaVA-OV1.5 against Qwen2.5-VL and Gemini/Claude under the same truncation protocol. | **Partly** — same question as our E1↔E2, but cross-*model*, not within-backbone. |
| C8 | **ROI-TimeChat-Online.** Training-free hand/gaze-ROI-prioritised KV token selection under a fixed 6 K budget at ~85 % drop rate. | No — we do not build a method. |
| C9 | **Memory-proxy sensitivity (Table A7).** MCQ accuracy under memory horizons 1–4× τ_m at fixed dense clip. **Accuracy only — no confidence.** | Adjacent — the cell we fill. |
| C10 | Human baseline (10 % subsample), blind text-only baseline, hard/absurd-negative validation, option-order re-shuffle, cross-scenario SFT transfer. | No. |
| C11 | **Novelty flag.** Table A9 asserts EgoSAT is the only benchmark with an "Answerability Eval." column ticked. | **Positioning collision** — see §4. |

---

## 2. The four places we actually collide

### 2.1 The confidence probe (C4) — hard collision, cite and move on

Their Eq. 6 is our definition verbatim: max softmax over the option-letter set at the forced
selection position, no generation. The project notes at the time presented this as a design decision of ours.
It is now published prior art in the egocentric-streaming setting, and it in turn traces to
Tsvilodub et al. 2024. **Action:** cite both; describe the probe as *standard*, not as a choice we
made. Nothing about our results changes.

### 2.2 The confidence-vs-distance slope (C5) — same estimand, non-overlapping scale

Their retrospective lag axis is τ/2τ/3τ = **8 / 16 / 24 s**. Our Δt grid is **1–60 min of stream
time** over a 44 h life-log with gap-aware coordinates. Their entire 3τ lag fits inside our first
bin with 36 s to spare. Two consequences:

- No number of theirs can be quoted against a number of ours as a direct comparison.
- Their lag axis grows the observed prefix by roughly 5 % (24 s of a ~300 s session), so it is
  *nearly* a pure elapsed-time manipulation — which makes it the closest thing in their paper to
  our `tokmatch_evkeep` arm, and worth comparing qualitatively (§3.1).

### 2.3 Answerability-conditioned confidence (C6) — the strongest overlap, and it favours us

Table 3 splits short-horizon anticipation into predictable and unpredictable queries. That is not
our manipulation — ours removes past evidence from the window, theirs varies the intrinsic
predictability of the future — but the structural claim is the same: *the variable that determines
whether the answer is derivable moves accuracy a lot and confidence almost not at all.* See §3.2,
where it replicates cleanly.

### 2.4 Online vs offline (C7) — same question, weaker design

Their §4.2 concludes streaming lags offline "primarily from the streaming memory constraint …
incremental caching, which inevitably leads to information loss" (Table 1: TimeChat-Online-7B
31.63 vs Qwen2.5-VL-7B 34.46 on multi-step retrospection; 37.84 vs 42.79 short-horizon).

**This inference does not follow from their design.** TimeChat-Online-7B and Qwen2.5-VL-7B differ
in backbone, training data, and instruction tuning, not only in memory management. The gap is
attributed to caching but is not isolated from anything else.

Ours is the controlled version: EgoStream's FlexibleCache is a plug-in memory manager over the
**same frozen Qwen3-VL-8B weights**, so streaming-vs-full-reprocess is within-backbone. What we
find there is that the accuracy cost of streaming is not resolvable on the shared contrasts
(`evidence_vs_none` Δacc difference −1.6 pp [−6.7, +3.5]; `context_length_4x_tokens` +0.8 pp
[−4.0, +5.4]) while the **confidence behaviour changes a great deal** (elapsed-time response
+3.5 pp [+1.7, +5.4]; context-length response −6.1 pp [−8.1, −4.1]). Their headline is a level
comparison across models; ours is a contrast comparison within one.

---

## 3. Cross-validation against our figures

### 3.1 Does confidence decline with retrospective distance? — *partial agreement, and it lands on E2*

Their Table 2, multi-step retrospection (derived column: monotonicity of the accuracy triple):

| model | acc τ→3τ | acc monotone ↓ | conf slope |
| --- | ---: | :---: | ---: |
| Qwen2.5-VL-72B | −7.41 | yes | −0.16 |
| Qwen2.5-VL-32B | −4.07 | yes | ≈ +0.01 |
| Qwen2.5-VL-7B | −2.59 | yes | +0.86 |
| Video-LLaVA-7B | −3.34 | yes | +1.44 |
| **TimeChat-Online-7B** *(streaming)* | **−1.48** | yes | **+0.99** |
| **ROI-TimeChat-Online** *(streaming)* | **+4.05** | no | **+4.61** |
| TimeChat-Online-7B (SFT) | −4.08 | no | ≈ +0.01 |
| ROI-TimeChat-Online (SFT) | +1.93 | no | +0.40 |

Their own reading: "confidence slopes are often inconsistent with the expected negative trend" —
six of eight are ≥ 0.

**Against E2 (revised 2026-08-20: *consistent but uninformative*, not agreement) (`streaming_headline.pdf`, panel A, elapsed-time block): agreement.** Our
`elapsed_time_fixed_frames` gives Δconf −0.1 pp [−1.7, +1.4] — flat. Their two native streaming
models give +0.99 and +4.61 — flat-to-positive. Both regimes say: *under a real bounded cache,
confidence does not fall with retrospective distance.* This corroborates E2's most surprising
result from an independent direction, and it is worth saying so, because E2's own caveat is that
we attribute the flatness to **unavailability** (X1b: the injected clock is illegible once the
cache saturates) rather than to a null. Their data is equally unable to separate the two — no
streaming paper we know of has a readable-clock control — which strengthens the case for
INTERPRETATION.md §2.8 item 1 as the next experiment.

**Against E1 (`headline.pdf`, panel B): apparent disagreement, three reasons it is not one.**
E1 finds confidence falling with elapsed time at fixed tokens and evidence (−3.7 pp [−4.9, −2.4]
for +45 min). Their offline models mostly do not. But (a) 45 min vs 16 s; (b) their slope is
conditioned on **all three probes correct**, a correctness-selected subsample that both truncates
the range and selects high-confidence items, while ours is unconditional over all 373 questions;
(c) their point estimates carry no interval, and half the table sits within rounding distance of
zero. This is a difference in estimand, not a contradiction — but it is the first thing a
reviewer will pit against us, so E1's write-up should name their Table 2 and state the three
differences explicitly.

### 3.2 Does confidence under-react to whether the answer is derivable? — *clean replication*

Derived from their Table 3 (predictable − unpredictable; `cc − cw` from Table 3's
`conf_correct`/`conf_wrong` columns):

| model | Δaccuracy | Δconfidence | under-reaction | cc−cw predictable | cc−cw unpredictable |
| --- | ---: | ---: | :---: | ---: | ---: |
| Qwen2.5-VL-72B | +9.45 | **−4.57** | sign-flipped | +5.83 | +1.41 |
| Qwen2.5-VL-32B | +6.98 | **−0.69** | sign-flipped | +4.05 | +4.63 |
| Qwen2.5-VL-7B | +10.05 | +1.75 | 5.7× | +2.87 | **−4.86** |
| Video-LLaVA-7B | −0.86 | −1.67 | (no acc effect) | +1.08 | −0.03 |
| TimeChat-Online-7B | +4.51 | +0.80 | 5.6× | +2.78 | +1.34 |
| ROI-TimeChat-Online | +5.42 | +0.74 | 7.3× | −0.35 | **−2.62** |
| TimeChat-Online-7B (SFT) | +10.20 | +3.34 | 3.1× | +14.91 | +7.84 |
| ROI-TimeChat-Online (SFT) | +11.64 | +2.06 | 5.7× | +10.23 | +8.19 |
| **ours — E1 (full-reprocess)** | **+9.7 [+4.8, +14.5]** | **+3.3 [+1.2, +5.4]** | **2.9×** | — | — |
| **ours — E2 (streaming)** | **+8.1 [+2.9, +13.1]** | **+0.5 [−1.5, +2.6]** | **16×** | — | — |

Three things line up, on a different dataset (Ego4D vs EgoLife), different backbones
(Qwen2.5-VL 7/32/72B, Video-LLaVA, TimeChat-Online vs our frozen Qwen3-VL-8B), and a different
manipulation of answerability (future predictability vs presence of past evidence):

1. **Accuracy effect size.** Theirs +4.5 to +11.6 pp; ours +9.7 (E1) and +8.1 (E2). Same band.
2. **Confidence under-reaction.** Theirs −4.6 to +3.3 pp, i.e. ratios of 3–7× or an outright sign
   flip on the two largest offline models. Ours 2.9× (E1) and 16× / CI-through-zero (E2). Our E1
   is the *mildest* case in the table; our E2 is at the extreme end alongside their 72B.
3. **Streaming is the worse case, in both.** Their two native streaming models under-react by
   5.6–7.3× and have `cc − cw` separations of +2.78 and **−0.35** — i.e. confidence carries no
   usable ranking signal at all. Our E2 AUROC decays from 0.61 at Δt = 1 min to 0.58 at 60 min
   (Δ −0.073 [−0.143, −0.003], p = 0.039), against a no-evidence floor of ≈ 0.56
   (`streaming_headline.pdf`, panel E). Their `cc − cw` is a coarse two-point version of the same
   statistic and tells the same story.

This is the single most useful thing in the paper for us: **our headline is no longer a
one-dataset, one-model result.** It should be cited as convergent evidence in exactly those terms.

Honest caveat on the analogy: their split is over *intrinsic future predictability*, ours over
*presence of past evidence in the window*. They are the same claim only if you accept
"answerability" as the shared latent. Say that explicitly rather than eliding it.

### 3.3 Does more memory buy accuracy? — *agreement, and they leave our cell empty*

Their Table A7 varies the sparse memory horizon at fixed dense clip (τ_m = 10 s):

| model | D10S10 | D10S20 | D10S30 | D10S40 | span |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemini 2.5 Pro, Ms.retrieval | 40.42 | 43.60 | 40.23 | 40.00 | 3.6 pp, non-monotone |
| Qwen2.5-VL-7B, Ms.retrieval | 26.10 | 28.27 | 27.60 | 28.52 | 2.4 pp, non-monotone |

Their conclusion — "increasing the memory horizon does not lead to uniform gains" — is our E2
budget arm's accuracy row: a 17× live-token budget is worth −1.9 pp [−6.4, +2.4], flat within
noise, and flat in **every** Δt bin (`streaming_headline.pdf`, panel C, blue trace).

**They report no confidence for this manipulation.** Our −15.2 pp [−17.5, −13.1] confidence
response at 17×, monotone across the budget ladder with accuracy pinned, is the finding that
occupies that gap. It is also the cleanest single result either project has: a manipulation that
moves confidence by 15 points and competence by nothing.

### 3.4 Does accuracy fall with retrospective distance? — *the one real tension*

Their offline models decline monotonically over 8 → 24 s (−2.6 to −7.4 pp). Our token-matched arms
say elapsed time costs no accuracy: E1 +0.3 pp [−2.9, +3.2] for +45 min at fixed tokens and
evidence; E2 −1.7 pp [−5.1, +1.6].

Their decline over **16 seconds** is the same magnitude as our entire raw 1 → 60 min sweep
(−6.4 pp). That should be read as a warning about their axis, not about ours:

- Their lag axis is **not** controlled for context volume, hard-negative difficulty, or query
  position. Negatives for multi-step retrospection are "sampled near the query timestamp," so
  moving the query later changes the distractor set as well as the lag.
- We ran precisely this decomposition in E1 (`headline.pdf`, panel A; `m8_teaser.md` §"The Δt
  sweep, decomposed") and the answer was that **the accuracy decline is entirely the
  context-length term** (−2.4 of −2.1 pp observed) with the elapsed-time term at +0.3 pp. E2
  reproduces it arithmetically (`streaming_headline.pdf`, panel D: −6.71 of −6.82 pp of the
  confidence change is context, with no time term left over).
- Note the internal evidence in their own table: their *streaming* models — the ones whose cache
  saturates, so context volume stops growing — show the smallest or non-monotone declines
  (−1.48, +4.05, +1.93). That is what our account predicts.

So the defensible framing is: **their uncontrolled lag axis reproduces our uncontrolled raw sweep,
and we have the control that says what it is made of.** That is a contribution against published
work, and it is worth stating in exactly one sentence in the paper rather than a paragraph.

---

## 4. What has to change on our side

*Status as of 2026-08-20 — all **must fix** items below are actioned; see the pointers.*

| # | landed in |
| --- | --- |
| 1 | No document made the claim — checked repo-wide. The positioning is now explicit in the root `README.md`: evidence age and the elapsed-time-vs-context-volume decomposition |
| 2 | `CLAUDE.md` *Design decisions that held → Confidence* |
| 3 | No document used the phrase — checked repo-wide; recorded so it stays that way |
| 4 | `Planning/[v1+] …` §A, dated bullet |

**Must fix (novelty / attribution):**

1. Drop any "first to evaluate confidence in egocentric streaming" framing. EgoSAT holds that flag
   (Table A9 explicitly claims the Answerability-Eval column). Position on **evidence age and the
   elapsed-time-vs-context-volume decomposition**, not on "we measure confidence under streaming."
2. Attribute the option-letter probe (CLAUDE.md *Design decisions that held* → **Confidence**) to
   Tsvilodub et al. 2024 / EgoSAT Eq. 6.
3. Retire "confidently wrong" as phrasing — it is their abstract's term.
4. Add EgoSAT to the citation index in `Planning/[v1+] …`, alongside the works it surfaces that we
   do not currently cite: TimeChat-Online (arXiv, ACM MM 2025), Flash-VStream (ICCV 2025),
   EWO/Eyes Wide Open (NeurIPS 2025), IPIBench (arXiv 2605.27074), StreamingBench, OmniMMI.

**Keeps its value (differentiators, in the order they are worth defending):**

1. **The token-matched decomposition.** Nobody in this literature separates elapsed time from
   context volume, and §3.4 shows the omission is load-bearing for a published conclusion.
2. **Within-backbone streaming vs full-reprocess.** Their C7 is cross-model and cannot support the
   causal claim it makes.
3. **Time scale.** 1–60 min of gap-aware *stream* time on a 44 h life-log, against their 8–24 s in
   5-minute Ego4D clips — 40–450× the horizon, and the regime where a life-logging assistant
   actually operates.
4. **Statistics.** Paired within-question contrasts, cluster bootstrap over questions (5,000
   draws), decomposition residual assertions, AUROC and ECE. Their Tables 2/3/A7 are point
   estimates with no intervals and no pairing; several of their conclusions rest on differences
   that would not survive one.
5. **A token-matched no-evidence floor.** Their blind baseline removes vision entirely (Table A4),
   which changes the token budget and the modality at once. Our `prior` arm holds frame count and
   token count fixed and removes only the evidence.
6. **Confidence on the memory-budget axis** (§3.3) — measured by us, absent from them.
7. **Post-hoc recalibration at ~zero cost** vs their LoRA SFT. Their B.2 result ("SFT primarily
   improves calibration by suppressing overconfident errors rather than elevating absolute
   confidence"; `cc − cw` +2.78 → +14.91 for TimeChat-Online) is *supporting evidence* that the
   discriminative signal is recoverable — which is what our AUROC-intact finding in E1 predicts,
   and it makes INTERPRETATION.md §2.8 item 2 (recalibrate on `live_tokens`, free, records on
   disk) more attractive, not less.

**Newly de-risked:** our claims-ladder item "accuracy responds to whether the evidence is in
context and confidence does not" now has independent cross-dataset, cross-backbone support (§3.2).
It can move from a two-regime internal replication to a genuinely external one.
