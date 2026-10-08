# Metamemory in streaming egocentric video-LLMs

**Where this stands, 2026-08-20.** Two experiments are complete. The project's original claim had
two halves; **the first is rejected** and the second needs revision on both of its variables. What
replaced it is a sharper result, and this document is the entry point to it.

Target venue: **ICLR 2027** — abstract 19 Sep 2026, full paper 24 Sep 2026 AoE.

> **The original claim.** *Streaming egocentric video-LLMs exhibit a widening gap between
> confidence and accuracy as evidence ages, and a Δt/evidence-conditioned recalibration recovers
> most of the reliability at near-zero cost.*
>
> The widening gap is **not there** — tested twice, in two memory regimes, and absent in both.

---

## The one-paragraph version

Confidence is computed from **surface properties of the context** — how much visual input the model
has processed, and how long ago it says that was — **not from whether the information needed to
answer is present.** Accuracy responds to evidence. Confidence does not, in either regime. When one
surface cue is removed, confidence latches onto another; what it never latches onto is evidence.
That is a failure of *what is being tracked*, and it is a harder and more useful problem than the
"fails to update" story the project set out to find.

**The two figures to show:**

| | figure | what it says |
| --- | --- | --- |
| **E1** | [`Teaser-Experiment/figures/headline.pdf`](Teaser-Experiment/figures/headline.pdf) | Confidence tracks the wrong variable: it slides with elapsed time, which costs no accuracy, and under-reacts to context length and evidence, which do. |
| **E2** | [`Teaser-Experiment/figures/streaming_headline.pdf`](Teaser-Experiment/figures/streaming_headline.pdf) | Accuracy tracks the evidence; confidence tracks how full the cache is. The same dissociation, rotated onto a different driver. |

E2's figure is purpose-built (module X10: it reads the committed metrics and **computes nothing**,
so it cannot drift) and names every arm and held-fixed variable on the page. E1's is M7's generic
template — still correct, but not self-documenting. Giving E1 a matching figure is a roadmap item,
not a gap in the result.

---

## The two experiments

Same 373 EgoLifeQA questions, same frame bank, same **frozen Qwen3-VL-8B-Instruct**. They differ
only in how the context reaches the model, which is what makes the comparison identifiable —
EgoStream is a plug-in memory manager over the same backbone, so no bridge condition is needed.

| | E1 | E2 |
| --- | --- | --- |
| regime | full-reprocess: every row re-encodes its whole window | streaming KV cache: one ingestion, queried at checkpoints against a bounded cache |
| records | 5,540 scored | 17,311 |
| modules | M0–M9 | X0–X12 |
| cost | ~6 GPU-h | ~14 GPU-h |
| **authoritative artifact** | [`artifacts/m8_teaser.md`](Teaser-Experiment/artifacts/m8_teaser.md) | [`artifacts/x9_streaming.md`](Teaser-Experiment/artifacts/x9_streaming.md) |
| verdict, selected by the measured intervals | *D — confidence tracks the wrong variable* | *Outcome 3 — the dissociation replicates* |

*(E1's raw record files hold 5,562 unique ids; M7 filters to rows in the current plan
(`m7_analyze.py:1027-1035`), dropping 22 that belong to questions the plan later excluded. Both
numbers are correct and appear in different artifacts.)*

Both artifacts are **generated** from the metrics JSON, so they cannot drift. Prose anywhere in
this repo — including this file — is interpretation and may round. **When they disagree, the
artifact wins.**

---

## Current results

**E1 — the dissociation.** Elapsed time is the variable confidence responds to, and it is the one
that costs the model nothing: +45 min at fixed visual tokens *and* fixed evidence moves accuracy
+0.3 pp and confidence −3.7 pp. Reverse it — hold elapsed time fixed and cut the context — and
accuracy moves while confidence barely does. The largest accuracy effect in the experiment, having
the evidence in context at all (+9.7 pp), buys only +3.3 pp of confidence: a 2.9× under-reaction.
The confidence *ranking* is intact across the sweep (AUROC flat), and the decline is item-blind —
a flat discount, not a signal about what was lost.

**E2 — the same dissociation, rotated.** Accuracy responds to whether the evidence is in the window
(+8.1 pp) and to nothing else. Confidence responds to how much visual input was processed (up to
−15.2 pp) and to nothing else, with a clean monotone dose–response and accuracy flat in every bin.
The elapsed-time axis is **unavailable, not null**: X1b established — *before* the contrast was
computed — that the injected clock is illegible once the cache saturates. Two things are new here:
with real eviction the confidence *ranking* degrades (AUROC −0.073, p = 0.039) where it was flat in
E1, and the item-level test moves toward genuine metamemory without reaching significance
(p = 0.118 on 107 lost items).

**The pair.** Evidence-blindness is the invariant: it holds in both regimes and is *deeper* under
streaming (confidence difference across regimes −2.8 pp, excluding zero). What rotates is the
driver — E1's elapsed-time effect is absent in E2 (+3.5 pp difference) and E2's context-volume
effect is absent in E1 (−6.1 pp difference), both excluding zero. On the one manipulation both
regimes share, **the accuracy cost is identical and the confidence response is not**, which rules
out the deflationary reading that E2 is merely a harder task.

> Every number above is a paired within-question contrast, cluster-bootstrapped over questions
> (5,000 draws, seed 0). **Read
> [`Teaser-Experiment/INTERPRETATION.md`](Teaser-Experiment/INTERPRETATION.md) before quoting any
> of them** — it carries the ground rules, the *null* / *unavailable* / *not confirmed* distinction,
> and the things not to claim.

---

## Current conclusions

The full ladder lives in
[`INTERPRETATION.md` §2.7](Teaser-Experiment/INTERPRETATION.md) and is canonical. In short:

**Safe to assert** — accuracy responds to evidence presence and confidence does not, in both
regimes; the predicted widening overconfidence gap is absent in both; on the shared context-length
manipulation the accuracy cost is the same across regimes while the confidence response is not;
under a bounded cache confidence falls monotonically with the token budget while accuracy does not
move; E2's confidence decline decomposes entirely onto presented volume.

**Defensible with the stated caveat** — the surface-properties account (a synthesis across two
regimes, not one measured quantity); the AUROC decay (one test, p = 0.039, wants replication);
permanent eviction in E2's default and large conditions but explicitly *not* in `small`; and the
denominator sensitivity on E2's evidence contrast.

**Not supported** — "streaming models don't know they forgot" in the original sense; "confidence
tracks the items actually lost"; "streaming makes the model more overconfident" (a level,
confounded with a prompt change); "elapsed time does not affect confidence in streaming systems"
(the axis was unavailable, which is not a null).

**What this does to the original claim.** Half 1 is rejected — twice, and the second time in the
setting the hypothesis most deserved, where evidence is genuinely evicted rather than merely
diluted. It should be retired, not tested a third way. Half 2 needs revision on both variables:
**Δt is the wrong conditioner** (it worked in E1 by accident and has no purchase in E2, where the
model cannot read it) and **`live_tokens` is the right one** — observable at inference and free;
while "evidence-conditioned" is not implementable at all, since knowing whether the evidence is in
context is the very thing the model cannot report. And the headroom differs by regime: recalibrating
a *level* only works if the *ranking* survives, which it does in E1 and does not in E2. **The cheap
fix gets cheaper-looking the less realistic the memory system is** — the most consequential thing
the pair says, and visible only because both regimes were run.

**How to position this.** On **evidence age and the elapsed-time-vs-context-volume decomposition**,
which nobody in this literature separates. *Not* on "we measure confidence under streaming" —
EgoSAT (arXiv 2606.24422) holds that flag, and the collision analysis is in
[`memo/RELATED-EGOSAT.md`](Report/Week1-Briefing/memo/RELATED-EGOSAT.md).

---

## Next steps

The canonical, ranked roadmap is
[`INTERPRETATION.md` §2.8](Teaser-Experiment/INTERPRETATION.md) — ten items in three cost tiers.
The shape of it:

- **Free, CPU only** — recalibrate on `live_tokens` per regime (the corrected Half-2 claim, and the
  AUROC asymmetry makes it a sharp test); off-letter mass as a reported quantity; `x11`/`x12` to
  turn the memos' tables into generated artifacts; three small schema and labelling fixes.
- **~1 GPU-h** — **strip the timestamps from E1 and re-score the token-matched cells.** Highest
  value per GPU-hour on the list: E1's clock is ordinary *text*, so this separates "confidence
  responds to elapsed time" from "confidence responds to printed numbers", and separates the two
  live explanations for E2's clock failure at the same time. Also: a top-k vocabulary dump on the
  `prior` arm, to find out what the model hedges *with*.
- **Larger** — restore a readable clock under streaming (the missing cell of the design, best run
  after the timestamp strip); explicit abstention (~2.4 GPU-h); an option-permutation bias check;
  and powering the item-level test.

### Open decisions — these need a human

1. Retire the original hypothesis outright and lead with the dissociation?
2. Which follow-up first? (Proposed: timestamps-off, then recalibration.)
3. How hard to position against EgoSAT — convergent evidence, or competing framing?
4. Is the abstention channel worth a paper section, given it rides on a 1.4 % tail?

---

## Where things live

| | |
| --- | --- |
| [`Teaser-Experiment/`](Teaser-Experiment/) | Both experiments: code (`egovm/` = E1, `egovm/streaming/` = E2), committed artifacts, figures, and the design documents. Named for the 2-day teaser it started as; both experiments are still a teaser for the wider project, so the name stays. |
| [`Report/Week1-Briefing/`](Report/Week1-Briefing/) | The mentor briefing deck, built so no slide can quote a stale number, plus three memos that audit the confidence probe, the streaming cache, and the EgoSAT collision. The memos are primary analysis, not slide notes. |
| [`Planning/`](Planning/) | Superseded pre-execution planning, kept as a record. Each file carries a banner saying what execution falsified. `[v1+]` is still useful as a verified literature and citation index. |
| `$EGOVM_DATA` | Everything large (~114 GB): video, the 0.2 fps frame bank, the raw records. **Not committed** — EgoLife is S-Lab License 1.0, research-use, non-redistributable. Pinned by checksum in `artifacts/m6_records_manifest.json`. |

### Document precedence

0. [`Teaser-Experiment/INTERPRETATION.md`](Teaser-Experiment/INTERPRETATION.md) — how to read both
   experiments and what not to claim. **Start here before quoting any number.**
1. The two generated artifacts — `m8_teaser.md` (E1) and `x9_streaming.md` (E2). Authoritative.
2. [`PLAN.md`](Teaser-Experiment/PLAN.md) (E1 design) and
   [`EXTENDING.md`](Teaser-Experiment/EXTENDING.md) (E2 design and result).
3. The three memos under `Report/Week1-Briefing/memo/` — each audits one construct against source
   and ends in a *what has to change* list. As of 2026-08-20 every must-fix item is actioned.
4. [`Teaser-Experiment/README.md`](Teaser-Experiment/README.md) — how to run it, and the cluster
   gotchas that cost real time.
5. `Planning/` — superseded. Do not take design decisions from any of it.

---

## Verify a checkout

```bash
# the records the committed metrics were computed from, both regimes
python Teaser-Experiment/scripts/records_manifest.py --check   # 37 files, 22,873 ids

# the deck still builds and every number on it still derives
cd Report/Week1-Briefing && make && make verify                 # 191 macros, nine checks
```

Neither needs a GPU. Reproducing the analysis itself from the raw records — also GPU-free — is in
[`Teaser-Experiment/README.md`](Teaser-Experiment/README.md).
