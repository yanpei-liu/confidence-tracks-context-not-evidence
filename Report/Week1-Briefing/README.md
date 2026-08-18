# Week 1 briefing deck

`build/week1-briefing.pdf` — a beamer deck for the mentor/PI briefing, covering the three
agenda items:

1. **What the two experiments say** — E1 (full-reprocess) and E2 (streaming KV cache), how to
   read them together, and what they do to the project's original claim.
2. **Is the confidence probe the right construct?** — the option-letter softmax, audited.
3. **EgoSAT** — the June 2026 benchmark that overlaps us: what it costs us, and where it
   independently replicates our central finding.

Built with `make`. Checked with `make verify` (nine checks, listed below).

## Why it is built this way

The deck follows the repository's own convention: **modules separated along
verifiable-artifact boundaries**. Each module emits a file that can be checked on its own, and
the next stage consumes only that file. The point is that **no number on any slide is typed by
hand** — every one is a macro whose value and provenance are recorded — so the deck cannot
drift from the experiments the way prose does.

| module | script | artifact | what checks it |
| --- | --- | --- | --- |
| **B1** numbers | `modules/b1_numbers.py` | `generated/numbers.tex`, `generated/tables.tex`, `generated/numbers.json` | `--check` re-derives all 128 macros from the metrics JSON and fails if the committed output is stale. Values that exist only in a memo carry a literal string that must still appear in that memo. |
| **B2** figures | `modules/b2_figures.py` | `figures/*.pdf`, `figures/figures_manifest.json` | `--check` re-cuts every panel into a temp dir and fails if a source figure's sha256 or a crop box changed. Panels are **crops, never redraws**, so a panel on a slide is bit-for-bit the panel in the committed figure. |
| **B3** shell | `preamble.tex` | theme, palette, semantic macros | — |
| **B4** content | `slides/*.tex` | one file per agenda item | `00-title`, `01-results`, `02-confidence`, `03-egosat`, `04-next`, `90-backup` |
| **B5** verify | `modules/b5_verify.py` | the checked PDF | nine checks: B1 current, B2 current, every macro defined, every figure present, **each slide file compiles on its own**, the PDF built, and the log clean (no errors, no undefined refs, no overfull box above 12 pt). |

## Sources

Numbers come from the artifacts CLAUDE.md makes authoritative, never from prose:

* `Teaser-Experiment/artifacts/m7_metrics.json` — E1
* `Teaser-Experiment/artifacts/m7_metrics_stream.json` — E2 per-bin, AUROC, decomposition
* `Teaser-Experiment/artifacts/x9_regime_comparison.json` — E2 contrasts and cross-regime
* `memo/CONFIDENCE-PROBE.md`, `memo/RELATED-EGOSAT.md`, `memo/STREAMING-CACHE.md` — this
  week's analyses, which are the only source for their own tables

Figures are cut from `Teaser-Experiment/figures/headline.pdf` (M7) and
`streaming_headline.pdf` (X10).

Interpretation follows `Teaser-Experiment/INTERPRETATION.md`: contrasts not levels, and the
*unavailable* / *null* / *not confirmed* distinction is preserved on the slides that need it.

## Building

```bash
make            # regenerate numbers + panels if their sources changed, then compile
make verify     # the nine checks above; run this before sending the deck
make numbers    # B1 only
make figures    # B2 only
make clean      # drop build/
```

Requires `pdflatex`/`latexmk` (beamer, booktabs, lmodern), `gs` and `pdfinfo`. No Python
packages beyond the standard library, and no GPU.

## If an experiment is re-run

`make` rebuilds; `make verify` fails until it has been rebuilt. If a memo is edited, B1 fails
loudly naming the macro whose literal no longer matches, rather than letting a slide quote a
number the memo no longer supports.
