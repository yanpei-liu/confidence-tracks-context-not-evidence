# Week 1 briefing deck -- see README.md for the module map.
PY      ?= python3
LATEX   ?= latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build
DECK     = week1-briefing
SLIDES   = $(wildcard slides/*.tex) preamble.tex $(DECK).tex
GENED    = generated/numbers.tex generated/tables.tex generated/numbers.json

.PHONY: all numbers figures pdf verify view clean distclean

all: pdf

numbers: $(GENED)
$(GENED): modules/b1_numbers.py \
          ../../Teaser-Experiment/artifacts/m7_metrics.json \
          ../../Teaser-Experiment/artifacts/m7_metrics_stream.json \
          ../../Teaser-Experiment/artifacts/x9_regime_comparison.json \
          ../../Teaser-Experiment/artifacts/x11_evidence_blindness.json \
          memo/CONFIDENCE-PROBE.md memo/RELATED-EGOSAT.md memo/STREAMING-CACHE.md
	$(PY) modules/b1_numbers.py

figures: figures/figures_manifest.json
figures/figures_manifest.json: modules/b2_figures.py \
          ../../Teaser-Experiment/figures/headline.pdf \
          ../../Teaser-Experiment/figures/streaming_headline.pdf \
          ../../Teaser-Experiment/figures/evidence_blind.pdf \
          ../../Teaser-Experiment/figures/evidence_blind_panelA.pdf
	$(PY) modules/b2_figures.py

pdf: build/$(DECK).pdf
build/$(DECK).pdf: $(SLIDES) $(GENED) figures/figures_manifest.json
	$(LATEX) $(DECK).tex

verify: pdf
	$(PY) modules/b5_verify.py

view: pdf
	@echo build/$(DECK).pdf

clean:
	rm -rf build

distclean: clean
	rm -f generated/*.tex generated/*.json figures/*.pdf figures/figures_manifest.json
