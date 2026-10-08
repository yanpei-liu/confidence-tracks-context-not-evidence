# E2 briefing deck -- reuses Week1-Briefing's generated macros and figures.
LATEX   ?= latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build
DECK     = e2-briefing
SLIDES   = $(wildcard slides/*.tex) preamble.tex $(DECK).tex
W1       = ../Week1-Briefing
GENED    = $(W1)/generated/numbers.tex $(W1)/generated/tables.tex
FIGS     = $(W1)/figures/figures_manifest.json

.PHONY: all pdf view clean

all: pdf

pdf: build/$(DECK).pdf
build/$(DECK).pdf: $(SLIDES) $(GENED) $(FIGS)
	$(LATEX) $(DECK).tex

view: pdf
	@echo build/$(DECK).pdf

clean:
	rm -rf build
