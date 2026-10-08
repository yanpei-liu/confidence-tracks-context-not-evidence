"""X10 — the standalone headline figure for the streaming experiment.

X9 copies M7's headline figure out under a streaming name. That figure was designed for the
full-reprocess sweep and its panel titles assert E1's findings ("the ranking is untouched by
Δt"), two of which are *false* on these records: the ranking degrades, and elapsed time is not
a manipulation here at all but an input the model cannot read (EXTENDING.md §4a). It also
inherits E1's gloss for the token-matched arms, which are degenerate as a token match once the
cache saturates (EXTENDING.md §3b).

This module draws the streaming result on its own terms, with every element's meaning spelled
out on the page. Nothing is computed here: every number is read from the two committed
artifacts, so the figure cannot drift from the metrics.

    artifacts/x9_regime_comparison.json   the paired contrasts, the budget arm, per-cell levels
                                          -- authoritative for the streaming experiment
    artifacts/m7_metrics_stream.json      per-bin levels + CIs, the decomposition, AUROC,
                                          item transitions (M7 run on the streaming records)

Panels, in reading order:

    A  every manipulation as a paired contrast, grouped by which variable it moves; filled
       marks are intervals excluding zero. This is the dissociation, and the figure's point
    B  the raw Δt sweep, with the no-evidence floor and the token counts that grow with it --
       the descriptive picture, labelled as confounded by construction
    C  the budget arm's dose-response: confidence tracks the live-token ratio monotonically
       while accuracy stays flat
    D  M7's decomposition of the 15 -> 60 min decline into elapsed time and context length
    E  AUROC over the sweep -- the one phenomenon full-reprocess could not produce

Artifacts: figures/streaming_headline.{pdf,png}
"""
from __future__ import annotations

import argparse
import json

from ..config import ARTIFACTS, FIGURES

# Same palette as M7's figures, so the two experiments' figures read as one system:
# blue = accuracy, orange = confidence (option-letter softmax), purple = no-evidence floor.
C_ACC = "#2a78d6"
C_CONF = "#eb6834"
C_PRIOR = "#4a3aa7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
FAINT = "#c9c8c3"
SURFACE = "#fcfcfb"
BAND = "#eceae4"

DT_MIN = (1, 5, 15, 30, 60)


def pp(d: dict) -> str:
    """A bootstrapped Δ as percentage points with its 95 % interval."""
    return (f"{100 * d['mean']:+.1f} "
            f"[{100 * d['lo95']:+.1f}, {100 * d['hi95']:+.1f}]")


def sig(d: dict) -> bool:
    """Does the 95 % interval exclude zero? Drives filled vs hollow marks."""
    return d["lo95"] > 0 or d["hi95"] < 0


def ktok(v: float) -> str:
    return f"{v / 1000:.1f}k" if v < 10_000 else f"{v / 1000:.0f}k"


def rows_spec(x9: dict) -> list[tuple]:
    """The panel-B rows: (group header or None, name, exact contrast, source dict).

    Each row's second line names the cells actually differenced, because "elapsed time" and
    "context length" mean different things here than they do under full-reprocess and the
    reader cannot be expected to carry EXTENDING.md §3b in their head.
    """
    c = x9["streaming_contrasts"]
    b60 = x9["budget_by_dt"]["3600"]
    ctx = c["context_length_4x_tokens"]
    tok_ratio = ctx["tokens_a"] / ctx["tokens_b"]
    live_ratio = b60["live_tokens_large"] / b60["live_tokens_small"]
    return [
        ("EVIDENCE IN THE WINDOW  —  accuracy moves, confidence does not", None, None, None),
        (None, "evidence present vs absent",
         "main@5 min − prior@5 min · same frame count, same tokens, same Δt",
         c["evidence_vs_none"]),
        ("CONTEXT VOLUME IN THE WINDOW  —  confidence moves, accuracy does not",
         None, None, None),
        (None, f"{live_ratio:.0f}× larger live KV cache",
         f"large − small at Δt = 60 min · {ktok(b60['live_tokens_large'])} vs "
         f"{ktok(b60['live_tokens_small'])} live tokens, same frames",
         b60),
        (None, f"{tok_ratio:.1f}× more frames presented",
         f"main@60 min − evkeep@60 min · {ktok(c['context_length_4x_tokens']['tokens_a'])} vs "
         f"{ktok(c['context_length_4x_tokens']['tokens_b'])} presented tokens, Δt fixed",
         c["context_length_4x_tokens"]),
        (None, "evidence frames thinned away",
         "uniform@60 min − evkeep@60 min · same frame count, evidence diluted",
         c["evidence_thinned"]),
        ("ELAPSED TIME  —  an input the model cannot read here (X1b)", None, None, None),
        (None, "+45 min of elapsed time",
         "evkeep@60 min − main@15 min · frames, tokens and evidence all fixed",
         c["elapsed_time_fixed_frames"]),
        (None, "the injected clock, on vs off",
         "main − nots, pooled over Δt · the markers cost ≈5 % of the token budget",
         c["timestamps_on_vs_off"]),
        ("FOR REFERENCE  —  confounded, and shown as such", None, None, None),
        (None, "the raw Δt sweep",
         "main@1 min − main@60 min · elapsed time and context volume grow together",
         c["elapsed_time_raw"]),
        (None, "a no-memory question instead",
         "control − memory · identical frames, answerable from the frames alone",
         c["control_vs_memory"]),
    ]


def panel_sweep(ax, m7: dict) -> None:
    """B — the descriptive sweep, and why it cannot be read as forgetting."""
    mem = m7["bins"]["memory/main"]
    prior = m7["bins"]["memory/prior"]
    x = list(DT_MIN)

    for field, ci, col, lbl, y_off in (("accuracy", "accuracy_ci", C_ACC, "accuracy", -0.035),
                                       ("mean_confidence", "confidence_ci", C_CONF,
                                        "confidence", 0.028)):
        lo = [mem[str(d * 60)][ci]["lo95"] for d in x]
        hi = [mem[str(d * 60)][ci]["hi95"] for d in x]
        ys = [mem[str(d * 60)][field] for d in x]
        ax.fill_between(x, lo, hi, color=col, alpha=0.13, lw=0, zorder=2)
        ax.plot(x, ys, "-o", color=col, lw=2, ms=7, mec=SURFACE, mew=1.4, zorder=4)
        ax.text(x[0] * 0.92, ys[0] + y_off, lbl, color=col, fontsize=8.4, ha="left",
                va="bottom" if y_off > 0 else "top", weight="bold")

    pdts = sorted(int(k) for k in prior)
    px = [d / 60 for d in pdts]
    ax.plot(px, [prior[str(d)]["accuracy"] for d in pdts], ls=(0, (4, 2)), marker="o",
            color=C_PRIOR, lw=1.6, ms=6.5, mfc=SURFACE, mew=1.6, zorder=5)
    ax.text(px[0] * 1.12, prior[str(pdts[0])]["accuracy"] - 0.014,
            "no-evidence floor (prior arm)", color=C_PRIOR,
            fontsize=7.0, ha="left", va="top")
    ax.axhline(0.25, color=MUTED, lw=1, ls=(0, (2, 3)), zorder=1)
    ax.text(x[0] * 0.9, 0.25 - 0.014, "chance (4-way MCQ)", fontsize=7.2, color=MUTED, va="top")


def panel_sweep_annotate(ax, m7: dict, x9: dict) -> None:
    """The token block inside panel A: what actually grows across the sweep."""
    mem = m7["bins"]["memory/main"]
    cells = x9["cells"]
    live = {d: cells[f"memory/main/{d * 60}"]["live_tokens"] for d in DT_MIN}
    cap = max(live.values())
    # tokens presented at 100/frame; the cache holds at most `cap` of them at query time.
    overhead = live[5] - mem["300"]["mean_visual_tokens"]
    sat_frames = (cap - overhead) / 100.0
    sat_min = sat_frames * 5.0 / 60.0

    ax.axvline(sat_min, color=MUTED, lw=1, ls=(0, (1, 2)), zorder=1)
    ax.text(sat_min * 1.1, 0.945,
            f"the live cache fills here:\n{cap:,.0f} tokens ≈ {sat_frames:.0f} frames",
            fontsize=6.9, color=INK2, ha="left", va="top", linespacing=1.35)

    ax.text(DT_MIN[0] * 0.72, 0.685, "visual tokens presented", fontsize=6.9, color=INK2,
            ha="left", va="bottom")
    ax.text(DT_MIN[0] * 0.72, 0.565, "live KV tokens at query time (grey: at the cap)",
            fontsize=6.9, color=INK2, ha="left", va="bottom")
    for d in DT_MIN:
        ax.text(d, 0.640, ktok(mem[str(d * 60)]["mean_visual_tokens"]), fontsize=7.2,
                color=INK, ha="center", va="center")
        ax.text(d, 0.520, ktok(live[d]), fontsize=7.2,
                color=INK if live[d] < cap else MUTED, ha="center", va="center")

    ax.set_xscale("log")
    ax.set_xticks(DT_MIN)
    ax.set_xticklabels([str(d) for d in DT_MIN])
    ax.set_xlim(0.62, 110)
    ax.set_ylim(0.20, 0.98)
    ax.set_xlabel("Δt — stream time from evidence to question (min, log)", fontsize=8.4)
    ax.set_ylabel("accuracy  /  mean confidence", fontsize=8.4)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.13, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("B  The sweep alone settles nothing: both\n"
                 "     candidate causes rise with Δt together",
                 fontsize=9.2, loc="left", color=INK)


def panel_map(ax, x9: dict) -> None:
    """A — every manipulation, paired within question, grouped by what it moves."""
    import matplotlib.transforms as mtransforms

    spec = rows_spec(x9)
    n = len(spec)
    ys = {i: n - 1 - i for i in range(n)}
    # group headings are left-aligned to the page, not to the axes: they are longer than the
    # row labels and would otherwise run off the canvas.
    page = mtransforms.blended_transform_factory(ax.figure.transFigure, ax.transData)

    for i, (header, name, cells, src) in enumerate(spec):
        y = ys[i]
        if header is not None:
            ax.axhline(y + 0.62, color=FAINT, lw=0.8, zorder=1)
            ax.text(0.010, y, header, transform=page, ha="left", va="center", fontsize=7.8,
                    color=INK, weight="bold", clip_on=False)
            continue
        ax.text(-0.017, y + 0.16, name, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=8.0, color=INK)
        ax.text(-0.017, y - 0.20, cells, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=6.6, color=MUTED)
        for key, col, dy, tag in (("delta_accuracy", C_ACC, 0.19, "acc"),
                                  ("delta_confidence", C_CONF, -0.19, "conf")):
            d = src[key]
            ax.plot([100 * d["lo95"], 100 * d["hi95"]], [y + dy, y + dy], color=col, lw=2.4,
                    alpha=0.85, solid_capstyle="round", zorder=3)
            filled = sig(d)
            ax.plot([100 * d["mean"]], [y + dy], "o", ms=7.5, color=col if filled else SURFACE,
                    mec=col, mew=1.8, zorder=5)
            ax.annotate(f"{tag} {pp(d)}", xy=(1.015 if tag == "acc" else 1.20, y + dy),
                        xycoords=ax.get_yaxis_transform(), fontsize=7.2,
                        color=col if filled else MUTED, ha="left", va="center",
                        annotation_clip=False,
                        weight="bold" if filled else "normal")

    # the elapsed-time group is shaded: the manipulation was run, but the input is unreadable
    elapsed_ys = [ys[i] for i, r in enumerate(spec)
                  if r[0] is None and r[1] in ("+45 min of elapsed time",
                                               "injected clock on vs off")]
    ax.axhspan(min(elapsed_ys) - 0.55, max(elapsed_ys) + 0.55, color=BAND, alpha=0.55, lw=0,
               zorder=0)

    ax.axvline(0, color=INK2, lw=0.9, zorder=2)
    ax.set_ylim(-0.75, n - 0.25)
    ax.set_xlim(-21, 26)
    ax.set_yticks([])
    ax.set_xlabel("change when the first condition replaces the second (pp) · "
                  "n = 373 questions on every row", fontsize=8.4)
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", alpha=0.13, lw=0.7)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title("A  What each manipulation moves.  Accuracy responds to whether the evidence "
                 "is in the window and to nothing else;\n"
                 "     confidence responds to how much context is in the window and to nothing "
                 "else — the two measures are driven by disjoint variables",
                 fontsize=10.2, loc="left", color=INK)

    import matplotlib.lines as mlines
    handles = [mlines.Line2D([], [], color=C_ACC, marker="o", ms=7.5, lw=2.4, mec=C_ACC,
                             mew=1.8, label="Δ accuracy, with its 95 % interval"),
               mlines.Line2D([], [], color=C_CONF, marker="o", ms=7.5, lw=2.4, mec=C_CONF,
                             mew=1.8, label="Δ confidence (option-letter softmax)"),
               mlines.Line2D([], [], color=MUTED, marker="o", ms=7.5, lw=0, mfc=SURFACE,
                             mec=MUTED, mew=1.8,
                             label="hollow mark: the interval includes zero")]
    ax.legend(handles=handles, frameon=False, fontsize=7.4, loc="lower left",
              bbox_to_anchor=(0.002, 0.005), handlelength=1.9, labelspacing=0.35,
              borderaxespad=0.2)


def panel_dose(ax, x9: dict) -> None:
    """C — the budget arm: a monotone confidence response with a flat accuracy response."""
    b = x9["budget_by_dt"]
    x = list(DT_MIN)
    for key, col, lbl, lx, ly in (("delta_accuracy", C_ACC, "Δ accuracy", 1.9, 3.6),
                                  ("delta_confidence", C_CONF, "Δ confidence", 1.8, -5.8)):
        ys = [100 * b[str(d * 60)][key]["mean"] for d in x]
        lo = [100 * b[str(d * 60)][key]["lo95"] for d in x]
        hi = [100 * b[str(d * 60)][key]["hi95"] for d in x]
        ax.fill_between(x, lo, hi, color=col, alpha=0.13, lw=0, zorder=2)
        ax.plot(x, ys, "-o", color=col, lw=2, ms=7, mec=SURFACE, mew=1.4, zorder=4)
        ax.text(lx, ly, lbl, color=col, fontsize=8.0, ha="left", va="top", weight="bold")
    ax.axhline(0, color=INK2, lw=0.9, zorder=1)

    for d in x:
        r = b[str(d * 60)]["live_tokens_large"] / b[str(d * 60)]["live_tokens_small"]
        ax.text(d, 5.0, f"{r:.0f}×" if r >= 10 else f"{r:.1f}×", fontsize=7.2, color=INK2,
                ha="center", va="center")
    ax.text(x[0] * 0.68, 7.0, "live-token ratio, large : small", fontsize=6.9, color=INK2,
            ha="left", va="center")
    ax.text(x[0] * 0.68, -16.6,
            "one paired contrast per point: same question, same\n"
            "frames, same Δt — only mutable_token_size differs",
            fontsize=6.8, color=INK2, ha="left", va="top", linespacing=1.5)

    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in x])
    ax.set_xlim(0.62, 110)
    ax.set_ylim(-20.5, 8.6)
    ax.set_xlabel("Δt bin (min, log) — held fixed within each contrast", fontsize=8.4)
    ax.set_ylabel("large − small (pp)", fontsize=8.4)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.13, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("C  A bigger cache buys no accuracy and\n"
                 "     costs a great deal of confidence",
                 fontsize=9.2, loc="left", color=INK)


def panel_decomp(ax, m7: dict) -> None:
    """D — the 15 → 60 min decline split into elapsed time and context length."""
    dec = m7["decomposition"]
    lo, hi = dec["dt_ref_s"] // 60, dec["dt_hi_s"] // 60
    rows = (("accuracy", C_ACC, 1.0), ("confidence", C_CONF, 0.0))
    h = 0.34
    for field, col, y in rows:
        d = dec[field]
        e, c = 100 * d["elapsed_time"], 100 * d["context_length"]
        ax.barh(y, e, height=h, color=col, alpha=0.30, lw=0, zorder=3)
        ax.barh(y, c, left=e, height=h, color=col, lw=0, zorder=3)
        # a surface gap between the two segments rather than a shared edge
        ax.plot([e, e], [y - h / 2, y + h / 2], color=SURFACE, lw=2.2, zorder=4)
        ax.plot([100 * d["observed"]], [y], "D", ms=6, color=SURFACE, mec=INK, mew=1.4,
                zorder=6)
        ax.text(e + c / 2, y, f"{c:+.2f}", fontsize=7.6, color=SURFACE, ha="center",
                va="center", weight="bold", zorder=7)
        if y > 0:  # the confidence row's elapsed segment is too small to label — see the note
            ax.annotate(f"{e:+.2f}", xy=(e / 2, y + h / 2), xytext=(-24, 13),
                        textcoords="offset points", fontsize=7.6, color=INK2, ha="center",
                        va="bottom", zorder=7,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkB=1))
            # label the observed mark once, on the top row
            ax.annotate("observed", xy=(100 * d["observed"], y), xytext=(-8, 0),
                        textcoords="offset points", fontsize=6.6, color=INK2, ha="right",
                        va="center")
        ax.text(0.3, y + 0.05, field, fontsize=8.4, color=col, ha="left", va="center",
                weight="bold")
        ax.text(0.3, y - 0.22, f"observed {100 * d['observed']:+.2f} pp\n"
                               f"residual {100 * d['residual']:+.2f} pp",
                fontsize=6.8, color=MUTED, ha="left", va="top", linespacing=1.45)

    ax.text(-8.1, -0.42,
            "pale = elapsed-time term, solid = context\n"
            "the confidence bar's elapsed-time term is\n"
            f"{100 * dec['confidence']['elapsed_time']:+.2f} pp — invisible here, and that is\n"
            "the finding, not a drawing artefact",
            fontsize=6.6, color=INK2, ha="left", va="top", linespacing=1.5)

    # the zero line spans the bars only, so it does not rule through the note below them
    ax.plot([0, 0], [-0.32, 1.55], color=INK2, lw=0.9, zorder=2)
    ax.set_ylim(-1.30, 1.72)
    ax.set_xlim(-8.2, 4.2)
    ax.set_yticks([])
    ax.set_xlabel(f"contribution to the {lo} → {hi} min change (pp)", fontsize=8.4)
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", alpha=0.13, lw=0.7)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    ax.set_title("D  The same split arithmetically: the decline is\n"
                 "     all context, with no time term left over",
                 fontsize=9.2, loc="left", color=INK)


def panel_auroc(ax, m7: dict) -> None:
    """E — the ranking decays, which is the phenomenon full-reprocess could not produce."""
    disc = m7["discrimination"]
    main = sorted((int(k.split("/")[-1]) // 60, v["auroc"]) for k, v in disc.items()
                  if k.startswith("memory/main/"))
    prior = sorted((int(k.split("/")[-1]) // 60, v["auroc"]) for k, v in disc.items()
                   if k.startswith("memory/prior/"))
    ax.plot([d for d, _ in main], [a for _, a in main], "-o", color=C_CONF, lw=2, ms=7,
            mec=SURFACE, mew=1.4, zorder=4)
    ax.text(main[0][0] * 1.1, main[0][1] - 0.008, "recall questions\n(evidence in the window)",
            color=C_CONF, fontsize=7.2, ha="left", va="top", linespacing=1.35, weight="bold")
    ax.plot([d for d, _ in prior], [a for _, a in prior], ls=(0, (4, 2)), marker="o",
            color=C_PRIOR, lw=1.6, ms=6.5, mfc=SURFACE, mew=1.6, zorder=5)
    ax.text(prior[0][0] * 1.1, prior[-1][1] - 0.012, "no-evidence floor (prior arm)",
            color=C_PRIOR, fontsize=7.2, ha="left", va="top")
    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (2, 3)), zorder=1)
    ax.text(0.68, 0.505, "chance", fontsize=7.2, color=MUTED, va="bottom")

    d = disc["delta_auroc_sweep"]
    ax.text(0.68, 0.742,
            f"1 → 60 min:  Δ AUROC {d['mean']:+.3f}\n"
            f"[{d['lo95']:+.3f}, {d['hi95']:+.3f}], p = {d['p_two_sided_vs_0']:.3f}",
            fontsize=7.8, color=INK, ha="left", va="top", weight="bold", linespacing=1.45)

    ax.set_xscale("log")
    ax.set_xticks(DT_MIN)
    ax.set_xticklabels([str(dd) for dd in DT_MIN])
    ax.text(0.68, 0.468,
            "the change is the finding, not the level: by 60 min the\n"
            "ranking is barely better than with no evidence at all.",
            fontsize=6.8, color=INK2, ha="left", va="top", linespacing=1.5)

    ax.set_xlim(0.62, 110)
    ax.set_ylim(0.432, 0.755)
    ax.set_xlabel("Δt (min, log)", fontsize=8.4)
    ax.set_ylabel("AUROC — confidence vs correctness", fontsize=8.4)
    ax.tick_params(labelsize=8)
    ax.set_yticks([0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    ax.grid(alpha=0.13, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("E  With real eviction the confidence ordering\n"
                 "     decays too, not only its level",
                 fontsize=9.2, loc="left", color=INK)


def subtitle(m7: dict, x9: dict) -> str:
    n_q = m7["bins"]["memory/main"]["60"]["n"]
    return (f"Streaming KV cache (EgoStream FlexibleCache + ReKV retrieval) over a frozen "
            f"Qwen3-VL-8B-Instruct · {n_q} EgoLifeQA questions, 0.2 fps frames at 320×320 "
            f"(100 visual tokens/frame) · {x9['n_streaming_records']:,} scored records\n"
            "Confidence is max softmax over the option-letter logits {A,B,C,D} at the first "
            "generated position — a model output, not a calibrated probability. Every Δ is a "
            "paired within-question contrast,\ncluster-bootstrapped over questions "
            "(5,000 draws, seed 0); bands and bars are 95 % intervals. The cache is reset per "
            "question, so this is streaming over the window, not over the life-log.")


def footer(m7: dict) -> str:
    it = m7["item_transitions"]
    return (
        "Arms  ·  main: the Δt window ending at the question, evidence inside it   "
        "prior: an evidence-free window of the same frame count (the floor)   "
        "evkeep / uniform: the Δt = 15 min frame count re-drawn at Δt = 60 min, evidence frames "
        "kept vs uniformly thinned\n"
        "          control: a no-memory question answerable from the same frames   "
        "small / large: mutable_token_size 1,024 / 32,768 against the 8,192 default (live caps "
        "1,940 / 33,684 vs 9,108)   nots: the injected clock removed\n"
        "Read with care  ·  the elapsed-time rows of A are an unavailable axis, not a null: X1b "
        "established, before this sweep ran, that the injected clock is illegible once the cache "
        "saturates — so this experiment cannot test whether confidence tracks elapsed time, only "
        "that this cache leaves it nothing to track.\n"
        "          In C, the large and small arms are both at their caps by Δt = 30 min, so the "
        "live-token ratio stops growing and what still grows is eviction pressure. Levels are not "
        "comparable with the full-reprocess sweep (different system prompt, different ingestion "
        "path); contrasts are.\n"
        f"          The no-evidence floor is measured at every Δt on the grid "
        f"({', '.join(str(d) for d in sorted(int(k) // 60 for k in m7['bins']['memory/prior']))} min); "
        "nothing about it is assumed. One participant (EgoLife A1), one "
        "model, one cache policy (EgoStream's shipped pruning pipeline plus ReKV retrieval).\n"
        f"Not shown, not a finding  ·  item-level metamemory: among items correct at the "
        f"earlier bin, confidence fell {100 * it['delta_confidence_kept']['mean']:+.1f} pp on "
        f"the {it['n_kept']} kept and {100 * it['delta_confidence_lost']['mean']:+.1f} pp on the "
        f"{it['n_lost']} lost; the difference is {pp(it['difference'])} pp, "
        f"p = {it['difference']['p_two_sided_vs_0']:.3f} — suggestive, underpowered.\n"
        "Sources  ·  artifacts/x9_regime_comparison.json (contrasts, budget arm, per-cell "
        "levels) and artifacts/m7_metrics_stream.json (per-bin levels, decomposition, AUROC, "
        "item transitions). Drawn by egovm/streaming/x10_figure.py; no number is computed here."
    )


def build(m7: dict, x9: dict, stem) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15.6, 10.6))
    # explicit axes rectangles rather than a gridspec: panel A needs a wide left gutter for its
    # row labels and a right gutter for its two value columns, and the title block above the
    # panels is a fixed height that tight_layout cannot see.
    ax_a = fig.add_axes([0.300, 0.505, 0.480, 0.355])
    ax_b = fig.add_axes([0.055, 0.190, 0.185, 0.225])
    ax_c = fig.add_axes([0.305, 0.190, 0.185, 0.225])
    ax_d = fig.add_axes([0.550, 0.190, 0.185, 0.225])
    ax_e = fig.add_axes([0.795, 0.190, 0.185, 0.225])

    panel_map(ax_a, x9)
    panel_sweep(ax_b, m7)
    panel_sweep_annotate(ax_b, m7, x9)
    panel_dose(ax_c, x9)
    panel_decomp(ax_d, m7)
    panel_auroc(ax_e, m7)

    fig.suptitle("Accuracy tracks the evidence; confidence tracks how full the cache is",
                 fontsize=15.5, x=0.007, ha="left", y=0.985, color=INK, weight="bold")
    fig.text(0.007, 0.958, subtitle(m7, x9), fontsize=8.2, color=INK2, ha="left", va="top",
             linespacing=1.55)
    fig.text(0.007, 0.128, footer(m7), fontsize=6.9, color=INK2, ha="left", va="top",
             linespacing=1.6)

    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=180, facecolor=SURFACE)
    plt.close(fig)
    print(f"streaming headline figure -> {stem}.pdf/.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", default=str(ARTIFACTS / "m7_metrics_stream.json"))
    ap.add_argument("--comparison", default=str(ARTIFACTS / "x9_regime_comparison.json"))
    ap.add_argument("--out", default=str(FIGURES / "streaming_headline"))
    args = ap.parse_args()

    m7 = json.loads(open(args.metrics).read())["all_questions"]
    x9 = json.loads(open(args.comparison).read())
    build(m7, x9, args.out)


if __name__ == "__main__":
    main()
