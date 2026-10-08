"""X12 — the evidence-blindness figure for the E2 briefing.

Reads `x11_evidence_blindness.json` and **computes nothing**, exactly as X10 does, so the
figure cannot drift from the metrics it draws. Two stacked panels, sized for a beamer half
column:

  A  the tracks    — accuracy and confidence against Δt, with and without the evidence in the
                     window. The accuracy pair separates; the confidence pair lies on top of
                     itself. The band between each pair is the effect, drawn to scale.
  B  the contrast  — Δaccuracy and Δconfidence per bin with their intervals. Filled marks where
                     the interval excludes zero, hollow where it does not.

Kept out of `streaming_headline.pdf` on purpose: adding a panel there would change that file's
sha256 and therefore every crop the deck already takes from it.

A second, compact rendering of panel A is emitted for the slide. A crop of the full figure
would not do: the deck shows it about 2.7 in wide against the figure's authored 6.4 in, so
every label would land at a third of its intended size. The compact variant is authored at the
size it is shown, which is the only way the type comes out right. It draws the same numbers
from the same file.

Artifacts: figures/evidence_blind.{pdf,png}, figures/evidence_blind_panelA.{pdf,png}
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from ..config import ARTIFACTS, FIGURES

# X10's palette, so the two E2 figures read as one system.
C_ACC = "#2a78d6"
C_CONF = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
FAINT = "#c9c8c3"
SURFACE = "#fcfcfb"


def pp(d: dict) -> str:
    return f"{100 * d['mean']:+.1f} [{100 * d['lo95']:+.1f}, {100 * d['hi95']:+.1f}]"


def sig(d: dict) -> bool:
    """Does the 95 % interval exclude zero? Drives filled vs hollow marks."""
    return d["lo95"] > 0 or d["hi95"] < 0


def panel_tracks(ax, x11: dict) -> None:
    """A — the same four numbers per bin the table gives, drawn so the two gaps compare."""
    dts = x11["prior_dts_s"]
    x = [d / 60 for d in dts]
    by = x11["by_dt"]

    for field_a, field_b, col, lbl in (("accuracy_a", "accuracy_b", C_ACC, "accuracy"),
                                       ("confidence_a", "confidence_b", C_CONF, "confidence")):
        ya = [by[str(d)][field_a] for d in dts]
        yb = [by[str(d)][field_b] for d in dts]
        ax.fill_between(x, ya, yb, color=col, alpha=0.16, lw=0, zorder=2)
        ax.plot(x, ya, "-o", color=col, lw=2, ms=6.5, mec=SURFACE, mew=1.4, zorder=4)
        ax.plot(x, yb, ls=(0, (4, 2)), marker="o", color=col, lw=1.6, ms=5.8,
                mfc=SURFACE, mew=1.6, zorder=4)
        ax.annotate(lbl, xy=(x[0], ya[0]), xytext=(0, 9), textcoords="offset points",
                    color=col, fontsize=8.2, ha="left", va="bottom", weight="bold")

    # A two-entry key in the empty band between the two measures, rather than leader lines:
    # at these levels every anchor point sits next to something else.
    for i, (ls, mfc, txt) in enumerate(((("-"), None, "evidence in the window"),
                                        ((0, (4, 2)), SURFACE,
                                         "evidence-free window (prior arm)"))):
        y = 0.680 - 0.072 * i
        ax.plot([x[0] * 0.72, x[0] * 1.15], [y, y], ls=ls, color=MUTED, lw=1.6,
                marker="o", ms=5.2, mfc=mfc if mfc else MUTED, mec=MUTED, mew=1.4, zorder=4)
        ax.text(x[0] * 1.32, y, txt, fontsize=6.9, color=INK2, ha="left", va="center")

    ax.axhline(0.25, color=MUTED, lw=1, ls=(0, (2, 3)), zorder=1)
    ax.text(x[0] * 0.66, 0.262, "chance (4-way MCQ)", fontsize=6.8, color=MUTED,
            va="bottom", ha="left")

    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d // 60}" for d in dts])
    ax.set_xlim(x[0] * 0.60, x[-1] * 1.35)
    ax.set_ylim(0.18, 0.98)
    ax.set_xlabel("Δt — stream time from evidence to question (min, log)", fontsize=8.2)
    ax.set_ylabel("accuracy  /  mean confidence", fontsize=8.2)
    ax.tick_params(labelsize=7.8)
    ax.grid(alpha=0.13, lw=0.7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("A  Removing the evidence opens a gap in accuracy,\n"
                 "     and almost none in confidence",
                 fontsize=9.0, loc="left", color=INK)


def panel_contrast(ax, x11: dict) -> None:
    """B — the same contrast as a paired difference, with the intervals that carry it."""
    dts = x11["prior_dts_s"]
    x = [d / 60 for d in dts]
    by = x11["by_dt"]

    ax.axhline(0, color=INK, lw=1.0, zorder=3)
    for key, col, lbl, dx in (("delta_accuracy", C_ACC, "Δaccuracy", 0.955),
                              ("delta_confidence", C_CONF, "Δconfidence", 1.047)):
        xs = [v * dx for v in x]
        ys = [100 * by[str(d)][key]["mean"] for d in dts]
        for xi, d in zip(xs, dts):
            b = by[str(d)][key]
            ax.plot([xi, xi], [100 * b["lo95"], 100 * b["hi95"]], color=col, lw=1.6,
                    solid_capstyle="round", zorder=4)
        filled = [sig(by[str(d)][key]) for d in dts]
        for xi, yi, f in zip(xs, ys, filled):
            ax.plot([xi], [yi], "o", color=col, ms=6.4, mfc=col if f else SURFACE,
                    mec=col, mew=1.6, zorder=5)
        ax.plot(xs, ys, "-", color=col, lw=1.1, alpha=0.55, zorder=3)
        ax.text(xs[-1] * 1.06, ys[-1], lbl, color=col, fontsize=8.2, ha="left",
                va="center", weight="bold")

    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d // 60}" for d in dts])
    ax.set_xlim(x[0] * 0.60, x[-1] * 2.35)
    ax.set_xlabel("Δt — stream time from evidence to question (min, log)", fontsize=8.2)
    ax.set_ylabel("paired Δ, main − prior (pp)", fontsize=8.2)
    ax.tick_params(labelsize=7.8)
    ax.grid(alpha=0.13, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("B  Filled = interval excludes zero. Every accuracy mark\n"
                 "     is filled; no confidence mark is",
                 fontsize=9.0, loc="left", color=INK)


def panel_tracks_compact(ax, x11: dict) -> None:
    """Panel A again, laid out for the size a slide actually shows it at.

    Same series, same numbers; the title, the legend block and the chance annotation are
    dropped or shortened because at 3.3 in wide there is no room for them and the slide
    supplies that text itself.
    """
    dts = x11["prior_dts_s"]
    x = [d / 60 for d in dts]
    by = x11["by_dt"]

    for field_a, field_b, col, lbl in (("accuracy_a", "accuracy_b", C_ACC, "accuracy"),
                                       ("confidence_a", "confidence_b", C_CONF, "confidence")):
        ya = [by[str(d)][field_a] for d in dts]
        yb = [by[str(d)][field_b] for d in dts]
        ax.fill_between(x, ya, yb, color=col, alpha=0.18, lw=0, zorder=2)
        ax.plot(x, ya, "-o", color=col, lw=2.2, ms=6.0, mec=SURFACE, mew=1.3, zorder=4)
        ax.plot(x, yb, ls=(0, (3, 2)), marker="o", color=col, lw=1.8, ms=5.4,
                mfc=SURFACE, mew=1.5, zorder=4)
        ax.annotate(lbl, xy=(x[0], ya[0]), xytext=(-2, 8), textcoords="offset points",
                    color=col, fontsize=8.6, ha="left", va="bottom", weight="bold")

    # a two-row key in the empty band between the two measures: at this size the gap between
    # the accuracy pair is only ~15 pt, too tight to hang two labels off the lines themselves
    for i, (ls, mfc, txt) in enumerate((("-", MUTED, "evidence in window"),
                                        ((0, (3, 2)), SURFACE, "no evidence"))):
        y = 0.655 - 0.090 * i
        ax.plot([x[0] * 1.45, x[0] * 2.75], [y, y], ls=ls, color=MUTED, lw=1.7,
                marker="o", ms=4.8, mfc=mfc, mec=MUTED, mew=1.4, zorder=4)
        ax.text(x[0] * 3.25, y, txt, fontsize=7.2, color=INK2, ha="left", va="center")

    ax.axhline(0.25, color=MUTED, lw=1, ls=(0, (2, 3)), zorder=1)
    ax.text(x[0] * 0.72, 0.242, "chance", fontsize=7.0, color=MUTED, va="top", ha="left")

    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d // 60}" for d in dts])
    ax.set_xlim(x[0] * 0.66, x[-1] * 1.30)
    ax.set_ylim(0.15, 1.00)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("Δt from evidence to question (min, log)", fontsize=8.2)
    ax.set_ylabel("accuracy / confidence", fontsize=8.2)
    ax.tick_params(labelsize=8.0)
    ax.grid(alpha=0.13, lw=0.7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def build_panel_a(x11: dict, stem) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(3.3, 2.8))
    panel_tracks_compact(fig.add_axes([0.175, 0.175, 0.800, 0.800]), x11)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=180, facecolor=SURFACE)
    plt.close(fig)
    print(f"evidence-blindness panel A (slide size) -> {stem}.pdf/.png")


def subtitle(x11: dict) -> str:
    dts = x11["prior_dts_s"]
    by = x11["by_dt"]
    accs = [by[str(d)]["delta_accuracy"]["mean"] for d in dts]
    n = by[str(dts[0])]["n_questions"]
    return textwrap.fill(
        f"Paired within each of {n} questions and token-matched — same duration, same frame count, "
        f"same presented and live token counts. The two windows differ only in whether the "
        f"evidence is inside. The accuracy effect varies {max(accs) / min(accs):.1f}× across "
        f"the bins; the confidence effect follows none of that variation and crosses zero at "
        f"every one of them.", width=128)


def footer(x11: dict) -> str:
    det = x11["detectors"]["pooled"]
    dts = x11["prior_dts_s"]
    ov = " / ".join(f"{x11['by_dt'][str(d)]['confidence_overlap']:.2f}" for d in dts)
    kind = x11["reference_manipulations"]["control_vs_memory"]["delta_confidence"]
    return textwrap.fill(
        f"Confidence distributions overlap {ov} across the bins, and confidence separates the "
        f"two conditions at AUROC {det['confidence']['mean']:.3f} "
        f"[{det['confidence']['lo95']:.3f}, {det['confidence']['hi95']:.3f}] on "
        f"{det['n_pairs']} pairs, against 0.500 for chance. The probe is not inert: the same "
        f"measure moves {pp(kind)} on question kind. Confidence is the option-letter softmax, "
        f"not a calibrated probability.", width=130) + (
        "\nDrawn by egovm/streaming/x12_evidence_figure.py from x11_evidence_blindness.json; "
        "no number is computed here.")


def build(x11: dict, stem) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6.4, 7.2))
    # explicit rectangles rather than a gridspec: the title and footer blocks are fixed
    # heights that tight_layout cannot see.
    ax_a = fig.add_axes([0.125, 0.570, 0.840, 0.280])
    ax_b = fig.add_axes([0.125, 0.212, 0.840, 0.198])

    panel_tracks(ax_a, x11)
    panel_contrast(ax_b, x11)

    fig.suptitle("Accuracy sees the evidence; confidence does not",
                 fontsize=12.8, x=0.016, ha="left", y=0.984, color=INK, weight="bold")
    fig.text(0.016, 0.955, subtitle(x11), fontsize=6.6, color=INK2, ha="left", va="top",
             linespacing=1.5)
    fig.text(0.016, 0.140, footer(x11), fontsize=6.0, color=INK2, ha="left", va="top",
             linespacing=1.55)

    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=180, facecolor=SURFACE)
    plt.close(fig)
    print(f"evidence-blindness figure -> {stem}.pdf/.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", default=str(ARTIFACTS / "x11_evidence_blindness.json"))
    ap.add_argument("--out", default=str(FIGURES / "evidence_blind"))
    a = ap.parse_args()
    x11 = json.loads(Path(a.metrics).read_text())
    build(x11, a.out)
    build_panel_a(x11, f"{a.out}_panelA")


if __name__ == "__main__":
    main()
