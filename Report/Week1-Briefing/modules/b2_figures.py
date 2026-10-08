#!/usr/bin/env python3
"""B2 -- the figure assets the deck includes.

The two headline figures are far too wide and dense for a 16:9 slide, so this module
cuts each one into the panels the deck argues from.  Nothing is redrawn: a crop is the
same page with a narrower CropBox, so a panel on a slide is bit-for-bit the panel in
the committed figure and cannot drift from it.

Emits
    figures/*.pdf                 one file per panel, plus the two full figures
    figures/figures_manifest.json source sha256, crop box, output size, output sha256

--check re-cuts into a temporary directory and fails if anything differs, so a
regenerated headline figure is caught rather than silently reused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
BRIEF = HERE.parent
REPO = BRIEF.parent.parent
SRC = REPO / "Teaser-Experiment" / "figures"
OUT = BRIEF / "figures"

# Crops are given as fractions of the source page: (x0, x1) from the left,
# (y0, y1) from the *top*, which is how the figures read.
PANELS: dict[str, tuple[str, tuple[float, float, float, float], str]] = {
    # name:              (source,               x0,    x1,    ytop0, ytop1, caption)
    "e1-full": ("headline.pdf", (0.000, 1.000, 0.000, 1.000), "E1 headline, whole figure"),
    "e1-head": ("headline.pdf", (0.000, 1.000, 0.000, 0.135), "E1 title and subtitle"),
    "e1-a": ("headline.pdf", (0.018, 0.335, 0.140, 1.000), "E1 panel A -- pin the context, the accuracy decline goes away"),
    "e1-b": ("headline.pdf", (0.330, 0.722, 0.140, 1.000), "E1 panel B -- what each manipulation moves"),
    "e1-c": ("headline.pdf", (0.715, 1.000, 0.140, 0.500), "E1 panel C upper -- kept vs lost items"),
    "e1-d": ("headline.pdf", (0.715, 1.000, 0.495, 1.000), "E1 panel C lower -- AUROC across the sweep"),
    "e2-full": ("streaming_headline.pdf", (0.000, 1.000, 0.000, 1.000), "E2 headline, whole figure"),
    "e2-head": ("streaming_headline.pdf", (0.000, 1.000, 0.000, 0.100), "E2 title and subtitle"),
    "e2-a": ("streaming_headline.pdf", (0.000, 1.000, 0.098, 0.535), "E2 panel A -- what each manipulation moves"),
    "e2-b": ("streaming_headline.pdf", (0.035, 0.255, 0.535, 0.868), "E2 panel B -- the raw sweep settles nothing"),
    "e2-c": ("streaming_headline.pdf", (0.252, 0.500, 0.535, 0.868), "E2 panel C -- a bigger cache buys no accuracy"),
    "e2-d": ("streaming_headline.pdf", (0.505, 0.762, 0.535, 0.868), "E2 panel D -- the decline is all context"),
    "e2-e": ("streaming_headline.pdf", (0.745, 1.000, 0.535, 0.868), "E2 panel E -- the confidence ordering decays"),
    # X12's standalone figure. Kept out of streaming_headline.pdf so that figure's sha256 --
    # and every crop above that depends on it -- is untouched by this addition.
    "e2-ev": ("evidence_blind.pdf", (0.000, 1.000, 0.000, 1.000), "E2 evidence blindness, whole figure"),
    "e2-ev-plots": ("evidence_blind.pdf", (0.000, 1.000, 0.098, 0.845), "E2 evidence blindness, both panels without the caption blocks"),
    # authored at slide size rather than cropped -- see x12_evidence_figure.py's docstring
    "e2-ev-slide": ("evidence_blind_panelA.pdf", (0.000, 1.000, 0.000, 1.000), "E2 evidence blindness panel A, authored at the size the deck shows it"),
}


def sha256(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def page_size(p: pathlib.Path) -> tuple[float, float]:
    out = subprocess.run(["pdfinfo", str(p)], capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if line.startswith("Page size:"):
            parts = line.split()
            return float(parts[2]), float(parts[4])
    raise SystemExit(f"B2: no page size for {p}")


def cut(src: pathlib.Path, box: tuple[float, float, float, float], dst: pathlib.Path) -> list[float]:
    w, h = page_size(src)
    x0, x1, yt0, yt1 = box
    # PDF y runs from the bottom; the spec gives distances from the top.
    crop = [round(x0 * w, 2), round(h - yt1 * h, 2), round(x1 * w, 2), round(h - yt0 * h, 2)]
    pdfmark = f"[/CropBox [{crop[0]} {crop[1]} {crop[2]} {crop[3]}] /PAGES pdfmark"
    subprocess.run(
        ["gs", "-q", "-o", str(dst), "-sDEVICE=pdfwrite", "-dQUIET",
         "-dAutoRotatePages=/None", "-c", pdfmark, "-f", str(src)],
        check=True, capture_output=True,
    )
    return crop


def generate(outdir: pathlib.Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {"source_figures": {}, "panels": {}}
    for src_name in sorted({v[0] for v in PANELS.values()}):
        src = SRC / src_name
        if not src.exists():
            raise SystemExit(f"B2: missing source figure {src}")
        w, h = page_size(src)
        manifest["source_figures"][src_name] = {
            "sha256": sha256(src), "width_pt": w, "height_pt": h,
            "path": str(src.relative_to(REPO)),
        }
    for name, (src_name, box, caption) in PANELS.items():
        dst = outdir / f"{name}.pdf"
        crop = cut(SRC / src_name, box, dst)
        w, h = page_size(dst)
        manifest["panels"][name] = {
            "source": src_name, "caption": caption,
            "crop_box_pt": crop, "width_pt": w, "height_pt": h,
            "aspect": round(w / h, 3), "sha256": sha256(dst),
        }
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    mpath = OUT / "figures_manifest.json"

    if args.check:
        if not mpath.exists():
            print("B2 FAIL: no figures_manifest.json -- run `make figures`", file=sys.stderr)
            return 1
        have = json.loads(mpath.read_text())
        with tempfile.TemporaryDirectory() as td:
            fresh = generate(pathlib.Path(td))
        if have["source_figures"] != fresh["source_figures"]:
            print("B2 FAIL: a source headline figure changed -- run `make figures`", file=sys.stderr)
            return 1
        missing = [n for n in fresh["panels"] if not (OUT / f"{n}.pdf").exists()]
        if missing:
            print(f"B2 FAIL: missing panel pdfs: {', '.join(missing)}", file=sys.stderr)
            return 1
        drift = [n for n, v in fresh["panels"].items()
                 if have["panels"].get(n, {}).get("crop_box_pt") != v["crop_box_pt"]]
        if drift:
            print(f"B2 FAIL: crop boxes drifted: {', '.join(drift)}", file=sys.stderr)
            return 1
        print(f"B2 ok: {len(fresh['panels'])} panels current against {len(fresh['source_figures'])} source figures")
        return 0

    if OUT.exists():
        for stale in OUT.glob("*.pdf"):
            stale.unlink()
    manifest = generate(OUT)
    mpath.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"B2: cut {len(manifest['panels'])} panels into {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
