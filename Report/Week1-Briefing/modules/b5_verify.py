#!/usr/bin/env python3
"""B5 -- check the built deck against everything it was built from.

Six checks, each of which has caught something at least once:
  1. B1 is current       -- no slide quotes a number the artifacts no longer support
  2. B2 is current       -- no panel is a crop of a headline figure that has since changed
  3. macros all defined  -- every \\Macro a slide uses exists in the preamble or in B1's output
  4. figures all present -- every \\includegraphics target exists on disk
  5. the pdf built       -- and has the expected structure
  6. the log is clean    -- no LaTeX errors, no undefined references, no bad overfull boxes
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
BRIEF = HERE.parent
PDF = BRIEF / "build" / "week1-briefing.pdf"
LOG = BRIEF / "build" / "week1-briefing.log"
MIN_PAGES = 25
MAX_OVERFULL_PT = 12.0   # anything larger is visible on a slide

FAILS: list[str] = []
NOTES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def run_module(script: str) -> tuple[bool, str]:
    r = subprocess.run([sys.executable, str(HERE / script), "--check"], capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout or r.stderr) else ""


def main() -> int:
    print("B5 verifying the Week 1 briefing deck\n")

    ok, msg = run_module("b1_numbers.py")
    check("B1 numbers are current", ok, msg)
    ok, msg = run_module("b2_figures.py")
    check("B2 figure panels are current", ok, msg)

    # -- macros
    defined: set[str] = set()
    for f in [BRIEF / "preamble.tex", BRIEF / "generated" / "numbers.tex", BRIEF / "generated" / "tables.tex"]:
        defined |= set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", f.read_text(encoding="utf-8")))
    used: dict[str, str] = {}
    for f in sorted((BRIEF / "slides").glob("*.tex")):
        body = f.read_text(encoding="utf-8")
        # only our own macros: they all start with an upper-case letter
        for m in re.findall(r"\\([A-Z][A-Za-z]*)\b", body):
            used.setdefault(m, f.name)
    latex_builtins = {
        "TeX", "LaTeX", "Delta", "Large", "Huge", "Big", "Bigg", "Bigl", "Bigr",
        "S", "P", "AtBeginSection", "Roman", "Alph",
    }
    unknown = sorted(m for m in used if m not in defined and m not in latex_builtins)
    check("every macro a slide uses is defined", not unknown,
          ("undefined: " + ", ".join(f"\\{m} ({used[m]})" for m in unknown)) if unknown else f"{len(used)} macros used")

    # -- figures
    missing = []
    for f in sorted((BRIEF / "slides").glob("*.tex")):
        for target in re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", f.read_text(encoding="utf-8")):
            if not (BRIEF / target).exists():
                missing.append(f"{target} ({f.name})")
    check("every included figure exists", not missing, ", ".join(missing))

    # -- each slide file compiles on its own
    check("each slide file compiles alone", *standalone())

    # -- the pdf itself
    if not PDF.exists():
        check("the pdf built", False, "run `make`")
        return report()
    info = subprocess.run(["pdfinfo", str(PDF)], capture_output=True, text=True).stdout
    pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
    check("the pdf built", pages >= MIN_PAGES, f"{pages} pages")

    # -- the log
    log = LOG.read_text(errors="ignore")
    errors = [l for l in log.splitlines() if l.startswith("! ")]
    check("no LaTeX errors", not errors, "; ".join(errors[:3]))
    undef = re.findall(r"LaTeX Warning: (Reference|Citation)[^\n]*", log)
    check("no undefined references", not undef, "; ".join(undef[:3]))
    bad = [(p, s) for p, s in overfull(log) if s > MAX_OVERFULL_PT]
    check(f"no overfull box above {MAX_OVERFULL_PT:.0f}pt", not bad,
          ", ".join(f"p{p} ({s:.0f}pt)" for p, s in bad[:6]))

    return report()


def standalone() -> tuple[bool, str]:
    """Each B4 file is a module: it must build without the others."""
    out = BRIEF / "build" / "standalone"
    out.mkdir(parents=True, exist_ok=True)
    broken = []
    for f in sorted((BRIEF / "slides").glob("*.tex")):
        wrapper = out / f"{f.stem}.tex"
        wrapper.write_text(
            "\\documentclass[aspectratio=169,10pt]{beamer}\n"
            "\\input{preamble.tex}\n"
            "\\title{t}\\subtitle{s}\\author{a}\\date{d}\n"
            "\\begin{document}\n"
            f"\\input{{slides/{f.name}}}\n"
            "\\end{document}\n",
            encoding="utf-8",
        )
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             f"-output-directory={out}", str(wrapper)],
            cwd=BRIEF, capture_output=True, text=True,
        )
        if r.returncode != 0:
            broken.append(f.name)
    return not broken, (", ".join(broken) if broken else f"{len(list((BRIEF / 'slides').glob('*.tex')))} files")


def overfull(log: str) -> list[tuple[int, float]]:
    page, out = 0, []
    for m in re.finditer(r"(\[(\d+)[^\]]*\]|Overfull \\[hv]box \(([\d.]+)pt)", log):
        if m.group(2):
            page = int(m.group(2))
        else:
            out.append((page + 1, float(m.group(3))))
    return out


def report() -> int:
    print()
    if FAILS:
        print(f"B5 FAIL: {len(FAILS)} check(s) failed: {', '.join(FAILS)}")
        return 1
    print("B5 ok: the deck is consistent with every artifact it was built from")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
