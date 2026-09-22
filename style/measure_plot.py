#!/usr/bin/env python3
"""Print-readiness gate for result figures (IEEE Transactions).

Checks the two things that silently ruin an otherwise correct plot:

  * **width** — the artwork must already be at column width, so LaTeX does not
    rescale it. A 7in plot dropped into a 3.5in column is reproduced at 0.5x and
    every label halves with it.
  * **smallest text** — measured from the PDF itself, not from the rc settings,
    because tick labels and annotations often bypass them.

    python measure_plot.py fig.pdf [--double-column]
"""
import argparse
import sys
from collections import Counter

import fitz

COL_SINGLE = 252.0   # 3.5 in
COL_DOUBLE = 516.0   # 7.16 in
WIDTH_TOL = 0.06     # +/-6% of the target column width
MIN_PT = 7.0         # smallest acceptable text on paper


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--double-column", action="store_true")
    ap.add_argument("--min-pt", type=float, default=MIN_PT)
    args = ap.parse_args()

    target = COL_DOUBLE if args.double_column else COL_SINGLE
    label = "double" if args.double_column else "single"

    doc = fitz.open(args.pdf)
    page = doc[0]
    w, h = page.rect.width, page.rect.height

    sizes = Counter()
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("text", "").strip():
                    sizes[round(span["size"], 1)] += len(span["text"].strip())

    scale = target / w if w else 0.0
    print(f"file        : {args.pdf}")
    print(f"artwork     : {w:.1f} x {h:.1f} pt   (aspect {w/h:.2f})")
    print(f"target      : {label} column = {target:.0f} pt  ->  placed at {scale:.2f}x")

    fails = []
    if abs(w - target) / target > WIDTH_TOL:
        verb = "wider" if w > target else "narrower"
        fails.append(
            f"WIDTH {w:.0f}pt is {verb} than the {label} column ({target:.0f}pt). "
            f"LaTeX will rescale by {scale:.2f}x and every label scales with it. "
            f"Set figsize to {target/72:.2f}in wide instead of resizing later."
        )

    if not sizes:
        print("text sizes  : (none found — all text may be outlined)")
        fails.append("NO TEXT FOUND: fonts appear outlined; keep pdf.fonttype 42 so "
                     "text stays selectable and measurable.")
    else:
        print("text sizes  : " + ", ".join(
            f"{s}pt x{n}" for s, n in sorted(sizes.items())))

        # Judge the smallest size that carries real content. Mathtext sub/superscripts
        # render at ~70% of their parent, so a lone 4.9pt star inside a 7pt label is
        # normal typography, not a legibility defect — failing on it would train people
        # to avoid math in labels. A size only counts if it carries >=2% of the glyphs.
        total = sum(sizes.values())
        body = {s: n for s, n in sizes.items() if n / total >= 0.02}
        smallest = min(body) if body else min(sizes)
        absolute = min(sizes)
        on_paper = smallest * scale
        if absolute < smallest:
            print(f"            : ignoring {absolute}pt x{sizes[absolute]} "
                  f"(<2% of glyphs — sub/superscript)")
        print(f"smallest    : {smallest}pt in artwork  ->  {on_paper:.1f}pt on paper "
              f"(floor {args.min_pt}pt)")
        # 0.1pt slack: an artwork 1-2pt off the column width scales 7pt to ~6.96pt,
        # which is a rounding artefact, not a legibility problem. The check exists to
        # catch the 3.7pt disasters, so do not let float noise fire it.
        if on_paper < args.min_pt - 0.1:
            worst = sizes[smallest]
            fails.append(
                f"TEXT TOO SMALL: {smallest}pt x{worst} chars lands at {on_paper:.1f}pt. "
                f"Raise it, or widen the artwork so no rescale is needed."
            )

    print()
    if fails:
        print("PRINT-READY: FAIL")
        for f in fails:
            print("  - " + f)
        return 1
    print("PRINT-READY: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
