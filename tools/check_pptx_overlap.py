#!/usr/bin/env python3
"""Sanity-check generated PPTX decks for layout overlaps and off-slide content.

The native builder (tools/spec2pptx.py) flows most content but places callouts
at fixed Y positions, so a tall figure/table can collide with the callout below
it, and content can run past the bottom edge. This checker opens each .pptx and
flags the two failure modes that actually bite:

  A. OFF-SLIDE  — any shape whose bottom edge runs past the slide bottom.
  B. COLLISION  — the callout box overlapping the figure / table / caption above
                  it (the classic "boxes overlapping" bug).

It also emits softer WARNs for figure/figure and callout/text overlaps that are
usually — but not always — intentional (e.g. amber annotation boxes drawn on an
image are excluded automatically).

Usage:
    python tools/check_pptx_overlap.py slides/pptx/lecture05_cnns.pptx
    python tools/check_pptx_overlap.py --all        # every deck in slides/pptx/
Exit code is nonzero if any FAIL is found, so it can gate a build.
"""

import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Emu
from pptx.enum.shapes import MSO_SHAPE_TYPE

ROOT = Path(__file__).resolve().parent.parent
PPTX_DIR = ROOT / "slides" / "pptx"

EMU_PER_IN = 914400.0
TOL = 0.04            # inches of slack before we call it an overlap
EDGE_TOL = 0.03       # inches a shape may poke past the slide edge


def _in(v):
    return (v or 0) / EMU_PER_IN


class Box:
    __slots__ = ("name", "kind", "x0", "y0", "x1", "y1")

    def __init__(self, name, kind, left, top, width, height):
        self.name = name
        self.kind = kind
        self.x0 = _in(left)
        self.y0 = _in(top)
        self.x1 = _in(left) + _in(width)
        self.y1 = _in(top) + _in(height)

    @property
    def w(self):
        return self.x1 - self.x0

    @property
    def h(self):
        return self.y1 - self.y0

    def contains(self, o, frac=0.6):
        ix = max(0.0, min(self.x1, o.x1) - max(self.x0, o.x0))
        iy = max(0.0, min(self.y1, o.y1) - max(self.y0, o.y0))
        inter = ix * iy
        oa = max(o.w * o.h, 1e-6)
        return inter / oa >= frac

    def overlaps(self, o, tol=TOL):
        return (
            self.x1 - o.x0 > tol
            and o.x1 - self.x0 > tol
            and self.y1 - o.y0 > tol
            and o.y1 - self.y0 > tol
        )


def classify(shape):
    """Return (kind, Box) or None for shapes we ignore (background, connectors)."""
    try:
        left, top, width, height = shape.left, shape.top, shape.width, shape.height
    except Exception:
        return None
    if left is None or top is None or not width or not height:
        return None
    b = Box(shape.name or "", "?", left, top, width, height)

    st = shape.shape_type
    # Full-slide background rectangle — ignore.
    if b.x0 <= 0.01 and b.y0 <= 0.01 and b.w >= 13.2 and b.h >= 7.4:
        return None
    if st == MSO_SHAPE_TYPE.PICTURE:
        b.kind = "picture"
        return b
    if getattr(shape, "has_table", False):
        b.kind = "table"
        return b
    if st == MSO_SHAPE_TYPE.LINE:
        return None
    # Callout signature from _callout(): a ~1.0in-tall wide rectangle low on the slide.
    if 0.8 <= b.h <= 1.35 and b.w >= 5.0:
        b.kind = "callout"
        return b
    if getattr(shape, "has_text_frame", False) and (shape.text_frame.text or "").strip():
        b.kind = "text"
        return b
    b.kind = "shape"
    return b


def check_slide(slide, idx, slide_h):
    boxes = [b for b in (classify(s) for s in slide.shapes) if b is not None]
    pics = [b for b in boxes if b.kind == "picture"]
    fails, warns = [], []

    # Exclude annotation shapes drawn on top of a picture (amber boxes/labels).
    def is_annotation(b):
        return b.kind in ("shape", "text") and any(p.contains(b) for p in pics)

    active = [b for b in boxes if not is_annotation(b)]

    # A. off-slide
    for b in active:
        if b.y1 > slide_h + EDGE_TOL:
            fails.append(f"OFF-SLIDE: {b.kind} bottom={b.y1:.2f}in > slide {slide_h:.2f}in")

    # B. callout collisions + figure overlaps
    callouts = [b for b in active if b.kind == "callout"]
    figs = [b for b in active if b.kind in ("picture", "table")]
    texts = [b for b in active if b.kind == "text"]
    for c in callouts:
        for f in figs:
            if c.overlaps(f):
                fails.append(
                    f"COLLISION: callout (top={c.y0:.2f}) overlaps {f.kind} "
                    f"(bottom={f.y1:.2f})"
                )
        for t in texts:
            if c.overlaps(t):
                warns.append(
                    f"callout (top={c.y0:.2f}) overlaps text (bottom={t.y1:.2f})"
                )
    # figure/figure overlap (two-col images are in separate columns → shouldn't hit)
    for i in range(len(figs)):
        for j in range(i + 1, len(figs)):
            if figs[i].overlaps(figs[j]):
                warns.append(f"{figs[i].kind}/{figs[j].kind} overlap")
    return fails, warns


def check_deck(path):
    prs = Presentation(str(path))
    slide_h = _in(prs.slide_height)
    n_fail = n_warn = 0
    slides = list(prs.slides)
    for i, slide in enumerate(slides):
        fails, warns = check_slide(slide, i, slide_h)
        for m in fails:
            print(f"  FAIL  {path.name} slide {i + 1}: {m}")
            n_fail += 1
        for m in warns:
            print(f"  warn  {path.name} slide {i + 1}: {m}")
            n_warn += 1
    status = "FAIL" if n_fail else ("warn" if n_warn else "ok")
    print(f"[{status}] {path.name}: {n_fail} fail, {n_warn} warn, {len(slides)} slides")
    return n_fail


def main(argv):
    args = argv[1:]
    if not args or args[0] == "--all":
        paths = sorted(p for p in PPTX_DIR.glob("*.pptx") if not p.name.startswith("~$"))
    else:
        paths = [Path(a) for a in args if not Path(a).name.startswith("~$")]
    if not paths:
        print("no .pptx found")
        return 1
    total_fail = 0
    for p in paths:
        if not p.exists():
            print(f"  missing: {p}")
            total_fail += 1
            continue
        total_fail += check_deck(p)
    print(f"\n{'FAIL' if total_fail else 'PASS'}: {total_fail} overlap failure(s) across {len(paths)} deck(s)")
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
