#!/usr/bin/env python3
"""Sanity-check the figure toolkit for text that overflows a hand-drawn box.

This is a DIFFERENT bug from the PowerPoint-box overflow checked by
tools/check_text_overflow.py. Here the offending text is baked INTO a matplotlib
figure PNG: a figure function draws a fixed-size box patch (FancyBboxPatch /
Rectangle) and then draws a SEPARATE ax.text(...) centred over it, at a font size
whose rendered width/height exceeds the patch -- so the text spills outside the
box in the final PNG (e.g. ce_formula's "so only the TRUE class survives" box).

How it works (non-invasive): it imports tools.make_slide_figures, monkeypatches
the module's `_save` so that, right before each real figure is written and
closed, we draw the Agg canvas, measure every Text artist's rendered pixel
extent, and for each Text find the enclosing hand-drawn box patch (a
FancyBboxPatch or Rectangle whose pixel bounds contain the text's anchor point).
A text that has its OWN auto-fitting bbox (t.get_bbox_patch()) is skipped -- that
box always fits by construction. If an enclosing patch exists and the text's
window_extent is not contained within the patch's window_extent (minus TOL_PX on
each side), it is FLAGGED.

The DEFINING symptom of this bug is horizontal: the rendered text is WIDER than
the box that is supposed to wrap it, so it spills past the left/right edge. We
flag only that -- a text whose pixel width exceeds its enclosing box's width by
more than TOL_PX. This deliberately ignores two look-alikes that are NOT the
bug: (a) text sitting near the edge of a large CONTAINER panel / axes-background
rectangle (the text fits the width, it is just tall or near an edge), and
(b) LaTeX-math labels whose bounding box is inflated vertically by \sum /
superscripts while the glyphs fit comfortably inside the box.

It then runs the toolkit's real build_all() so every figure is exercised exactly
as it is normally generated (correct args, correct PNG name) -- but figures are
NOT written to disk during the check (the monkeypatched _save closes the fig
without saving), so running the checker never mutates slides/assets/img/.

Usage:
    python tools/check_figure_overflow.py            # check every figure
Exit code is nonzero if any box overflows its text.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import make_slide_figures as msf  # noqa: E402

# Tolerance: how many pixels a text may poke past its box before we flag it.
# 3px absorbs antialiasing / rounding; real overflows here are tens of pixels.
TOL_PX = 3.0

# The figure paper background (matplotlib RGBA of msf.PAPER "#FBFBF8"). A
# full-span rectangle in this colour is the AXES BACKGROUND, not a wrap-box, so
# a free caption drawn over the paper must never be treated as "inside a box."
PAPER_RGBA = (0.98, 0.98, 0.97)


def _is_wrapbox(patch, axbb, renderer):
    """A patch counts as a text-wrapping box only if it is NOT the axes-background
    paper panel: it must not be paper-coloured AND not span (almost) the full
    axes width. Real wrap-boxes have a distinct fill (grey/teal/pink/white card)
    and are narrower than the whole plot."""
    try:
        fc = patch.get_facecolor()
    except Exception:
        fc = None
    if fc is not None and patch.get_fill():
        if all(abs(fc[k] - PAPER_RGBA[k]) < 0.015 for k in range(3)):
            return False
    try:
        bb = patch.get_window_extent(renderer=renderer)
    except Exception:
        return True
    if axbb.width > 1 and bb.width >= 0.95 * axbb.width:
        return False
    return True

_flags = []  # (png_name, text_snippet, side->overflow_px, text_bbox, patch_bbox)


def _bbox_contains_point(bb, x, y):
    return bb.x0 <= x <= bb.x1 and bb.y0 <= y <= bb.y1


def _snippet(s, n=48):
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "\u2026"


def _check_fig(fig, name):
    """Measure every Text against any hand-drawn box that encloses it."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for ax in fig.get_axes():
        axbb = ax.get_window_extent(renderer=renderer)
        texts = [c for c in ax.get_children() if isinstance(c, Text)]
        boxes = [
            c
            for c in ax.get_children()
            if isinstance(c, (FancyBboxPatch, Rectangle))
            and _is_wrapbox(c, axbb, renderer)
        ]
        for t in texts:
            if not t.get_text().strip():
                continue
            if not t.get_visible():
                continue
            # A text with its OWN auto-fitting bbox always fits -> skip.
            if t.get_bbox_patch() is not None:
                continue
            try:
                tbb = t.get_window_extent(renderer=renderer)
            except Exception:
                continue
            tcx = 0.5 * (tbb.x0 + tbb.x1)
            tcy = 0.5 * (tbb.y0 + tbb.y1)

            # Find the SMALLEST hand-drawn box whose pixel bounds contain the
            # text's centre point (the intended enclosing box).
            enclosing = None
            enc_area = None
            for b in boxes:
                if not b.get_visible():
                    continue
                try:
                    bbb = b.get_window_extent(renderer=renderer)
                except Exception:
                    continue
                if bbb.width <= 1 or bbb.height <= 1:
                    continue
                if _bbox_contains_point(bbb, tcx, tcy):
                    area = bbb.width * bbb.height
                    if enc_area is None or area < enc_area:
                        enclosing, enc_area = bbb, area
            if enclosing is None:
                continue

            # The bug is HORIZONTAL: text wider than the box that wraps it.
            # Require the text to actually be wider than the box (the defining
            # symptom) before flagging -- this excludes container panels and
            # tall math bounding boxes whose glyphs fit the width fine.
            if tbb.width <= enclosing.width + 2 * TOL_PX:
                continue
            over = {
                "left": enclosing.x0 - tbb.x0,
                "right": tbb.x1 - enclosing.x1,
            }
            worst = max(over.values())
            if worst > TOL_PX:
                sides = {k: round(v, 1) for k, v in over.items() if v > TOL_PX}
                _flags.append(
                    (
                        name,
                        _snippet(t.get_text()),
                        sides,
                        (round(tbb.x0), round(tbb.y0), round(tbb.x1), round(tbb.y1)),
                        (
                            round(enclosing.x0),
                            round(enclosing.y0),
                            round(enclosing.x1),
                            round(enclosing.y1),
                        ),
                    )
                )


def main():
    checked = {"n": 0}
    real_save = msf._save

    def fake_save(fig, name, dpi=150):
        checked["n"] += 1
        try:
            _check_fig(fig, name)
        finally:
            plt.close(fig)

    msf._save = fake_save
    # Silence the per-figure "wrote ..." spam from build_all's own prints by
    # keeping them -- they help show progress; but they say "wrote" even though
    # we do not write. Suppress to avoid confusion.
    import builtins

    real_print = builtins.print

    def quiet_print(*a, **k):
        msg = " ".join(str(x) for x in a)
        if msg.strip().startswith("wrote "):
            return
        real_print(*a, **k)

    builtins.print = quiet_print
    try:
        msf.build_all()
    finally:
        builtins.print = real_print
        msf._save = real_save

    real_print(f"\nchecked {checked['n']} figure render(s)")
    if not _flags:
        real_print("PASS: 0 overflowing box(es)")
        return 0

    real_print(f"\nFAIL: {len(_flags)} overflowing box(es):")
    for name, snip, sides, tbb, pbb in _flags:
        side_str = ", ".join(f"{k} +{v}px" for k, v in sides.items())
        real_print(f"  [{name}] \"{snip}\"")
        real_print(f"      overflow: {side_str}")
        real_print(f"      text  bbox px: {tbb}")
        real_print(f"      patch bbox px: {pbb}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
