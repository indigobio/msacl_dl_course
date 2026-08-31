#!/usr/bin/env python3
"""Build NATIVE, editable PowerPoint decks from a compact YAML spec.

Why this exists (replaces the html->screenshot pipeline):
  - the old tools/html2pptx.py rendered each HTML slide to a PNG and pasted it
    full-bleed, so the .pptx was flat images: not editable, web-styled, and
    every deck was 700-900 lines of hand-written HTML/CSS/SVG.
  - this builder reads a small YAML deck spec (~10-15 lines per slide) and emits
    real PowerPoint objects: native text boxes, tables, shapes, connectors and
    placed images. Titles/body stay editable in PowerPoint; look is driven by a
    shared theme here, not CSS. Real licensed images live under slides/assets/img
    and are placed with `image:`; hand-drawn SVG is used only where rule 6 (the
    numeric mechanics) truly needs it.

Spec contract (see slides/spec/*.yaml for a worked example):

    deck:
      title: "Lecture 5 ..."           # core doc title
      eyebrow: "MSACL · DS301 ..."     # default eyebrow shown top-left
    slides:
      - type: title
        eyebrow: "..."                  # overrides deck.eyebrow
        title: "Big title"
        subtitle: "one line"
        notes: "speaker notes"

      - type: bullets
        eyebrow: "..."
        title: "Slide title"
        bullets:                        # str, or {text, time, sub}
          - text: "Why CNNs fit spectra"
            time: "8 min"
        callout: {kind: ms, text: "..."}   # kind: ms|check|pair|plain
        notes: "..."

      - type: two-col
        title: "..."
        left:  {image: img/sparrow.jpg, caption: "photo: ..., CC BY-SA 2.0"}
        right: {lead: ["para 1", "para 2"], callout: {kind: ms, text: "..."}}
        notes: "..."

      - type: image                    # one big image + caption + optional callout
        title: "..."
        image: img/vgg16.png
        img_h: 4.2                       # optional: taller image box (default 3.5in)
        caption: "VGG-16 · CC BY-SA 4.0"
        calc: "every filter is 3x3 ..."   # mono strip under the figure
        callout: {kind: plain, text: "..."}

      - type: pipeline                 # stage boxes joined by arrows
        title: "The whole CNN"
        stages:                        # {label, sub, kind: plain|conv|pool|dashed}
          - {label: input, sub: spectrum}
          - {label: conv, sub: pattern maps, kind: conv}
        callout: {kind: plain, text: "..."}

      - type: grid                     # numeric table (conv/pooling mechanics, rule 6)
        title: "..."
        table:
          data: [[1,0,0],[0,1,0]]
          highlight: [[0,0],[1,1]]     # cells drawn in amber (the sliding window)
        calc: "3x3 filter, stride 1 -> 4x4 map"

Usage:
    python tools/spec2pptx.py slides/spec/lecture05_cnns.yaml
    python tools/spec2pptx.py --all
"""

import sys
from pathlib import Path

import yaml
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
SPEC_DIR = ROOT / "slides" / "spec"
PPTX_DIR = ROOT / "slides" / "pptx"
ASSETS = ROOT / "slides" / "assets"

# ---- theme (mirrors slides/assets/theme.css, but native) --------------------
PAPER = RGBColor(0xFB, 0xFB, 0xF8)
INK = RGBColor(0x1B, 0x20, 0x25)
INK_SOFT = RGBColor(0x3D, 0x44, 0x4C)
MUTED = RGBColor(0x8A, 0x90, 0x99)
HAIRLINE = RGBColor(0xD9, 0xD9, 0xD2)
SIGNAL = RGBColor(0x0E, 0x7C, 0x7B)
SIGNAL_SOFT = RGBColor(0xE3, 0xF0, 0xEF)
ROI = RGBColor(0xE0, 0x9E, 0x2F)
ROI_SOFT = RGBColor(0xFB, 0xF0, 0xDC)
ROI_INK = RGBColor(0xA9, 0x72, 0x1A)
NEG = RGBColor(0xC4, 0x53, 0x4F)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

SANS = "Avenir Next"
MONO = "Menlo"

EMU_W, EMU_H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.6)

CALLOUT_STYLE = {  # kind -> (bar color, prefix)
    "ms": (SIGNAL, "MASS SPEC · "),
    "check": (ROI, "CHECKPOINT · "),
    "pair": (ROI, "PAIR · "),
    "plain": (INK_SOFT, ""),
}


# ---- text-height estimation (prevents text overflowing its box) -------------
# python-pptx does not measure text, and there is no LibreOffice on the build
# host, so we estimate wrapped text height deterministically and size boxes to
# fit. Constants are calibrated against the real decks (Avenir Next, a humanist
# sans a touch wider than average): ADVANCE is the mean glyph advance as a
# fraction of the point size, LH the line-height multiple. Kept slightly
# conservative so we never UNDER-estimate (which would let text spill). The
# throwaway checker tools/_verify_textfit.py uses the identical formula.
ADVANCE = 0.55
LH = 1.2
EMU_PER_IN = 914400.0


def _est_lines(text, usable_in, font_pt):
    """Estimate how many wrapped lines `text` needs in `usable_in` inches at
    `font_pt`. Word-aware; respects explicit newlines."""
    if not text:
        return 0
    cw = ADVANCE * font_pt / 72.0
    cpl = max(usable_in / cw, 1.0)
    lines = 0
    for para in text.split("\n"):
        words = para.split(" ")
        cur = 0
        pl = 1
        for w in words:
            add = (len(w) + 1) if cur > 0 else len(w)
            if cur + add > cpl and cur > 0:
                pl += 1
                cur = len(w)
            else:
                cur += add
        lines += pl
    return lines


def _est_height_in(segments, box_w_in, margin_lr_in=0.1, pad_in=0.14,
                   space_after_pt=0.0):
    """Estimate the total height (inches) a stack of text segments needs.
    `segments` is a list of (text, font_pt) rendered as separate paragraphs;
    a single wrapped run is one segment. `pad_in` is total top+bottom padding."""
    usable = max(box_w_in - 2 * margin_lr_in, 0.5)
    total = pad_in
    for text, fpt in segments:
        lines = _est_lines(text, usable, fpt)
        total += lines * (fpt * LH / 72.0)
        total += space_after_pt / 72.0
    return total


def _set(run, size, color, bold=False, italic=False, font=SANS):
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = font


def _bg(slide, prs):
    r = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    r.fill.solid()
    r.fill.fore_color.rgb = PAPER
    r.line.fill.background()
    r.shadow.inherit = False
    slide.shapes._spTree.remove(r._element)
    slide.shapes._spTree.insert(2, r._element)
    return r


def _text(slide, left, top, width, height, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    return tb, tf


def _eyebrow(slide, text):
    if not text:
        return
    _, tf = _text(slide, MARGIN, Inches(0.35), Inches(12), Inches(0.4))
    r = tf.paragraphs[0].add_run()
    r.text = text.upper()
    _set(r, 13, MUTED, bold=True)
    tf.paragraphs[0].runs[0].font.name = MONO


def _title(slide, text, top=Inches(0.8)):
    # Grow the title box when the text wraps to 2 lines so it never spills; a
    # single-line title keeps the original 1.1in look. Returns the bottom edge.
    est = _est_height_in([(text, 34)], 12.1, margin_lr_in=0.1, pad_in=0.10)
    h = max(Inches(1.1), int(Inches(est)))
    _, tf = _text(slide, MARGIN, top, Inches(12.1), h)
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    _set(r, 34, INK, bold=True)
    return int(top) + int(h)


CALLOUT_MIN_H = Inches(1.0)
CALLOUT_MAX_H = Inches(2.6)
CALLOUT_MARGIN_LR = 0.2


def _callout_height(spec, width=Inches(11.5)):
    """Estimate the height (EMU, integer) a callout box needs for its text so
    callers can reserve space / position it without overflow. Mirrors _callout.
    Short callouts stay at the 1.0in floor to preserve the current look."""
    if not spec:
        return 0
    kind = spec.get("kind", "plain")
    _, prefix = CALLOUT_STYLE.get(kind, CALLOUT_STYLE["plain"])
    text = (prefix or "") + spec["text"]
    box_w_in = int(width) / EMU_PER_IN
    # prefix run is mono 12pt, body 17pt; estimate the whole line at 17pt (the
    # larger, so slightly conservative) as a single wrapped paragraph.
    est_in = _est_height_in([(text, 17)], box_w_in,
                            margin_lr_in=CALLOUT_MARGIN_LR, pad_in=0.16)
    h = max(int(CALLOUT_MIN_H), int(Inches(est_in)))
    return min(h, int(CALLOUT_MAX_H))


def _callout(slide, spec, top, width=Inches(11.5), left=None):
    if not spec:
        return
    kind = spec.get("kind", "plain")
    bar, prefix = CALLOUT_STYLE.get(kind, CALLOUT_STYLE["plain"])
    left = left if left is not None else MARGIN
    height = _callout_height(spec, width)
    box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, int(left), int(top), int(width), height)
    box.fill.solid()
    box.fill.fore_color.rgb = SIGNAL_SOFT if kind == "ms" else RGBColor(0xF1, 0xF1, 0xEC)
    box.line.color.rgb = bar
    box.line.width = Pt(1)
    box.shadow.inherit = False
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.2)
    tf.margin_right = Inches(0.2)
    p = tf.paragraphs[0]
    if prefix:
        rp = p.add_run()
        rp.text = prefix
        _set(rp, 12, bar, bold=True, font=MONO)
    r = p.add_run()
    r.text = spec["text"]
    _set(r, 17, INK_SOFT)
    return box


# ---- slide builders ---------------------------------------------------------
def slide_title(slide, prs, s):
    _eyebrow(slide, s.get("eyebrow"))
    # Size the (middle-anchored) title box to its content so long titles or
    # multi-line subtitles never clip. Kept centered on ~y=3.5in.
    segs = [(s["title"], 44)]
    if s.get("subtitle"):
        segs.append((s["subtitle"], 22))
    est = _est_height_in(segs, 12.1, margin_lr_in=0.1, pad_in=0.10,
                         space_after_pt=16.0)
    box_h = max(int(Inches(2.2)), int(Inches(est)))
    box_top = int(Inches(3.5)) - box_h // 2
    _, tf = _text(slide, MARGIN, box_top, Inches(12.1), box_h, MSO_ANCHOR.MIDDLE)
    r = tf.paragraphs[0].add_run()
    r.text = s["title"]
    _set(r, 44, INK, bold=True)
    if s.get("subtitle"):
        p = tf.add_paragraph()
        p.space_before = Pt(16)
        r = p.add_run()
        r.text = s["subtitle"]
        _set(r, 22, SIGNAL)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, MARGIN, Inches(2.25), Inches(1.4), Inches(0.06))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ROI
    bar.line.fill.background()
    bar.shadow.inherit = False


def _bullets(tf, bullets, size=26):
    first = True
    for b in bullets:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(18)
        if isinstance(b, str):
            b = {"text": b}
        r = p.add_run()
        r.text = "•  "
        _set(r, size, SIGNAL, bold=True)
        r = p.add_run()
        r.text = b["text"]
        _set(r, size, INK)
        if b.get("time"):
            r = p.add_run()
            r.text = f"   · {b['time']}"
            _set(r, size - 5, MUTED, font=MONO)
        if b.get("sub"):
            sp = tf.add_paragraph()
            sp.space_after = Pt(18)
            r = sp.add_run()
            r.text = "     " + b["sub"]
            _set(r, size - 5, MUTED)


def _bullets_est_height(bullets, size=26, box_w_in=11.8):
    """Estimate height (in) a bullet list needs, mirroring _bullets: each bullet
    is a paragraph (prefix '•  ' + text) with 18pt space_after; a 'sub' adds a
    smaller paragraph."""
    segs = []
    for b in bullets:
        if isinstance(b, str):
            b = {"text": b}
        line = "\u2022  " + b["text"]
        if b.get("time"):
            line += f"   \u00b7 {b['time']}"
        segs.append((line, size))
        if b.get("sub"):
            segs.append(("     " + b["sub"], size - 5))
    return _est_height_in(segs, box_w_in, margin_lr_in=0.1, pad_in=0.10,
                          space_after_pt=18.0)


def slide_bullets(slide, prs, s):
    _eyebrow(slide, s.get("eyebrow"))
    title_bottom = _title(slide, s["title"])
    bullets = s.get("bullets", [])
    top = max(Inches(2.1), title_bottom + Inches(0.05))
    # Reserve room for the callout (if any) so bullets don't run into it.
    call_h = _callout_height(s["callout"]) if s.get("callout") else 0
    reserve_below = (call_h + int(Inches(0.15))) if call_h else int(Inches(0.1))
    avail = int(Inches(7.5)) - int(top) - reserve_below
    # Auto-shrink the bullet font (26 -> down to 16) if the list is too tall to
    # fit the available height, so text never spills off the box / slide.
    size = 26
    while size > 16 and int(Inches(_bullets_est_height(bullets, size, 11.8))) > avail:
        size -= 1
    est = _bullets_est_height(bullets, size, 11.8)
    box_h = min(max(int(Inches(est)), int(Inches(1.0))), avail)
    _, tf = _text(slide, MARGIN, int(top), Inches(11.8), box_h)
    _bullets(tf, bullets, size=size)
    if s.get("callout"):
        # Sit the callout just below the bullets, clamped above the slide bottom.
        cy = int(top) + box_h + int(Inches(0.15))
        max_cy = int(Inches(7.5)) - call_h - int(Inches(0.1))
        _callout(slide, s["callout"], min(cy, max_cy))


def _annotate(slide, img_x, img_y, img_w, img_h, boxes):
    """Draw amber bounding boxes (and optional labels) over a placed image.
    Each box: {x, y, w, h} as fractions [0-1] of the image, optional 'label'
    plus optional label anchor 'lx','ly' (fractions); label defaults above box."""
    for b in boxes:
        bx = int(img_x + b["x"] * img_w)
        by = int(img_y + b["y"] * img_h)
        bw = int(b["w"] * img_w)
        bh = int(b["h"] * img_h)
        rect = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, bx, by, bw, bh)
        rect.fill.background()
        rect.line.color.rgb = ROI
        rect.line.width = Pt(2.5)
        rect.shadow.inherit = False
        if b.get("label"):
            lx = int(img_x + b.get("lx", b["x"]) * img_w)
            ly = int(img_y + b.get("ly", max(b["y"] - 0.09, 0)) * img_h)
            lab = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE, lx, ly, Inches(1.9), Inches(0.34)
            )
            lab.fill.solid()
            lab.fill.fore_color.rgb = WHITE
            lab.line.color.rgb = ROI
            lab.line.width = Pt(0.75)
            lab.shadow.inherit = False
            tf = lab.text_frame
            tf.margin_top = tf.margin_bottom = Inches(0.02)
            tf.word_wrap = False
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            r.text = b["label"]
            _set(r, 12, ROI_INK, bold=True, font=MONO)


def _place_image(slide, path, left, top, max_w, max_h, caption=None, boxes=None):
    """Place an image scaled into (max_w, max_h) and return the Y of its bottom
    edge (below the caption, if any) so callers can flow more content beneath."""
    img = ASSETS / path
    from PIL import Image

    with Image.open(img) as im:
        iw, ih = im.size
    ratio = min(max_w / iw, max_h / ih)
    w, h = int(iw * ratio), int(ih * ratio)
    x = int(left + (max_w - w) // 2)
    slide.shapes.add_picture(str(img), x, int(top), width=w, height=h)
    if boxes:
        _annotate(slide, x, int(top), w, h, boxes)
    bottom = int(top) + h
    if caption:
        cap_top = bottom + Inches(0.06)
        cap_h = Inches(0.45)
        _, tf = _text(slide, left, cap_top, max_w, cap_h)
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = caption
        _set(r, 12, MUTED, italic=True)
        bottom = cap_top + cap_h
    return bottom


def slide_two_col(slide, prs, s):
    _eyebrow(slide, s.get("eyebrow"))
    _title(slide, s["title"])
    colw = Inches(5.9)
    left = s.get("left", {})
    right = s.get("right", {})
    # Vertical room a lead column has before the callout row (~5.8in).
    lead_avail = int(Inches(5.75)) - int(Inches(2.2))

    def _lead(paras, x, colw_):
        # Auto-shrink the lead font (22 -> down to 15) so multi-paragraph lead
        # text never spills into the callout / off the column.
        colw_in = int(colw_) / EMU_PER_IN
        size = 22
        while size > 15:
            est = _est_height_in([(p, size) for p in paras], colw_in,
                                 margin_lr_in=0.1, pad_in=0.10, space_after_pt=12.0)
            if int(Inches(est)) <= lead_avail:
                break
            size -= 1
        _, tf = _text(slide, x, Inches(2.2), colw_, Inches(4))
        for i, para in enumerate(paras):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.space_after = Pt(12)
            r = p.add_run()
            r.text = para
            _set(r, size, INK)

    if left.get("image"):
        _place_image(slide, left["image"], MARGIN, Inches(2.15), int(colw), Inches(3.2), left.get("caption"), left.get("boxes"))
    elif left.get("lead"):
        _lead(left["lead"], MARGIN, colw)
    rx = Inches(6.9)
    if right.get("image"):
        _place_image(slide, right["image"], rx, Inches(2.15), int(colw), Inches(3.2), right.get("caption"), right.get("boxes"))
    else:
        _lead(right.get("lead", []), rx, colw)
    if right.get("callout"):
        ch = _callout_height(right["callout"], width=colw)
        cy = min(int(Inches(5.9)), int(Inches(7.5)) - ch - int(Inches(0.1)))
        _callout(slide, right["callout"], cy, width=colw, left=rx)
    elif s.get("callout"):
        ch = _callout_height(s["callout"])
        cy = min(int(Inches(6.2)), int(Inches(7.5)) - ch - int(Inches(0.1)))
        _callout(slide, s["callout"], cy)


def slide_image(slide, prs, s):
    _eyebrow(slide, s.get("eyebrow"))
    _title(slide, s["title"])
    # img_h (optional) lets a tall/near-square figure use more vertical space;
    # default 3.5in keeps every existing deck (e.g. lecture05) byte-compatible.
    # Cap the image so the whole stack (image + caption + calc + callout) fits
    # ABOVE the slide bottom, so the callout can never be clamped on top of a
    # tall figure (the recurring "boxes overlapping" bug). See
    # tools/check_pptx_overlap.py.
    top = Inches(2.0)
    # Space that must remain below the image bottom (caption is added inside
    # _place_image, so reserve it here when a caption is present).
    reserve = Inches(0.1)
    if s.get("caption"):
        reserve += Inches(0.55)
    calc_h = 0
    if s.get("calc"):
        # calc strip (15pt mono, centered) can wrap to 2 lines -> size it.
        calc_est = _est_height_in([(s["calc"], 15)], 12.1, margin_lr_in=0.1,
                                  pad_in=0.10)
        calc_h = max(int(Inches(0.5)), int(Inches(calc_est)))
        reserve += calc_h + int(Inches(0.05))
    if s.get("callout"):
        # Reserve the ACTUAL (possibly taller) callout height + bottom margin so
        # a long callout shrinks the image instead of colliding / running off.
        reserve += _callout_height(s["callout"]) + int(Inches(0.15))
    avail_h = int(Inches(7.5)) - int(top) - int(reserve)
    max_h = min(int(Inches(s.get("img_h", 3.5))), avail_h)
    y = _place_image(slide, s["image"], MARGIN, top, Inches(12.1), max_h, s.get("caption"), s.get("boxes"))
    y += Inches(0.1)
    if s.get("calc"):
        _, tf = _text(slide, MARGIN, int(y), Inches(12.1), calc_h)
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        r = tf.paragraphs[0].add_run()
        r.text = s["calc"]
        _set(r, 15, INK_SOFT, font=MONO)
        y = int(y) + calc_h + int(Inches(0.05))
    if s.get("callout"):
        # Place the callout at the true content bottom, never above it.
        _callout(slide, s["callout"], int(y))


def slide_pipeline(slide, prs, s):
    _eyebrow(slide, s.get("eyebrow"))
    _title(slide, s["title"])
    stages = s["stages"]
    n = len(stages)
    gap = Inches(0.25)
    total = Inches(12.1)
    box_w = int((total - gap * (n - 1)) / n)
    y = Inches(3.0)
    box_h = Inches(1.5)
    kind_color = {"conv": SIGNAL, "pool": ROI, "dashed": NEG, "plain": INK_SOFT}
    for i, st in enumerate(stages):
        x = int(MARGIN + i * (box_w + gap))
        kind = st.get("kind", "plain")
        col = kind_color.get(kind, INK_SOFT)
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, box_w, box_h)
        box.fill.solid()
        box.fill.fore_color.rgb = WHITE
        box.line.color.rgb = col
        box.line.width = Pt(2.5 if kind != "plain" else 1.5)
        box.shadow.inherit = False
        tf = box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = st["label"]
        _set(r, 18, col if kind != "plain" else INK, bold=True)
        if st.get("sub"):
            sp = tf.add_paragraph()
            sp.alignment = PP_ALIGN.CENTER
            r = sp.add_run()
            r.text = st["sub"]
            _set(r, 12, MUTED)
        if i < n - 1:
            ax = int(x + box_w)
            cy = int(y + box_h / 2)
            conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, ax, cy, int(ax + gap), cy)
            conn.line.color.rgb = MUTED
            conn.line.width = Pt(1.5)
    if s.get("callout"):
        ch = _callout_height(s["callout"])
        cy = min(int(Inches(5.4)), int(Inches(7.5)) - ch - int(Inches(0.1)))
        _callout(slide, s["callout"], cy)


def slide_grid(slide, prs, s):
    _eyebrow(slide, s.get("eyebrow"))
    _title(slide, s["title"])
    spec = s["table"]
    data = spec["data"]
    highlight = {tuple(c) for c in spec.get("highlight", [])}
    rows, cols = len(data), len(data[0])
    cell = Inches(0.62)
    tw, th = cell * cols, cell * rows
    x0 = (EMU_W - tw) // 2
    y0 = Inches(2.4)
    gt = slide.shapes.add_table(rows, cols, x0, y0, tw, th).table
    for c in range(cols):
        gt.columns[c].width = cell
    for r_ in range(rows):
        gt.rows[r_].height = cell
    for r_ in range(rows):
        for c in range(cols):
            cellobj = gt.cell(r_, c)
            cellobj.text = str(data[r_][c])
            para = cellobj.text_frame.paragraphs[0]
            para.alignment = PP_ALIGN.CENTER
            run = para.runs[0]
            _set(run, 20, ROI_INK if (r_, c) in highlight else INK, bold=(r_, c) in highlight, font=MONO)
            cellobj.fill.solid()
            cellobj.fill.fore_color.rgb = ROI_SOFT if (r_, c) in highlight else WHITE
            cellobj.vertical_anchor = MSO_ANCHOR.MIDDLE
    if s.get("calc"):
        calc_est = _est_height_in([(s["calc"], 16)], 12.1, margin_lr_in=0.1,
                                  pad_in=0.10)
        calc_h = max(int(Inches(0.6)), int(Inches(calc_est)))
        _, tf = _text(slide, MARGIN, int(y0 + th + Inches(0.3)), Inches(12.1), calc_h)
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        r = tf.paragraphs[0].add_run()
        r.text = s["calc"]
        _set(r, 16, INK_SOFT, font=MONO)
    if s.get("callout"):
        ch = _callout_height(s["callout"])
        cy = min(int(Inches(6.2)), int(Inches(7.5)) - ch - int(Inches(0.1)))
        _callout(slide, s["callout"], cy)


BUILDERS = {
    "title": slide_title,
    "bullets": slide_bullets,
    "two-col": slide_two_col,
    "image": slide_image,
    "pipeline": slide_pipeline,
    "grid": slide_grid,
}


def build(spec_path: Path) -> Path:
    spec = yaml.safe_load(spec_path.read_text())
    deck = spec.get("deck", {})
    default_eyebrow = deck.get("eyebrow", "")

    prs = Presentation()
    prs.slide_width = EMU_W
    prs.slide_height = EMU_H
    blank = prs.slide_layouts[6]

    for s in spec["slides"]:
        slide = prs.slides.add_slide(blank)
        _bg(slide, prs)
        s.setdefault("eyebrow", default_eyebrow)
        builder = BUILDERS.get(s["type"])
        if not builder:
            raise ValueError(f"unknown slide type: {s['type']}")
        builder(slide, prs, s)
        if s.get("notes"):
            slide.notes_slide.notes_text_frame.text = s["notes"]

    if deck.get("title"):
        prs.core_properties.title = deck["title"]
    PPTX_DIR.mkdir(parents=True, exist_ok=True)
    dest = PPTX_DIR / (spec_path.stem + ".pptx")
    prs.save(str(dest))
    print(f"{spec_path.name}: {len(spec['slides'])} slide(s) -> {dest.relative_to(ROOT)}")
    return dest


def main(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    targets = sorted(SPEC_DIR.glob("*.yaml")) if argv[1] == "--all" else [Path(a).resolve() for a in argv[1:]]
    status = 0
    for spec in targets:
        try:
            build(spec)
        except Exception as exc:
            print(f"ERROR in {spec.name}: {exc}", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv))
