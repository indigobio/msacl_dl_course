#!/usr/bin/env python3
"""Generate the reusable slide figures for MSACL DS301 Deep Learning decks.

Why this exists (course rule 6 "numeric mechanics" + the visualize-aggressively
principle): the mathy I-do / you-do slides read far better as clean, labelled
figures than as walls of bullet-arithmetic. This module is one function per
figure, each writing a PNG into slides/assets/img/. Curves and schematics are
plotted with matplotlib/numpy; photo composites are assembled with PIL.

Everything uses the course palette (paper #FBFBF8, ink #1B2025, teal #0E7C7B,
amber #E09E2F, red #C4534F, muted #8A9099), a sans font, no chartjunk, a paper
background so figures blend into the deck, ~150 dpi, and a final downscale so
files stay small and the deck still renders offline.

Usage:
    python tools/make_slide_figures.py            # build everything
    python tools/make_slide_figures.py activations neuron   # build a subset
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "slides" / "assets" / "img"

# ---- course palette ---------------------------------------------------------
PAPER = "#FBFBF8"
INK = "#1B2025"
INK_SOFT = "#3D444C"
TEAL = "#0E7C7B"
TEAL_SOFT = "#E3F0EF"
AMBER = "#E09E2F"
AMBER_SOFT = "#FBF0DC"
ROI_INK = "#A9721A"  # dark amber for labels/arrows on light amber
HAIRLINE = "#D9D9D2"
RED = "#C4534F"
MUTED = "#8A9099"
WHITE = "#FFFFFF"

plt.rcParams.update(
    {
        "font.family": ["Avenir Next", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 15,
        "axes.edgecolor": INK_SOFT,
        "axes.linewidth": 1.1,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
    }
)

MAX_W = 1400  # cap the stored width; downscale anything wider


def _save(fig, name, dpi=150):
    """Save a matplotlib figure, then downscale in place if it is too wide."""
    path = IMG / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    _downscale(path)
    print(f"  wrote {name}")


def _downscale(path, max_w=MAX_W):
    with Image.open(path) as im:
        if im.width > max_w:
            h = round(im.height * max_w / im.width)
            im = im.convert("RGB").resize((max_w, h), Image.LANCZOS)
            im.save(path)


def _font(size):
    for cand in (
        "/System/Library/Fonts/Supplemental/Avenir Next.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(cand).exists():
            try:
                return ImageFont.truetype(cand, size)
            except OSError:
                continue
    return ImageFont.load_default()


# ---- activation curves ------------------------------------------------------
def _draw_activation(ax, kind, color):
    z = np.linspace(-3, 3, 400)
    if kind == "relu":
        y = np.maximum(0, z)
    elif kind == "sigmoid":
        y = 1 / (1 + np.exp(-z))
    else:  # linear
        y = 0.5 * z
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    ax.plot(z, y, color=color, lw=3.2, zorder=3, solid_capstyle="round")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1.6, 2.1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def activations(labeled=True, name="fig_activations.png",
                order=("relu", "sigmoid", "linear")):
    """Three activation curves. labeled=True names each (the I-do reference);
    labeled=False tags them (a)(b)(c) for the Q1 'name that activation' you-do."""
    names = {"relu": "ReLU", "sigmoid": "Sigmoid", "linear": "Linear"}
    colors = {"relu": TEAL, "sigmoid": AMBER, "linear": RED}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    tags = ["(a)", "(b)", "(c)"]
    for i, (ax, kind) in enumerate(zip(axes, order)):
        _draw_activation(ax, kind, colors[kind])
        label = names[kind] if labeled else tags[i]
        ax.set_title(label, color=INK, fontsize=22, fontweight="bold", pad=10)
    fig.subplots_adjust(wspace=0.12)
    _save(fig, name)


# ---- one-neuron schematic ---------------------------------------------------
def neuron_schematic(x1, x2, w1, w2, b, act, z_txt, y_txt, name):
    """Input nodes -> weighted edges -> sum node (z) -> activation -> output."""
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def node(cx, cy, r, face, edge, label, sub=None, lsize=20):
        ax.add_patch(Circle((cx, cy), r, facecolor=face, edgecolor=edge,
                            lw=2.4, zorder=3))
        ax.text(cx, cy, label, ha="center", va="center", color=INK,
                fontsize=lsize, fontweight="bold", zorder=4)
        if sub:
            ax.text(cx, cy - r - 0.42, sub, ha="center", va="center",
                    color=INK_SOFT, fontsize=16, zorder=4)

    def edge(x0, y0, x1_, y1_, label, lcolor):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1_, y1_),
                    arrowstyle="-|>", mutation_scale=16, lw=2.2,
                    color=INK_SOFT, zorder=2))
        mx, my = (x0 + x1_) / 2, (y0 + y1_) / 2
        ax.text(mx, my + 0.32, label, ha="center", va="center",
                color=lcolor, fontsize=17, fontweight="bold", zorder=5)

    # nodes
    node(1.2, 4.2, 0.55, TEAL_SOFT, TEAL, f"{x1:g}", sub=r"$x_1$")
    node(1.2, 1.8, 0.55, TEAL_SOFT, TEAL, f"{x2:g}", sub=r"$x_2$")
    node(5.5, 3.0, 0.85, WHITE, INK_SOFT, r"$\Sigma$", lsize=26)
    ax.text(5.5, 1.85, f"z = {z_txt}", ha="center", va="center",
            color=INK, fontsize=18, fontweight="bold")
    node(8.4, 3.0, 0.85, AMBER_SOFT, AMBER, act, lsize=18)
    node(11.0, 3.0, 0.6, TEAL_SOFT, TEAL, y_txt, sub="y")

    # edges
    edge(1.75, 4.2, 4.7, 3.25, rf"$w_1$={w1:g}", TEAL)
    edge(1.75, 1.8, 4.7, 2.75, rf"$w_2$={w2:g}", TEAL)
    # bias
    ax.text(4.15, 0.75, f"bias b = {b:g}", ha="center", va="center",
            color=RED, fontsize=16, fontweight="bold")
    ax.add_patch(FancyArrowPatch((4.4, 1.05), (5.15, 2.35),
                arrowstyle="-|>", mutation_scale=14, lw=2.0,
                color=RED, zorder=2))
    edge(6.35, 3.0, 7.55, 3.0, "bend", AMBER)
    edge(9.25, 3.0, 10.4, 3.0, "", INK_SOFT)

    ax.text(6.0, 5.4, "multiply  ·  add  ·  bend", ha="center", va="center",
            color=MUTED, fontsize=17, style="italic")
    _save(fig, name)


# ---- two-layer network schematic -------------------------------------------
def two_layer_net(x1, x2, name="fig_twolayer_youdo.png",
                  w1=None, b1=None, w2=None, b2=None,
                  hidden=None, y=None, arith=None):
    """input (2) -> hidden (2, ReLU) -> output (1). Two modes share the SAME
    2->2->1 node layout so the I-do / you-do pair matches visually (course
    rule 8). Both modes now print EVERY weight and bias on the figure so the
    room can compute straight from the picture (course rules 6/9 — numeric
    mechanics must be self-contained):
      layer-1 weights w1 (2x2, w1[i][j] = input j -> hidden i) sit on the four
      input->hidden edges; layer-1 bias b1 (len 2) sits under each hidden node;
      layer-2 weights w2 (len 2) sit on the hidden->output edges; bias b2 sits
      under the output node.
      you-do (hidden=None): hidden and output read '?' so the room computes
        them (Quiz 1 Q6) — weights are shown, the worked answer lives in notes.
      I-do (pass hidden=(h1,h2), y, arith): the hidden values, y, and the y
        arithmetic are filled in for the worked example."""
    worked = hidden is not None
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def node(cx, cy, r, face, edge, label, lsize=20):
        ax.add_patch(Circle((cx, cy), r, facecolor=face, edgecolor=edge,
                            lw=2.4, zorder=3))
        ax.text(cx, cy, label, ha="center", va="center", color=INK,
                fontsize=lsize, fontweight="bold", zorder=4)

    # column x-positions: input, hidden, output
    xin, xhid, xout = 1.6, 6.0, 10.4
    y_top, y_bot = 4.2, 1.8
    inputs = [(xin, y_top, f"{x1:g}"), (xin, y_bot, f"{x2:g}")]
    hidden_pos = [(xhid, y_top), (xhid, y_bot)]
    out = (xout, 3.0)

    # edges input -> hidden, with the layer-1 weight w1[i][j] on each edge. The
    # four labels stack at a fixed fraction near the input side (each on a paper
    # background) so the two crossing diagonals do not collide.
    for j, (ix, iy, _) in enumerate(inputs):
        for i, (hx, hy) in enumerate(hidden_pos):
            x0, x1_ = ix + 0.55, hx - 0.65
            ax.add_patch(FancyArrowPatch((x0, iy), (x1_, hy),
                        arrowstyle="-|>", mutation_scale=12, lw=1.6,
                        color=INK_SOFT, alpha=0.55, zorder=2))
            if w1 is not None:
                frac = 0.30
                lx = x0 + frac * (x1_ - x0)
                ly = iy + frac * (hy - iy)
                dy = 0.30 if iy == hy == y_top else (-0.32 if iy == hy else 0.0)
                ax.text(lx, ly + dy, f"{w1[i][j]:g}", ha="center", va="center",
                        color=TEAL, fontsize=14, fontweight="bold", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.12", fc=PAPER,
                                  ec="none"))
    # edges hidden -> output, with the layer-2 weight on each edge
    for i, (hx, hy) in enumerate(hidden_pos):
        ax.add_patch(FancyArrowPatch((hx + 0.65, hy), (out[0] - 0.6, out[1]),
                    arrowstyle="-|>", mutation_scale=12, lw=1.6,
                    color=INK_SOFT, alpha=0.55, zorder=2))
        if w2 is not None:
            mx = (hx + 0.65) + 0.42 * ((out[0] - 0.6) - (hx + 0.65))
            my = hy + 0.42 * (out[1] - hy)
            ax.text(mx, my + (0.3 if i == 0 else -0.3), rf"$w_2$={w2[i]:g}",
                    ha="center", va="center", color=TEAL, fontsize=15,
                    fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.1", fc=PAPER, ec="none"))

    hid_labels = [f"{v:g}" for v in hidden] if worked else ["?", "?"]
    y_label = f"{y:g}" if worked else "?"
    for (ix, iy, lab) in inputs:
        node(ix, iy, 0.55, TEAL_SOFT, TEAL, lab)
    for (hx, hy), lab in zip(hidden_pos, hid_labels):
        node(hx, hy, 0.6, AMBER_SOFT, AMBER, lab)
    node(out[0], out[1], 0.6, TEAL_SOFT, TEAL, y_label)

    # layer-1 biases under each hidden node; layer-2 bias under the output node
    # (red, matching the one-neuron figure's bias colour)
    if b1 is not None:
        for (hx, hy), bv in zip(hidden_pos, b1):
            ax.text(hx, hy - 0.95, f"b\u2081={bv:g}", ha="center", va="center",
                    color=RED, fontsize=13, fontweight="bold", zorder=6)
    if b2 is not None:
        ax.text(out[0], out[1] - 0.95, f"b\u2082={b2:g}", ha="center",
                va="center", color=RED, fontsize=13, fontweight="bold",
                zorder=6)

    ax.text(xin, 5.4, "input x", ha="center", color=INK_SOFT, fontsize=16)
    ax.text(xhid, 5.4, "hidden (ReLU)", ha="center", color=INK_SOFT, fontsize=16)
    ax.text(xout, 4.15, "output y", ha="center", color=INK_SOFT, fontsize=16)
    if worked:
        ax.text(6.0, 0.3, arith or "", ha="center", color=INK,
                fontsize=16, fontweight="bold")
    else:
        ax.text(6.0, 0.3, "layer 1 = W\u2081x + b\u2081, bend  ·  layer 2 = W\u2082h + b\u2082",
                ha="center", color=MUTED, fontsize=15, style="italic")
    _save(fig, name)


# ---- matrix x vector --------------------------------------------------------
def matvec(W, x, out_top, out_bottom, highlight, name):
    """2x3 weight matrix times a length-3 vector, color-coded by row so 'one row
    = one neuron' is visible. highlight in {1,2} bolds that row + its output."""
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    cell = 0.95
    row_face = {1: TEAL_SOFT, 2: AMBER_SOFT}
    row_edge = {1: TEAL, 2: AMBER}

    def grid(x0, y0, rows, cols, values, faces, edges, dim=None):
        for r in range(rows):
            for c in range(cols):
                cx = x0 + c * cell
                cy = y0 - r * cell
                on = (dim is None) or (r == dim)
                ax.add_patch(FancyBboxPatch(
                    (cx, cy - cell), cell * 0.92, cell * 0.92,
                    boxstyle="round,pad=0.02,rounding_size=0.08",
                    facecolor=faces[r] if on else WHITE,
                    edgecolor=edges[r] if on else "#D9D9D2",
                    lw=2.4 if on else 1.2, zorder=2))
                v = values[r][c]
                vtxt = f"{v:g}" if isinstance(v, (int, float)) else str(v)
                ax.text(cx + cell * 0.46, cy - cell * 0.46, vtxt,
                        ha="center", va="center",
                        color=INK if on else MUTED,
                        fontsize=20, fontweight="bold" if on else "normal",
                        zorder=3)

    on_row = None if highlight is None else highlight - 1
    grid(0.4, 4.2, 2, 3, W, {0: row_face[1], 1: row_face[2]},
         {0: row_edge[1], 1: row_edge[2]}, dim=on_row)
    ax.text(0.4 + 1.5 * cell, 4.55, "W  (2 neurons x 3 inputs)",
            ha="center", color=INK_SOFT, fontsize=15)

    ax.text(3.55, 3.25, r"$\times$", ha="center", va="center",
            fontsize=26, color=INK_SOFT)

    xvals = [[v] for v in x]
    grid(4.05, 4.2, 3, 1, xvals, {i: "#EFEFEA" for i in range(3)},
         {i: INK_SOFT for i in range(3)})
    ax.text(4.05 + cell * 0.46, 4.55, "x", ha="center",
            color=INK_SOFT, fontsize=15)

    ax.text(5.6, 3.25, "=", ha="center", va="center",
            fontsize=26, color=INK_SOFT)

    out = [[out_top], [out_bottom]]
    grid(6.2, 4.2, 2, 1, out, {0: row_face[1], 1: row_face[2]},
         {0: row_edge[1], 1: row_edge[2]}, dim=on_row)
    ax.text(6.2 + cell * 0.46, 4.55, "Wx", ha="center",
            color=INK_SOFT, fontsize=15)

    # spelled-out row arithmetic
    r1 = f"row 1  =  ({W[0][0]:g})({x[0]:g}) + ({W[0][1]:g})({x[1]:g}) + ({W[0][2]:g})({x[2]:g})  =  {out_top}"
    r2 = f"row 2  =  ({W[1][0]:g})({x[0]:g}) + ({W[1][1]:g})({x[1]:g}) + ({W[1][2]:g})({x[2]:g})  =  {out_bottom}"
    ax.text(8.7, 3.55, r1, ha="left", va="center", color=TEAL,
            fontsize=15.5, fontweight="bold" if highlight == 1 else "normal")
    ax.text(8.7, 2.6, r2, ha="left", va="center", color=AMBER,
            fontsize=15.5, fontweight="bold" if highlight == 2 else "normal")
    ax.text(8.7, 1.45, "one row = one neuron", ha="left", va="center",
            color=MUTED, fontsize=15, style="italic")
    _save(fig, name)


# ---- universal approximation ------------------------------------------------
def universal_approx(name="fig_universal_approx.png"):
    """A wiggly target curve approximated by a sum of ReLU bumps/steps."""
    x = np.linspace(0, 10, 600)
    target = (np.sin(1.1 * x) + 0.45 * np.sin(2.7 * x + 0.6)
              + 0.15 * x - 0.6)

    def relu(t):
        return np.maximum(0, t)

    # a handful of ReLU "hat" bumps summed into a coarse approximation
    knots = np.linspace(0.4, 9.6, 11)
    coeffs = np.interp(knots, x, target)
    approx = np.full_like(x, coeffs[0])
    prev_slope = 0.0
    for i in range(1, len(knots)):
        slope = (coeffs[i] - coeffs[i - 1]) / (knots[i] - knots[i - 1])
        approx = approx + (slope - prev_slope) * relu(x - knots[i - 1])
        prev_slope = slope

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(x, target, color=INK, lw=3.4, label="target function",
            zorder=3, solid_capstyle="round")
    ax.plot(x, approx, color=TEAL, lw=2.6, ls="--", label="sum of ReLU pieces",
            zorder=4)
    # a few faint individual ReLU pieces (thin, so the picture stays calm) —
    # sample ~5 knots rather than drawing all of them
    show = np.linspace(1, len(knots) - 1, 5).round().astype(int)
    for i in show:
        slope = (coeffs[i] - coeffs[i - 1]) / (knots[i] - knots[i - 1])
        piece = coeffs[0] + slope * relu(x - knots[i - 1])
        ax.plot(x, piece, color=AMBER, lw=0.8, alpha=0.28, zorder=2)
    ax.set_xlim(0, 10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=15)
    ax.text(0.5, -0.12, "more neurons -> more bends -> any shape",
            transform=ax.transAxes, ha="center", color=MUTED,
            fontsize=15, style="italic")
    _save(fig, name)


# ---- photo composites (PIL) -------------------------------------------------
def _fit_fill(im, w, h):
    """Center-crop im to exactly (w, h) preserving aspect (cover)."""
    im = im.convert("RGB")
    scale = max(w / im.width, h / im.height)
    nw, nh = round(im.width * scale), round(im.height * scale)
    im = im.resize((nw, nh), Image.LANCZOS)
    left = (nw - w) // 2
    top = (nh - h) // 2
    return im.crop((left, top, left + w, top + h))


def hook_4panel(name="hook_4panel.png"):
    """2x2 grid of DL-in-your-world photos, each with a caption label bar."""
    panels = [
        ("alphafold_protein.png", "AlphaFold"),
        ("alphago_go_board.jpg", "AlphaGo"),
        ("chatgpt_conversation.png", "ChatGPT / LLMs"),
        ("selfdriving_waymo.jpg", "Self-driving cars"),
    ]
    cw, ch = 560, 360
    bar = 46
    gap = 14
    W = cw * 2 + gap * 3
    H = (ch + bar) * 2 + gap * 3
    canvas = Image.new("RGB", (W, H), PAPER)
    draw = ImageDraw.Draw(canvas)
    font = _font(30)
    for i, (fn, label) in enumerate(panels):
        r, c = divmod(i, 2)
        x0 = gap + c * (cw + gap)
        y0 = gap + r * (ch + bar + gap)
        photo = _fit_fill(Image.open(IMG / fn), cw, ch)
        canvas.paste(photo, (x0, y0))
        # label bar
        draw.rectangle([x0, y0 + ch, x0 + cw, y0 + ch + bar], fill=INK)
        tb = draw.textbbox((0, 0), label, font=font)
        tw = tb[2] - tb[0]
        draw.text((x0 + (cw - tw) / 2, y0 + ch + (bar - (tb[3] - tb[1])) / 2 - tb[1]),
                  label, fill=WHITE, font=font)
    canvas.save(IMG / name)
    _downscale(IMG / name)
    print(f"  wrote {name}")


def cat_hierarchy(name="cat_hierarchy.png"):
    """cat photo -> edges -> parts -> whole, left to right, with arrows."""
    stages = [
        ("cat.jpg", "raw pixels"),
        ("cat_l1_edges.png", "layer 1: edges"),
        ("cat_l2_parts.png", "layer 2: parts"),
        ("cat_l3_whole.png", "layer 3: whole"),
    ]
    ph = 300
    pw = 226
    bar = 40
    arrow = 74
    gap = 10
    W = pw * 4 + arrow * 3 + gap * 2
    H = ph + bar + gap * 2
    canvas = Image.new("RGB", (W, H), PAPER)
    draw = ImageDraw.Draw(canvas)
    font = _font(24)
    x = gap
    y = gap
    for i, (fn, label) in enumerate(stages):
        photo = _fit_fill(Image.open(IMG / fn), pw, ph)
        canvas.paste(photo, (x, y))
        draw.rectangle([x, y + ph, x + pw, y + ph + bar], fill=INK)
        tb = draw.textbbox((0, 0), label, font=font)
        tw = tb[2] - tb[0]
        draw.text((x + (pw - tw) / 2, y + ph + (bar - (tb[3] - tb[1])) / 2 - tb[1]),
                  label, fill=WHITE, font=font)
        x += pw
        if i < len(stages) - 1:
            ay = y + ph // 2
            draw.line([x + 12, ay, x + arrow - 12, ay], fill=TEAL, width=6)
            draw.polygon([(x + arrow - 12, ay - 11), (x + arrow - 12, ay + 11),
                          (x + arrow, ay)], fill=TEAL)
            x += arrow
    canvas.save(IMG / name)
    _downscale(IMG / name)
    print(f"  wrote {name}")


def maldi_real(name="maldi_tof_real.png"):
    """Crop the top row (three full-range real MALDI-TOF spectra) from the
    Wikimedia source and downscale to ~1200px wide for the MS-hierarchy slide.
    Source (kept in-repo for reproducibility): _maldi_src.webp — Martisius et
    al. 2020, Wikimedia Commons, CC BY 4.0."""
    src = IMG / "_maldi_src.webp"
    if not src.exists():
        print(f"  skip {name} (source {src.name} absent; keeping existing PNG)")
        return
    with Image.open(src) as im:
        im = im.convert("RGB")
        top = im.crop((0, 0, im.width, 392))  # three spectra, 1000-3500 m/z
        w = 1200
        h = round(top.height * w / top.width)
        top = top.resize((w, h), Image.LANCZOS)
        top.save(IMG / name)
    print(f"  wrote {name}")


def chatgpt_panel(name="chatgpt_conversation.png"):
    """Synthesize a clean ENGLISH chat panel in the course palette for the hook
    composite. We author this ourselves (no third-party screenshot), so it is
    fully license-clean and English — the earlier French Wikimedia screenshot
    was wrong for an English course. A short prompt bubble + reply bubble,
    MS-flavoured, headed 'ChatGPT'."""
    W, H = 700, 450
    canvas = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(canvas)
    head_f = _font(30)
    body_f = _font(24)

    # header strip
    draw.rectangle([0, 0, W, 62], fill=INK)
    dot_cx = 34
    draw.ellipse([dot_cx - 13, 18, dot_cx + 13, 44], fill=TEAL)
    draw.text((60, 16), "ChatGPT", fill=WHITE, font=head_f)

    def bubble(x0, y0, text, fill, txtcol, align_right=False):
        # word-wrap to a max pixel width
        max_w = 430
        words = text.split()
        lines, cur = [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            if draw.textlength(trial, font=body_f) <= max_w:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        line_h = 34
        pad = 18
        bw = max(draw.textlength(ln, font=body_f) for ln in lines) + 2 * pad
        bh = line_h * len(lines) + 2 * pad - 6
        if align_right:
            x0 = W - 24 - bw
        draw.rounded_rectangle([x0, y0, x0 + bw, y0 + bh], radius=16, fill=fill)
        ty = y0 + pad - 3
        for ln in lines:
            draw.text((x0 + pad, ty), ln, fill=txtcol, font=body_f)
            ty += line_h
        return y0 + bh

    # user prompt (right, teal) then assistant reply (left, soft grey)
    y = 92
    y = bubble(0, y, "What is a mass spectrum?",
               fill=TEAL, txtcol=WHITE, align_right=True)
    y += 26
    bubble(24, y,
           "A plot of ion intensity vs. mass-to-charge (m/z). Each peak marks a "
           "molecule's mass — together the peaks form a fingerprint of the sample.",
           fill="#EFEFEA", txtcol=INK)
    canvas.save(IMG / name)
    _downscale(IMG / name)
    print(f"  wrote {name} (synthetic English chat panel)")


def driams_mrsa_spectrum(name="fig_driams_mrsa.png"):
    """Plot ONE real clinical MALDI-TOF spectrum from the DRIAMS S. aureus +
    oxacillin (MRSA) slice — intensity vs. m/z in the course palette, with a
    couple of peak regions flagged so the slide can walk the resistance
    hierarchy (raw intensities → peaks → signature → R/S call).
    Source: DRIAMS, Dryad doi:10.5061/dryad.bzkh1899q, CC0."""
    npz = ROOT / "data" / "slices" / "driams_c_saureus_oxacillin.npz"
    if not npz.exists():
        print(f"  skip {name} (slice {npz.name} absent; keeping existing PNG)")
        return
    d = np.load(npz, allow_pickle=True)
    X, y = d["X"].astype(float), d["y"]
    # binned_6000: 6000 fixed bins spanning 2000-20000 Da (see data/README.md)
    mz_full = np.linspace(2000, 20000, X.shape[1])
    # pick a resistant spectrum with strong total signal (a clear fingerprint)
    res = np.where(y == 1)[0]
    idx = res[np.argmax(X[res].sum(1))]
    # bacterial fingerprint region carries ~90% of signal below 12000 Da
    win = mz_full <= 12000
    mz, sp = mz_full[win], X[idx][win]
    sp = sp / sp.max()  # scale to 0-1 for a clean axis

    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.plot(mz, sp, color=TEAL, lw=1.1, zorder=3)
    ax.fill_between(mz, sp, color=TEAL, alpha=0.12, zorder=2)

    # flag two peak regions that support the "peaks → signature" story;
    # labels sit on a clean shelf above the trace with a short connector down
    regions = [(2400, 3100, "peak cluster"), (6200, 7200, "peak cluster")]
    label_y = 1.22
    for lo, hi, lab in regions:
        m = (mz >= lo) & (mz <= hi)
        pk = np.argmax(sp[m])
        px = mz[m][pk]
        py = sp[m][pk]
        ax.annotate(lab, xy=(px, py + 0.02), xytext=(px, label_y),
                    ha="center", color=AMBER, fontsize=13, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=AMBER, lw=1.4))

    ax.set_xlim(2000, 12000)
    ax.set_ylim(0, 1.35)
    ax.set_xlabel("m/z (Da)", fontsize=14)
    ax.set_ylabel("relative intensity", fontsize=14)
    ax.set_yticks([0, 0.5, 1.0])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.text(0.99, 0.95, "S. aureus · oxacillin-resistant (MRSA)",
            transform=ax.transAxes, ha="right", va="top",
            color=INK_SOFT, fontsize=13, style="italic")
    _save(fig, name)


# =====================================================================
#  Lecture 2 · How Networks Learn (loss, gradient descent, backprop,
#  the training loop). One function per figure, course palette, rule 6.
# =====================================================================

def _lossbowl(ax, wmin=0.0, k=1.0, floor=0.1, wlo=-3.2, whi=3.2):
    """Draw a clean loss 'bowl' (parabola) on ax and return the loss(w) fn."""
    w = np.linspace(wlo, whi, 400)
    loss = k * (w - wmin) ** 2 + floor
    ax.plot(w, loss, color=TEAL, lw=3.0, zorder=3, solid_capstyle="round")
    ax.set_xlim(wlo, whi)
    ax.set_ylim(0, k * (whi - wmin) ** 2 * 1.05 + floor)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return lambda ww: k * (ww - wmin) ** 2 + floor


# ---- loss = MSE intuition ---------------------------------------------------
def loss_mse(name="fig_loss_mse.png"):
    """Truth vs. guess for a handful of samples, red gap = error; the caption
    says MSE = average of those gaps, squared. The 'how wrong' number."""
    truth = np.array([0.85, 0.35, 0.62, 0.20, 0.72])
    guess = np.array([0.55, 0.60, 0.50, 0.44, 0.66])
    xs = np.arange(1, len(truth) + 1)
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    for xi, t, g in zip(xs, truth, guess):
        ax.plot([xi, xi], [t, g], color=RED, lw=3.0, zorder=2,
                solid_capstyle="round")
    ax.scatter(xs, truth, s=150, color=TEAL, zorder=4, label="truth")
    ax.scatter(xs, guess, s=150, color=AMBER, zorder=4, label="guess")
    # one labelled error gap
    ax.annotate("error", xy=(xs[0], (truth[0] + guess[0]) / 2),
                xytext=(xs[0] + 0.35, (truth[0] + guess[0]) / 2 + 0.02),
                color=RED, fontsize=15, fontweight="bold", va="center")
    ax.set_xlim(0.4, len(truth) + 0.6)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"sample {i}" for i in xs], fontsize=12)
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=14, ncol=2,
              handletextpad=0.3, columnspacing=1.0)
    ax.text(0.5, -0.13, "MSE  =  average of (guess \u2212 truth)\u00b2",
            transform=ax.transAxes, ha="center", color=INK,
            fontsize=17, fontweight="bold")
    _save(fig, name)


# ---- gradient descent = downhill ball --------------------------------------
def loss_bowl(name="fig_loss_bowl.png"):
    """A ball high on the loss bowl taking downhill steps toward the minimum,
    with a tangent line (the slope) and the update rule. Gradient descent."""
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    L = _lossbowl(ax, wmin=0.0, k=0.5, floor=0.2, wlo=-3.4, whi=3.4)
    ax.set_ylim(0, 6.4)
    # stepping ball positions closing on the minimum
    ws = [-2.7, -1.6, -0.85, -0.4, -0.1]
    for i, ww in enumerate(ws):
        alpha = 0.4 + 0.6 * i / (len(ws) - 1)
        ax.scatter([ww], [L(ww)], s=200 if i == 0 else 130,
                   color=AMBER, edgecolor=ROI_INK if i == 0 else "none",
                   linewidth=1.6, zorder=5, alpha=alpha)
    for a, b in zip(ws[:-1], ws[1:]):
        ax.annotate("", xy=(b, L(b)), xytext=(a, L(a)),
                    arrowprops=dict(arrowstyle="-|>", color=ROI_INK,
                                    lw=1.8, shrinkA=7, shrinkB=7), zorder=4)
    # short tangent line (the slope) at the starting ball
    w0 = ws[0]
    slope = 2 * 0.5 * w0
    tx = np.array([w0 - 0.85, w0 + 0.85])
    ax.plot(tx, L(w0) + slope * (tx - w0), color=RED, lw=2.4, ls="--",
            zorder=3)
    ax.annotate("slope here\n(points uphill)", xy=(w0, L(w0)),
                xytext=(w0 - 0.2, L(w0) + 1.5), color=RED, fontsize=12.5,
                fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="-", color=RED, lw=1.0))
    ax.scatter([0.0], [L(0.0)], s=130, color=TEAL, zorder=6)
    ax.annotate("lowest loss", xy=(0.0, L(0.0)), xytext=(1.0, L(0.0) + 1.1),
                color=TEAL, fontsize=12.5, fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="-", color=TEAL, lw=1.0))
    ax.set_xlabel("weight value", fontsize=13)
    ax.set_ylabel("loss", fontsize=13, color=MUTED)
    ax.text(0.5, -0.17,
            "new weight  =  weight  \u2212  lr \u00d7 slope   (step the opposite way of the slope)",
            transform=ax.transAxes, ha="center", color=INK,
            fontsize=15.5, fontweight="bold")
    _save(fig, name)


# ---- learning rate: three failure modes ------------------------------------
def lr_three_modes(name="fig_lr_three_modes.png"):
    """Same bowl, three step sizes: too small (crawls / can stall), just right
    (steady), too big (overshoots, diverges). The learning-rate dial."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    titles = [("too small", ROI_INK), ("just right", TEAL), ("too big", RED)]
    for ax, (title, col) in zip(axes, titles):
        L = _lossbowl(ax, wmin=0.0, k=0.5, floor=0.15, wlo=-3.3, whi=3.3)
        ax.set_title(title, color=col, fontsize=19, fontweight="bold", pad=8)
    # too small: tiny crawling steps, never reaches bottom
    ws = [-2.9, -2.55, -2.28, -2.06, -1.9]
    Ls = _lossbowl(axes[0], wmin=0.0, k=0.5, floor=0.15, wlo=-3.3, whi=3.3)
    for a, b in zip(ws[:-1], ws[1:]):
        axes[0].annotate("", xy=(b, Ls(b)), xytext=(a, Ls(a)),
                         arrowprops=dict(arrowstyle="-|>", color=ROI_INK,
                                         lw=1.6, shrinkA=5, shrinkB=5))
    axes[0].scatter(ws, [Ls(w) for w in ws], s=70, color=AMBER, zorder=5)
    axes[0].text(0.5, -0.14, "crawls \u2014 can stall short",
                 transform=axes[0].transAxes, ha="center", color=MUTED,
                 fontsize=13)
    # just right: brisk steps to the floor
    wr = [-2.9, -1.5, -0.6, -0.15, 0.0]
    Lr = _lossbowl(axes[1], wmin=0.0, k=0.5, floor=0.15, wlo=-3.3, whi=3.3)
    for a, b in zip(wr[:-1], wr[1:]):
        axes[1].annotate("", xy=(b, Lr(b)), xytext=(a, Lr(a)),
                         arrowprops=dict(arrowstyle="-|>", color=TEAL,
                                         lw=1.8, shrinkA=6, shrinkB=6))
    axes[1].scatter(wr, [Lr(w) for w in wr], s=80, color=TEAL, zorder=5)
    axes[1].text(0.5, -0.14, "reaches the bottom fast",
                 transform=axes[1].transAxes, ha="center", color=MUTED,
                 fontsize=13)
    # too big: overshoots, amplitude grows (diverging)
    wb = [-2.4, 2.7, -3.0, 3.2]
    Lb = _lossbowl(axes[2], wmin=0.0, k=0.5, floor=0.15, wlo=-3.3, whi=3.3)
    for a, b in zip(wb[:-1], wb[1:]):
        axes[2].annotate("", xy=(b, Lb(b)), xytext=(a, Lb(a)),
                         arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8,
                                         shrinkA=6, shrinkB=6,
                                         connectionstyle="arc3,rad=-0.25"))
    axes[2].scatter(wb, [Lb(w) for w in wb], s=80, color=RED, zorder=5)
    axes[2].text(0.5, -0.14, "overshoots \u2014 can blow up",
                 transform=axes[2].transAxes, ha="center", color=MUTED,
                 fontsize=13)
    fig.subplots_adjust(wspace=0.1)
    _save(fig, name)


# ---- loss curves: diagnose the learning rate -------------------------------
def _loss_curve(ax, kind, color):
    e = np.linspace(0, 30, 200)
    if kind == "just_right":
        y = 2.1 * np.exp(-0.28 * e) + 0.14
    elif kind == "too_low":
        y = 2.1 * np.exp(-0.018 * e) + 0.05
    elif kind == "too_high":  # diverging: growing oscillation, rising
        y = 0.9 + 0.05 * e + (0.18 + 0.055 * e) * np.abs(np.sin(1.15 * e))
    else:  # a_bit_high: noisy but converging
        y = 1.9 * np.exp(-0.13 * e) + 0.2 + (0.55 * np.exp(-0.11 * e)) * np.sin(1.9 * e)
        y = np.clip(y, 0.05, None)
    ax.plot(e, y, color=color, lw=2.6, zorder=3, solid_capstyle="round")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 3.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("epoch \u2192", fontsize=12, color=MUTED)


def loss_curves(labeled=True, name="fig_loss_curves_ido.png"):
    """labeled=True: three canonical shapes named (the I-do dashboard).
    labeled=False: the four Quiz 2 Q3 curves tagged (a)-(d), unlabeled."""
    if labeled:
        specs = [("too_low", ROI_INK, "too low"),
                 ("too_high", RED, "too high"),
                 ("just_right", TEAL, "just right")]
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    else:
        # order mirrors Quiz 2 Q3 exactly: (a) far too high, (b) too low,
        # (c) just right, (d) a bit too high
        specs = [("too_high", INK, "(a)"), ("too_low", INK, "(b)"),
                 ("just_right", INK, "(c)"), ("a_bit_high", INK, "(d)")]
        fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.2))
    for ax, (kind, col, title) in zip(axes, specs):
        _loss_curve(ax, kind, col)
        ax.set_title(title, color=col, fontsize=18, fontweight="bold", pad=8)
    axes[0].text(-0.08, 0.5, "loss", transform=axes[0].transAxes,
                 rotation=90, va="center", ha="center", color=MUTED,
                 fontsize=12)
    fig.subplots_adjust(wspace=0.14)
    _save(fig, name)


# ---- backprop on the two-weight net (the centerpiece) ----------------------
def backprop_two_weight(x, w1, w2, y, reveal, name):
    """One input -> one hidden (ReLU) -> one output, two weights. Numbers flow
    forward (teal, above) then gradients flow back (red, below).
      reveal='fwd'   : forward pass filled, backward lane empty (I-do forward)
      reveal='full'  : forward + backward gradients (I-do backward, the reveal)
      reveal='blank' : givens shown, everything to solve is '?' (you-do = Q1)
    """
    z = w1 * x
    h = max(0.0, z)
    yhat = w2 * h
    err = yhat - y
    loss = err ** 2
    relu_slope = 1 if z > 0 else 0
    signal = 2 * err
    g2 = signal * h
    g1 = signal * w2 * relu_slope * x
    blank = reveal == "blank"

    def q(v, fmt="{:g}"):
        return "?" if blank else fmt.format(v)

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    yc = 5.1
    xn, hn, yn, ln = 1.1, 4.4, 7.7, 10.7

    def node(cx, face, edge, label, lsize=17):
        ax.add_patch(Circle((cx, yc), 0.6, facecolor=face, edgecolor=edge,
                            lw=2.6, zorder=3))
        ax.text(cx, yc, label, ha="center", va="center", color=INK,
                fontsize=lsize, fontweight="bold", zorder=4)

    # forward arrows (teal, above)
    for a, b in [(xn, hn), (hn, yn), (yn, ln)]:
        ax.add_patch(FancyArrowPatch((a + 0.6, yc), (b - 0.72, yc),
                    arrowstyle="-|>", mutation_scale=15, lw=2.2,
                    color=TEAL, zorder=2))
    # nodes
    node(xn, TEAL_SOFT, TEAL, f"x={x:g}")
    node(hn, AMBER_SOFT, AMBER, "ReLU")
    node(yn, TEAL_SOFT, TEAL, "\u0177")
    ax.add_patch(FancyBboxPatch((ln - 0.68, yc - 0.4), 1.36, 0.8,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                facecolor="#F7E4E3", edgecolor=RED, lw=2.4, zorder=3))
    ax.text(ln, yc, "loss", ha="center", va="center", color=RED,
            fontsize=16, fontweight="bold", zorder=4)

    # weight labels on the forward edges
    ax.text((xn + hn) / 2, yc + 0.42, f"w\u2081={w1:g}", ha="center",
            color=TEAL, fontsize=16, fontweight="bold")
    ax.text((hn + yn) / 2, yc + 0.42, f"w\u2082={w2:g}", ha="center",
            color=TEAL, fontsize=16, fontweight="bold")

    # forward values under the nodes (compact, single row of the block)
    ax.text(hn, yc - 0.85, f"z=w\u2081x={q(z)}\nh=ReLU(z)={q(h)}",
            ha="center", va="top", color=INK_SOFT, fontsize=13)
    ax.text(yn, yc - 0.85, f"\u0177=w\u2082h={q(yhat)}", ha="center", va="top",
            color=INK_SOFT, fontsize=13)
    ax.text(ln, yc - 0.85, f"y={y:g},  error={q(err)}\nloss={q(loss)}",
            ha="center", va="top", color=INK_SOFT, fontsize=13)

    ax.text(xn - 0.55, yc + 1.1, "forward  \u2192", ha="left", color=TEAL,
            fontsize=15, fontweight="bold")

    if reveal == "full" or blank:
        # backward arrows (red, well below the forward value block)
        yb = 2.0
        for a, b in [(ln, yb), (yn, yb), (hn, yb)]:
            pass
        ax.add_patch(FancyArrowPatch((ln, yb), (xn, yb),
                    arrowstyle="-|>", mutation_scale=16, lw=2.2,
                    color=RED, zorder=2, alpha=0.9))
        ax.text(xn - 0.55, yb + 0.75, "\u2190  backward (multiply local effects)",
                ha="left", color=RED, fontsize=15, fontweight="bold")
        # the shared "how wrong" signal near the loss end
        ax.text(ln, yb + 0.4, f"how wrong = 2\u00d7error = {q(signal)}",
                ha="right", va="bottom", color=RED, fontsize=13,
                fontweight="bold")
        # per-weight gradients on two rows below the arrow (short path first)
        ax.text(xn - 0.55, yb - 0.45,
                f"grad w\u2082 = (2\u00d7err) \u00d7 h = {q(g2)}",
                ha="left", va="top", color=RED, fontsize=13, fontweight="bold")
        ax.text(xn - 0.55, yb - 1.15,
                f"grad w\u2081 = (2\u00d7err) \u00d7 w\u2082 \u00d7 ReLU' \u00d7 x = {q(g1)}",
                ha="left", va="top", color=RED, fontsize=13, fontweight="bold")
    _save(fig, name)


# ---- backprop: the one idea (chain of local effects) -----------------------
def backprop_chain(name="fig_backprop_chain.png"):
    """Boxes joined by x: the shared 'how wrong' signal times one local slope at
    each arrow = the gradient for a weight. The chain rule, no symbols."""
    fig, ax = plt.subplots(figsize=(11.5, 3.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        ("how wrong", "2 \u00d7 error", TEAL_SOFT, TEAL),
        ("local effect", "slope at\nan arrow", WHITE, INK_SOFT),
        ("local effect", "slope at\nnext arrow", WHITE, INK_SOFT),
        ("gradient", "for that\nweight", TEAL_SOFT, TEAL),
    ]
    bw, gap = 2.3, 0.75
    x0 = 0.5
    seps = ["\u00d7", "\u00d7", "="]
    for i, (cap, val, face, edge) in enumerate(boxes):
        cx = x0 + i * (bw + gap)
        ax.add_patch(FancyBboxPatch((cx, 0.7), bw, 1.6,
                    boxstyle="round,pad=0.02,rounding_size=0.12",
                    facecolor=face, edgecolor=edge, lw=2.2, zorder=2))
        ax.text(cx + bw / 2, 2.02, cap, ha="center", va="center",
                color=MUTED, fontsize=12, fontweight="bold")
        ax.text(cx + bw / 2, 1.35, val, ha="center", va="center",
                color=edge if edge != INK_SOFT else INK, fontsize=15,
                fontweight="bold")
        if i < len(seps):
            ax.text(cx + bw + gap / 2, 1.5, seps[i], ha="center", va="center",
                    color=INK_SOFT, fontsize=22, fontweight="bold")
    ax.text(6, 0.25, "multiply the local slopes along the path  =  the chain rule (no symbols needed)",
            ha="center", color=MUTED, fontsize=13, style="italic")
    _save(fig, name)


# ---- one gradient-descent step by hand -------------------------------------
def onestep_update(rows, loss_pair, name):
    """One gradient-descent step laid out as 'w <- old - lr x grad = new'.
    rows: list of (wname, expr, new) — new may be '?' for the you-do.
    loss_pair: (before, after) draws a loss-falls bar chart; None hides it
    (the you-do leaves the loss re-check to the room)."""
    if loss_pair is None:
        fig, axl = plt.subplots(figsize=(9.5, 3.4))
    else:
        fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 3.8),
                                       gridspec_kw={"width_ratios": [1.7, 1]})
    axl.set_xlim(0, 10)
    axl.set_ylim(0, 4)
    axl.axis("off")
    axl.text(0.2, 3.6, "new weight = weight \u2212 lr \u00d7 gradient   (lr = 0.1)",
             color=INK, fontsize=14, fontweight="bold")
    for i, (wn, expr, new) in enumerate(rows):
        yy = 2.5 - i * 1.3
        blank = new == "?"
        axl.text(0.4, yy, wn, color=TEAL, fontsize=20, fontweight="bold",
                 va="center")
        axl.text(1.3, yy, f"\u2190  {expr}  =", color=INK_SOFT, fontsize=17,
                 va="center")
        axl.add_patch(FancyBboxPatch((6.7, yy - 0.45), 1.5, 0.9,
                     boxstyle="round,pad=0.02,rounding_size=0.12",
                     facecolor=AMBER_SOFT if blank else TEAL_SOFT,
                     edgecolor=AMBER if blank else TEAL, lw=2.2))
        axl.text(7.45, yy, new, color=AMBER if blank else TEAL, fontsize=20,
                 fontweight="bold", ha="center", va="center")
        if not blank:
            axl.annotate("", xy=(8.9, yy - 0.3), xytext=(8.9, yy + 0.3),
                         arrowprops=dict(arrowstyle="-|>", color=ROI_INK, lw=2.2))
            axl.text(9.3, yy, "down", color=ROI_INK, fontsize=12, va="center")
    if loss_pair is not None:
        b, a = loss_pair
        axr.bar([0, 1], [b, a], color=[AMBER, TEAL], width=0.6, zorder=3)
        for xi, v in zip([0, 1], [b, a]):
            axr.text(xi, v + 0.01 * b / 0.25, f"{v:g}", ha="center",
                     va="bottom", color=INK, fontsize=15, fontweight="bold")
        axr.set_xticks([0, 1])
        axr.set_xticklabels(["before", "after\n1 step"], fontsize=12)
        axr.set_ylim(0, b * 1.2)
        axr.set_yticks([])
        for sp in ("top", "right", "left"):
            axr.spines[sp].set_visible(False)
        axr.set_title("loss falls", color=INK_SOFT, fontsize=14, pad=6)
    _save(fig, name)


# ---- code figures (training loop) ------------------------------------------
def _code_figure(rows, name, figsize=(11, 3.6), comment_x=0.46):
    """Render monospace 'code' rows on a clean card. Each row is a dict:
      {code, comment?, num?, dim?}. num draws a highlighted step number box."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96,
                boxstyle="round,pad=0.005,rounding_size=0.02",
                facecolor=WHITE, edgecolor=HAIRLINE, lw=1.2, zorder=1))
    n = len(rows)
    top, bot = 0.86, 0.14
    dy = (top - bot) / max(n - 1, 1)
    for i, row in enumerate(rows):
        yy = top - i * dy
        cx = 0.05
        if row.get("num") is not None:
            ax.add_patch(FancyBboxPatch((0.035, yy - 0.055), 0.05, 0.11,
                        boxstyle="round,pad=0.005,rounding_size=0.03",
                        facecolor=TEAL, edgecolor="none", zorder=3))
            ax.text(0.06, yy, str(row["num"]), color=WHITE, fontsize=14,
                    fontweight="bold", ha="center", va="center", zorder=4,
                    family="monospace")
            cx = 0.11
        elif row.get("blanknum"):
            ax.add_patch(FancyBboxPatch((0.035, yy - 0.055), 0.05, 0.11,
                        boxstyle="round,pad=0.005,rounding_size=0.03",
                        facecolor=AMBER_SOFT, edgecolor=AMBER, lw=1.6, zorder=3))
            cx = 0.11
        ax.text(cx, yy, row["code"], color=MUTED if row.get("dim") else INK,
                fontsize=15.5, ha="left", va="center", family="monospace",
                fontweight="bold" if row.get("num") else "normal")
        if row.get("comment"):
            ax.text(comment_x, yy, row["comment"], color=MUTED, fontsize=13,
                    ha="left", va="center", family="monospace")
    _save(fig, name)


def training_code(name="fig_training_code.png"):
    """The five-line loop, annotated (forward -> loss -> zero -> backward -> step)."""
    rows = [
        {"code": "for epoch in range(20):", "comment": "# repeat many times",
         "dim": True},
        {"code": "pred = model(x)", "comment": "# FORWARD  \u2014 make a guess",
         "num": 1},
        {"code": "loss = loss_fn(pred, y)", "comment": "# LOSS     \u2014 how wrong?",
         "num": 2},
        {"code": "optimizer.zero_grad()", "comment": "# ZERO     \u2014 clear old grads",
         "num": 3},
        {"code": "loss.backward()", "comment": "# BACKWARD \u2014 backprop the blame",
         "num": 4},
        {"code": "optimizer.step()", "comment": "# STEP     \u2014 nudge every weight",
         "num": 5},
    ]
    _code_figure(rows, name, figsize=(11, 3.8), comment_x=0.50)


def loop_order(name="fig_loop_order.png"):
    """The five lines shuffled with blank number boxes (you-do = Quiz 2 Q4).
    Order matches the quiz exactly so the slide is a faithful cue."""
    rows = [
        {"code": "loss.backward()", "comment": "# backprop the blame",
         "blanknum": True},
        {"code": "pred = model(x)", "comment": "# forward \u2014 make a guess",
         "blanknum": True},
        {"code": "optimizer.step()", "comment": "# nudge every weight",
         "blanknum": True},
        {"code": "loss = loss_fn(pred, y)", "comment": "# how wrong?",
         "blanknum": True},
        {"code": "optimizer.zero_grad()", "comment": "# clear old gradients",
         "blanknum": True},
    ]
    _code_figure(rows, name, figsize=(11, 3.5), comment_x=0.50)


def autograd_check(name="fig_autograd.png"):
    """The same two-weight example in PyTorch + console output matching the
    by-hand gradients (w1.grad=3.0, w2.grad=1.0)."""
    rows = [
        {"code": "x, y = torch.tensor(2.), torch.tensor(1.)", "dim": False},
        {"code": "w1 = torch.tensor(.5, requires_grad=True)"},
        {"code": "w2 = torch.tensor(1.5, requires_grad=True)"},
        {"code": "h    = torch.relu(w1 * x)", "comment": "# hidden"},
        {"code": "pred = w2 * h", "comment": "# guess"},
        {"code": "loss = (pred - y) ** 2", "comment": "# sq. error"},
        {"code": "loss.backward()", "comment": "# fills .grad"},
    ]
    fig, (axc, axo) = plt.subplots(1, 2, figsize=(11.5, 3.5),
                                   gridspec_kw={"width_ratios": [1.6, 1]})
    # left: code card (reuse the same styling inline)
    axc.set_xlim(0, 1)
    axc.set_ylim(0, 1)
    axc.axis("off")
    axc.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96,
                 boxstyle="round,pad=0.005,rounding_size=0.02",
                 facecolor=WHITE, edgecolor=HAIRLINE, lw=1.2))
    top, bot = 0.88, 0.12
    dy = (top - bot) / (len(rows) - 1)
    for i, row in enumerate(rows):
        yy = top - i * dy
        axc.text(0.05, yy, row["code"], color=INK, fontsize=13.5,
                 ha="left", va="center", family="monospace")
        if row.get("comment"):
            axc.text(0.70, yy, row["comment"], color=MUTED, fontsize=11.5,
                     ha="left", va="center", family="monospace")
    # right: console output (dark card)
    axo.set_xlim(0, 1)
    axo.set_ylim(0, 1)
    axo.axis("off")
    axo.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96,
                 boxstyle="round,pad=0.005,rounding_size=0.02",
                 facecolor="#22272D", edgecolor="#22272D", lw=1.0))
    axo.text(0.08, 0.82, "console", color=MUTED, fontsize=12,
             family="monospace", va="center")
    axo.text(0.08, 0.58, "pred = 1.5", color="#7FD1CF", fontsize=15,
             family="monospace", va="center", fontweight="bold")
    axo.text(0.08, 0.44, "loss = 0.25", color="#7FD1CF", fontsize=15,
             family="monospace", va="center", fontweight="bold")
    axo.text(0.08, 0.24, "w1.grad = 3.0", color="#F0C674", fontsize=15,
             family="monospace", va="center", fontweight="bold")
    axo.text(0.08, 0.10, "w2.grad = 1.0", color="#F0C674", fontsize=15,
             family="monospace", va="center", fontweight="bold")
    _save(fig, name)


# ---- backprop at scale (one red backward path on a full net) ---------------
def backprop_scale_net(name="fig_backprop_scale.png"):
    """A small MLP with faint forward edges and ONE highlighted red backward
    path — the two-weight chain we just did, now one thread among thousands.
    This is what loss.backward() runs for every weight at once."""
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    layers = [
        [(1.3, 4.4), (1.3, 1.6)],
        [(4.3, 5.0), (4.3, 3.0), (4.3, 1.0)],
        [(7.3, 4.4), (7.3, 1.6)],
        [(10.4, 3.0)],
    ]
    # faint forward edges
    for li in range(len(layers) - 1):
        for (ax0, ay0) in layers[li]:
            for (ax1, ay1) in layers[li + 1]:
                ax.plot([ax0, ax1], [ay0, ay1], color=TEAL, lw=1.2,
                        alpha=0.28, zorder=1)
    # one highlighted backward path (output <- ... <- an input)
    path = [layers[3][0], layers[2][0], layers[1][0], layers[0][0]]
    for a, b in zip(path[:-1], path[1:]):
        ax.annotate("", xy=b, xytext=a,
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.6,
                                    shrinkA=14, shrinkB=14), zorder=4)
    # nodes
    for li, layer in enumerate(layers):
        for (nx, ny) in layer:
            if li == 0:
                face, edge = TEAL_SOFT, TEAL
            elif li == len(layers) - 1:
                face, edge = "#F7E4E3", RED
            else:
                face, edge = AMBER_SOFT, AMBER
            ax.add_patch(Circle((nx, ny), 0.42, facecolor=face,
                                edgecolor=edge, lw=2.2, zorder=3))
    ax.text(layers[3][0][0], layers[3][0][1], "\u0177", ha="center",
            va="center", color=INK, fontsize=15, fontweight="bold", zorder=5)
    ax.text(1.3, 5.5, "forward  \u2192", ha="center", color=TEAL, fontsize=14,
            fontweight="bold")
    ax.text(10.4, 5.5, "\u2190  backward", ha="center", color=RED,
            fontsize=14, fontweight="bold")
    ax.text(6, 0.35, "one red path = the chain you just did · every weight gets its own, in one sweep",
            ha="center", color=MUTED, fontsize=13, style="italic")
    _save(fig, name)


# ---- the training loop as a cycle ------------------------------------------
def training_loop_cycle(name="fig_training_loop.png"):
    """Five steps arranged as a clean labelled cycle with a 'repeat' loop-back."""
    import math
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    cx, cy, Rx, Ry = 6, 3.5, 4.3, 2.5
    steps = [
        ("1 · Forward", "make a guess", TEAL),
        ("2 · Loss", "how wrong?", AMBER),
        ("3 · zero_grad", "clear old grads", INK_SOFT),
        ("4 · Backward", "backprop blame", RED),
        ("5 · Step", "nudge weights", TEAL),
    ]
    pos = []
    for i in range(5):
        ang = math.radians(90 - i * 72)
        pos.append((cx + Rx * math.cos(ang), cy + Ry * math.sin(ang)))
    # arrows between consecutive steps (clockwise), plus loop-back 5->1
    for i in range(5):
        a = pos[i]
        b = pos[(i + 1) % 5]
        col = TEAL if i < 4 else MUTED
        style = "arc3,rad=-0.18"
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>",
                    mutation_scale=16, lw=1.8, color=col, zorder=1,
                    shrinkA=34, shrinkB=34, connectionstyle=style,
                    linestyle="--" if i == 4 else "-"))
    for (bx, by), (name_, sub, col) in zip(pos, steps):
        ax.add_patch(FancyBboxPatch((bx - 1.15, by - 0.55), 2.3, 1.1,
                    boxstyle="round,pad=0.02,rounding_size=0.12",
                    facecolor=WHITE, edgecolor=col, lw=2.4, zorder=3))
        ax.text(bx, by + 0.16, name_, ha="center", va="center", color=col,
                fontsize=14.5, fontweight="bold", zorder=4)
        ax.text(bx, by - 0.28, sub, ha="center", va="center", color=MUTED,
                fontsize=11.5, zorder=4)
    ax.text(cx, cy, "repeat\nevery batch", ha="center", va="center",
            color=MUTED, fontsize=13, style="italic")
    _save(fig, name)


# ---- a healthy loss curve over training ------------------------------------
def loss_fall(name="fig_loss_fall.png"):
    """The healthy training signature: steep early drop, then a gentle flatten.
    Narrated as a tiny classifier on clinical spectra (Lab 1 runs it for real)."""
    e = np.linspace(0, 20, 200)
    y = 1.9 * np.exp(-0.32 * e) + 0.12 + 0.015 * np.cos(2.2 * e) * np.exp(-0.2 * e)
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.plot(e, y, color=TEAL, lw=3.0, zorder=3, solid_capstyle="round")
    ax.scatter([e[0]], [y[0]], s=120, color=AMBER, zorder=5)
    ax.scatter([e[-1]], [y[-1]], s=120, color=TEAL, zorder=5)
    ax.annotate("random start", xy=(e[0], y[0]), xytext=(1.5, y[0] + 0.05),
                color=ROI_INK, fontsize=13, fontweight="bold")
    ax.annotate("trained", xy=(e[-1], y[-1]), xytext=(14.5, y[-1] + 0.28),
                color=TEAL, fontsize=13, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=TEAL, lw=1.2))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 2.2)
    ax.set_xlabel("epoch", fontsize=13)
    ax.set_ylabel("loss", fontsize=13)
    ax.set_yticks([])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(0.98, 0.9, "steep drop, then a gentle flatten",
            transform=ax.transAxes, ha="right", color=MUTED, fontsize=13,
            style="italic")
    _save(fig, name)


# ---- random-weights bridge from Lecture 1 ----------------------------------
def random_weights_net(name="fig_random_weights.png"):
    """The Lecture 1 network with every weight a '?' \u2014 the honest starting
    state before training. Bridges 'a network runs forward' to 'who set them?'"""
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    xin, xhid, xout = 1.6, 6.0, 10.4
    inputs = [(xin, 4.2), (xin, 1.8)]
    hidden = [(xhid, 4.2), (xhid, 1.8)]
    out = (xout, 3.0)
    for (ix, iy) in inputs:
        for (hx, hy) in hidden:
            ax.add_patch(FancyArrowPatch((ix + 0.5, iy), (hx - 0.55, hy),
                        arrowstyle="-", lw=1.5, color=INK_SOFT, alpha=0.45,
                        zorder=1))
    for (hx, hy) in hidden:
        ax.add_patch(FancyArrowPatch((hx + 0.55, hy), (out[0] - 0.55, out[1]),
                    arrowstyle="-", lw=1.5, color=INK_SOFT, alpha=0.45,
                    zorder=1))
    # weight '?' labels on a couple of edges
    ax.text((xin + xhid) / 2, 3.35, "w = ?", color=RED, fontsize=15,
            fontweight="bold", ha="center")
    ax.text((xhid + xout) / 2, 3.9, "w = ?", color=RED, fontsize=15,
            fontweight="bold", ha="center")

    def node(cx, cy, face, edge, label):
        ax.add_patch(Circle((cx, cy), 0.55, facecolor=face, edgecolor=edge,
                            lw=2.4, zorder=3))
        ax.text(cx, cy, label, ha="center", va="center", color=INK,
                fontsize=15, fontweight="bold", zorder=4)
    for (ix, iy) in inputs:
        node(ix, iy, TEAL_SOFT, TEAL, "x")
    for (hx, hy) in hidden:
        node(hx, hy, AMBER_SOFT, AMBER, "ReLU")
    node(out[0], out[1], TEAL_SOFT, TEAL, "\u0177")
    ax.text(xin, 5.3, "input", ha="center", color=INK_SOFT, fontsize=14)
    ax.text(xhid, 5.3, "hidden", ha="center", color=INK_SOFT, fontsize=14)
    ax.text(xout, 4.1, "output", ha="center", color=INK_SOFT, fontsize=14)
    ax.text(6, 0.5, "every arrow carries a weight \u2014 but what number should it be?",
            ha="center", color=MUTED, fontsize=14, style="italic")
    _save(fig, name)


# =====================================================================
#  Lecture 4 · Making Training Actually Work (loss choice, tuning the
#  knobs, overfitting). One function per figure, course palette, rules 6/9.
# =====================================================================

# ---- Part 1 · softmax: scores -> probabilities -----------------------------
def softmax_bars(name="fig_softmax_bars.png"):
    """Raw class scores squashed by softmax into probabilities that sum to 1.
    The MALDI R/S example: scores 2.0, 0.5 -> 0.82, 0.18."""
    labels = ["R (resistant)", "S (susceptible)"]
    probs = [0.82, 0.18]
    colors = [TEAL, AMBER]
    fig, ax = plt.subplots(figsize=(9.4, 3.6))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.6, 1.6)
    for i, (lab, p, c) in enumerate(zip(labels, probs, colors)):
        yy = 1.0 - i
        ax.add_patch(FancyBboxPatch((0, yy - 0.28), 1.0, 0.56,
                    boxstyle="round,pad=0.005,rounding_size=0.03",
                    facecolor="#F1F1EC", edgecolor=HAIRLINE, lw=1.2, zorder=1))
        ax.add_patch(FancyBboxPatch((0, yy - 0.28), max(p, 0.02), 0.56,
                    boxstyle="round,pad=0.005,rounding_size=0.03",
                    facecolor=c, edgecolor="none", zorder=2))
        ax.text(-0.02, yy, lab, ha="right", va="center", color=INK_SOFT,
                fontsize=15, fontweight="bold")
        ax.text(p + 0.03, yy, f"{p:.2f}", ha="left", va="center", color=c,
                fontsize=16, fontweight="bold")
    ax.text(0.5, 1.42, "raw scores  R = 2.0,  S = 0.5   \u2192   softmax   \u2192   probabilities (sum to 1)",
            ha="center", va="center", color=INK, fontsize=14.5,
            fontweight="bold")
    ax.text(0.5, -0.48, "\u201c82% sure it\u2019s resistant\u201d  \u2014  any number of classes works the same way",
            ha="center", va="center", color=MUTED, fontsize=13,
            style="italic")
    ax.axis("off")
    _save(fig, name)


# ---- Part 1 · cross-entropy penalty curve ----------------------------------
def crossentropy_curve(name="fig_ce_curve.png"):
    """penalty = -log(probability you gave the TRUE class). Right (p→1) tiny,
    left (p→0, confidently wrong) explodes. The 'surprise' curve."""
    p = np.linspace(0.02, 1.0, 400)
    loss = -np.log(p)
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    ax.plot(p, loss, color=TEAL, lw=3.2, zorder=3, solid_capstyle="round")
    for px, col, lab, dy in [(0.95, TEAL, "p\u22481 \u2192 ~0", 0.25),
                             (0.5, AMBER, "p=0.5 \u2192 0.69", 0.35),
                             (0.1, RED, "confident & wrong \u2192 huge", 0.2)]:
        ax.scatter([px], [-np.log(px)], s=90, color=col, zorder=5)
        ax.annotate(lab, xy=(px, -np.log(px)), xytext=(px + 0.03, -np.log(px) + dy),
                    color=col, fontsize=13, fontweight="bold", va="bottom")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 4.1)
    ax.set_xlabel("probability you gave the TRUE class \u2192", fontsize=13)
    ax.set_ylabel("penalty (loss)", fontsize=13)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(0.5, -0.2, "being sure and wrong is the worst thing a classifier can do",
            transform=ax.transAxes, ha="center", color=MUTED, fontsize=13,
            style="italic")
    _save(fig, name)


# ---- Part 1 · the explicit cross-entropy formula ---------------------------
def crossentropy_formula(name="fig_ce_formula.png"):
    """Write the loss out so every later number is COMPUTED, not eyeballed:
    CE = -sum_i y_i log(yhat_i); for a one-hot truth only the true class
    survives, so CE = -log(p_true) (natural log). One worked substitution
    (-ln 0.90 = 0.105) anchors the arithmetic used on the next slides."""
    fig, ax = plt.subplots(figsize=(10.6, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # general definition
    ax.text(0.5, 0.88, r"$\mathrm{CE}\;=\;-\sum_i\, y_i\,\log(\hat{y}_i)$",
            ha="center", va="center", fontsize=27, color=INK)
    ax.text(0.5, 0.68,
            r"truth is one-hot: the true class has $y=1$, every other class $y=0$",
            ha="center", va="center", fontsize=14, color=MUTED)
    # reduction
    ax.add_patch(FancyBboxPatch((0.20, 0.40), 0.60, 0.16,
                boxstyle="round,pad=0.01,rounding_size=0.03",
                facecolor="#F1F1EC", edgecolor=HAIRLINE, lw=1.3))
    ax.text(0.5, 0.48,
            r"so only the TRUE class survives:  $\mathrm{CE}\;=\;-\log(p_{\mathrm{true}})$",
            ha="center", va="center", fontsize=22, color=TEAL, fontweight="bold")
    # worked substitution anchor
    ax.text(0.5, 0.20,
            r"e.g. truth is R and the model says $p_{\mathrm{true}}=0.90$:   "
            r"$\mathrm{CE}=-\ln(0.90)=0.105$",
            ha="center", va="center", fontsize=16, color=INK_SOFT)
    ax.text(0.5, 0.05,
            "natural log (ln); higher confidence on the truth \u2192 smaller loss",
            ha="center", va="center", fontsize=12.5, color=MUTED, style="italic")
    _save(fig, name)


# ---- Part 1 · worked cross-entropy on an MS call ---------------------------
def crossentropy_worked(name="fig_ce_worked.png", cards=None, reveal=True):
    """Two models both CALL the truly-resistant isolate R (both argmax R, so
    accuracy counts both 'correct'), at different confidence. Every loss is
    computed straight from CE = -ln(p_true) (the formula on the prior slide):
    -ln(0.90) = 0.105; -ln(0.55) = 0.598 — about 5.7x the loss for the same
    call, which is exactly the difference accuracy can't see.
    reveal=False leaves each penalty as '?' for the you-do (Quiz 4 Q1): the
    room plugs each probability into -ln(p_true)."""
    if cards is None:
        cards = [("Model A \u2014 sure and right", 0.90, 0.105, TEAL),
                 ("Model B \u2014 a hedge, still right", 0.55, 0.598, ROI_INK)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (title, pr, pen, col) in zip(axes, cards):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.02, 0.04), 0.96, 0.92,
                    boxstyle="round,pad=0.01,rounding_size=0.03",
                    facecolor=WHITE, edgecolor=HAIRLINE, lw=1.3))
        ax.text(0.5, 0.86, title, ha="center", color=col, fontsize=15,
                fontweight="bold")
        for i, (lab, val, c) in enumerate([("R (true)", pr, col),
                                           ("S", 1 - pr, AMBER)]):
            yy = 0.62 - i * 0.16
            ax.add_patch(FancyBboxPatch((0.28, yy - 0.05), 0.6, 0.1,
                        boxstyle="round,pad=0.002,rounding_size=0.01",
                        facecolor="#F1F1EC", edgecolor=HAIRLINE, lw=1.0))
            ax.add_patch(FancyBboxPatch((0.28, yy - 0.05), 0.6 * max(val, 0.02),
                        0.1, boxstyle="round,pad=0.002,rounding_size=0.01",
                        facecolor=c, edgecolor="none"))
            ax.text(0.26, yy, lab, ha="right", va="center", color=INK_SOFT,
                    fontsize=12)
            ax.text(0.9, yy, f"{val:.2f}", ha="left", va="center", color=c,
                    fontsize=12, fontweight="bold")
        pen_txt = f"\u2212ln({pr:.2f}) = {pen:.3f}" if reveal \
            else f"\u2212ln({pr:.2f}) = ?"
        ax.text(0.5, 0.2, pen_txt, ha="center",
                color=col, fontsize=16, fontweight="bold")
    footer = ("both call R (accuracy counts both \u2018correct\u2019) \u2014 but \u2212ln charges B\u2019s hedge 5.7\u00d7 the loss"
              if reveal else
              "same truth (R) \u2014 plug each into CE = \u2212ln(p_true); which loss is bigger?  \u00b7  Quiz 4 Q1")
    fig.text(0.5, -0.02, footer,
             ha="center", color=INK, fontsize=13.5, fontweight="bold")
    _save(fig, name)


# ---- Part 2 · mini-batch vs full-batch (how data is fed) -------------------
def minibatch_paths(name="fig_minibatch.png"):
    """Same valley: full batch takes one careful step per pass; mini-batch takes
    many quick, jittery steps per pass."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.add_patch(plt.matplotlib.patches.Ellipse((5, 5), 8, 9,
                    fill=False, edgecolor=HAIRLINE, lw=1.4))
        ax.add_patch(plt.matplotlib.patches.Ellipse((5, 5), 4.6, 5.2,
                    fill=False, edgecolor=HAIRLINE, lw=1.2))
    # full batch: one smooth arc top -> bottom
    axes[0].set_title("FULL BATCH", color=TEAL, fontsize=15, fontweight="bold")
    axes[0].annotate("", xy=(5, 4.6), xytext=(5, 8.4),
                     arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2.6,
                                     connectionstyle="arc3,rad=0.12"))
    axes[0].scatter([5], [8.4], s=90, color=TEAL, zorder=5)
    axes[0].scatter([5], [4.6], s=90, color=TEAL, zorder=5)
    axes[0].text(5, 1.4, "smooth \u2014 but 1 step / pass", ha="center",
                 color=MUTED, fontsize=12.5)
    # mini-batch: jittery zigzag
    axes[1].set_title("MINI-BATCH", color=AMBER, fontsize=15, fontweight="bold")
    xs = [5, 4.0, 6.1, 4.3, 6.0, 4.7, 5.2]
    ys = [8.4, 7.4, 6.6, 5.9, 5.4, 5.0, 4.7]
    axes[1].plot(xs, ys, color=AMBER, lw=2.4, zorder=4)
    axes[1].scatter([xs[0]], [ys[0]], s=90, color=AMBER, zorder=5)
    axes[1].scatter([xs[-1]], [ys[-1]], s=90, color=TEAL, zorder=5)
    axes[1].annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                     arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=2.4))
    axes[1].text(5, 1.4, "noisier \u2014 but many steps / pass", ha="center",
                 color=ROI_INK, fontsize=12.5)
    fig.text(0.5, -0.01, "20 batches = 20 updates per epoch (vs. 1 for full-batch): faster, and the noise finds flatter, better-generalizing minima",
             ha="center", color=INK_SOFT, fontsize=12.5, style="italic")
    _save(fig, name)


# ---- Part 2 · Adam = adaptive rate + momentum ------------------------------
def adam_momentum(name="fig_adam.png"):
    """A hillside with a small bump: a plain step stalls on the bump; momentum
    carries the ball over it toward the valley floor (low loss)."""
    x = np.linspace(0, 10, 400)
    hill = (2.6 * np.exp(-0.5 * (x - 1.2) ** 2)
            + 0.9 * np.exp(-1.2 * (x - 5.2) ** 2)  # the bump/plateau
            + 0.15 * x)
    hill = hill.max() - hill  # invert so it descends to a valley floor
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.plot(x, hill, color=MUTED, lw=2.0, zorder=2)
    # plain ball stuck just before the bump
    bx = 4.4
    by = np.interp(bx, x, hill)
    ax.scatter([bx], [by], s=180, facecolor=WHITE, edgecolor=RED, lw=2.6,
               zorder=5)
    ax.annotate("plain step: stuck on the bump", xy=(bx, by),
                xytext=(bx - 1.9, by + 0.9), color=RED, fontsize=12.5,
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=RED, lw=1.0))
    # momentum ball: trail then carried over the bump
    for tx, al in [(2.6, 0.25), (3.3, 0.45), (4.0, 0.65)]:
        ax.scatter([tx], [np.interp(tx, x, hill)], s=70, color=TEAL,
                   alpha=al, zorder=4)
    mx = 6.6
    ax.scatter([mx], [np.interp(mx, x, hill)], s=150, color=TEAL, zorder=6)
    ax.annotate("momentum carries it through", xy=(mx, np.interp(mx, x, hill)),
                xytext=(mx - 0.3, np.interp(mx, x, hill) + 1.0), color=TEAL,
                fontsize=12.5, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=TEAL, lw=1.0))
    ax.scatter([x[-1]], [hill[-1]], s=110, color=TEAL, zorder=6)
    ax.text(x[-1], hill[-1] + 0.5, "valley floor\n= low loss", ha="right",
            color=TEAL, fontsize=12, fontweight="bold")
    ax.set_xlim(0, 10)
    ax.set_ylim(hill.min() - 0.5, hill.max() + 1.4)
    ax.axis("off")
    ax.text(0.5, -0.04, "Adam = adaptive learning rate (per weight)  +  momentum (inertia through bumps)",
            transform=ax.transAxes, ha="center", color=INK, fontsize=14,
            fontweight="bold")
    _save(fig, name)


# ---- Part 2 · learning-rate schedule (warmup-stable-decay) ------------------
def lr_schedule(name="fig_lr_schedule.png"):
    """Learning rate vs training step: warmup ramps up, stable holds high, decay
    drops to tiny steps to settle in. The Lecture 2 learning-rate callback."""
    step = np.linspace(0, 30, 400)
    warm, stable = 4.0, 20.0
    lr = np.where(step < warm, step / warm,
                  np.where(step < stable, 1.0,
                           np.maximum(0.06, 1.0 - (step - stable) / (30 - stable))))
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    ax.plot(step, lr, color=TEAL, lw=3.2, zorder=3, solid_capstyle="round")
    ax.fill_between(step, lr, color=TEAL, alpha=0.10, zorder=1)
    for xb in (warm, stable):
        ax.axvline(xb, color=HAIRLINE, lw=1.2, zorder=1)
    ax.text(warm / 2, 1.12, "warmup", ha="center", color=ROI_INK, fontsize=13,
            fontweight="bold")
    ax.text((warm + stable) / 2, 1.12, "stable (hold high)", ha="center",
            color=TEAL, fontsize=13, fontweight="bold")
    ax.text((stable + 30) / 2, 1.12, "decay", ha="center", color=RED,
            fontsize=13, fontweight="bold")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1.3)
    ax.set_xlabel("training step \u2192", fontsize=13)
    ax.set_ylabel("learning rate (step size)", fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(0.5, -0.16, "ramp up gently \u00b7 travel fast \u00b7 shrink the step to settle in  (cosine is the smooth-decay cousin)",
            transform=ax.transAxes, ha="center", color=MUTED, fontsize=12.5,
            style="italic")
    _save(fig, name)


# ---- Part 2 · activation choices (ReLU vs sigmoid slope) -------------------
def activation_choices(name="fig_activation_choices.png"):
    """Why ReLU is the hidden-layer default: its slope is 1 for positive inputs
    (passes the gradient), while sigmoid's slope is at most 0.25 and ~0 in the
    tails (squashes the gradient) — the seed of vanishing gradients."""
    z = np.linspace(-4, 4, 400)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    # ReLU
    axes[0].plot(z, np.maximum(0, z), color=TEAL, lw=3.2, solid_capstyle="round")
    axes[0].set_title("ReLU \u2014 hidden-layer default", color=TEAL, fontsize=15,
                      fontweight="bold", pad=8)
    axes[0].text(0.5, -0.2, "slope = 1 for positives \u2192 passes the gradient",
                 transform=axes[0].transAxes, ha="center", color=INK_SOFT,
                 fontsize=13, fontweight="bold")
    # Sigmoid
    sig = 1 / (1 + np.exp(-z))
    axes[1].plot(z, sig, color=AMBER, lw=3.2, solid_capstyle="round")
    axes[1].set_title("Sigmoid \u2014 only for a 0\u20131 output", color=ROI_INK,
                      fontsize=15, fontweight="bold", pad=8)
    axes[1].text(0.5, -0.2, "slope \u2264 0.25, ~0 in the tails \u2192 squashes the gradient",
                 transform=axes[1].transAxes, ha="center", color=RED,
                 fontsize=13, fontweight="bold")
    # flat-tail shading to make the squash visible
    axes[1].fill_between(z, sig, where=(np.abs(z) > 2.2), color=RED, alpha=0.12)
    for ax in axes:
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-1.4, 4.1) if ax is axes[0] else ax.set_ylim(-0.15, 1.15)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.subplots_adjust(wspace=0.14, bottom=0.16)
    _save(fig, name)


# ---- Part 2 · vanishing / exploding gradient chain -------------------------
def _grad_chain(ax, y, factor, n, start, color, reveal_last, label):
    """One horizontal chain of layer values, each = previous x factor. Gradient
    flows right->left visually but we lay it left(input)->right(loss). Draws n+1
    node values; if reveal_last is False the final (input-side) value is '?'."""
    vals = [start * factor ** i for i in range(n + 1)]
    xs = np.linspace(0.8, 11.2, n + 1)
    for i, (xx, v) in enumerate(zip(xs, vals)):
        show_q = (i == n) and not reveal_last
        txt = "?" if show_q else (f"{v:g}" if v >= 1 else f"{v:.3g}")
        fs = 13 + 3 * (v / max(vals)) if factor > 1 else 13 + 3 * (vals[0] and (v / max(vals)))
        ax.text(xx, y, txt, ha="center", va="center", color=color,
                fontsize=16 if show_q else 15, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc=AMBER_SOFT if show_q else PAPER,
                          ec=AMBER if show_q else "none"))
        if i < n:
            ax.annotate("", xy=(xs[i + 1] - 0.35, y), xytext=(xx + 0.35, y),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6,
                                        alpha=0.55))
    ax.text(11.9, y, label, ha="left", va="center", color=color, fontsize=12,
            fontweight="bold")


def vanishing_chain(reveal, name):
    """Backprop's chain-multiply across many layers (Lecture 2's 'multiply the
    local effects', now deep). reveal='ido': both a vanishing (×0.5) and an
    exploding (×2) row, fully worked. reveal='youdo' (= Quiz 4 Q4): one ×0.5
    chain of 5 layers with the input-side value left as '?' for the room."""
    if reveal == "ido":
        fig, ax = plt.subplots(figsize=(11.5, 3.8))
        ax.set_xlim(0, 15)
        ax.set_ylim(0, 4)
        ax.axis("off")
        ax.text(6, 3.65, "gradient travels back through every layer, multiplied by that layer\u2019s local slope",
                ha="center", color=INK_SOFT, fontsize=13, fontweight="bold")
        ax.text(0.2, 2.7, "\u00d70.5 each", ha="left", color=MUTED, fontsize=12)
        _grad_chain(ax, 2.4, 0.5, 6, 1.0, TEAL, True,
                    "\u2192 vanishes:\n   early layers\n   barely learn")
        ax.text(0.2, 1.2, "\u00d72 each", ha="left", color=MUTED, fontsize=12)
        _grad_chain(ax, 0.9, 2.0, 6, 1.0, RED, True,
                    "\u2192 explodes:\n   loss \u2192 NaN")
        ax.text(6, 0.05, "0.5 \u00d7 0.5 \u00d7 0.5 \u00d7 0.5 \u00d7 0.5 \u00d7 0.5  =  0.5\u2076  \u2248  0.016   \u2014  Lecture 2\u2019s chain, now across MANY layers",
                ha="center", color=INK, fontsize=13, fontweight="bold")
    else:  # youdo = Quiz 4 Q4
        fig, ax = plt.subplots(figsize=(11.5, 2.8))
        ax.set_xlim(0, 15)
        ax.set_ylim(0, 3)
        ax.axis("off")
        ax.text(6, 2.5, "a gradient of 1.0 leaves the loss and passes back through 5 layers, each \u00d7 0.5",
                ha="center", color=INK_SOFT, fontsize=13, fontweight="bold")
        _grad_chain(ax, 1.4, 0.5, 5, 1.0, TEAL, False,
                    "= ?  vanish or\n   explode?")
        ax.text(6, 0.2, "what gradient reaches the FIRST layer \u2014 and does it vanish or explode?",
                ha="center", color=MUTED, fontsize=12.5, style="italic")
    _save(fig, name)


# ---- Part 2 · hyperparameter search: grid vs random ------------------------
def hyperparam_search(name="fig_search.png"):
    """Grid vs random search over learning rate (x) x batch size (y): the same
    16-run budget tests only 4 distinct learning rates on a grid, but 16 on
    random — which wins when one knob dominates."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    # grid
    gx = np.linspace(0.15, 0.85, 4)
    gy = np.linspace(0.15, 0.85, 4)
    for x in gx:
        for y in gy:
            axes[0].scatter([x], [y], s=70, color=TEAL, zorder=3)
    axes[0].set_title("grid \u2014 16 runs, only 4 learning rates", color=TEAL,
                      fontsize=13.5, fontweight="bold", pad=8)
    # random
    rng = np.random.default_rng(4)
    rx = rng.uniform(0.1, 0.9, 16)
    ry = rng.uniform(0.1, 0.9, 16)
    axes[1].scatter(rx, ry, s=70, color=AMBER, zorder=3)
    axes[1].set_title("random \u2014 16 runs, 16 learning rates", color=ROI_INK,
                      fontsize=13.5, fontweight="bold", pad=8)
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("learning rate \u2192", fontsize=12.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("batch size", fontsize=12.5)
    fig.text(0.5, -0.02, "when one knob dominates, random samples far more of it for the same budget \u2014 smarter still: Bayesian search learns from past trials",
             ha="center", color=INK_SOFT, fontsize=12, style="italic")
    _save(fig, name)


# ---- Part 2 · batch-size trade-off (sharp vs flat minima) ------------------
def batch_tradeoff(name="fig_batch_tradeoff.png"):
    """Same loss plane, two batch sizes: small batch takes many small noisy steps
    into a wide FLAT basin (generalizes); large batch takes few big confident
    steps into a narrow SHARP basin (worse generalization)."""
    from matplotlib.patches import Ellipse
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    # small batch: wide flat basin, jittery path
    axes[0].set_title("SMALL BATCH \u2192 many small, noisy steps", color=ROI_INK,
                      fontsize=13, fontweight="bold")
    for rx, ry in [(8.8, 3.4), (6.0, 2.2), (3.2, 1.2)]:
        axes[0].add_patch(Ellipse((5, 3), rx, ry, fill=False,
                          edgecolor=HAIRLINE, lw=1.3))
    px = [0.6, 1.6, 1.2, 2.4, 2.1, 3.2, 3.9, 4.4]
    py = [5.4, 4.7, 3.9, 4.0, 3.3, 3.2, 2.9, 3.0]
    axes[0].plot(px, py, color=RED, lw=1.8, zorder=3)
    axes[0].scatter(px, py, s=22, color=RED, zorder=4)
    axes[0].scatter([5], [3], s=70, color=TEAL, zorder=5)
    axes[0].text(5, 3 - 1.0, "flat basin", ha="center", color=TEAL, fontsize=11)
    # large batch: narrow sharp basin, few straight steps
    axes[1].set_title("LARGE BATCH \u2192 few big, confident steps", color=RED,
                      fontsize=13, fontweight="bold")
    for rx, ry in [(4.6, 3.0), (2.6, 1.7), (1.1, 0.7)]:
        axes[1].add_patch(Ellipse((5, 3), rx, ry, fill=False,
                          edgecolor=HAIRLINE, lw=1.3))
    axes[1].plot([1.0, 3.2, 4.7], [5.4, 4.0, 3.2], color=TEAL, lw=2.4, zorder=3)
    axes[1].scatter([1.0, 3.2], [5.4, 4.0], s=30, color=TEAL, zorder=4)
    axes[1].annotate("", xy=(4.85, 3.1), xytext=(4.6, 3.25),
                     arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2.2))
    axes[1].scatter([5], [3], s=70, color=TEAL, zorder=5)
    axes[1].text(5, 3 - 1.1, "sharp basin", ha="center", color=TEAL, fontsize=11)
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6.2)
        ax.axis("off")
    fig.text(0.5, -0.02, "a flat basin is forgiving (a small train\u2192new-data shift barely changes the loss) \u2014 a sharp basin generalizes worse",
             ha="center", color=INK_SOFT, fontsize=12, style="italic")
    _save(fig, name)


# ---- Part 2 · batch size on the loss curve (the you-do cue) ----------------
def batch_loss_curve(name="fig_batch_youdo.png"):
    """Batch size's fingerprint on the loss curve: large batch = smooth descent
    but plateaus higher; small batch = jagged but reaches lower. Smoothness is
    not the goal — where it ends up is. The visual cue for Quiz 4 Q2."""
    e = np.linspace(0, 30, 240)
    large = 1.9 * np.exp(-0.16 * e) + 0.55
    rng = np.random.default_rng(1)
    small = 1.9 * np.exp(-0.16 * e) + 0.2 + 0.16 * rng.standard_normal(e.size) * np.exp(-0.03 * e)
    small = np.clip(small, 0.12, None)
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    ax.plot(e, large, color=TEAL, lw=2.8, zorder=3, label="large batch")
    ax.plot(e, small, color=AMBER, lw=1.8, zorder=2, label="small batch")
    ax.annotate("plateaus higher", xy=(e[-1], large[-1]), xytext=(20, large[-1] + 0.35),
                color=TEAL, fontsize=12, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=TEAL, lw=1.0))
    ax.annotate("reaches lower", xy=(e[-1], small[-1]), xytext=(19, small[-1] - 0.45),
                color=ROI_INK, fontsize=12, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=ROI_INK, lw=1.0))
    ax.plot([e[-1], e[-1]], [small[-1], large[-1]], color=RED, lw=1.4,
            ls=(0, (3, 3)))
    ax.text(e[-1] - 0.6, (small[-1] + large[-1]) / 2, "gap", ha="right",
            color=RED, fontsize=11)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 2.4)
    ax.set_xlabel("epoch \u2192", fontsize=13)
    ax.set_ylabel("loss", fontsize=13)
    ax.set_yticks([])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=13)
    ax.text(0.5, -0.27, "a clean curve that stalls high is the too-big fingerprint \u2014 judge a run by where it ENDS UP, not how smooth it looks",
            transform=ax.transAxes, ha="center", color=MUTED, fontsize=12,
            style="italic")
    fig.subplots_adjust(bottom=0.2)
    _save(fig, name)


# ---- Part 3 · fit trio (underfit / just right / overfit) -------------------
def fit_trio(name="fig_fit_trio.png"):
    """Same seven noisy points, three models: a line that misses the trend
    (underfit), a smooth trend curve (just right), a wiggle through every point
    (overfit)."""
    xp = np.array([1.0, 2.0, 3.0, 4.2, 5.4, 6.6, 7.6])
    yp = np.array([1.2, 2.1, 1.7, 3.0, 2.7, 3.9, 3.6])
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.3))
    titles = [("underfitting", ROI_INK, "too simple \u2014 misses the trend"),
              ("just right", TEAL, "captures the trend, ignores the wiggle"),
              ("overfitting", RED, "threads every noisy point \u2014 memorizes")]
    xs = np.linspace(0.7, 7.9, 300)
    # underfit: straight line fit
    m, b = np.polyfit(xp, yp, 1)
    axes[0].plot(xs, m * xs + b, color=ROI_INK, lw=2.6)
    # just right: smooth quadratic-ish trend
    c = np.polyfit(xp, yp, 2)
    axes[1].plot(xs, np.polyval(c, xs), color=TEAL, lw=2.6)
    # overfit: high-degree wiggle through the points
    c6 = np.polyfit(xp, yp, 6)
    axes[2].plot(xs, np.polyval(c6, xs), color=RED, lw=2.6)
    for ax, (title, col, sub) in zip(axes, titles):
        ax.scatter(xp, yp, s=45, color=INK_SOFT, zorder=5)
        ax.set_title(title, color=col, fontsize=15, fontweight="bold", pad=6)
        ax.set_xlim(0.4, 8.2)
        ax.set_ylim(0, 5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, -0.12, sub, transform=ax.transAxes, ha="center",
                color=MUTED, fontsize=11.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.subplots_adjust(wspace=0.12, bottom=0.14)
    _save(fig, name)


# ---- Part 3 · train / validation curve gallery -----------------------------
def _tv_panel(ax, kind):
    e = np.linspace(0, 30, 200)
    if kind == "underfit":
        # high-and-flat: barely moves, matching the quiz's Q2(a) tikz panel
        tr = 0.35 * np.exp(-0.06 * e) + 1.80
        va = tr + 0.12
    elif kind == "just_right":
        tr = 1.9 * np.exp(-0.22 * e) + 0.2
        va = tr + 0.14
    elif kind == "overfit":
        tr = 2.0 * np.exp(-0.2 * e) + 0.08
        va = 1.9 * np.exp(-0.28 * e) + 0.3 + 0.9 * (1 / (1 + np.exp(-(e - 16) / 3)))
    else:  # lr_too_high: single jagged diverging training curve
        rng = np.random.default_rng(3)
        va = None
        tr = 1.0 + 0.03 * e + (0.35 + 0.02 * e) * np.abs(np.sin(1.1 * e))
    ax.plot(e, tr, color=TEAL, lw=2.6, zorder=3, solid_capstyle="round")
    if va is not None:
        ax.plot(e, va, color=RED, lw=2.4, ls=(0, (5, 4)), zorder=3)
        if kind == "overfit":
            imin = int(np.argmin(va))
            ax.scatter([e[imin]], [va[imin]], s=40, color=ROI_INK, zorder=5)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 3.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("epoch \u2192", fontsize=11, color=MUTED)


def trainval_curves(labeled=True, name="fig_trainval_ido.png"):
    """labeled=True (I-do): three named signatures — underfitting (both high),
    just right (both low, small gap), overfitting (train down, val up).
    labeled=False (you-do = Quiz 4 Q2): four pairs (a)-(d) in the quiz's exact
    order — (a) underfit, (b) just right, (c) overfit, (d) LR too high."""
    if labeled:
        specs = [("underfit", ROI_INK, "underfitting"),
                 ("just_right", TEAL, "just right"),
                 ("overfit", RED, "overfitting")]
        fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.3))
    else:
        specs = [("underfit", INK, "(a)"), ("just_right", INK, "(b)"),
                 ("overfit", INK, "(c)"), ("lr_too_high", INK, "(d)")]
        fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.1))
    for ax, (kind, col, title) in zip(axes, specs):
        _tv_panel(ax, kind)
        ax.set_title(title, color=col, fontsize=17, fontweight="bold", pad=8)
    # legend on the first panel
    axes[0].plot([], [], color=TEAL, lw=2.6, label="training")
    axes[0].plot([], [], color=RED, lw=2.4, ls=(0, (5, 4)), label="validation")
    axes[0].legend(loc="upper right", frameon=False, fontsize=10.5)
    fig.subplots_adjust(wspace=0.14, bottom=0.12)
    _save(fig, name)


# ---- Part 3 · early stopping -----------------------------------------------
def early_stopping(name="fig_early_stopping.png"):
    """Training loss keeps falling while validation bottoms out then rises; stop
    at the validation minimum."""
    e = np.linspace(0, 30, 200)
    tr = 2.0 * np.exp(-0.16 * e) + 0.08
    va = 1.9 * np.exp(-0.3 * e) + 0.35 + 0.8 * (1 / (1 + np.exp(-(e - 13) / 2.5)))
    imin = int(np.argmin(va))
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    ax.plot(e, tr, color=TEAL, lw=2.8, zorder=3, label="training")
    ax.plot(e, va, color=RED, lw=2.6, ls=(0, (5, 4)), zorder=3, label="validation")
    ax.axvline(e[imin], color=ROI_INK, lw=2.0, ls=(0, (4, 4)), zorder=2)
    ax.scatter([e[imin]], [va[imin]], s=70, color=ROI_INK, zorder=5)
    ax.text(e[imin], 2.55, "STOP here", ha="center", color=ROI_INK,
            fontsize=13, fontweight="bold")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 2.7)
    ax.set_xlabel("epoch \u2192", fontsize=13)
    ax.set_ylabel("loss", fontsize=13)
    ax.set_yticks([])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=13)
    ax.text(0.5, -0.27, "keep the model from the epoch where validation loss bottoms out \u2014 free, and always worth doing",
            transform=ax.transAxes, ha="center", color=MUTED, fontsize=12,
            style="italic")
    fig.subplots_adjust(bottom=0.2)
    _save(fig, name)


# ---- Part 3 · dropout (neurons switched off) -------------------------------
def _dropout_panel(ax, dropped, caption):
    """2 inputs -> 4 hidden -> 1 output; hidden units in `dropped` are X'd out."""
    ins = [(1.0, 3.4), (1.0, 1.6)]
    hid = [(4.0, 4.4), (4.0, 3.2), (4.0, 2.0), (4.0, 0.8)]
    out = (7.0, 2.6)
    for (ix, iy) in ins:
        for j, (hx, hy) in enumerate(hid):
            al = 0.12 if j in dropped else 0.4
            ax.plot([ix + 0.3, hx - 0.3], [iy, hy], color=HAIRLINE, lw=1.1,
                    alpha=al, zorder=1)
    for j, (hx, hy) in enumerate(hid):
        al = 0.12 if j in dropped else 0.5
        ax.plot([hx + 0.3, out[0] - 0.3], [hy, out[1]], color=HAIRLINE,
                lw=1.1, alpha=al, zorder=1)
    for (ix, iy) in ins:
        ax.add_patch(Circle((ix, iy), 0.28, facecolor=WHITE, edgecolor=INK_SOFT,
                            lw=2.0, zorder=3))
    for j, (hx, hy) in enumerate(hid):
        if j in dropped:
            ax.add_patch(Circle((hx, hy), 0.3, facecolor=WHITE, edgecolor=MUTED,
                                lw=1.6, ls=(0, (2, 2)), alpha=0.5, zorder=3))
            ax.plot([hx - 0.2, hx + 0.2], [hy - 0.2, hy + 0.2], color=RED, lw=2.0,
                    zorder=4)
            ax.plot([hx - 0.2, hx + 0.2], [hy + 0.2, hy - 0.2], color=RED, lw=2.0,
                    zorder=4)
        else:
            ax.add_patch(Circle((hx, hy), 0.3, facecolor=TEAL_SOFT,
                                edgecolor=TEAL, lw=2.0, zorder=3))
    ax.add_patch(Circle((out[0], out[1]), 0.32, facecolor=TEAL, edgecolor=TEAL,
                        lw=2.0, zorder=3))
    ax.text(out[0], out[1], "\u0177", ha="center", va="center", color=WHITE,
            fontsize=13, fontweight="bold", zorder=4)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    ax.text(4, 0.05, caption, ha="center", color=MUTED, fontsize=12)


def dropout_panels(name="fig_dropout.png"):
    """Same net, two training steps: a different random ~half of the hidden
    neurons is switched off each step, so no neuron becomes a linchpin."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    _dropout_panel(axes[0], {1, 3}, "training step 1 \u2014 drop h2, h4")
    _dropout_panel(axes[1], {0, 2}, "step 2 \u2014 a different random set drops (h1, h3)")
    fig.text(0.5, -0.01, "the net can\u2019t lean on any one neuron \u2192 redundant, robust features (a committee that votes) \u00b7 at test time all neurons are back on",
             ha="center", color=INK_SOFT, fontsize=12, style="italic")
    _save(fig, name)


# ---- Part 3 · L2 = weight decay --------------------------------------------
def weight_decay(name="fig_weight_decay.png"):
    """Same four noisy points: big weights let the curve thread every one exactly
    (output = noise, memorized); small weights keep it smooth (output ≈ signal)."""
    xp = np.array([0.6, 1.7, 2.8, 3.9])
    yp = np.array([1.0, 2.2, 1.4, 2.0])
    xs = np.linspace(0.4, 4.1, 300)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    # big weights: high-degree interpolant through every point
    c3 = np.polyfit(xp, yp, 3)
    axes[0].plot(xs, np.polyval(c3, xs), color=RED, lw=2.6)
    axes[0].set_title("big weights", color=RED, fontsize=14, fontweight="bold")
    axes[0].text(0.5, -0.12, "output = noise (memorized)", transform=axes[0].transAxes,
                 ha="center", color=MUTED, fontsize=12)
    # small weights: gentle linear-ish trend
    m, b = np.polyfit(xp, yp, 1)
    axes[1].plot(xs, m * xs + b, color=TEAL, lw=2.6)
    axes[1].set_title("small weights", color=TEAL, fontsize=14, fontweight="bold")
    axes[1].text(0.5, -0.12, "output \u2248 signal (generalizes)", transform=axes[1].transAxes,
                 ha="center", color=MUTED, fontsize=12)
    for ax in axes:
        ax.scatter(xp, yp, s=45, color=INK_SOFT, zorder=5)
        ax.set_xlim(0.2, 4.3)
        ax.set_ylim(0, 3)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.text(0.5, -0.02, "total loss = fit the data + \u03bb \u00b7 \u03a3 w\u00b2  \u2014  every step nudges weights toward 0 (\u201cweight decay\u201d), so the curve can\u2019t bend through the noise",
             ha="center", color=INK_SOFT, fontsize=12, style="italic")
    fig.subplots_adjust(wspace=0.12, bottom=0.16)
    _save(fig, name)


# ---- Part 3 · class imbalance (real DRIAMS counts) -------------------------
def class_imbalance(name="fig_imbalance.png"):
    """Real DRIAMS S. aureus + oxacillin class counts (697 susceptible vs 41
    resistant): 'always predict susceptible' scores ~94% accuracy while catching
    ZERO resistant isolates. Why plain accuracy misleads on a rare class.
    Source: DRIAMS, Dryad doi:10.5061/dryad.bzkh1899q, CC0."""
    npz = ROOT / "data" / "slices" / "driams_c_saureus_oxacillin.npz"
    if npz.exists():
        y = np.load(npz, allow_pickle=True)["y"]
        n_s = int((y == 0).sum())
        n_r = int((y == 1).sum())
    else:
        n_s, n_r = 697, 41
    total = n_s + n_r
    acc = 100 * n_s / total
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    bars = ax.bar([0, 1], [n_s, n_r], width=0.55, color=[TEAL, RED], zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"susceptible (S)\n{n_s}  \u2014  {100*n_s/total:.0f}%",
                        f"resistant (R)\n{n_r}  \u2014  {100*n_r/total:.0f}%"],
                       fontsize=13)
    for b, v in zip(bars, [n_s, n_r]):
        ax.text(b.get_x() + b.get_width() / 2, v + 12, str(v), ha="center",
                color=INK, fontsize=14, fontweight="bold")
    ax.set_ylim(0, n_s * 1.18)
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.annotate("a model that ALWAYS says \u201csusceptible\u201d\n"
                f"scores {acc:.0f}% accuracy \u2014 and catches 0 of {n_r} resistant",
                xy=(1, n_r), xytext=(1.05, n_s * 0.6), ha="center",
                color=RED, fontsize=12.5, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6))
    ax.text(0.5, 0.97, "DRIAMS \u00b7 S. aureus + oxacillin \u00b7 real class counts",
            transform=ax.transAxes, ha="center", va="top", color=INK_SOFT,
            fontsize=12, style="italic")
    _save(fig, name)


# =====================================================================
#  Lecture 5 · Convolutional Networks (why conv, the conv mechanic cell-
#  by-cell in the Hung-yi-Lee style, stride, padding, output size,
#  pooling, 1D on spectra, detection). One function per figure, course
#  palette, rules 6/9. Every number here is recomputed exact so the deck,
#  the slide notes, and the Quiz 5 key all agree (rule 3).
# =====================================================================

def _cellgrid(ax, data, x0, ytop, cell, reveal=None, hi=None,
              face=WHITE, edge=HAIRLINE, txt=INK, fontsize=17,
              hi_face=AMBER_SOFT, hi_edge=AMBER, na="?"):
    """Draw a grid of numeric cells with row 0 at the TOP (top-left origin).
    reveal: set of (r,c) to show a value; cells not in reveal show `na` (?).
            None means reveal everything. hi: set of (r,c) drawn amber.
    Returns (left, top, right, bottom) in data coords."""
    rows, cols = len(data), len(data[0])
    hi = hi or set()
    for r in range(rows):
        for c in range(cols):
            cx = x0 + c * cell
            cy = ytop - r * cell
            on = (r, c) in hi
            ax.add_patch(FancyBboxPatch(
                (cx, cy - cell), cell * 0.92, cell * 0.92,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                facecolor=hi_face if on else face,
                edgecolor=hi_edge if on else edge,
                lw=2.6 if on else 1.3, zorder=3))
            shown = (reveal is None) or ((r, c) in reveal)
            v = data[r][c]
            vtxt = (f"{v:g}" if isinstance(v, (int, float)) else str(v)) if shown else na
            ax.text(cx + cell * 0.46, cy - cell * 0.46, vtxt,
                    ha="center", va="center",
                    color=(ROI_INK if on else txt) if shown else AMBER,
                    fontsize=fontsize, fontweight="bold" if on or not shown else "normal",
                    zorder=4, family="monospace")
    return (x0, ytop, x0 + cols * cell, ytop - rows * cell)


def _prodstr(win, filt):
    """Spelled-out multiply-add string for one window (rule 6/3): every product
    written out, then the sum. Uses unicode dot and minus."""
    k = len(filt)
    terms = []
    for i in range(k):
        for j in range(k):
            b = filt[i][j]
            bstr = f"{b}" if b >= 0 else f"({b})"
            terms.append(f"{win[i][j]}\u00b7{bstr}")
    return "  +  ".join(terms).replace("-", "\u2212")


def _convmap(image, filt, stride):
    n, k = len(image), len(filt)
    o = (n - k) // stride + 1
    return [[sum(image[r * stride + i][c * stride + j] * filt[i][j]
                 for i in range(k) for j in range(k))
             for c in range(o)] for r in range(o)], o


def conv_walk(image, filt, stride=1, window=None, reveal=None, calc="none",
              name="fig_conv.png", filt_note="", map_label="feature map",
              note=None, calc_text=None):
    """The Hung-yi-Lee sliding-window walkthrough, computed cell by cell:
    image (n\u00d7n)  \u2217  filter (k\u00d7k)  =  feature map (o\u00d7o).
    window=(r,c): highlight that output cell + its receptive field in the image.
    reveal: set of output (r,c) to fill; None = all; empty set = all '?'.
    calc in {'full','short','rule','none'}: what to print under the grids."""
    n, k = len(image), len(filt)
    M, o = _convmap(image, filt, stride)
    reveal = set(reveal) if reveal is not None else None

    fig, ax = plt.subplots(figsize=(12.6, 5.3))
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0, 7)
    ax.axis("off")
    cell = 0.66

    # receptive-field cells in the image for the current window
    hi_img = set()
    if window is not None:
        wr, wc = window
        for i in range(k):
            for j in range(k):
                hi_img.add((wr * stride + i, wc * stride + j))

    # image grid (left)
    ix0, iytop = 0.4, 6.15
    _cellgrid(ax, image, ix0, iytop, cell, hi=hi_img, fontsize=16)
    ax.text(ix0 + n * cell / 2, iytop + 0.32, f"image  {n}\u00d7{n}",
            ha="center", color=INK_SOFT, fontsize=14, fontweight="bold")
    if window is not None:
        wr, wc = window
        rx = ix0 + wc * stride * cell
        rtop = iytop - wr * stride * cell
        ax.add_patch(FancyBboxPatch((rx - 0.03, rtop - k * cell - 0.03),
                    k * cell + 0.06, k * cell + 0.06,
                    boxstyle="round,pad=0.01,rounding_size=0.04",
                    fill=False, edgecolor=AMBER, lw=3.0, zorder=6))

    # star
    ax.text(ix0 + n * cell + 0.42, 4.5, "\u2217", ha="center", va="center",
            fontsize=30, color=INK_SOFT)

    # filter grid (middle)
    fx0, fytop = ix0 + n * cell + 0.95, 5.55
    _cellgrid(ax, filt, fx0, fytop, cell, txt=TEAL, edge=TEAL,
              fontsize=16)
    ax.text(fx0 + k * cell / 2, fytop + 0.32, f"filter  {k}\u00d7{k}",
            ha="center", color=TEAL, fontsize=14, fontweight="bold")
    if filt_note:
        ax.text(fx0 + k * cell / 2, fytop - k * cell - 0.28, filt_note,
                ha="center", color=MUTED, fontsize=12, style="italic")

    # equals
    ax.text(fx0 + k * cell + 0.5, 4.5, "=", ha="center", va="center",
            fontsize=28, color=INK_SOFT)

    # feature map (right)
    mx0, mytop = fx0 + k * cell + 1.05, 5.55
    hi_map = {window} if window is not None else set()
    _cellgrid(ax, M, mx0, mytop, cell, reveal=reveal, hi=hi_map, fontsize=17)
    ax.text(mx0 + o * cell / 2, mytop + 0.32, f"{map_label}  {o}\u00d7{o}",
            ha="center", color=INK_SOFT, fontsize=14, fontweight="bold")

    # calc / note line(s)
    if calc_text is not None:
        line = calc_text
    elif calc == "full" and window is not None:
        wr, wc = window
        win = [[image[wr * stride + i][wc * stride + j] for j in range(k)]
               for i in range(k)]
        line = f"{_prodstr(win, filt)}   =   {M[wr][wc]:g}"
    elif calc == "short" and window is not None:
        wr, wc = window
        line = f"overlay \u00b7 multiply \u00b7 add  =  {M[wr][wc]:g}"
    elif calc == "rule":
        line = "overlay the filter on the window  \u00b7  multiply matching cells  \u00b7  add the products  \u2192  one output cell"
    else:
        line = ""
    if line:
        fam = "monospace" if calc == "full" else "sans-serif"
        ax.text(7.6, 1.5, line, ha="center", va="center", color=INK,
                fontsize=13 if calc == "full" else 14.5, fontweight="bold",
                family=fam)
    if note:
        ax.text(7.6, 0.75, note, ha="center", va="center", color=MUTED,
                fontsize=12.5, style="italic")
    _save(fig, name)


def maxpool_grid(mp, stride=2, reveal=True, name="fig_maxpool.png",
                 note=None):
    """Max pooling on a 4\u00d74 map: 2\u00d72 non-overlapping windows, keep each
    window's strongest response \u2192 a 2\u00d72 output. reveal=False leaves the
    pooled cells as '?' for the you-do; the four windows are always colour-coded
    so the drill is self-evident."""
    m = len(mp)
    k = stride
    o = m // k
    out = [[max(mp[r * k + i][c * k + j] for i in range(k) for j in range(k))
            for c in range(o)] for r in range(o)]
    block_face = [TEAL_SOFT, AMBER_SOFT, "#EDE7F3", "#F2E6E6"]
    block_edge = [TEAL, AMBER, "#7C6BA8", RED]

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")
    cell = 0.72
    ix0, iytop = 0.6, 5.1
    # draw the 4x4 with each 2x2 block shaded by its colour
    for r in range(m):
        for c in range(m):
            b = (r // k) * o + (c // k)
            cx = ix0 + c * cell
            cy = iytop - r * cell
            ax.add_patch(FancyBboxPatch(
                (cx, cy - cell), cell * 0.92, cell * 0.92,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                facecolor=block_face[b], edgecolor=block_edge[b],
                lw=1.6, zorder=3))
            ax.text(cx + cell * 0.46, cy - cell * 0.46, f"{mp[r][c]:g}",
                    ha="center", va="center", color=INK, fontsize=17,
                    zorder=4, family="monospace")
    ax.text(ix0 + m * cell / 2, iytop + 0.34, f"feature map  {m}\u00d7{m}",
            ha="center", color=INK_SOFT, fontsize=14, fontweight="bold")

    ax.annotate("", xy=(ix0 + m * cell + 1.5, 3.4),
                xytext=(ix0 + m * cell + 0.35, 3.4),
                arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, lw=2.2))
    ax.text(ix0 + m * cell + 0.93, 3.75, f"max-pool\n{k}\u00d7{k}", ha="center",
            va="bottom", color=INK_SOFT, fontsize=12, fontweight="bold")

    # output 2x2
    ox0, oytop = ix0 + m * cell + 2.0, 4.4
    ocell = 0.86
    for r in range(o):
        for c in range(o):
            b = r * o + c
            cx = ox0 + c * ocell
            cy = oytop - r * ocell
            ax.add_patch(FancyBboxPatch(
                (cx, cy - ocell), ocell * 0.92, ocell * 0.92,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                facecolor=block_face[b], edgecolor=block_edge[b],
                lw=2.4, zorder=3))
            txt = f"{out[r][c]:g}" if reveal else "?"
            ax.text(cx + ocell * 0.46, cy - ocell * 0.46, txt,
                    ha="center", va="center",
                    color=AMBER if not reveal else INK,
                    fontsize=22, fontweight="bold", zorder=4,
                    family="monospace")
    ax.text(ox0 + o * ocell / 2, oytop + 0.34, f"pooled  {o}\u00d7{o}",
            ha="center", color=INK_SOFT, fontsize=14, fontweight="bold")

    footer = (note if note else
              "each colour = one 2\u00d72 window \u00b7 keep its MAX \u00b7 half the size, strongest response preserved")
    ax.text(7, 0.5, footer, ha="center", color=MUTED, fontsize=12.5,
            style="italic")
    _save(fig, name)


def padding_grid(n=5, name="fig_conv_padding.png"):
    """Zero padding: a ring of zeros around an n\u00d7n image makes an
    (n+2)\u00d7(n+2) input, so a 3\u00d73 stride-1 conv returns a map the SAME size
    as the original \u2014 and edge cells get looked at as often as the middle."""
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")
    cell = 0.62
    P = n + 2
    ix0, iytop = 0.7, 5.3
    for r in range(P):
        for c in range(P):
            edge_ring = (r == 0 or c == 0 or r == P - 1 or c == P - 1)
            cx = ix0 + c * cell
            cy = iytop - r * cell
            ax.add_patch(FancyBboxPatch(
                (cx, cy - cell), cell * 0.92, cell * 0.92,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                facecolor=AMBER_SOFT if edge_ring else WHITE,
                edgecolor=AMBER if edge_ring else HAIRLINE,
                lw=1.8 if edge_ring else 1.2, zorder=3))
            val = "0" if edge_ring else "\u00b7"
            ax.text(cx + cell * 0.46, cy - cell * 0.46, val, ha="center",
                    va="center", color=ROI_INK if edge_ring else MUTED,
                    fontsize=15, fontweight="bold" if edge_ring else "normal",
                    zorder=4, family="monospace")
    ax.text(ix0 + P * cell / 2, iytop + 0.34,
            f"image padded to {P}\u00d7{P}  (amber ring = added zeros)",
            ha="center", color=INK_SOFT, fontsize=14, fontweight="bold")

    ax.text(9.3, 4.0,
            f"3\u00d73 filter, stride 1, padding 1:\n\n"
            f"({n} \u2212 3 + 2\u00b71) / 1 + 1  =  {n}",
            ha="left", va="center", color=INK, fontsize=16, fontweight="bold")
    ax.text(9.3, 1.9,
            "output map is the SAME size as the\noriginal \u2014 edges stay in play",
            ha="left", va="center", color=MUTED, fontsize=13, style="italic")
    _save(fig, name)


def output_size(mode="ido", name="fig_output_size.png"):
    """The one formula of the hour: out = (n \u2212 k + 2p) / s + 1, drawn with a
    labelled 1D strip so every letter is concrete before it is used (rule 3:
    the rule is on the deck before the exercise). mode='ido' shows three worked
    counts on the 5\u00d75 grid; mode='youdo' poses the Quiz 5 Q3 scenario blank."""
    fig, ax = plt.subplots(figsize=(11.6, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(6, 5.4, "out  =  ( n \u2212 k + 2p ) / s  +  1", ha="center",
            va="center", fontsize=26, color=INK, fontweight="bold",
            family="monospace")
    # labelled strip: n cells (white), a padding cell each side (amber dashed),
    # one filter window (teal) of width k
    cell = 0.62
    n_disp = 5
    y = 3.4
    x0 = 6 - (n_disp + 2) * cell / 2
    # padding left
    for idx, kind in enumerate(["p"] + ["n"] * n_disp + ["p"]):
        cx = x0 + idx * cell
        pad = kind == "p"
        ax.add_patch(FancyBboxPatch((cx, y - cell / 2), cell * 0.9, cell,
                    boxstyle="round,pad=0.01,rounding_size=0.04",
                    facecolor=AMBER_SOFT if pad else WHITE,
                    edgecolor=AMBER if pad else INK_SOFT,
                    lw=1.6, ls=(0, (3, 2)) if pad else "solid", zorder=3))
    # filter window over first k cells (teal)
    kk = 3
    ax.add_patch(FancyBboxPatch((x0 + cell, y - cell / 2 - 0.08),
                kk * cell, cell + 0.16,
                boxstyle="round,pad=0.01,rounding_size=0.04", fill=False,
                edgecolor=TEAL, lw=2.8, zorder=6))
    ax.text(x0 + cell + kk * cell / 2, y + cell / 2 + 0.34, "k",
            ha="center", color=TEAL, fontsize=15, fontweight="bold")
    ax.text(x0 + cell + n_disp * cell / 2, y - cell / 2 - 0.42, "n",
            ha="center", color=INK, fontsize=15, fontweight="bold")
    ax.text(x0 + cell / 2, y - cell / 2 - 0.42, "p", ha="center",
            color=ROI_INK, fontsize=13, fontweight="bold")
    ax.text(x0 + (n_disp + 1.5) * cell, y - cell / 2 - 0.42, "p", ha="center",
            color=ROI_INK, fontsize=13, fontweight="bold")

    if mode == "ido":
        ax.text(6, 1.55,
                "our stride-1 walk:   (5 \u2212 3 + 0) / 1 + 1  =  3   \u2713",
                ha="center", color=INK, fontsize=15, fontweight="bold",
                family="monospace")
        ax.text(6, 0.95,
                "stride 2:   (5 \u2212 3 + 0) / 2 + 1  =  2      "
                "padding 1:   (5 \u2212 3 + 2) / 1 + 1  =  5",
                ha="center", color=INK_SOFT, fontsize=14,
                family="monospace")
    else:
        ax.text(6, 1.35,
                "Quiz 5 Q3:   n = 7,  k = 3,  p = 1,  s = 1   \u2192   out = ?",
                ha="center", color=ROI_INK, fontsize=16, fontweight="bold",
                family="monospace")
        ax.text(6, 0.7, "plug into the formula above \u2014 what length comes out?",
                ha="center", color=MUTED, fontsize=13, style="italic")
    _save(fig, name)


def dense_explosion(name="fig_dense_explosion.png"):
    """Why a fully-connected layer doesn't scale to images/spectra: wiring every
    pixel to every neuron explodes the weight count, versus a conv filter that
    reuses 9 shared weights slid everywhere."""
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
    # left: dense
    axL = axes[0]
    axL.set_xlim(0, 10)
    axL.set_ylim(0, 8)
    axL.axis("off")
    axL.set_title("fully connected", color=RED, fontsize=15,
                  fontweight="bold")
    # small pixel block
    for r in range(4):
        for c in range(4):
            axL.add_patch(FancyBboxPatch((0.5 + c * 0.5, 5.5 - r * 0.5), 0.44,
                        0.44, boxstyle="round,pad=0.01,rounding_size=0.04",
                        facecolor="#F1F1EC", edgecolor=HAIRLINE, lw=1.0))
    neurons = [(7.6, 6.2), (7.6, 5.2), (7.6, 4.2), (7.6, 3.2)]
    for (nx, ny) in neurons:
        for r in range(4):
            for c in range(4):
                axL.plot([0.94 + c * 0.5, nx - 0.28], [5.72 - r * 0.5, ny],
                         color=RED, lw=0.4, alpha=0.25, zorder=1)
    for (nx, ny) in neurons:
        axL.add_patch(Circle((nx, ny), 0.28, facecolor=WHITE, edgecolor=INK_SOFT,
                             lw=1.8, zorder=3))
    axL.text(2.5, 2.4, "100\u00d7100\u00d73 pixels \u2192 1,000 neurons",
             ha="center", color=INK_SOFT, fontsize=12)
    axL.text(2.5, 1.5, "= 30,000,000 weights", ha="center", color=RED,
             fontsize=17, fontweight="bold")
    axL.text(2.5, 0.7, "in ONE layer \u2014 each needs data to learn",
             ha="center", color=MUTED, fontsize=11.5, style="italic")
    # right: conv
    axR = axes[1]
    axR.set_xlim(0, 10)
    axR.set_ylim(0, 8)
    axR.axis("off")
    axR.set_title("one convolution filter", color=TEAL, fontsize=15,
                  fontweight="bold")
    for r in range(3):
        for c in range(3):
            axR.add_patch(FancyBboxPatch((3.7 + c * 0.6, 6.0 - r * 0.6), 0.52,
                        0.52, boxstyle="round,pad=0.01,rounding_size=0.04",
                        facecolor=TEAL_SOFT, edgecolor=TEAL, lw=1.8))
    axR.text(5.2, 6.7, "3\u00d73 window", ha="center", color=TEAL, fontsize=12,
             fontweight="bold")
    axR.annotate("", xy=(7.4, 4.6), xytext=(6.6, 5.0),
                 arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=2.0))
    axR.text(5.0, 2.4, "9 shared weights", ha="center", color=TEAL,
             fontsize=17, fontweight="bold")
    axR.text(5.0, 1.5, "slid across the WHOLE image", ha="center",
             color=INK_SOFT, fontsize=12)
    axR.text(5.0, 0.7, "local windows + weight sharing", ha="center",
             color=MUTED, fontsize=11.5, style="italic")
    _save(fig, name)


def conv_real_filters(name="fig_conv_real_filters.png"):
    """Rule 7 payoff: REAL convolutions of the sparrow photo with named 3\u00d73
    kernels (Sobel x / Sobel y / Laplacian / box blur) \u2014 the maps were
    computed, not illustrated. A CNN learns kernels like these on its own."""
    panels = [
        ("sparrow_gray.jpg", "input (grayscale)"),
        ("sparrow_vedge.jpg", "vertical edges"),
        ("sparrow_hedge.jpg", "horizontal edges"),
        ("sparrow_blob.jpg", "blobs / spots"),
        ("sparrow_blur.jpg", "smoothing (blur)"),
    ]
    pw, ph, bar, gap = 240, 300, 40, 12
    W = pw * len(panels) + gap * (len(panels) + 1)
    H = ph + bar + gap * 2
    canvas = Image.new("RGB", (W, H), (251, 251, 248))
    draw = ImageDraw.Draw(canvas)
    font = _font(24)
    x = gap
    for fn, label in panels:
        photo = _fit_fill(Image.open(IMG / fn), pw, ph)
        canvas.paste(photo, (x, gap))
        draw.rectangle([x, gap + ph, x + pw, gap + ph + bar], fill=INK)
        tb = draw.textbbox((0, 0), label, font=font)
        draw.text((x + (pw - (tb[2] - tb[0])) / 2,
                   gap + ph + (bar - (tb[3] - tb[1])) / 2 - tb[1]),
                  label, fill=(255, 255, 255), font=font)
        x += pw + gap
    canvas.save(IMG / name)
    _downscale(IMG / name)
    print(f"  wrote {name}")


def _spectrum(xg):
    """A clean synthetic MALDI-style spectrum (deterministic) with a few
    Gaussian peaks on a low baseline \u2014 used for the 1D conv walkthrough."""
    peaks = [(1.6, 1.0, 0.05), (3.0, 0.55, 0.05), (5.2, 0.85, 0.05),
             (6.1, 0.4, 0.04), (8.0, 0.65, 0.05)]
    y = 0.05 + 0.02 * np.sin(xg * 1.3)
    for mu, a, w in peaks:
        y = y + a * np.exp(-((xg - mu) ** 2) / (2 * w))
    return y


def conv1d_spectrum(name="fig_conv1d_spectrum.png"):
    """The same overlay-multiply-add, one dimension: a short filter window slides
    along a spectrum's m/z axis, and the feature map below lights up where the
    peak SHAPE is found. Deletes one dimension from the 2D drill."""
    xg = np.linspace(0, 10, 800)
    sp = _spectrum(xg)
    # a peak-shape matched filter response (normalized cross-correlation-ish):
    # slide a small Gaussian template and score the local match
    tmpl = np.exp(-((np.linspace(-1, 1, 61)) ** 2) / (2 * 0.12))
    tmpl = tmpl - tmpl.mean()
    fmap = np.convolve(sp - sp.mean(), tmpl[::-1], mode="same")
    fmap = np.clip(fmap, 0, None)
    fmap = fmap / fmap.max()

    fig, (axs, axf) = plt.subplots(2, 1, figsize=(10.5, 5.0), sharex=True,
                                   gridspec_kw={"height_ratios": [1.1, 1]})
    axs.plot(xg, sp, color=INK, lw=2.0, zorder=3)
    # the filter window box over one peak
    axs.add_patch(FancyBboxPatch((4.7, -0.02), 1.0, sp.max() * 0.9 + 0.05,
                boxstyle="round,pad=0.01,rounding_size=0.02", fill=True,
                facecolor=AMBER_SOFT, edgecolor=AMBER, lw=2.2, alpha=0.8,
                zorder=2))
    axs.text(5.2, sp.max() + 0.06, "filter window (k bins)", ha="center",
             color=ROI_INK, fontsize=12, fontweight="bold")
    axs.annotate("slides along m/z \u2192", xy=(7.2, sp.max() * 0.6),
                 xytext=(2.0, sp.max() * 0.92), color=MUTED, fontsize=12,
                 style="italic",
                 arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4))
    axs.set_ylabel("spectrum", fontsize=12)
    axs.set_yticks([])
    axs.set_ylim(-0.05, sp.max() + 0.2)
    for spn in ("top", "right"):
        axs.spines[spn].set_visible(False)

    axf.plot(xg, fmap, color=TEAL, lw=2.2, zorder=3)
    axf.fill_between(xg, fmap, color=TEAL, alpha=0.12)
    axf.set_ylabel('feature map\n"peak here"', fontsize=11)
    axf.set_yticks([])
    axf.set_ylim(0, 1.15)
    axf.set_xlabel("m/z  \u2192", fontsize=12)
    axf.set_xticks([])
    for spn in ("top", "right"):
        axf.spines[spn].set_visible(False)
    fig.text(0.5, -0.01,
             "one detector \u00b7 a handful of weights \u00b7 every position on the axis \u2014 the 2D drill with one dimension deleted",
             ha="center", color=INK_SOFT, fontsize=12, style="italic")
    fig.subplots_adjust(hspace=0.12)
    _save(fig, name)


def conv1d_code(name="fig_conv1d_code.png"):
    """nn.Conv1d + MaxPool1d with every argument annotated back to a by-hand
    dial, plus the output-length check via the formula."""
    rows = [
        {"code": "nn.Conv1d(", "dim": True},
        {"code": "    in_channels=1,", "comment": "# one raw spectrum"},
        {"code": "    out_channels=8,", "comment": "# 8 filters \u2192 8 maps"},
        {"code": "    kernel_size=15,", "comment": "# window k = 15 bins"},
        {"code": "    stride=1, padding=7,", "comment": "# keep length 6000"},
        {"code": ")", "dim": True},
        {"code": "nn.MaxPool1d(2)", "comment": "# halve: 6000 \u2192 3000"},
    ]
    _code_figure(rows, name, figsize=(11, 3.9), comment_x=0.52)


def detection_grid(name="fig_detection_grid.png"):
    """Object detection = boxes + confidence scores from the SAME conv backbone;
    only the output head changes. Two objects, two amber boxes, two scores."""
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.4, 0.4), 9.2, 5.8,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                facecolor="#F1F1EC", edgecolor=MUTED, lw=1.2))
    from matplotlib.patches import Ellipse
    ax.add_patch(Ellipse((3.0, 3.5), 2.4, 1.7, facecolor=TEAL, alpha=0.28))
    ax.add_patch(Ellipse((7.0, 4.4), 1.9, 1.4, facecolor=RED, alpha=0.25))
    ax.add_patch(FancyBboxPatch((1.6, 2.5), 2.9, 2.1, fill=False,
                edgecolor=AMBER, lw=3.0, boxstyle="square,pad=0"))
    ax.add_patch(FancyBboxPatch((5.9, 3.5), 2.3, 1.8, fill=False,
                edgecolor=AMBER, lw=3.0, boxstyle="square,pad=0"))
    ax.text(1.7, 4.75, "cell \u00b7 0.94", color=ROI_INK, fontsize=14,
            fontweight="bold", family="monospace")
    ax.text(6.0, 5.45, "debris \u00b7 0.71", color=ROI_INK, fontsize=14,
            fontweight="bold", family="monospace")
    ax.text(5.0, 0.0, "one CNN pass \u2192 every BOX (where) + every SCORE (how sure)",
            ha="center", color=INK_SOFT, fontsize=12.5, style="italic")
    _save(fig, name)


def peak_detection(reveal=True, name="fig_peak_detection.png"):
    """A chromatogram is an image whose objects are peaks: the bounding box is the
    retention-time integration window, the score is reviewer confidence. The
    manual 'peak review' step is a detection problem a CNN already solves.
    reveal=False blanks the box/score labels for the Quiz 5 Q4 you-do."""
    t = np.linspace(0, 10, 800)
    peaks = [(3.0, 1.0, 0.10), (6.6, 0.62, 0.13)]
    y = 0.04 + 0.015 * np.sin(t * 1.1)
    for mu, a, w in peaks:
        y = y + a * np.exp(-((t - mu) ** 2) / (2 * w))
    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    ax.plot(t, y, color=INK, lw=2.2, zorder=3)
    ax.fill_between(t, y, color=TEAL, alpha=0.08)
    boxes = [(2.15, 3.85, 1.12, "peak", "0.97"),
             (5.75, 7.45, 0.72, "peak", "0.83")]
    for x0, x1, top, lab, score in boxes:
        ax.add_patch(FancyBboxPatch((x0, 0.0), x1 - x0, top + 0.08,
                    boxstyle="round,pad=0.005,rounding_size=0.02", fill=False,
                    edgecolor=AMBER, lw=3.0, zorder=5))
        tag = f"{lab} \u00b7 {score}" if reveal else "box = ?  \u00b7  score = ?"
        ax.text((x0 + x1) / 2, top + 0.2, tag, ha="center", color=ROI_INK,
                fontsize=13, fontweight="bold", family="monospace")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.45)
    ax.set_xlabel("retention time  \u2192", fontsize=13)
    ax.set_yticks([])
    ax.set_xticks([])
    for spn in ("top", "right"):
        ax.spines[spn].set_visible(False)
    footer = ("object \u2192 peak  \u00b7  bounding box \u2192 RT integration window  \u00b7  score \u2192 how sure it's a real peak"
              if reveal else
              "name the two boxed peaks as a detector would: what is the BOX? what is the SCORE?")
    ax.text(0.5, -0.22, footer, transform=ax.transAxes, ha="center",
            color=INK_SOFT, fontsize=12, style="italic")
    fig.subplots_adjust(bottom=0.2)
    _save(fig, name)


# ---- runner -----------------------------------------------------------------
def build_all():
    print("figures:")
    # web-sourced images first (composites depend on chatgpt_panel)
    maldi_real()
    chatgpt_panel()
    driams_mrsa_spectrum()
    # generated plots
    activations(labeled=True, name="fig_activations.png",
                order=("relu", "sigmoid", "linear"))
    activations(labeled=False, name="fig_activations_quiz.png",
                order=("sigmoid", "relu", "linear"))
    # one-neuron: I-do (positive, passes) then you-do = Quiz Q2 (negative, silent)
    neuron_schematic(3, 2, 2, -1, 1, "ReLU", "5", "5",
                     "fig_neuron_ido.png")
    neuron_schematic(2, 3, 0.5, -1, 1, "ReLU", "?", "?",
                     "fig_neuron_youdo.png")
    # matrix = Quiz Q3 numbers; row 1 I-do, row 2 you-do
    W = [[1, 0, 2], [-1, 3, 1]]
    x = [2, 1, 0]
    matvec(W, x, 2, "?", 1, "fig_matvec_row1.png")
    matvec(W, x, 2, "?", 2, "fig_matvec_row2.png")
    universal_approx()
    # two-layer beat = matched I-do / you-do pair, same 2->2->1 node layout,
    # EVERY weight + bias printed on both figures (rules 6/9: self-contained).
    # I-do (worked): x=(2,1); W1=[[1,-1],[0,1]], b1=(0,0) -> hidden pre (1,1),
    # ReLU keeps both active -> h=(1,1); layer 2 w2=(2,1), b2=-1 -> y=2.
    two_layer_net(2, 1, "fig_twolayer_ido.png",
                  w1=((1, -1), (0, 1)), b1=(0, 0), w2=(2, 1), b2=-1,
                  hidden=(1, 1), y=2,
                  arith="y = 2\u00b71 + 1\u00b71 \u2212 1 = 2")
    # you-do = Quiz 1 Q6 (all weights shown; hidden + output are '?'):
    # x=(2,1); W1=[[1,-1],[-1,1]], b1=(0,-4); W2=(3,2), b2=-1 -> h=(1,0), y=2.
    two_layer_net(2, 1, "fig_twolayer_youdo.png",
                  w1=((1, -1), (-1, 1)), b1=(0, -4), w2=(3, 2), b2=-1)
    # composites
    hook_4panel()
    cat_hierarchy()
    # ---- Lecture 2 ----
    random_weights_net()
    loss_mse()
    loss_bowl()
    lr_three_modes()
    loss_curves(labeled=True, name="fig_loss_curves_ido.png")
    loss_curves(labeled=False, name="fig_loss_curves_youdo.png")
    # backprop centerpiece: I-do numbers (x=2, w1=0.5, w2=1.5, y=1)
    backprop_two_weight(2, 0.5, 1.5, 1, "fwd", "fig_backprop_forward.png")
    backprop_two_weight(2, 0.5, 1.5, 1, "full", "fig_backprop_backward.png")
    # you-do = Quiz 2 Q1 (different numbers: x=1, w1=2, w2=0.5, y=0)
    backprop_two_weight(1, 2, 0.5, 0, "blank", "fig_backprop_youdo.png")
    backprop_chain()
    # one gradient-descent step: I-do (worked numbers) then you-do = Quiz 2 Q2
    onestep_update([("w\u2082", "1.5 \u2212 0.1 \u00d7 1.0", "1.4"),
                    ("w\u2081", "0.5 \u2212 0.1 \u00d7 3.0", "0.2")],
                   (0.25, 0.19), "fig_onestep.png")
    onestep_update([("w\u2082", "0.5 \u2212 0.1 \u00d7 ?", "?"),
                    ("w\u2081", "2 \u2212 0.1 \u00d7 ?", "?")],
                   None, "fig_onestep_youdo.png")
    backprop_scale_net()
    training_loop_cycle()
    training_code()
    loop_order()
    autograd_check()
    loss_fall()
    # ---- Lecture 4 ----
    softmax_bars()
    crossentropy_curve()
    crossentropy_formula()
    crossentropy_worked()
    # CE you-do = Quiz 4 Q1 (same two-card shape, different confidences, ? penalty)
    crossentropy_worked("fig_ce_youdo.png",
                        cards=[("Model A \u2014 0.8 on R", 0.80, None, TEAL),
                               ("Model B \u2014 0.2 on R", 0.20, None, RED)],
                        reveal=False)
    minibatch_paths()
    adam_momentum()
    lr_schedule()
    activation_choices()
    # vanishing/exploding chain: I-do (worked) then you-do = Quiz 4 Q4
    vanishing_chain("ido", "fig_vanishing_ido.png")
    vanishing_chain("youdo", "fig_vanishing_youdo.png")
    hyperparam_search()
    batch_tradeoff()
    batch_loss_curve()
    fit_trio()
    # train/val curves: I-do (3 named) then you-do = Quiz 4 Q2 (4 pairs a-d)
    trainval_curves(True, "fig_trainval_ido.png")
    trainval_curves(False, "fig_trainval_youdo.png")
    early_stopping()
    dropout_panels()
    weight_decay()
    class_imbalance()
    # ---- Lecture 5 ----
    _lecture5_figures()
    # ---- Lecture 7 ----
    _lecture7_figures()
    # ---- Lecture 8 ----
    _lecture8_figures()


# deck I-do convolution (5x5 image, 3x3 diagonal detector) and the matched
# Quiz 5 you-do (5x5 image, 3x3 X-detector) — different numbers, same drill.
CONV_IMG_IDO = [
    [1, 0, 0, 1, 0],
    [0, 1, 0, 0, 1],
    [0, 0, 1, 0, 0],
    [1, 0, 0, 1, 0],
    [0, 1, 0, 0, 1],
]
CONV_FILT_IDO = [[1, -1, -1], [-1, 1, -1], [-1, -1, 1]]  # diagonal detector
CONV_IMG_QUIZ = [
    [1, 1, 0, 1, 0],
    [0, 1, 1, 0, 1],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 1],
    [1, 0, 1, 0, 1],
]
CONV_FILT_QUIZ = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]  # X / corner+centre detector
MAXPOOL_IDO = [[3, 1, 2, 4], [0, 2, 5, 1], [6, 0, 1, 3], [2, 4, 2, 0]]
MAXPOOL_QUIZ = [[1, 3, 2, 0], [4, 2, 1, 5], [0, 1, 6, 2], [2, 3, 1, 4]]


def _lecture5_figures():
    dense_explosion()
    conv_walk(CONV_IMG_IDO, CONV_FILT_IDO, stride=1, window=(0, 0),
              reveal=set(), calc="rule", name="fig_conv_setup.png",
              filt_note="hunts diagonals",
              note="the 9 filter values are learned weights \u2014 today we hand-pick them so we can compute")
    conv_walk(CONV_IMG_IDO, CONV_FILT_IDO, stride=1, window=(0, 0),
              reveal={(0, 0)}, calc="full", name="fig_conv_ido1.png",
              filt_note="hunts diagonals",
              note="nine products \u2014 six are zero, three 1\u00b71 hits = a strong diagonal match")
    conv_walk(CONV_IMG_IDO, CONV_FILT_IDO, stride=1, window=(0, 1),
              reveal={(0, 0), (0, 1)}, calc="full", name="fig_conv_ido2.png",
              filt_note="hunts diagonals",
              note="same 9 weights, new window \u2014 that reuse IS weight sharing")
    conv_walk(CONV_IMG_IDO, CONV_FILT_IDO, stride=1, window=None,
              reveal=None, calc="none", name="fig_conv_map.png",
              filt_note="hunts diagonals",
              calc_text="9 window sums \u2192 a 3\u00d73 map \u00b7 large value = 'diagonal found here' \u00b7 only 9 weights",
              note="the feature map is itself an image \u2014 so another conv can stack on top")
    conv_walk(CONV_IMG_IDO, CONV_FILT_IDO, stride=2, window=(0, 0),
              reveal=None, calc="none", name="fig_conv_stride2.png",
              filt_note="hunts diagonals", map_label="feature map",
              calc_text="stride 2: the window jumps 2 \u2014 (5 \u2212 3)/2 + 1 = 2 \u2192 a 2\u00d72 map (a quarter of the outputs)")
    conv_walk(CONV_IMG_QUIZ, CONV_FILT_QUIZ, stride=1, window=(0, 0),
              reveal=set(), calc="rule", name="fig_conv_youdo.png",
              filt_note="corners + centre",
              note="your turn \u2014 same overlay-multiply-add, new image and filter (Quiz 5 Q1)")
    padding_grid(n=5)
    output_size(mode="ido", name="fig_output_size.png")
    output_size(mode="youdo", name="fig_output_size_youdo.png")
    conv_real_filters()
    maxpool_grid(MAXPOOL_IDO, reveal=True, name="fig_maxpool_ido.png")
    maxpool_grid(MAXPOOL_QUIZ, reveal=False, name="fig_maxpool_youdo.png",
                 note="your turn \u2014 keep each 2\u00d72 window's MAX (Quiz 5 Q2)")
    conv1d_spectrum()
    conv1d_code()
    detection_grid()
    peak_detection(reveal=True, name="fig_peak_detection.png")
    peak_detection(reveal=False, name="fig_peak_detection_youdo.png")


# ============================================================================
#  Lecture 7 · Sequence Models and Attention
#  Hand-authored numeric mechanics (rule 6) + bespoke MS-anchored schematics
#  (rule 9). Canonical published diagrams (RNN unfold, LSTM cell, multi-head)
#  are REAL licensed images placed directly in the deck, not drawn here.
# ============================================================================

def _chip(ax, x, y, w, h, label, face, txt=WHITE, fs=16, edge="none", lw=1.4):
    """A rounded token chip centred at (x + w/2, y)."""
    ax.add_patch(FancyBboxPatch(
        (x, y - h / 2), w, h,
        boxstyle="round,pad=0.004,rounding_size=0.03",
        facecolor=face, edgecolor=edge, lw=lw, zorder=3))
    ax.text(x + w / 2, y, label, ha="center", va="center",
            color=txt, fontsize=fs, fontweight="bold", zorder=4)


def seq_order_matters(name="fig_seq_order.png"):
    """Why order matters for sequence data (MS-first). Left: the SAME four amino
    acids in two orders are two different peptides (same mass, different
    sequence → different molecule). Right: a QC-drift trace across runs — the
    signal you care about lives in the time order; shuffle it and the drift is
    gone."""
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    ax = axes[0]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.95, "a peptide is an ordered sequence", ha="center",
            fontsize=14, color=INK, fontweight="bold")

    def row(y, toks, color, cap):
        n = len(toks); w = 0.15; gap = 0.025
        total = n * w + (n - 1) * gap; x0 = 0.5 - total / 2
        for i, t in enumerate(toks):
            _chip(ax, x0 + i * (w + gap), y, w, 0.13, t, color, fs=17)
        ax.text(0.5, y - 0.12, cap, ha="center", va="top",
                fontsize=12, color=INK_SOFT)
    row(0.66, ["S", "A", "G", "K"], TEAL, "peptide  SAGK")
    row(0.30, ["G", "K", "A", "S"], RED, "shuffled  GKAS — a different peptide")
    ax.annotate("", xy=(0.5, 0.40), xytext=(0.5, 0.50),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2))
    ax.text(0.5, 0.06, "same four residues, same mass — order is the identity",
            ha="center", fontsize=11.5, color=MUTED, style="italic")

    ax = axes[1]
    runs = np.arange(1, 9)
    qc = np.array([100, 101, 103, 104, 106, 108, 109, 111], dtype=float)
    ax.plot(runs, qc, "-o", color=TEAL, lw=2.6, ms=7, zorder=3,
            solid_capstyle="round")
    ax.annotate("drift", xy=(7.2, 110), xytext=(4.4, 111.4), color=RED,
                fontsize=13, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
    ax.set_xlim(0.5, 8.5); ax.set_ylim(98, 113)
    ax.set_xlabel("run number (time order) →", fontsize=12)
    ax.set_ylabel("QC value", fontsize=12)
    ax.set_yticks([])
    ax.set_xticks(runs)
    ax.set_title("QC drift across runs", fontsize=14, color=INK,
                 fontweight="bold")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(0.5, -0.22, "shuffle the runs and the drift vanishes — the\ninformation is in the order",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=11.5, color=MUTED, style="italic")
    fig.subplots_adjust(bottom=0.2, wspace=0.25)
    _save(fig, name)


def tokenize_peptide(name="fig_tokenize.png"):
    """Tokenization by analogy: a sentence splits into word tokens; a peptide
    splits into amino-acid tokens, each mapped to an integer ID from a fixed
    vocabulary. 'A peptide is a sentence; amino acids are the words.'"""
    fig, ax = plt.subplots(figsize=(11.6, 4.3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def strip(y, items, color, label, ids=None):
        n = len(items); w = 0.135; gap = 0.02
        total = n * w + (n - 1) * gap; x0 = 0.52 - total / 2
        ax.text(0.06, y, label, ha="left", va="center", fontsize=13,
                color=INK_SOFT, fontweight="bold")
        for i, t in enumerate(items):
            x = x0 + i * (w + gap)
            _chip(ax, x, y, w, 0.12, t, color, fs=15)
            if ids is not None:
                ax.text(x + w / 2, y - 0.10, f"id {ids[i]}", ha="center",
                        va="top", fontsize=10, color=MUTED, family="monospace")
    strip(0.80, ["The", "assay", "flags", "MRSA"], INK_SOFT,
          "sentence →")
    ax.annotate("", xy=(0.52, 0.60), xytext=(0.52, 0.70),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2))
    ax.text(0.60, 0.65, "same idea", ha="left", va="center",
            fontsize=11, color=MUTED, style="italic")
    strip(0.40, list("ACDEFG"), TEAL, "peptide →",
          ids=[1, 2, 3, 4, 5, 6])
    ax.text(0.5, 0.14,
            "each token → an integer ID from a fixed vocabulary  ·  "
            "a peptide is a sentence, amino acids are the words",
            ha="center", fontsize=12, color=INK_SOFT)
    ax.text(0.5, 0.04,
            "modified residues (phospho-S, oxidised-M) just get their own token IDs",
            ha="center", fontsize=10.5, color=MUTED, style="italic")
    _save(fig, name)


def embedding_space(name="fig_embedding.png"):
    """Embeddings: each token becomes coordinates, and chemically similar
    residues end up near each other — learned from data, not hand-set. Four
    plain clusters (hydrophobic / acidic / basic / polar)."""
    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    groups = {
        "hydrophobic": (TEAL, [("A", 1.1, 1.3), ("V", 1.5, 1.9),
                               ("L", 1.9, 1.4), ("I", 1.4, 1.0),
                               ("F", 2.1, 1.9)]),
        "acidic (−)": (RED, [("D", 4.5, 3.9), ("E", 4.9, 3.4)]),
        "basic (+)": (AMBER, [("K", 4.4, 1.2), ("R", 4.9, 1.7),
                              ("H", 4.2, 1.9)]),
        "polar": (MUTED, [("S", 1.4, 3.8), ("T", 1.9, 3.4),
                          ("N", 1.1, 3.1), ("Q", 2.0, 4.1)]),
    }
    for gname, (color, pts) in groups.items():
        xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
        cx, cy = np.mean(xs), np.mean(ys)
        ax.add_patch(plt.matplotlib.patches.Ellipse(
            (cx, cy), max(np.ptp(xs), 0.8) + 1.2, max(np.ptp(ys), 0.8) + 1.2,
            facecolor=color, alpha=0.10, edgecolor=color, lw=1.4,
            zorder=1))
        ax.text(cx, cy + max(np.ptp(ys), 0.8) / 2 + 0.75, gname, ha="center",
                color=color, fontsize=12.5, fontweight="bold", zorder=2)
        for lab, x, y in pts:
            ax.scatter([x], [y], s=520, color=color, zorder=3,
                       edgecolor=WHITE, linewidth=1.5)
            ax.text(x, y, lab, ha="center", va="center", color=WHITE,
                    fontsize=13, fontweight="bold", zorder=4)
    ax.set_xlim(0, 6); ax.set_ylim(0, 5.4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("embedding dimension 1 →", fontsize=12)
    ax.set_ylabel("embedding dimension 2 →", fontsize=12)
    ax.set_title("each amino-acid token → coordinates; similar residues sit near each other",
                 fontsize=13.5, color=INK, fontweight="bold")
    ax.text(0.5, -0.12,
            "the coordinates are LEARNED from data — the model discovers the chemistry, we don’t hand it in",
            transform=ax.transAxes, ha="center", color=MUTED, fontsize=11.5,
            style="italic")
    fig.subplots_adjust(bottom=0.16)
    _save(fig, name)


def selfattn_all_to_all(name="fig_selfattn_alltoall.png"):
    """The big idea: every token looks at every other token and decides what is
    relevant. One query token's links to all tokens are highlighted; the rest
    of the all-pairs mesh is faint."""
    # Same 4-token sentence as the heat-map I-do (fig_attn_heatmap_ido) so the
    # illustrative example is continuous slide-to-slide (no dropped "the").
    toks = ["peptide", "eluted", "because", "it"]
    n = len(toks)
    xs = np.linspace(1.2, 10.8, n)
    y = 1.4
    fig, ax = plt.subplots(figsize=(11.6, 4.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.4); ax.axis("off")
    q = n - 1  # the query token "it" (last token)
    # faint all-pairs mesh
    for i in range(n):
        for j in range(i + 1, n):
            ax.plot([xs[i], xs[j]], [y, y], color=HAIRLINE, lw=1.0,
                    zorder=1)
    # highlighted arcs from the query to every token
    for j in range(n):
        if j == q:
            continue
        xm = (xs[q] + xs[j]) / 2
        rad = 0.55 if xs[j] < xs[q] else 0.55
        ax.annotate("", xy=(xs[j], y + 0.28), xytext=(xs[q], y + 0.28),
                    arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2,
                                    connectionstyle=f"arc3,rad={0.45 if xs[j]<xs[q] else -0.45}"),
                    zorder=2)
    for i, t in enumerate(toks):
        face = AMBER if i == q else TEAL_SOFT
        txt = WHITE if i == q else INK
        _chip(ax, xs[i] - 0.55, y, 1.1, 0.5, t, face, txt=txt, fs=15,
              edge=AMBER if i == q else HAIRLINE, lw=1.5)
    ax.text(xs[q], 0.55, "query: “it” looks at every token", ha="center",
            color=ROI_INK, fontsize=12.5, fontweight="bold")
    ax.text(6, 3.9, "self-attention: every token looks at every other and decides what’s relevant",
            ha="center", color=INK, fontsize=13.5, fontweight="bold")
    _save(fig, name)


def qkv_lookup(reveal=True, name="fig_qkv.png"):
    """The query/key/value soft-lookup analogy. Three role cards carry the
    plain-language phrases; a one-line pipeline shows how they combine. When
    reveal=False the Q/K/V names are blanked ('?') — the you-do match (Quiz 7
    Q3)."""
    roles = [
        ("Query", "what I'm looking for", TEAL),
        ("Key", "what I contain, so others\ncan decide if I'm relevant", AMBER),
        ("Value", "what I give you once\nyou've decided I'm relevant", RED),
    ]
    if not reveal:
        # You-do (Quiz 7 Q3): neutralize the box colors and shuffle the phrase
        # order so the match is on the LANGUAGE, not the palette or position —
        # the I-do reveal (teal/amber/red in Q/K/V order) must not telegraph it.
        roles = [
            ("Value", roles[2][1], MUTED),
            ("Query", roles[0][1], MUTED),
            ("Key",   roles[1][1], MUTED),
        ]
    fig, ax = plt.subplots(figsize=(11.6, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    cx = [2.2, 6.0, 9.8]
    for (role, phrase, color), x in zip(roles, cx):
        ax.add_patch(FancyBboxPatch((x - 1.7, 2.3), 3.4, 2.2,
                    boxstyle="round,pad=0.02,rounding_size=0.08",
                    facecolor=WHITE, edgecolor=color, lw=2.4, zorder=2))
        ax.add_patch(FancyBboxPatch((x - 1.7, 3.85), 3.4, 0.65,
                    boxstyle="round,pad=0.02,rounding_size=0.08",
                    facecolor=color, edgecolor=color, lw=2.4, zorder=3))
        head = role if reveal else "?"
        ax.text(x, 4.16, head, ha="center", va="center", color=WHITE,
                fontsize=18, fontweight="bold", zorder=4)
        ax.text(x, 3.05, phrase, ha="center", va="center", color=INK_SOFT,
                fontsize=13, zorder=4)
    # one-line pipeline of how they combine (role names hidden in the you-do so
    # the match isn't given away)
    pipe = ("score = Query · Key   →   softmax → weights   →   output = Σ weight × Value"
            if reveal else
            "compare, then blend:   score →   softmax → weights   →   output = Σ weight × (what a token gives)")
    ax.text(6, 1.5, pipe, ha="center", va="center", color=INK, fontsize=14,
            fontweight="bold")
    ax.text(6, 0.7,
            "a soft lookup: match my query against every key, then blend the values by how well they matched",
            ha="center", va="center", color=MUTED, fontsize=11.5,
            style="italic")
    if not reveal:
        ax.text(6, 4.85, "match each phrase to Query · Key · Value",
                ha="center", color=ROI_INK, fontsize=12.5, fontweight="bold")
    _save(fig, name)


def attention_heatmap(kind="sentence", name="fig_attn_heatmap.png"):
    """Reading an attention heat map. kind='sentence' draws the full query×key
    matrix for a 4-token sentence and rings the 'it' query row; kind='peptide'
    draws the single query=K row over A-C-D-E-K — the exact numbers on Quiz 7
    Q1 (weights 0.05, 0.05, 0.35, 0.45, 0.10), so figure and key agree."""
    if kind == "sentence":
        toks = ["peptide", "eluted", "because", "it"]
        M = [[0.55, 0.20, 0.05, 0.20],
             [0.30, 0.45, 0.10, 0.15],
             [0.15, 0.30, 0.40, 0.15],
             [0.60, 0.15, 0.10, 0.15]]
        n = len(toks); cell = 1.0
        fig, ax = plt.subplots(figsize=(8.6, 5.0))
        ax.set_xlim(-2.4, n + 0.4); ax.set_ylim(-1.4, n + 0.6)
        ax.axis("off")
        for r in range(n):
            for c in range(n):
                w = M[r][c]
                op = min(w * 1.7, 0.92)
                x = c; yy = n - 1 - r
                ax.add_patch(plt.Rectangle((x, yy), 0.94, 0.94,
                             facecolor=TEAL, alpha=op, edgecolor=HAIRLINE,
                             lw=1.0, zorder=2))
                ax.text(x + 0.47, yy + 0.47, f"{w:.2f}", ha="center",
                        va="center", fontsize=13,
                        color=WHITE if op > 0.5 else INK_SOFT,
                        fontweight="bold", zorder=3)
            ax.text(-0.15, n - 1 - r + 0.47, toks[r], ha="right", va="center",
                    fontsize=13, color=INK, fontweight="bold")
        for c in range(n):
            ax.text(c + 0.47, n + 0.12, toks[c], ha="center", va="bottom",
                    fontsize=12, color=INK_SOFT, rotation=30)
        # ring the query row for "it" (last token, bottom row)
        ax.add_patch(plt.Rectangle((-0.03, -0.03), n * cell + 0.0, 0.94 + 0.06,
                     fill=False, edgecolor=AMBER, lw=3.0, zorder=5))
        ax.text(n + 0.25, 0.47, "query = “it”", ha="left", va="center",
                color=ROI_INK, fontsize=12.5, fontweight="bold")
        ax.text((n) / 2.0, -1.15,
                "rows = query token · columns = key token · darker = more · each row sums to 1",
                ha="center", fontsize=11.5, color=INK_SOFT)
        _save(fig, name)
    else:  # peptide single row — mirrors Quiz 7 Q1 exactly
        toks = ["A", "C", "D", "E", "K"]
        wts = [0.05, 0.05, 0.35, 0.45, 0.10]
        n = len(toks)
        fig, ax = plt.subplots(figsize=(10.4, 3.4))
        ax.set_xlim(-0.4, n + 0.6); ax.set_ylim(-1.5, 1.9)
        ax.axis("off")
        for i, (t, w) in enumerate(zip(toks, wts)):
            op = min(w * 1.8, 0.92)
            ax.add_patch(plt.Rectangle((i, 0), 0.94, 0.94, facecolor=TEAL,
                         alpha=op, edgecolor=HAIRLINE, lw=1.1, zorder=2))
            ax.text(i + 0.47, 0.47, f"{w:.2f}", ha="center", va="center",
                    fontsize=15, color=WHITE if op > 0.5 else INK_SOFT,
                    fontweight="bold", zorder=3)
            ax.text(i + 0.47, 1.25, t, ha="center", va="center", fontsize=18,
                    color=INK, fontweight="bold")
        ax.annotate("", xy=(n - 0.05, -0.15), xytext=(n - 0.94 + 0.05, -0.15),
                    arrowprops=dict(arrowstyle="-", color=ROI_INK, lw=2))
        ax.text(n - 0.5, -0.55, "query = K", ha="center", va="top",
                color=ROI_INK, fontsize=13, fontweight="bold")
        ax.text(n / 2.0, -1.25,
                "how much the last token K attends to each residue  ·  darker = more  ·  weights sum to 1",
                ha="center", fontsize=12, color=INK_SOFT)
        _save(fig, name)


def attention_formula(name="fig_attn_formula.png"):
    """State the two rules BEFORE any slide asks a number of them (rule 3):
    weights = softmax(scores)  and  output = Σ weight×value. The softmax formula
    is a Lecture 4 callback; the weighted average is the new, hand-computable
    step the quiz drills."""
    fig, ax = plt.subplots(figsize=(10.8, 4.3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.92, "attention output, in two steps", ha="center",
            color=INK, fontsize=16, fontweight="bold")
    ax.add_patch(FancyBboxPatch((0.08, 0.60), 0.84, 0.18,
                boxstyle="round,pad=0.01,rounding_size=0.03",
                facecolor="#F1F1EC", edgecolor=HAIRLINE, lw=1.3))
    ax.text(0.5, 0.69,
            r"1.  weights $=$ softmax(scores),   "
            r"$\mathrm{softmax}_i=\dfrac{e^{s_i}}{\sum_j e^{s_j}}$",
            ha="center", va="center", fontsize=17, color=INK_SOFT)
    ax.text(0.5, 0.52, "(the Lecture 4 softmax — turns scores into weights that sum to 1)",
            ha="center", va="center", fontsize=11.5, color=MUTED,
            style="italic")
    ax.add_patch(FancyBboxPatch((0.08, 0.26), 0.84, 0.18,
                boxstyle="round,pad=0.01,rounding_size=0.03",
                facecolor=TEAL_SOFT, edgecolor=TEAL, lw=1.6))
    ax.text(0.5, 0.35,
            r"2.  output $=\sum_i$ weight$_i\times$value$_i$",
            ha="center", va="center", fontsize=20, color=TEAL,
            fontweight="bold")
    ax.text(0.5, 0.14,
            "a weighted average of the values — trust the highly-weighted tokens more",
            ha="center", va="center", fontsize=13, color=INK_SOFT)
    _save(fig, name)


def attention_weighted_avg(weights, values, output, name,
                           scores=None, reveal=True, tokens=None):
    """The centerpiece numeric mechanic (rule 6): scores → softmax → weights,
    then output = Σ weight×value, worked cell by cell for 3 tokens.
    scores=None hides the softmax step (the you-do gives weights directly,
    matching Quiz 7 Q2). reveal=False leaves the output as '?'."""
    n = len(values)
    tokens = tokens or [f"token {i+1}" for i in range(n)]
    xs = np.linspace(2.4, 9.6, n)
    fig, ax = plt.subplots(figsize=(11.4, 5.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.4); ax.axis("off")

    def label(y, txt, color=INK_SOFT):
        ax.text(0.5, y, txt, ha="left", va="center", fontsize=12.5,
                color=color, fontweight="bold")
    # token headers
    for i, t in enumerate(tokens):
        ax.text(xs[i], 5.05, t, ha="center", va="center", fontsize=12,
                color=MUTED)
    y_score, y_w, y_v = 4.4, 3.4, 2.4
    if scores is not None:
        label(y_score, "score  s")
        for i, s in enumerate(scores):
            _chip(ax, xs[i] - 0.5, y_score, 1.0, 0.55, f"{s:g}", INK_SOFT,
                  fs=15)
        ax.text(6, (y_score + y_w) / 2 + 0.02, "softmax  ↓", ha="center",
                color=TEAL, fontsize=12.5, fontweight="bold", style="italic")
    label(y_w, "weight  w")
    for i, w in enumerate(weights):
        _chip(ax, xs[i] - 0.5, y_w, 1.0, 0.55, f"{w:g}", TEAL, fs=15)
    label(y_v, "value  v")
    for i, v in enumerate(values):
        _chip(ax, xs[i] - 0.5, y_v, 1.0, 0.55, f"{v:g}", AMBER, fs=15)
    # computation line
    terms = "  +  ".join(f"{w:g}·{v:g}" for w, v in zip(weights, values))
    if reveal:
        prods = [w * v for w, v in zip(weights, values)]
        prodstr = "  +  ".join(f"{p:g}" for p in prods)
        comp = f"output = {terms} = {prodstr} = {output:g}"
    else:
        comp = f"output = {terms} = ?"
    ax.add_patch(FancyBboxPatch((0.4, 0.75), 11.2, 0.95,
                boxstyle="round,pad=0.02,rounding_size=0.04",
                facecolor=TEAL_SOFT if reveal else "#F1F1EC",
                edgecolor=TEAL if reveal else HAIRLINE, lw=1.6))
    ax.text(6, 1.22, comp, ha="center", va="center",
            color=INK if reveal else INK_SOFT, fontsize=13.5,
            fontweight="bold", family="monospace")
    tag = ("blend the values, trusting the highly-weighted token most"
           if reveal else "weights are given — just blend the values")
    ax.text(6, 0.28, tag, ha="center", color=MUTED, fontsize=11.5,
            style="italic")
    _save(fig, name)


def rnn_vs_attention(name="fig_rnn_vs_attention.png"):
    """Why attention beat recurrence, in one comparison. Left: recurrence passes
    a hidden state token by token — sequential (slow) and the earliest token
    fades (forgetful). Right: self-attention links all pairs at once — parallel
    and long-range (token 1 ↔ token 5 in one hop)."""
    toks = ["t1", "t2", "t3", "t4", "t5"]
    n = len(toks)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    xs = np.linspace(0.9, 5.1, n)
    y = 2.3
    # left: recurrence
    ax = axes[0]
    ax.set_xlim(0, 6); ax.set_ylim(0, 4.4); ax.axis("off")
    ax.text(3, 4.1, "recurrence (RNN / LSTM)", ha="center", color=INK,
            fontsize=14, fontweight="bold")
    for i in range(n - 1):
        ax.annotate("", xy=(xs[i + 1] - 0.4, y), xytext=(xs[i] + 0.4, y),
                    arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, lw=2))
    for i, t in enumerate(toks):
        _chip(ax, xs[i] - 0.4, y, 0.8, 0.6, t, TEAL_SOFT, txt=INK, fs=13,
              edge=TEAL, lw=1.4)
    # fading long-range arc t1 -> t5
    ax.annotate("", xy=(xs[4], y + 0.5), xytext=(xs[0], y + 0.5),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4,
                                alpha=0.4, connectionstyle="arc3,rad=-0.4"))
    ax.text(3, y + 1.35, "early tokens fade (forgetful)", ha="center",
            color=RED, fontsize=11.5, fontweight="bold")
    ax.text(3, 0.7, "step 5 must wait for steps 1–4 (slow, one at a time)",
            ha="center", color=MUTED, fontsize=11.5, style="italic")
    # right: self-attention
    ax = axes[1]
    ax.set_xlim(0, 6); ax.set_ylim(0, 4.4); ax.axis("off")
    ax.text(3, 4.1, "self-attention", ha="center", color=INK, fontsize=14,
            fontweight="bold")
    for i in range(n):
        for j in range(i + 1, n):
            rad = -0.25 * (j - i)
            ax.annotate("", xy=(xs[j], y + 0.32), xytext=(xs[i], y + 0.32),
                        arrowprops=dict(arrowstyle="-", color=TEAL, lw=1.0,
                                        alpha=0.45,
                                        connectionstyle=f"arc3,rad={rad}"),
                        zorder=1)
    for i, t in enumerate(toks):
        _chip(ax, xs[i] - 0.4, y, 0.8, 0.6, t, TEAL_SOFT, txt=INK, fs=13,
              edge=TEAL, lw=1.4)
    ax.annotate("", xy=(xs[4], y + 0.55), xytext=(xs[0], y + 0.55),
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2.2,
                                connectionstyle="arc3,rad=-0.5"))
    ax.text(3, y + 1.35, "token 1 ↔ token 5 in one hop (long-range)",
            ha="center", color=TEAL, fontsize=11.5, fontweight="bold")
    ax.text(3, 0.7, "all pairs computed at once (parallel, fast on a GPU)",
            ha="center", color=MUTED, fontsize=11.5, style="italic")
    _save(fig, name)


def _lecture7_figures():
    seq_order_matters()
    tokenize_peptide()
    embedding_space()
    selfattn_all_to_all()
    qkv_lookup(reveal=True, name="fig_qkv_ido.png")
    qkv_lookup(reveal=False, name="fig_qkv_youdo.png")
    attention_heatmap("sentence", "fig_attn_heatmap_ido.png")
    attention_heatmap("peptide", "fig_attn_heatmap_youdo.png")
    attention_formula()
    # I-do: scores [1,2,0] -> softmax weights [0.24,0.67,0.09]; values [5,10,0]
    #  output = 0.24·5 + 0.67·10 + 0.09·0 = 1.2 + 6.7 + 0 = 7.9
    attention_weighted_avg([0.24, 0.67, 0.09], [5, 10, 0], 7.9,
                           "fig_attn_weightedavg_ido.png",
                           scores=[1, 2, 0], reveal=True,
                           tokens=["token 1", "token 2", "token 3"])
    # you-do = Quiz 7 Q2: weights [0.5,0.3,0.2] given, values [4,10,1]
    #  output = 0.5·4 + 0.3·10 + 0.2·1 = 2 + 3 + 0.2 = 5.2
    attention_weighted_avg([0.5, 0.3, 0.2], [4, 10, 1], 5.2,
                           "fig_attn_weightedavg_youdo.png",
                           scores=None, reveal=False,
                           tokens=["token 1", "token 2", "token 3"])
    rnn_vs_attention()


# ============================================================================
#  Lecture 8 · The Transformer, Assembled (+ transfer learning)
#  Hand-authored numeric mechanics (rule 6) + bespoke MS-anchored schematics
#  (rule 9). The canonical full encoder–decoder diagram is a REAL licensed
#  image (dvgodoy, Wikimedia Commons, CC BY 4.0) placed directly in the deck.
# ============================================================================

def positional_encoding(name="fig_posenc.png"):
    """Positional encoding's job, by picture (rule 6/9). Left: attention is
    order-blind — the SAME residues in two orders give the same weighted
    average, so attention can't tell them apart. Right: the fix, an exact
    element-wise ADD of a per-position signal to a token's embedding, so the
    same token at a new position becomes a different vector."""
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5),
                             gridspec_kw={"width_ratios": [1, 1.15]})
    # ---- left: attention is order-blind ----
    ax = axes[0]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.96, "attention is order-blind", ha="center", fontsize=14,
            color=INK, fontweight="bold")

    def orow(y, toks, cap):
        n = len(toks); w = 0.17; gap = 0.03
        total = n * w + (n - 1) * gap; x0 = 0.5 - total / 2
        for i, t in enumerate(toks):
            _chip(ax, x0 + i * (w + gap), y, w, 0.14, t, TEAL_SOFT, txt=INK,
                  fs=15, edge=TEAL, lw=1.4)
        ax.text(0.5, y - 0.12, cap, ha="center", va="top", fontsize=11.5,
                color=INK_SOFT)
    orow(0.74, ["D", "E", "K"], "peptide  D–E–K")
    orow(0.44, ["K", "E", "D"], "shuffled  K–E–D")
    ax.text(0.5, 0.20,
            "same tokens → same weighted average\nattention can’t tell the two orders apart",
            ha="center", va="top", fontsize=11, color=RED, fontweight="bold")
    # ---- right: the fix — add a position signal ----
    ax = axes[1]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.96, "the fix: ADD a position signal", ha="center",
            fontsize=14, color=INK, fontweight="bold")
    emb = [0.2, 0.9, 0.5, 0.1]
    pos = [0.1, 0.0, 0.5, 1.0]
    tot = [round(a + b, 1) for a, b in zip(emb, pos)]

    def vec(y, label, vals, color, lcol, txt=WHITE):
        ax.text(0.03, y + 0.13, label, ha="left", va="center", fontsize=11,
                color=lcol, fontweight="bold")
        n = len(vals); w = 0.155; gap = 0.02
        total = n * w + (n - 1) * gap; x0 = 0.97 - total
        for i, v in enumerate(vals):
            _chip(ax, x0 + i * (w + gap), y, w, 0.13, f"{v:g}", color,
                  txt=txt, fs=13)
    vec(0.72, "token “peptide” embedding", emb, TEAL, TEAL)
    ax.text(0.645, 0.605, "+", ha="center", fontsize=20, color=INK,
            fontweight="bold")
    vec(0.49, "position-3 signal (a fixed wave)", pos, AMBER, ROI_INK)
    ax.plot([0.32, 0.97], [0.37, 0.37], color=INK_SOFT, lw=1.3)
    vec(0.26, "= position-aware input", tot, INK_SOFT, INK)
    ax.text(0.5, 0.075,
            "same token at a new position → a different vector, so order becomes visible",
            ha="center", va="top", fontsize=10, color=MUTED, style="italic")
    fig.subplots_adjust(wspace=0.16)
    _save(fig, name)


def norm_axes(mode="ido", name="fig_norm_axes.png"):
    """Batch norm vs. layer norm at the 'which axis gets averaged' level
    (the Lecture 8 learning outcome). Rows = samples in the batch, columns =
    features. LAYER norm averages across a token's features (one teal row);
    BATCH norm averages down a feature across the batch (one amber column).
    mode='ido' shows the worked means and a full, exact layer-norm normalize;
    mode='youdo' blanks the answers for Quiz 8 Q4 (parallel matrix, new
    numbers)."""
    if mode == "ido":
        data = [[2, 2, 6, 6], [4, 4, 8, 8], [0, 0, 4, 4]]
        clabels = ["f1", "f2", "f3", "f4"]; rlabels = ["tok1", "tok2", "tok3"]
    else:
        data = [[2, 4, 6], [4, 6, 8]]
        clabels = ["f1", "f2", "f3"]; rlabels = ["tok1", "tok2"]
    rows = len(data); cols = len(data[0])
    fig, ax = plt.subplots(figsize=(11.6, 5.0 if mode == "ido" else 4.2))
    ax.set_xlim(-1.7, cols + 6.4)
    ax.set_ylim(-2.5 if mode == "ido" else -1.9, rows + 1.3)
    ax.axis("off")
    for c in range(cols):
        ax.text(c + 0.48, rows + 0.28, clabels[c], ha="center", va="bottom",
                fontsize=12, color=INK_SOFT, fontweight="bold")
    for r in range(rows):
        yy = rows - 1 - r
        ax.text(-0.22, yy + 0.48, rlabels[r], ha="right", va="center",
                fontsize=12, color=INK_SOFT, fontweight="bold")
        for c in range(cols):
            face = TEAL_SOFT if r == 0 else WHITE
            ax.add_patch(plt.Rectangle((c, yy), 0.96, 0.96, facecolor=face,
                         edgecolor=HAIRLINE, lw=1.2, zorder=2))
            ax.text(c + 0.48, yy + 0.48, f"{data[r][c]:g}", ha="center",
                    va="center", fontsize=15, color=INK, fontweight="bold",
                    zorder=3)
    # amber box around column f1 (batch-norm axis)
    ax.add_patch(plt.Rectangle((-0.04, -0.04), 1.04, rows + 0.08, fill=False,
                 edgecolor=AMBER, lw=3.0, zorder=5))
    # teal arrow across the top row (layer-norm axis)
    ax.annotate("", xy=(cols + 1.0, rows - 0.5), xytext=(cols + 0.15, rows - 0.5),
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2.4))
    if mode == "ido":
        ax.text(cols + 1.15, rows - 0.5,
                "LAYER norm — average across a\ntoken’s features:  mean(2,2,6,6) = 4",
                ha="left", va="center", fontsize=11.5, color=TEAL,
                fontweight="bold")
    else:
        ax.text(cols + 1.15, rows - 0.5,
                "LAYER norm averages across ___?\nmean of tok1 = ?",
                ha="left", va="center", fontsize=11.5, color=TEAL,
                fontweight="bold")
    # amber arrow down column f1 (batch-norm axis)
    ax.annotate("", xy=(0.48, -1.05), xytext=(0.48, -0.18),
                arrowprops=dict(arrowstyle="-|>", color=ROI_INK, lw=2.4))
    if mode == "ido":
        ax.text(0.48, -1.2, "BATCH norm — average down\nthe batch:  mean(2,4,0) = 2",
                ha="center", va="top", fontsize=11.5, color=ROI_INK,
                fontweight="bold")
    else:
        ax.text(0.48, -1.15, "BATCH norm averages down ___?\nmean of f1 = ?",
                ha="center", va="top", fontsize=11.5, color=ROI_INK,
                fontweight="bold")
    if mode == "ido":
        strip = ("normalize tok1:  [2, 2, 6, 6]  − mean 4  →  [−2, −2, 2, 2]"
                 "  ÷ std 2  →  [−1, −1, 1, 1]")
        ax.text(cols / 2.0, -2.1, strip, ha="center", va="center",
                fontsize=12, color=INK, fontweight="bold", family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc=TEAL_SOFT, ec=TEAL,
                          lw=1.4))
    _save(fig, name)


def transformer_block(labeled=True, name="fig_transformer_block.png"):
    """A simplified transformer (encoder) block as a vertical stack of the four
    blocks the room already owns, so Quiz 8 Q1 is labelable from the taught
    diagram. labeled=True names each block (the I-do reference); labeled=False
    blanks them 1-4 with a word bank (the you-do = Quiz 8 Q1). Same layout and
    colours in both so the pair matches visually (rule 8)."""
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    ax.set_xlim(0, 8); ax.set_ylim(0, 11); ax.axis("off")
    cx = 3.2; bw = 5.2
    names = (["Positional encoding", "Multi-head self-attention",
              "Add & Norm  (layer norm)", "Feed-forward network"]
             if labeled else ["①", "②", "③", "④"])
    styles = [(TEAL_SOFT, TEAL), (AMBER_SOFT, AMBER), (WHITE, INK_SOFT),
              ("#F7E4E3", RED)]
    ys = [1.0, 3.2, 5.4, 7.6]
    bh = 1.3

    def arrow(y0, y1):
        ax.annotate("", xy=(cx, y1), xytext=(cx, y0),
                    arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, lw=2.0))
    ax.text(cx, 0.35, "input tokens", ha="center", fontsize=11, color=MUTED)
    arrow(0.6, 1.0)
    for i, (lab, (face, edge), y) in enumerate(zip(names, styles, ys)):
        ax.add_patch(FancyBboxPatch((cx - bw / 2, y), bw, bh,
                    boxstyle="round,pad=0.02,rounding_size=0.08",
                    facecolor=face, edgecolor=edge, lw=2.4, zorder=3))
        ax.text(cx, y + bh / 2, lab, ha="center", va="center",
                color=INK if labeled else edge,
                fontsize=12.5 if labeled else 20, fontweight="bold", zorder=4)
        if i < len(ys) - 1:
            arrow(y + bh, ys[i + 1])
    arrow(ys[-1] + bh, ys[-1] + bh + 0.5)
    ax.text(cx, ys[-1] + bh + 0.75, "to the next block  →  output",
            ha="center", fontsize=11, color=MUTED)
    # right-hand rail: 'residual skip' note echoing the '+' in the real diagram
    ax.annotate("", xy=(cx + bw / 2 + 0.35, ys[2] + bh / 2),
                xytext=(cx + bw / 2 + 0.35, ys[1] - 0.1),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6,
                                connectionstyle="arc3,rad=0.5"))
    ax.text(cx + bw / 2 + 0.5, (ys[1] + ys[2]) / 2 + 0.3, "residual\nskip",
            ha="left", va="center", fontsize=9.5, color=MUTED, style="italic")
    if not labeled:
        bank = ("word bank:   Feed-forward network  ·  Multi-head self-attention"
                "  ·  Add & Norm (layer norm)  ·  Positional encoding")
        ax.text(cx, -0.15, bank, ha="center", va="top", fontsize=10,
                color=INK_SOFT, wrap=True)
        ax.set_ylim(-1.0, 11)
    _save(fig, name)


def bert_vs_gpt(name="fig_bert_vs_gpt.png"):
    """Encoder-only (BERT: read the whole sequence → one label / embedding) vs.
    decoder-only (GPT: generate the next token, left to right). MS anchors
    underneath each."""
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
    toks = ["A", "C", "D", "E", "K"]
    xs = np.linspace(1.0, 5.0, 5)
    # ---- left: encoder-only ----
    ax = axes[0]; ax.set_xlim(0, 6); ax.set_ylim(0, 5); ax.axis("off")
    ax.text(3, 4.7, "Encoder-only  (BERT)", ha="center", fontsize=14,
            color=TEAL, fontweight="bold")
    ax.add_patch(FancyBboxPatch((1.0, 3.0), 4.0, 0.75,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=TEAL_SOFT, edgecolor=TEAL, lw=2.2))
    ax.text(3, 3.375, "one label for the whole input", ha="center",
            va="center", fontsize=11.5, color=INK, fontweight="bold")
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            ax.annotate("", xy=(xs[j], 1.95), xytext=(xs[i], 1.95),
                        arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.8,
                        alpha=0.4, connectionstyle=f"arc3,rad={-0.18*(j-i)}"))
    for x, t in zip(xs, toks):
        _chip(ax, x - 0.33, 1.3, 0.66, 0.55, t, TEAL_SOFT, txt=INK, fs=13,
              edge=TEAL, lw=1.3)
    ax.text(3, 0.62, "reads the WHOLE sequence, both directions → classify / embed",
            ha="center", va="top", fontsize=10, color=INK_SOFT)
    ax.text(3, 0.18, "MS: whole-spectrum call — resistant / susceptible",
            ha="center", va="top", fontsize=10, color=TEAL, fontweight="bold")
    # ---- right: decoder-only ----
    ax = axes[1]; ax.set_xlim(0, 6); ax.set_ylim(0, 5); ax.axis("off")
    ax.text(3, 4.7, "Decoder-only  (GPT)", ha="center", fontsize=14,
            color=RED, fontweight="bold")
    gen = ["A", "C", "D", "E", "?"]
    for k in range(4):
        ax.annotate("", xy=(xs[k + 1] - 0.32, 1.6), xytext=(xs[k] + 0.32, 1.6),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6))
    for i, (x, t) in enumerate(zip(xs, gen)):
        face = "#F7E4E3" if i < 4 else WHITE
        edge = RED if i < 4 else AMBER
        _chip(ax, x - 0.33, 1.3, 0.66, 0.55, t, face, txt=INK, fs=13,
              edge=edge, lw=1.3)
    ax.add_patch(FancyBboxPatch((3.6, 2.7), 2.2, 0.7,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=AMBER_SOFT, edgecolor=AMBER, lw=2.0))
    ax.text(4.7, 3.05, "predict next", ha="center", va="center", fontsize=11,
            color=ROI_INK, fontweight="bold")
    ax.annotate("", xy=(xs[4], 1.65), xytext=(4.7, 2.7),
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.6))
    ax.text(3, 0.62, "generates one token at a time, left to right → generate",
            ha="center", va="top", fontsize=10, color=INK_SOFT)
    ax.text(3, 0.18, "MS: de novo peptide — predict the next residue (Casanovo)",
            ha="center", va="top", fontsize=10, color=RED, fontweight="bold")
    fig.subplots_adjust(wspace=0.12)
    _save(fig, name)


def _mini_spectrum(ax, peaks, x_hi=10.0):
    mz = np.linspace(0, x_hi, 500)
    y = np.zeros_like(mz)
    for c, h in peaks:
        y += h * np.exp(-((mz - c) / 0.13) ** 2)
    ax.plot(mz, y, color=TEAL, lw=1.3, zorder=3)
    ax.fill_between(mz, y, color=TEAL, alpha=0.10, zorder=2)
    return mz, y


def tokenization_strategies(name="fig_tokenization_strategies.png"):
    """Three ways to turn ONE spectrum into transformer tokens, side by side:
    (a) chunk into patches + linear projection (ViT-style); (b) a small CNN
    front-end that emits token embeddings (Lecture 5 callback); (c) each peak is
    a token — m/z + intensity embedded (Casanovo)."""
    peaks = [(1.8, 1.0), (3.0, 0.45), (4.8, 0.8), (6.6, 0.3), (8.2, 0.6)]
    fig, axes = plt.subplots(3, 1, figsize=(11.2, 6.6))
    titles = [
        ("(a) chunk into patches → linear projection   —  ViT-style", TEAL),
        ("(b) small CNN front-end → token embeddings   —  Lecture 5 callback", AMBER),
        ("(c) each peak = one token: (m/z, intensity)   —  Casanovo", RED),
    ]
    for row, (ax, (title, col)) in enumerate(zip(axes, titles)):
        ax.set_xlim(0, 15.2); ax.set_ylim(-0.35, 1.5); ax.axis("off")
        ax.text(0.0, 1.42, title, ha="left", va="top", fontsize=12.5,
                color=col, fontweight="bold")
        mz, y = _mini_spectrum(ax, peaks)
        ax.plot([0, 10], [0, 0], color=HAIRLINE, lw=1.0)
        # token chips on the right
        tx = np.linspace(11.2, 14.4, 4)
        if row == 0:  # patches: vertical dashed splits
            for xb in np.linspace(2.5, 10, 4):
                ax.axvline(xb, ymin=0.10, ymax=0.86, color=ROI_INK, ls="--",
                           lw=1.2, alpha=0.8)
            ax.text(5, -0.28, "equal-width chunks", ha="center", fontsize=9.5,
                    color=MUTED, style="italic")
            for x in tx:
                _chip(ax, x - 0.42, 0.55, 0.84, 0.45, "tok", TEAL, fs=11)
        elif row == 1:  # CNN window sliding
            ax.add_patch(FancyBboxPatch((1.2, 0.02), 2.2, 1.05,
                        boxstyle="round,pad=0.02,rounding_size=0.05",
                        fill=False, edgecolor=AMBER, lw=2.0, ls="--"))
            ax.annotate("", xy=(6.0, 1.15), xytext=(2.3, 1.15),
                        arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.6))
            ax.text(5, -0.28, "conv window slides along m/z", ha="center",
                    fontsize=9.5, color=MUTED, style="italic")
            for x in tx:
                _chip(ax, x - 0.42, 0.55, 0.84, 0.45, "tok", AMBER, fs=11)
        else:  # peaks as tokens
            for c, h in peaks:
                ax.plot([c], [h], marker="o", ms=7, color=RED, zorder=5)
            ax.text(5, -0.28, "only the peaks become tokens", ha="center",
                    fontsize=9.5, color=MUTED, style="italic")
            ptx = np.linspace(10.9, 14.6, 5)
            for x in ptx:
                _chip(ax, x - 0.34, 0.55, 0.68, 0.45, "pk", RED, fs=10)
        ax.annotate("", xy=(11.0, 0.55), xytext=(10.2, 0.55),
                    arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, lw=1.6))
    fig.subplots_adjust(hspace=0.55)
    _save(fig, name)


def tokenization_decision(mode="ido", name="fig_tokenization_decision.png"):
    """A which-strategy-for-which-data chart. mode='ido' shows the three
    data→strategy rules; mode='youdo' presents the three Quiz 8 Q2 data-type
    cards with a blank arrow and a word bank of the strategies."""
    fig, ax = plt.subplots(figsize=(11.4, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    if mode == "ido":
        ax.text(6, 5.6, "which strategy for which data?", ha="center",
                fontsize=14, color=INK, fontweight="bold")
        rowspec = [
            ("2D image\n(imaging-MS, LC×MS heatmap)", TEAL,
             "cut into patches +\nlinear projection  (ViT)"),
            ("dense 1D signal\n(binned spectrum, chromatogram)", AMBER,
             "CNN front-end →\ntoken embeddings"),
            ("sparse peak list\n(m/z + intensity peaks)", RED,
             "peak-as-token\n(Casanovo)"),
        ]
    else:
        ax.text(6, 5.6, "your turn: match each data type to a strategy",
                ha="center", fontsize=14, color=ROI_INK, fontweight="bold")
        rowspec = [
            ("imaging-MS image", TEAL, "?"),
            ("raw binned spectrum", AMBER, "?"),
            ("peak list", RED, "?"),
        ]
    ys = [4.3, 2.9, 1.5]
    for (dtype, col, strat), y in zip(rowspec, ys):
        ax.add_patch(FancyBboxPatch((0.3, y - 0.55), 4.4, 1.1,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor=WHITE, edgecolor=col, lw=2.2))
        ax.text(2.5, y, dtype, ha="center", va="center", fontsize=11.5,
                color=INK, fontweight="bold")
        ax.annotate("", xy=(6.9, y), xytext=(4.9, y),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2))
        face = col if mode == "ido" else "#F1F1EC"
        tcol = WHITE if mode == "ido" else MUTED
        ax.add_patch(FancyBboxPatch((7.0, y - 0.55), 4.6, 1.1,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor=face, edgecolor=col, lw=2.2))
        ax.text(9.3, y, strat, ha="center", va="center",
                fontsize=12 if mode == "ido" else 20,
                color=tcol if mode == "ido" else col, fontweight="bold")
    if mode == "youdo":
        bank = ("word bank:   patches + linear projection (ViT)   ·   "
                "CNN front-end   ·   peak-as-token (Casanovo)")
        ax.text(6, 0.35, bank, ha="center", va="center", fontsize=10.5,
                color=INK_SOFT)
    _save(fig, name)


def transfer_learning(name="fig_transfer_learning.png"):
    """Transfer learning: keep a pretrained body (FROZEN, trained on millions),
    bolt on a small NEW head, train only the head on your 500–800 labels."""
    fig, ax = plt.subplots(figsize=(11.4, 4.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    # frozen pretrained body (big teal stack)
    ax.add_patch(FancyBboxPatch((3.6, 0.7), 4.8, 3.2,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=TEAL_SOFT, edgecolor=TEAL, lw=2.6))
    ax.text(6.0, 2.9, "pretrained transformer body", ha="center",
            va="center", fontsize=14, color=INK, fontweight="bold")
    ax.text(6.0, 2.3, "FROZEN — weights kept as-is", ha="center", va="center",
            fontsize=12, color=TEAL, fontweight="bold")
    ax.text(6.0, 1.5, "already learned general features\nfrom MILLIONS of spectra / sequences",
            ha="center", va="center", fontsize=11, color=INK_SOFT)
    # new head (small amber box on top)
    ax.add_patch(FancyBboxPatch((4.6, 4.3), 2.8, 1.1,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=AMBER_SOFT, edgecolor=AMBER, lw=2.6))
    ax.text(6.0, 5.05, "NEW head", ha="center", va="center", fontsize=13,
            color=ROI_INK, fontweight="bold")
    ax.text(6.0, 4.6, "trainable", ha="center", va="center", fontsize=11,
            color=ROI_INK)
    ax.annotate("", xy=(6.0, 4.3), xytext=(6.0, 3.9),
                arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, lw=2.0))
    ax.annotate("", xy=(6.0, 0.7), xytext=(6.0, 0.2),
                arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, lw=2.0))
    ax.text(6.0, 0.05, "your spectrum in", ha="center", va="top",
            fontsize=10.5, color=MUTED)
    # right note: your small labeled set trains the head
    ax.annotate("", xy=(7.5, 4.85), xytext=(9.4, 4.85),
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=2.0))
    ax.text(9.5, 4.85, "train on your\n500–800 labels", ha="left", va="center",
            fontsize=11, color=ROI_INK, fontweight="bold")
    ax.text(9.5, 2.3,
            "from scratch would need\nmillions of labels — you\nhave hundreds → fine-tune",
            ha="left", va="center", fontsize=10.5, color=INK_SOFT)
    ax.text(1.0, 2.3, "borrow the\nbody →", ha="center", va="center",
            fontsize=12, color=TEAL, fontweight="bold")
    _save(fig, name)


def _lecture8_figures():
    positional_encoding()
    norm_axes("ido", "fig_norm_axes.png")
    norm_axes("youdo", "fig_norm_youdo.png")
    transformer_block(True, "fig_transformer_block.png")
    transformer_block(False, "fig_transformer_block_youdo.png")
    bert_vs_gpt()
    tokenization_strategies()
    tokenization_decision("ido", "fig_tokenization_decision.png")
    tokenization_decision("youdo", "fig_tokenization_youdo.png")
    transfer_learning()


FUNCS = {
    "maldi": maldi_real,
    "chatgpt": chatgpt_panel,
    "driams": driams_mrsa_spectrum,
    "activations": lambda: (
        activations(True, "fig_activations.png", ("relu", "sigmoid", "linear")),
        activations(False, "fig_activations_quiz.png", ("sigmoid", "relu", "linear")),
    ),
    "neuron": lambda: (
        neuron_schematic(3, 2, 2, -1, 1, "ReLU", "5", "5", "fig_neuron_ido.png"),
        neuron_schematic(2, 3, 0.5, -1, 1, "ReLU", "?", "?", "fig_neuron_youdo.png"),
    ),
    "matvec": lambda: (
        matvec([[1, 0, 2], [-1, 3, 1]], [2, 1, 0], 2, "?", 1, "fig_matvec_row1.png"),
        matvec([[1, 0, 2], [-1, 3, 1]], [2, 1, 0], 2, "?", 2, "fig_matvec_row2.png"),
    ),
    "universal": universal_approx,
    "twolayer": lambda: (
        two_layer_net(2, 1, "fig_twolayer_ido.png",
                      w1=((1, -1), (0, 1)), b1=(0, 0), w2=(2, 1), b2=-1,
                      hidden=(1, 1), y=2,
                      arith="y = 2\u00b71 + 1\u00b71 \u2212 1 = 2"),
        two_layer_net(2, 1, "fig_twolayer_youdo.png",
                      w1=((1, -1), (-1, 1)), b1=(0, -4), w2=(3, 2), b2=-1),
    ),
    "hook": hook_4panel,
    "cat": cat_hierarchy,
    # ---- Lecture 2 ----
    "random_weights": random_weights_net,
    "loss_mse": loss_mse,
    "loss_bowl": loss_bowl,
    "lr_modes": lr_three_modes,
    "loss_curves": lambda: (
        loss_curves(True, "fig_loss_curves_ido.png"),
        loss_curves(False, "fig_loss_curves_youdo.png"),
    ),
    "backprop": lambda: (
        backprop_two_weight(2, 0.5, 1.5, 1, "fwd", "fig_backprop_forward.png"),
        backprop_two_weight(2, 0.5, 1.5, 1, "full", "fig_backprop_backward.png"),
        backprop_two_weight(1, 2, 0.5, 0, "blank", "fig_backprop_youdo.png"),
    ),
    "backprop_chain": backprop_chain,
    "onestep": lambda: (
        onestep_update([("w\u2082", "1.5 \u2212 0.1 \u00d7 1.0", "1.4"),
                        ("w\u2081", "0.5 \u2212 0.1 \u00d7 3.0", "0.2")],
                       (0.25, 0.19), "fig_onestep.png"),
        onestep_update([("w\u2082", "0.5 \u2212 0.1 \u00d7 ?", "?"),
                        ("w\u2081", "2 \u2212 0.1 \u00d7 ?", "?")],
                       None, "fig_onestep_youdo.png"),
    ),
    "backprop_scale": backprop_scale_net,
    "loop_cycle": training_loop_cycle,
    "training_code": training_code,
    "loop_order": loop_order,
    "autograd": autograd_check,
    "loss_fall": loss_fall,
    # ---- Lecture 4 ----
    "softmax": softmax_bars,
    "ce_curve": crossentropy_curve,
    "ce_formula": crossentropy_formula,
    "ce_worked": lambda: (
        crossentropy_worked("fig_ce_worked.png"),
        crossentropy_worked("fig_ce_youdo.png",
                            cards=[("Model A \u2014 0.8 on R", 0.80, None, TEAL),
                                   ("Model B \u2014 0.2 on R", 0.20, None, RED)],
                            reveal=False),
    ),
    "minibatch": minibatch_paths,
    "adam": adam_momentum,
    "lr_schedule": lr_schedule,
    "activation_choices": activation_choices,
    "vanishing": lambda: (
        vanishing_chain("ido", "fig_vanishing_ido.png"),
        vanishing_chain("youdo", "fig_vanishing_youdo.png"),
    ),
    "search": hyperparam_search,
    "batch_tradeoff": batch_tradeoff,
    "batch_curve": batch_loss_curve,
    "fit_trio": fit_trio,
    "trainval": lambda: (
        trainval_curves(True, "fig_trainval_ido.png"),
        trainval_curves(False, "fig_trainval_youdo.png"),
    ),
    "early_stopping": early_stopping,
    "dropout": dropout_panels,
    "weight_decay": weight_decay,
    "imbalance": class_imbalance,
    # ---- Lecture 5 ----
    "cnn": _lecture5_figures,
    "dense_explosion": dense_explosion,
    "conv_real_filters": conv_real_filters,
    "conv1d": lambda: (conv1d_spectrum(), conv1d_code()),
    "detection": lambda: (
        detection_grid(),
        peak_detection(reveal=True, name="fig_peak_detection.png"),
        peak_detection(reveal=False, name="fig_peak_detection_youdo.png"),
    ),
    # ---- Lecture 7 ----
    "attention": _lecture7_figures,
    "seq_order": seq_order_matters,
    "tokenize": tokenize_peptide,
    "embedding": embedding_space,
    "selfattn": selfattn_all_to_all,
    "qkv": lambda: (
        qkv_lookup(reveal=True, name="fig_qkv_ido.png"),
        qkv_lookup(reveal=False, name="fig_qkv_youdo.png"),
    ),
    "attn_heatmap": lambda: (
        attention_heatmap("sentence", "fig_attn_heatmap_ido.png"),
        attention_heatmap("peptide", "fig_attn_heatmap_youdo.png"),
    ),
    "attn_formula": attention_formula,
    "attn_weightedavg": lambda: (
        attention_weighted_avg([0.24, 0.67, 0.09], [5, 10, 0], 7.9,
                               "fig_attn_weightedavg_ido.png",
                               scores=[1, 2, 0], reveal=True,
                               tokens=["token 1", "token 2", "token 3"]),
        attention_weighted_avg([0.5, 0.3, 0.2], [4, 10, 1], 5.2,
                               "fig_attn_weightedavg_youdo.png",
                               scores=None, reveal=False,
                               tokens=["token 1", "token 2", "token 3"]),
    ),
    "rnn_vs_attn": rnn_vs_attention,
    # ---- Lecture 8 ----
    "transformer": _lecture8_figures,
    "posenc": positional_encoding,
    "norm_axes": lambda: (
        norm_axes("ido", "fig_norm_axes.png"),
        norm_axes("youdo", "fig_norm_youdo.png"),
    ),
    "transformer_block": lambda: (
        transformer_block(True, "fig_transformer_block.png"),
        transformer_block(False, "fig_transformer_block_youdo.png"),
    ),
    "bert_vs_gpt": bert_vs_gpt,
    "tokenization": lambda: (
        tokenization_strategies(),
        tokenization_decision("ido", "fig_tokenization_decision.png"),
        tokenization_decision("youdo", "fig_tokenization_youdo.png"),
    ),
    "transfer": transfer_learning,
}


def main(argv):
    if len(argv) > 1:
        for key in argv[1:]:
            fn = FUNCS.get(key)
            if not fn:
                print(f"unknown figure: {key} (choose from {', '.join(FUNCS)})")
                return 1
            print(f"figure: {key}")
            fn()
    else:
        build_all()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
