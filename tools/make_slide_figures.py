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


def activation_apply(name="fig_activation_apply.png"):
    """You-do (apply/decide): three LABELED curves + a scenario (which one gives a
    0-1 probability?) and two concrete activation computations. Replaces the old
    'name that activation' recall task with reasoning + arithmetic."""
    names = {"relu": "ReLU", "sigmoid": "Sigmoid", "linear": "Linear"}
    colors = {"relu": TEAL, "sigmoid": AMBER, "linear": RED}
    order = ("sigmoid", "relu", "linear")
    prompts = {"sigmoid": "sigmoid(0) = ?", "relu": "ReLU(\u22123) = ?", "linear": ""}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    tags = ["(a)", "(b)", "(c)"]
    for i, (ax, kind) in enumerate(zip(axes, order)):
        _draw_activation(ax, kind, colors[kind])
        ax.set_title(f"{tags[i]}  {names[kind]}", color=INK, fontsize=20,
                     fontweight="bold", pad=10)
        if prompts[kind]:
            ax.text(0.5, -0.13, prompts[kind], transform=ax.transAxes,
                    ha="center", va="top", color=INK_SOFT, fontsize=16,
                    fontweight="bold")
    fig.suptitle("Which curve gives a 0\u20131 probability for the final neuron?",
                 color=TEAL, fontsize=18, fontweight="bold", y=1.03)
    fig.subplots_adjust(wspace=0.12, top=0.80, bottom=0.14)
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


def loop_bug(name="fig_loop_bug.png"):
    """You-do = Quiz 2 Q4 (spot-the-bug): zero_grad() is misplaced AFTER
    backward(), so it erases the fresh gradients and step() changes nothing.
    All five lines are present and plausible -- the task is reasoning, not order."""
    rows = [
        {"code": "for x, y in loader:", "comment": "# every batch", "dim": True},
        {"code": "    pred = model(x)", "comment": "# forward"},
        {"code": "    loss = loss_fn(pred, y)", "comment": "# how wrong?"},
        {"code": "    loss.backward()", "comment": "# backprop the blame"},
        {"code": "    optimizer.zero_grad()", "comment": "# clear gradients"},
        {"code": "    optimizer.step()", "comment": "# nudge every weight"},
    ]
    _code_figure(rows, name, figsize=(11, 3.8), comment_x=0.62)
    return


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
    """FORMAL definition of softmax (the foundation reused by attention in
    Lecture 7): softmax_i = e^{s_i} / sum_j e^{s_j}, worked on the MALDI R/S
    scores 2.0, 0.5 -> e^2.0=7.39, e^0.5=1.65, sum 9.04 -> 0.82, 0.18."""
    labels = ["R (resistant)", "S (susceptible)"]
    probs = [0.82, 0.18]
    colors = [TEAL, AMBER]
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-2.4, 1.7)
    # --- formal formula + worked exponentials (the definition) ---
    ax.text(0.5, 1.5,
            r"$\mathrm{softmax}(s)_i=\dfrac{e^{\,s_i}}{\sum_j e^{\,s_j}}$",
            ha="center", va="center", color=INK, fontsize=23)
    ax.text(0.5, 0.78,
            "exponentiate every score (makes it positive), then divide by the total",
            ha="center", va="center", color=MUTED, fontsize=12.5, style="italic")
    # bars
    for i, (lab, p, c) in enumerate(zip(labels, probs, colors)):
        yy = 0.05 - i * 0.72
        ax.add_patch(FancyBboxPatch((0, yy - 0.24), 1.0, 0.48,
                    boxstyle="round,pad=0.005,rounding_size=0.03",
                    facecolor="#F1F1EC", edgecolor=HAIRLINE, lw=1.2, zorder=1))
        ax.add_patch(FancyBboxPatch((0, yy - 0.24), max(p, 0.02), 0.48,
                    boxstyle="round,pad=0.005,rounding_size=0.03",
                    facecolor=c, edgecolor="none", zorder=2))
        ax.text(-0.02, yy, lab, ha="right", va="center", color=INK_SOFT,
                fontsize=14, fontweight="bold")
        ax.text(p + 0.03, yy, f"{p:.2f}", ha="left", va="center", color=c,
                fontsize=15, fontweight="bold")
    # worked arithmetic line beneath the bars
    ax.text(0.5, -1.55,
            r"scores $R=2.0,\ S=0.5$:   "
            r"$e^{2.0}=7.39,\ e^{0.5}=1.65$   (sum $=9.04$)",
            ha="center", va="center", color=INK_SOFT, fontsize=13.5)
    ax.text(0.5, -2.02,
            r"$\Rightarrow\ \frac{7.39}{9.04}=0.82,\ \ \frac{1.65}{9.04}=0.18$"
            "      \u201c82% sure it\u2019s resistant\u201d",
            ha="center", va="center", color=TEAL, fontsize=14.5,
            fontweight="bold")
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
    # reduction -- give the text its own auto-fitting box so it can never
    # overflow (a fixed FancyBboxPatch was too narrow for the 22pt line).
    ax.text(0.5, 0.48,
            r"so only the TRUE class survives:  $\mathrm{CE}\;=\;-\log(p_{\mathrm{true}})$",
            ha="center", va="center", fontsize=22, color=TEAL, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F1F1EC",
                      edgecolor=HAIRLINE, linewidth=1.3))
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


def conv_walk_two_strides(image, filt, name="fig_conv_youdo.png",
                          filt_note="", note=None, calc_text=None):
    """You-do twin of conv_walk: one image and one filter, TWO empty output maps
    to derive in full — the whole stride-1 map and the whole stride-2 map
    (Quiz 5 Q1). Both grids show '?' so the room fills every cell."""
    n, k = len(image), len(filt)
    M1, o1 = _convmap(image, filt, 1)
    M2, o2 = _convmap(image, filt, 2)

    fig, ax = plt.subplots(figsize=(13.0, 5.3))
    ax.set_xlim(0, 14.0)
    ax.set_ylim(0, 7)
    ax.axis("off")
    cell = 0.66

    # image grid (left)
    ix0, iytop = 0.4, 6.15
    _cellgrid(ax, image, ix0, iytop, cell, fontsize=16)
    ax.text(ix0 + n * cell / 2, iytop + 0.32, f"image  {n}×{n}",
            ha="center", color=INK_SOFT, fontsize=14, fontweight="bold")

    ax.text(ix0 + n * cell + 0.42, 4.5, "∗", ha="center", va="center",
            fontsize=30, color=INK_SOFT)

    # filter grid (middle)
    fx0, fytop = ix0 + n * cell + 0.95, 5.55
    _cellgrid(ax, filt, fx0, fytop, cell, txt=TEAL, edge=TEAL, fontsize=16)
    ax.text(fx0 + k * cell / 2, fytop + 0.32, f"filter  {k}×{k}",
            ha="center", color=TEAL, fontsize=14, fontweight="bold")
    if filt_note:
        ax.text(fx0 + k * cell / 2, fytop - k * cell - 0.28, filt_note,
                ha="center", color=MUTED, fontsize=12, style="italic")

    ax.text(fx0 + k * cell + 0.5, 4.5, "=", ha="center", va="center",
            fontsize=28, color=INK_SOFT)

    # the two blank maps to fill, side by side
    m1x0, mytop = fx0 + k * cell + 1.05, 5.55
    _cellgrid(ax, M1, m1x0, mytop, cell, reveal=set(), fontsize=17)
    ax.text(m1x0 + o1 * cell / 2, mytop + 0.32,
            f"stride 1  →  {o1}×{o1}",
            ha="center", color=INK_SOFT, fontsize=14, fontweight="bold")

    m2x0 = m1x0 + o1 * cell + 1.15
    _cellgrid(ax, M2, m2x0, mytop, cell, reveal=set(), fontsize=17)
    ax.text(m2x0 + o2 * cell / 2, mytop + 0.32,
            f"stride 2  →  {o2}×{o2}",
            ha="center", color=INK_SOFT, fontsize=14, fontweight="bold")

    line = calc_text or (
        f"derive EVERY cell: all {o1 * o1} at stride 1  ·  all {o2 * o2} at "
        f"stride 2  —  overlay · multiply · add")
    ax.text(7.0, 1.5, line, ha="center", va="center", color=INK,
            fontsize=14.5, fontweight="bold")
    if note:
        ax.text(7.0, 0.75, note, ha="center", va="center", color=MUTED,
                fontsize=12.5, style="italic")
    _save(fig, name)


# ---- channels: c_i slices per filter, c_o filters per layer ----------------
# The canonical worked example, numbers taken verbatim from Zhang et al.,
# "Dive into Deep Learning" §7.4 (CC BY-SA 4.0), fig. 7.4.1: a 2-channel 3×3
# input, a 2-channel 2×2 kernel, output [[56, 72], [104, 120]] with the
# top-left cell (19 + 37 = 56) the one worked on the slide.
D2L_X = [
    [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
]
D2L_K = [
    [[0, 1], [2, 3]],
    [[1, 2], [3, 4]],
]
# the matched you-do (Quiz 5 Q5): same shape, new numbers, partials 8 + 3 = 11
CHAN_QUIZ_X = [
    [[1, 2, 0], [0, 3, 1], [2, 1, 0]],
    [[2, 1, 1], [1, 0, 2], [0, 1, 3]],
]
CHAN_QUIZ_K = [
    [[1, 2], [0, 1]],
    [[0, 1], [2, 0]],
]
# the three filters of the multi-output example: K, K+1, K+2 (d2l §7.4)
D2L_K3 = [[[[v + d for v in row] for row in sl] for sl in D2L_K]
          for d in (0, 1, 2)]


def _corr_multi(X, K, stride=1):
    """Multi-channel cross-correlation: convolve each input channel with its
    own kernel slice and ADD the per-channel results. X is c×n×n, K is c×k×k."""
    c, n, k = len(X), len(X[0]), len(K[0])
    o = (n - k) // stride + 1
    return [[sum(X[ch][r * stride + i][c2 * stride + j] * K[ch][i][j]
                 for ch in range(c) for i in range(k) for j in range(k))
             for c2 in range(o)] for r in range(o)], o


def _partial(X_ch, K_ch, window=(0, 0)):
    """One channel's contribution to one output cell: its window dot its slice."""
    wr, wc = window
    k = len(K_ch)
    win = [[X_ch[wr + i][wc + j] for j in range(k)] for i in range(k)]
    return win, sum(win[i][j] * K_ch[i][j] for i in range(k) for j in range(k))


def channels_in(X=None, K=None, window=(0, 0), reveal=True,
                name="fig_channels_in.png", labels=None, credit=None):
    """Multiple INPUT channels, computed by hand: the SAME window is taken from
    every input channel, each is multiplied by that channel's own slice of ONE
    filter, and the per-channel partial sums are ADDED into a single output
    cell. reveal=False blanks the partials and the output for the you-do."""
    X = X or D2L_X
    K = K or D2L_K
    nc, n, k = len(X), len(X[0]), len(K[0])
    M, o = _corr_multi(X, K)
    labels = labels or [f"input channel {i + 1}" for i in range(nc)]
    hi_win = {(window[0] + i, window[1] + j) for i in range(k) for j in range(k)}

    cell = 0.58
    in_w, kn_w = n * cell, k * cell
    group_w = in_w + 0.56 + kn_w
    sep, out_gap = 0.62, 0.52
    ocell = 0.92
    width = 0.35 * 2 + nc * group_w + (nc - 1) * sep + out_gap + o * ocell
    height = 5.15

    fig, ax = plt.subplots(figsize=(width * 0.86, height * 0.86))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")

    gytop = 4.62                      # top of the input grids
    gmid = gytop - in_w / 2           # vertical centre of every grid

    x = 0.35
    for i in range(nc):
        _cellgrid(ax, X[i], x, gytop, cell, hi=hi_win, fontsize=15)
        ax.text(x + in_w / 2, gytop + 0.26, labels[i], ha="center",
                color=INK_SOFT, fontsize=13, fontweight="bold")
        ax.text(x + in_w + 0.28, gmid, "\u2217", ha="center", va="center",
                fontsize=22, color=INK_SOFT)
        kx = x + in_w + 0.56
        _cellgrid(ax, K[i], kx, gmid + kn_w / 2, cell, txt=TEAL, edge=TEAL,
                  fontsize=15)
        ax.text(kx + kn_w / 2, gmid + kn_w / 2 + 0.26,
                f"kernel slice {i + 1}", ha="center", color=TEAL,
                fontsize=13, fontweight="bold")
        x += group_w
        if i < nc - 1:
            ax.text(x + sep / 2, gmid, "+", ha="center", va="center",
                    fontsize=26, color=INK_SOFT)
            x += sep

    ax.text(x + out_gap / 2, gmid, "=", ha="center", va="center",
            fontsize=26, color=INK_SOFT)
    ox = x + out_gap
    shown = M if reveal else [["?" if (r, c) == (0, 0) else "\u00b7"
                               for c in range(o)] for r in range(o)]
    _cellgrid(ax, shown, ox, gmid + o * ocell / 2, ocell, hi={(0, 0)},
              fontsize=16 if reveal else 20)
    ax.text(ox + o * ocell / 2, gmid + o * ocell / 2 + 0.26,
            f"output  {o}\u00d7{o}", ha="center", color=INK_SOFT,
            fontsize=13, fontweight="bold")

    # the per-channel arithmetic, one line each, so the sums line up
    parts = [_partial(X[i], K[i], window)[1] for i in range(nc)]
    total = sum(parts)
    lx = 0.9
    for i in range(nc):
        win, part = _partial(X[i], K[i], window)
        ax.text(lx, 2.28 - i * 0.52,
                f"channel {i + 1}:   {_prodstr(win, K[i])}   =   {part:g}"
                if reveal else f"channel {i + 1}:   your partial sum  =  ?",
                ha="left", va="center", color=INK if reveal else ROI_INK,
                fontsize=13, family="monospace",
                fontweight="normal" if reveal else "bold")
    line = (" + ".join(f"{p:g}" for p in parts) + f" = {total:g}") if reveal \
        else " + ".join("?" for _ in parts) + " = ?"
    ax.text(width / 2, 1.02,
            f"{nc} channels in \u2192 {nc} partial sums \u2192 ADDED into ONE "
            f"number:   {line}", ha="center", va="center", color=INK,
            fontsize=14.5, fontweight="bold")
    if credit:
        ax.text(width / 2, 0.42, credit, ha="center", va="center",
                color=MUTED, fontsize=11.5, style="italic")
    _save(fig, name)


def channels_out(X=None, Ks=None, name="fig_channels_out.png", credit=None):
    """Multiple OUTPUT channels: the same c_i-channel input run through THREE
    filters (each one c_i slices deep) gives THREE feature maps — the output is
    a c_out-deep stack. Every map value is really computed (d2l §7.4). The two
    rules and the weight count live in the slide callout, so this figure stays
    all grids and prints big."""
    X = X or D2L_X
    Ks = Ks or D2L_K3
    nc, n, k = len(X), len(X[0]), len(Ks[0][0])
    co = len(Ks)
    _, o = _corr_multi(X, Ks[0])

    cell = 0.52
    in_w, kn_w = n * cell, k * cell
    ocell = 0.66
    out_w = o * ocell
    row_gap = 0.26
    in_h = nc * in_w + (nc - 1) * row_gap
    fstep = kn_w + 0.62
    mstep = out_w + 0.62

    x_in = 0.35
    x_star = x_in + in_w + 0.42
    x_f = x_star + 0.42
    x_eq = x_f + co * fstep - 0.62 + 0.42
    x_m = x_eq + 0.42
    width = x_m + co * mstep - 0.62 + 0.35
    height = in_h + 1.95

    fig, ax = plt.subplots(figsize=(width * 0.86, height * 0.86))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")

    band_top = height - 0.72
    cy = band_top - in_h / 2
    label_y = band_top + 0.3
    sub_y = band_top - in_h - 0.14

    # ---- the input: c_i channel grids, stacked (that stack IS c_in)
    in_tops = [band_top - i * (in_w + row_gap) for i in range(nc)]
    for i in range(nc):
        _cellgrid(ax, X[i], x_in, in_tops[i], cell, fontsize=14)
        ax.text(x_in + in_w + 0.1, in_tops[i] - in_w / 2, f"ch {i + 1}",
                ha="left", va="center", color=MUTED, fontsize=12)
    ax.text(x_in + in_w / 2, label_y, f"input  {nc}\u00d7{n}\u00d7{n}",
            ha="center", color=INK_SOFT, fontsize=15, fontweight="bold")

    ax.text(x_star, cy, "\u2217", ha="center", va="center", fontsize=26,
            color=INK_SOFT)

    # ---- c_o filters, each c_i slices deep, aligned with the input rows
    for j in range(co):
        x = x_f + j * fstep
        for i in range(nc):
            _cellgrid(ax, Ks[j][i], x, in_tops[i] - (in_w - kn_w) / 2, cell,
                      txt=TEAL, edge=TEAL, fontsize=14)
        ax.text(x + kn_w / 2, sub_y, f"filter {j + 1}", ha="center", va="top",
                color=TEAL, fontsize=13.5, fontweight="bold")
    ax.text(x_f + (co * fstep - 0.62) / 2, label_y,
            f"{co} filters  \u00b7  each {nc}\u00d7{k}\u00d7{k}", ha="center",
            color=TEAL, fontsize=15, fontweight="bold")

    ax.text(x_eq, cy, "=", ha="center", va="center", fontsize=26,
            color=INK_SOFT)

    # ---- c_o output maps, every value really computed
    for j in range(co):
        M, _ = _corr_multi(X, Ks[j])
        x = x_m + j * mstep
        _cellgrid(ax, M, x, cy + out_w / 2, ocell, face=AMBER_SOFT,
                  edge=AMBER, fontsize=15)
        ax.text(x + out_w / 2, sub_y, f"map {j + 1}", ha="center", va="top",
                color=ROI_INK, fontsize=13.5, fontweight="bold")
    ax.text(x_m + (co * mstep - 0.62) / 2, label_y,
            f"output  {co}\u00d7{o}\u00d7{o}  \u2014 a {co}-channel stack",
            ha="center", color=ROI_INK, fontsize=15, fontweight="bold")

    if credit:
        ax.text(width / 2, 0.22, credit, ha="center", va="center",
                color=MUTED, fontsize=12, style="italic")
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


def peak_detection_decide(name="fig_peak_detection_youdo.png"):
    """You-do = Quiz 5 Q4 (DECIDE, not recall): the detector boxes two
    candidates with DIFFERENT scores -- a tall sharp peak at 0.95 and a small
    shoulder at 0.30. The room decides which to integrate and what a 0.5 vs 0.2
    confidence threshold keeps. The scores are the GIVEN data for the decision,
    so both are shown; the thinking is the threshold call, not restating a map."""
    t = np.linspace(0, 10, 800)
    peaks = [(3.0, 1.0, 0.055), (6.7, 0.26, 0.16)]
    y = 0.04 + 0.015 * np.sin(t * 1.1)
    for mu, a, w in peaks:
        y = y + a * np.exp(-((t - mu) ** 2) / (2 * w))
    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    ax.plot(t, y, color=INK, lw=2.2, zorder=3)
    ax.fill_between(t, y, color=TEAL, alpha=0.08)
    # (x0, x1, box_top, score, box_color)
    boxes = [(2.35, 3.65, 1.12, "0.95", TEAL),
             (5.95, 7.45, 0.42, "0.30", RED)]
    for x0, x1, top, score, col in boxes:
        ax.add_patch(FancyBboxPatch((x0, 0.0), x1 - x0, top + 0.08,
                    boxstyle="round,pad=0.005,rounding_size=0.02", fill=False,
                    edgecolor=col, lw=3.0, zorder=5))
        ax.text((x0 + x1) / 2, top + 0.2, f"score = {score}", ha="center",
                color=col, fontsize=13, fontweight="bold", family="monospace")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.45)
    ax.set_xlabel("retention time  \u2192", fontsize=13)
    ax.set_yticks([])
    ax.set_xticks([])
    for spn in ("top", "right"):
        ax.spines[spn].set_visible(False)
    ax.text(0.5, -0.22,
            "each box = its RT window \u00b7 which do you integrate? which survives a 0.5 threshold? a 0.2 threshold?",
            transform=ax.transAxes, ha="center", color=INK_SOFT, fontsize=12,
            style="italic")
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
    # ---- Lecture 10 ----
    _lecture10_figures()
    # ---- Lecture 11 ----
    _lecture11_figures()
    # ---- Lecture 13 ----
    _lecture13_figures()
    # ---- Lecture 14 ----
    _lecture14_figures()


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
    conv_walk_two_strides(CONV_IMG_QUIZ, CONV_FILT_QUIZ,
                          name="fig_conv_youdo.png",
                          filt_note="corners + centre",
                          note="your turn \u2014 both whole maps, new image and filter (Quiz 5 Q1)")
    padding_grid(n=5)
    output_size(mode="ido", name="fig_output_size.png")
    output_size(mode="youdo", name="fig_output_size_youdo.png")
    conv_real_filters()
    # channels: c_i (one filter spans every input channel) then c_o (one
    # filter per output map), on d2l \u00a77.4's canonical numbers
    channels_in(name="fig_channels_in.png",
                credit="worked example after Zhang et al., Dive into Deep "
                       "Learning \u00a77.4 (CC BY-SA 4.0)")
    channels_out(name="fig_channels_out.png",
                 credit="filters K, K+1, K+2 and every map value after d2l "
                        "\u00a77.4 (CC BY-SA 4.0)")
    channels_in(CHAN_QUIZ_X, CHAN_QUIZ_K, reveal=False,
                name="fig_channels_youdo.png",
                credit="your turn \u2014 same drill, new numbers (Quiz 5 Q5)")
    maxpool_grid(MAXPOOL_IDO, reveal=True, name="fig_maxpool_ido.png")
    maxpool_grid(MAXPOOL_QUIZ, reveal=False, name="fig_maxpool_youdo.png",
                 note="your turn \u2014 keep each 2\u00d72 window's MAX (Quiz 5 Q2)")
    conv1d_spectrum()
    conv1d_code()
    detection_grid()
    peak_detection(reveal=True, name="fig_peak_detection.png")
    peak_detection_decide(name="fig_peak_detection_youdo.png")


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


# Lecture 7 runs on LANGUAGE examples (instructor, 2026-09-09): the clause the
# deck reads all hour, and the matched Quiz 7 Q1 clause. Both rows sum to 1.
ATTN_SENTENCE = ["cat", "drank", "milk", "because", "it"]
ATTN_SENTENCE_MAP = [
    [0.50, 0.20, 0.10, 0.05, 0.15],   # cat
    [0.35, 0.25, 0.25, 0.05, 0.10],   # drank
    [0.15, 0.30, 0.40, 0.05, 0.10],   # milk
    [0.15, 0.25, 0.15, 0.30, 0.15],   # because
    [0.55, 0.10, 0.15, 0.05, 0.15],   # it  → looks back at "cat"
]
ATTN_QUIZ_TOKENS = ["farmer", "fed", "chickens", "because", "they"]
ATTN_QUIZ_WEIGHTS = [0.10, 0.05, 0.60, 0.10, 0.15]


def seq_order_matters(name="fig_seq_order.png"):
    """Why order matters, taught on LANGUAGE (the official example) and then
    applied to a lab trace. Left: the same five words in two orders are two
    opposite statements. Right: a QC-drift trace across runs — the signal only
    exists because the runs are in time order."""
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    ax = axes[0]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.95, "a sentence is an ordered sequence", ha="center",
            fontsize=14, color=INK, fontweight="bold")

    def row(y, toks, color, cap):
        n = len(toks); w = 0.155; gap = 0.018
        total = n * w + (n - 1) * gap; x0 = 0.5 - total / 2
        for i, t in enumerate(toks):
            _chip(ax, x0 + i * (w + gap), y, w, 0.13, t, color, fs=14)
        ax.text(0.5, y - 0.12, cap, ha="center", va="top",
                fontsize=12, color=INK_SOFT)
    row(0.72, ["the", "dog", "bit", "the", "man"], TEAL,
        "one story")
    row(0.32, ["the", "man", "bit", "the", "dog"], RED,
        "same five words, shuffled — a different story")
    ax.annotate("", xy=(0.5, 0.43), xytext=(0.5, 0.51),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2))
    ax.text(0.5, 0.04,
            "identical words, identical counts — the ORDER carries the meaning",
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
    ax.set_title("in your lab: QC drift across runs", fontsize=14, color=INK,
                 fontweight="bold")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(0.5, -0.22, "shuffle the runs and the drift vanishes — the\ninformation is in the order",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=11.5, color=MUTED, style="italic")
    fig.subplots_adjust(bottom=0.2, wspace=0.25)
    _save(fig, name)


def tokenize_sentence(name="fig_tokenize.png"):
    """Tokenization on TEXT (the official example): a sentence is cut into word
    tokens with integer IDs from a fixed vocabulary, and a word the vocabulary
    has never seen is split into sub-word pieces. The lab applications (one
    token per residue, or a spectrum patch) are named, not worked."""
    fig, ax = plt.subplots(figsize=(11.6, 4.3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def strip(y, items, color, label, ids=None, fs=15):
        n = len(items); w = 0.135; gap = 0.02
        total = n * w + (n - 1) * gap; x0 = 0.58 - total / 2
        ax.text(0.02, y, label, ha="left", va="center", fontsize=13,
                color=INK_SOFT, fontweight="bold")
        for i, t in enumerate(items):
            x = x0 + i * (w + gap)
            _chip(ax, x, y, w, 0.12, t, color, fs=fs)
            if ids is not None:
                ax.text(x + w / 2, y - 0.10, f"id {ids[i]}", ha="center",
                        va="top", fontsize=10, color=MUTED, family="monospace")
    strip(0.86, ["the", "cat", "drank", "the", "milk"], TEAL,
          "sentence →", ids=[2, 41, 87, 2, 63])
    ax.text(0.58, 0.60,
            "every token gets an integer ID from a FIXED vocabulary  ·  "
            "the repeated “the” gets the same id 2",
            ha="center", fontsize=12, color=INK_SOFT)
    strip(0.40, ["chroma", "##to", "##graphy"], AMBER,
          "a word the\nvocabulary\nlacks →", ids=[915, 44, 208], fs=13)
    ax.text(0.58, 0.15,
            "unknown words are split into sub-word pieces, so nothing is ever "
            "out-of-vocabulary",
            ha="center", fontsize=12, color=INK_SOFT)
    ax.text(0.58, 0.04,
            "same recipe elsewhere: one token per amino acid, or one token per "
            "spectrum patch (Lecture 8)",
            ha="center", fontsize=10.5, color=MUTED, style="italic")
    _save(fig, name)


def embedding_space(name="fig_embedding.png"):
    """Embeddings on WORDS (the official example): each token becomes
    coordinates, and words used in similar ways end up near each other —
    learned from data, not hand-set."""
    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    groups = {
        "animals": (TEAL, [("cat", 1.15, 1.25), ("dog", 1.95, 1.55),
                           ("horse", 1.6, 0.62)]),
        "royalty": (RED, [("king", 4.6, 4.2), ("queen", 5.3, 3.8),
                          ("prince", 4.5, 3.4)]),
        "drinks": (AMBER, [("milk", 4.65, 1.25), ("water", 5.35, 1.55),
                           ("coffee", 4.35, 0.62)]),
        "verbs": (MUTED, [("drank", 1.3, 4.2), ("ate", 2.0, 3.8),
                          ("slept", 1.1, 3.4)]),
    }
    for gname, (color, pts) in groups.items():
        xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
        cx, cy = np.mean(xs), np.mean(ys)
        ax.add_patch(plt.matplotlib.patches.Ellipse(
            (cx, cy), max(np.ptp(xs), 0.8) + 1.7, max(np.ptp(ys), 0.8) + 1.3,
            facecolor=color, alpha=0.10, edgecolor=color, lw=1.4,
            zorder=1))
        ax.text(cx, cy + max(np.ptp(ys), 0.8) / 2 + 0.74, gname, ha="center",
                color=color, fontsize=12.5, fontweight="bold", zorder=5)
        for lab, x, y in pts:
            ax.text(x, y, lab, ha="center", va="center", color=WHITE,
                    fontsize=12.5, fontweight="bold", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.34", facecolor=color,
                              edgecolor=WHITE, linewidth=1.5))
    ax.set_xlim(0, 6.4); ax.set_ylim(0, 5.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("embedding dimension 1 →", fontsize=12)
    ax.set_ylabel("embedding dimension 2 →", fontsize=12)
    ax.set_title("each word token → coordinates; words used alike sit near each other",
                 fontsize=13.5, color=INK, fontweight="bold")
    ax.text(0.5, -0.12,
            "the coordinates are LEARNED from data — and the same thing happens to residue or analyte tokens",
            transform=ax.transAxes, ha="center", color=MUTED, fontsize=11.5,
            style="italic")
    fig.subplots_adjust(bottom=0.16)
    _save(fig, name)


def selfattn_all_to_all(name="fig_selfattn_alltoall.png"):
    """The big idea: every token looks at every other token and decides what is
    relevant. One query token's links to all tokens are highlighted; the rest
    of the all-pairs mesh is faint."""
    # Same 5-token clause as the heat-map I-do (fig_attn_heatmap_ido) so the
    # illustrative example runs continuously slide to slide.
    toks = ATTN_SENTENCE
    n = len(toks)
    xs = np.linspace(1.3, 10.7, n)
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
        _chip(ax, xs[i] - 0.62, y, 1.24, 0.5, t, face, txt=txt, fs=15,
              edge=AMBER if i == q else HAIRLINE, lw=1.5)
    ax.text(xs[q], 0.55, "query: “it” looks at every token", ha="center",
            color=ROI_INK, fontsize=12.5, fontweight="bold")
    ax.text(6, 3.9, "self-attention: every token looks at every other and decides what’s relevant",
            ha="center", color=INK, fontsize=13.5, fontweight="bold")
    _save(fig, name)


def qkv_lookup(reveal=True, name="fig_qkv.png"):
    """Query/Key/Value defined RIGOROUSLY (not 'what I'm looking for'): each is a
    learned linear projection of the token (q=x·W_Q, k=x·W_K, v=x·W_V), and each
    is defined by the OPERATION it takes part in — query·key is a relevance score,
    softmax makes weights, output is the weight-blended sum of values. Framed as
    the soft database key-value lookup of Vaswani et al. 2017 / d2l.ai."""
    roles = [
        ("Query", "q = x\u00b7W_Q",
         "the vector a token sends out\nto search — dotted with every\nkey to score how relevant\nthat token is to it", TEAL),
        ("Key", "k = x\u00b7W_K",
         "the vector a token exposes\nto be searched — query\u00b7key\nis its relevance score to\nthe querying token", AMBER),
        ("Value", "v = x\u00b7W_V",
         "the content a token offers —\nthe output is the sum of\nvalues, each weighted by\nits attention weight", RED),
    ]
    fig, ax = plt.subplots(figsize=(11.8, 5.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.9); ax.axis("off")
    ax.text(6, 5.62,
            "three learned projections of the same token  —  W_Q, W_K, W_V are trained",
            ha="center", va="center", color=INK_SOFT, fontsize=12.5,
            style="italic")
    cx = [2.2, 6.0, 9.8]
    for (role, proj, phrase, color), x in zip(roles, cx):
        ax.add_patch(FancyBboxPatch((x - 1.75, 2.15), 3.5, 3.0,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor=WHITE, edgecolor=color, lw=2.4, zorder=2))
        ax.add_patch(FancyBboxPatch((x - 1.75, 4.45), 3.5, 0.7,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor=color, edgecolor=color, lw=2.4, zorder=3))
        ax.text(x, 4.78, role, ha="center", va="center", color=WHITE,
                fontsize=18, fontweight="bold", zorder=4)
        ax.text(x, 4.02, proj, ha="center", va="center", color=color,
                fontsize=13.5, fontweight="bold", family="monospace", zorder=4)
        ax.text(x, 3.0, phrase, ha="center", va="center", color=INK_SOFT,
                fontsize=11.5, zorder=4, linespacing=1.35)
    ax.text(6, 1.42,
            "score = query \u00b7 key   →   softmax → weights   →   output = \u03a3 weight \u00d7 value",
            ha="center", va="center", color=INK, fontsize=14,
            fontweight="bold")
    ax.text(6, 0.62,
            "a soft database lookup: compare one query against every key, then read back a\n"
            "blend of the values — weighted by how well each key matched  (Vaswani et al. 2017)",
            ha="center", va="center", color=MUTED, fontsize=11.5,
            style="italic", linespacing=1.3)
    _save(fig, name)


def attention_heatmap(kind="sentence", name="fig_attn_heatmap.png"):
    """Reading an attention heat map, on language. kind='sentence' draws the
    full query×key matrix for the I-do clause and rings the "it" query row (its
    darkest cell is "cat" — coreference); kind='youdo' draws the single query
    row for the Quiz 7 Q1 clause, query "they" over farmer/fed/chickens/because
    /they with weights 0.10, 0.05, 0.60, 0.10, 0.15, so figure and key agree."""
    if kind == "sentence":
        toks = ATTN_SENTENCE
        M = ATTN_SENTENCE_MAP
        n = len(toks); cell = 1.0
        fig, ax = plt.subplots(figsize=(9.4, 5.4))
        ax.set_xlim(-2.6, n + 0.4); ax.set_ylim(-1.4, n + 0.8)
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
                        va="center", fontsize=12.5,
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
        ax.text(n / 2.0, -1.15,
                "rows = query token · columns = key token · darker = more · each row sums to 1",
                ha="center", fontsize=11.5, color=INK_SOFT)
        _save(fig, name)
    else:  # single query row — mirrors Quiz 7 Q1 exactly
        toks = ATTN_QUIZ_TOKENS
        wts = ATTN_QUIZ_WEIGHTS
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
            ax.text(i + 0.47, 1.25, t, ha="center", va="center", fontsize=15,
                    color=INK, fontweight="bold")
        ax.annotate("", xy=(n - 0.05, -0.15), xytext=(n - 0.94 + 0.05, -0.15),
                    arrowprops=dict(arrowstyle="-", color=ROI_INK, lw=2))
        ax.text(n - 0.5, -0.55, "query = “they”", ha="center", va="top",
                color=ROI_INK, fontsize=13, fontweight="bold")
        ax.text(n / 2.0, -1.25,
                "how much the last token “they” attends to each word  ·  darker = more  ·  weights sum to 1",
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


def attention_full_head(q, keys, values, name, reveal=True, tokens=None):
    """ONE attention head, end to end and hand-computable (rule 6): from a query
    and per-token keys, score s_i = q·k_i (dot product), soften into weights with
    the Lecture 4 softmax (an e^x table is given), then output = sum w_i*v_i.
    reveal=False blanks scores/weights/output for the you-do (= Quiz 7 Q2).
    Displayed weights are rounded to 2 dp and the output is computed from those
    rounded weights, so every number on the slide is internally exact."""
    import math
    n = len(values)
    tokens = tokens or [f"token {i+1}" for i in range(n)]
    scores = [sum(a * b for a, b in zip(q, k)) for k in keys]
    exps = [math.exp(s) for s in scores]
    Z = sum(exps)
    weights = [round(e / Z, 2) for e in exps]
    output = round(sum(w * v for w, v in zip(weights, values)), 2)
    xs = np.linspace(4.7, 10.2, n)
    fig, ax = plt.subplots(figsize=(11.8, 6.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.7); ax.axis("off")

    def rowlabel(y, txt):
        ax.text(0.25, y, txt, ha="left", va="center", fontsize=12.5,
                color=INK_SOFT, fontweight="bold")
    # query (top-left) and e^x table (top-right) — the givens
    ax.add_patch(FancyBboxPatch((0.25, 5.98), 3.5, 0.6,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                facecolor=TEAL_SOFT, edgecolor=TEAL, lw=1.4))
    ax.text(2.0, 6.28, f"query  q = ({', '.join(f'{v:g}' for v in q)})",
            ha="center", va="center", fontsize=14, color=TEAL,
            fontweight="bold", family="monospace")
    ax.add_patch(FancyBboxPatch((7.55, 5.98), 4.2, 0.6,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                facecolor="#F1F1EC", edgecolor=HAIRLINE, lw=1.3))
    ax.text(9.65, 6.28, "e\u2070=1.00   e\u00b9=2.72   e\u00b2=7.39",
            ha="center", va="center", fontsize=13, color=INK_SOFT,
            family="monospace")
    # token headers
    for i, t in enumerate(tokens):
        ax.text(xs[i], 5.5, t, ha="center", va="center", fontsize=12,
                color=MUTED)
    y_key, y_score, y_w, y_v = 4.95, 3.95, 2.75, 1.75
    # keys
    rowlabel(y_key, "key  k")
    for i, k in enumerate(keys):
        _chip(ax, xs[i] - 0.6, y_key, 1.2, 0.55,
              f"({', '.join(f'{v:g}' for v in k)})", "#F1F1EC", txt=INK_SOFT,
              fs=13, edge=HAIRLINE, lw=1.2)
    # scores = q . k
    rowlabel(y_score, "score  s = q\u00b7k")
    for i, s in enumerate(scores):
        _chip(ax, xs[i] - 0.5, y_score, 1.0, 0.55, f"{s:g}" if reveal else "?",
              INK_SOFT if reveal else "#F1F1EC", txt=("white" if reveal else MUTED),
              fs=15, edge=INK_SOFT, lw=1.2)
    ax.text(2.3, (y_score + y_w) / 2, "softmax  \u2193", ha="center",
            color=TEAL, fontsize=12.5, fontweight="bold", style="italic")
    # weights = softmax(scores)
    rowlabel(y_w, "weight  w")
    for i, w in enumerate(weights):
        _chip(ax, xs[i] - 0.5, y_w, 1.0, 0.55,
              f"{w:.2f}" if reveal else "?", TEAL if reveal else "#F1F1EC",
              txt=("white" if reveal else MUTED), fs=14,
              edge=TEAL, lw=1.3)
    # values
    rowlabel(y_v, "value  v")
    for i, v in enumerate(values):
        _chip(ax, xs[i] - 0.5, y_v, 1.0, 0.55, f"{v:g}", AMBER, txt="white",
              fs=15)
    # output box
    terms = "  +  ".join(f"{w:.2f}\u00b7{v:g}" for w, v in zip(weights, values))
    comp = (f"output = \u03a3 w\u1d62\u00b7v\u1d62 = {terms} = {output:g}"
            if reveal else "output = \u03a3 w\u1d62\u00b7v\u1d62 = ?")
    ax.add_patch(FancyBboxPatch((0.4, 0.5), 11.2, 0.85,
                boxstyle="round,pad=0.02,rounding_size=0.04",
                facecolor=TEAL_SOFT if reveal else "#F1F1EC",
                edgecolor=TEAL if reveal else HAIRLINE, lw=1.6))
    ax.text(6.0, 0.92, comp, ha="center", va="center",
            color=INK if reveal else INK_SOFT, fontsize=13.5,
            fontweight="bold", family="monospace")
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



def qkv_projection(name="fig_qkv_projection.png"):
    """Where q, k and v COME FROM (rule 6, worked): one token vector is pushed
    through three learned matrices W_Q, W_K, W_V to give its query, key and
    value. Same x on every row — three different projections, three different
    jobs. Course-drawn; the mechanism is Vaswani et al., NeurIPS 2017."""
    x = [[1, 2], [2, 0]]
    mats = [
        ("W_Q", [[1, 0], [1, 1]], "QUERIES", TEAL, TEAL_SOFT, "q"),
        ("W_K", [[0, 1], [1, 0]], "KEYS", AMBER, AMBER_SOFT, "k"),
        ("W_V", [[1, 1], [0, 1]], "VALUES", RED, "#F7E4E3", "v"),
    ]
    fig, ax = plt.subplots(figsize=(12.6, 5.9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    cell = 0.52
    col_x = [3.05, 6.05]
    mat_x = 9.6

    ax.text(0.35, 5.62, "EMBEDDING", ha="left", va="center", color=INK_SOFT,
            fontsize=12.5, fontweight="bold")
    for c, (cx, lab) in enumerate(zip(col_x, ["token 1", "token 2"])):
        ax.text(cx + cell, 6.18, lab, ha="center", color=INK, fontsize=13,
                fontweight="bold")
        _cellgrid(ax, [x[c]], cx, 5.88, cell, face=WHITE, edge=INK_SOFT,
                  fontsize=15)
        ax.text(cx - 0.18, 5.62, f"x{'\u2081' if c == 0 else '\u2082'}",
                ha="right", va="center", color=INK, fontsize=14,
                fontweight="bold")

    rows_y = [4.42, 2.82, 1.22]
    for (mname, M, rlab, colr, soft, sym), ry in zip(mats, rows_y):
        ax.text(0.35, ry, rlab, ha="left", va="center", color=colr,
                fontsize=12.5, fontweight="bold")
        out = [[sum(x[c][i] * M[i][j] for i in range(2)) for j in range(2)]
               for c in range(2)]
        for c, cx in enumerate(col_x):
            _cellgrid(ax, [out[c]], cx, ry + cell / 2, cell, face=soft,
                      edge=colr, txt=colr, fontsize=15)
            sub = "\u2081" if c == 0 else "\u2082"
            ax.text(cx - 0.18, ry, f"{sym}{sub}", ha="right", va="center",
                    color=colr, fontsize=14, fontweight="bold")
        _cellgrid(ax, M, mat_x, ry + cell, cell, face=WHITE, edge=colr,
                  txt=colr, fontsize=14)
        ax.text(mat_x + 2 * cell + 0.22, ry, mname, ha="left", va="center",
                color=colr, fontsize=15, fontweight="bold")
        ax.annotate("", xy=(mat_x - 0.22, ry), xytext=(col_x[1] + 2 * cell + 0.3, ry),
                    arrowprops=dict(arrowstyle="<|-", color=colr, lw=2.0))
        ax.text((mat_x + col_x[1] + 2 * cell) / 2 + 0.05, ry + 0.3,
                "learned \u00b7 shared by every token", ha="center", color=MUTED,
                fontsize=10.5, style="italic")

    ax.text(7.0, 0.28,
            "one worked cell:  q\u2081 = x\u2081\u00b7W_Q = (1\u00b71 + 2\u00b71,  1\u00b70 + 2\u00b71) = (3, 2)"
            "      \u2014 same x, three matrices, three roles",
            ha="center", va="center", color=INK, fontsize=12.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", fc=TEAL_SOFT, ec=TEAL, lw=1.6))
    _save(fig, name)


# A deterministic synthetic spectrum for the denoising panels: fixed peak
# positions/heights so every rebuild is byte-identical, and a seeded RNG for
# the corruption. No external data, so a clean checkout can regenerate them.
_DENOISE_PEAKS = [(118, 0.55, 3.0), (205, 0.95, 3.4), (262, 0.40, 2.8),
                  (348, 0.72, 3.2), (430, 0.30, 2.6)]


def _clean_spectrum(nbins=520, shift=0, scale=None):
    mz = np.arange(nbins, dtype=float)
    y = np.zeros(nbins)
    scale = scale or [1.0] * len(_DENOISE_PEAKS)
    for (c, h, w), k in zip(_DENOISE_PEAKS, scale):
        y += k * h * np.exp(-0.5 * ((mz - (c + shift)) / w) ** 2)
    return y


def _noisy_spectrum(y, rng, sigma=0.13):
    return np.clip(y + rng.normal(0, sigma, y.shape), 0, None)


def _reconstructed(y, rng, sigma=0.022):
    """What a trained denoiser returns: the peaks back, slightly smoothed and
    a little short — NOT a perfect copy of the target."""
    return np.clip(0.94 * y + rng.normal(0, sigma, y.shape), 0, None)


def denoising_autoencoder(name="fig_denoising_ae.png"):
    """The denoising autoencoder as a pipeline (rule 2, MS-anchored): corrupt a
    good spectrum on purpose, make the network rebuild the CLEAN one. The label
    is the original, so no human ever annotates anything. Course-drawn."""
    rng = np.random.default_rng(3)
    clean = _clean_spectrum()
    noisy = _noisy_spectrum(clean, rng)
    out = _reconstructed(clean, np.random.default_rng(11))

    fig = plt.figure(figsize=(13.0, 5.0))
    ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, 1); ov.set_ylim(0, 1)
    ov.axis("off")

    def spectrum(rect, y, colour, title, sub):
        ax = fig.add_axes(rect)
        ax.plot(y, color=colour, lw=1.3)
        ax.fill_between(np.arange(len(y)), y, color=colour, alpha=0.16)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylim(-0.05, 1.35)
        for sp in ax.spines.values():
            sp.set_edgecolor(HAIRLINE); sp.set_linewidth(1.2)
        ov.text(rect[0] + rect[2] / 2, rect[1] + rect[3] + 0.055, title,
                ha="center", va="bottom", color=colour, fontsize=12.5,
                fontweight="bold")
        ov.text(rect[0] + rect[2] / 2, rect[1] - 0.045, sub, ha="center",
                va="top", color=MUTED, fontsize=10.5, style="italic")

    spectrum([0.035, 0.42, 0.20, 0.34], noisy, AMBER,
             "NOISY INPUT", "a good run, + noise WE added")
    spectrum([0.775, 0.42, 0.20, 0.34], out, TEAL,
             "DENOISED OUTPUT", "scored against the ORIGINAL clean spectrum")

    # the hourglass between them: wide -> narrow -> code -> narrow -> wide
    mid = 0.59
    layers = [(0.300, 0.60), (0.360, 0.38), (0.420, 0.18),
              (0.480, 0.38), (0.540, 0.60)]
    for i, (lx, lh) in enumerate(layers):
        is_code = i == 2
        colr = RED if is_code else INK_SOFT
        ov.add_patch(FancyBboxPatch(
            (lx, mid - lh / 2), 0.034, lh,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            facecolor="#F7E4E3" if is_code else PAPER,
            edgecolor=colr, lw=2.0))
    ov.text(0.365, 0.20, "ENCODER", ha="center", color=INK_SOFT,
            fontsize=12, fontweight="bold")
    ov.text(0.437, 0.20, "CODE", ha="center", color=RED, fontsize=12,
            fontweight="bold")
    ov.text(0.522, 0.20, "DECODER", ha="center", color=INK_SOFT,
            fontsize=12, fontweight="bold")
    for xa, xb in [(0.245, 0.292), (0.582, 0.767)]:
        ov.annotate("", xy=(xb, mid), xytext=(xa, mid),
                    arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, lw=2.2))

    ov.text(0.5, 0.045,
            "the label is the spectrum we started from \u2014 nobody annotates "
            "anything, so training data is free",
            ha="center", va="center", color=INK, fontsize=12.5,
            fontweight="bold")
    _save(fig, name)


def denoising_results(name="fig_denoising_results.png"):
    """The payoff, three rows the room reads top to bottom: what the network
    SEES, what it PRODUCES, and the clean original it was scored against.
    Course-drawn schematic on synthetic spectra (not measured output)."""
    rng = np.random.default_rng(7)
    ncol = 4
    shifts = [0, 9, -7, 15]
    scales = [[1.0, 1.0, 1.0, 1.0, 1.0],
              [0.6, 0.8, 1.3, 0.7, 1.4],
              [1.3, 0.5, 0.9, 1.2, 0.6],
              [0.8, 1.2, 0.6, 1.0, 1.1]]
    cleans = [_clean_spectrum(shift=sh, scale=sc)
              for sh, sc in zip(shifts, scales)]
    noisys = [_noisy_spectrum(c, rng) for c in cleans]
    outs = [_reconstructed(c, rng) for c in cleans]

    rows = [("NOISY INPUT", noisys, AMBER, "what the network sees"),
            ("DENOISED OUTPUT", outs, TEAL, "what it produces"),
            ("CLEAN TARGET", cleans, INK_SOFT, "the original it is scored against")]

    fig = plt.figure(figsize=(13.0, 5.6))
    ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, 1); ov.set_ylim(0, 1)
    ov.axis("off")
    left, w, gap = 0.175, 0.185, 0.022
    rh = 0.225
    tops = [0.705, 0.425, 0.145]
    for (label, series, colour, sub), ry in zip(rows, tops):
        ov.text(left - 0.025, ry + rh / 2, label, ha="right", va="center",
                color=colour, fontsize=12.5, fontweight="bold")
        ov.text(left - 0.025, ry + rh / 2 - 0.055, sub, ha="right", va="center",
                color=MUTED, fontsize=10, style="italic")
        for c in range(ncol):
            ax = fig.add_axes([left + c * (w + gap), ry, w, rh])
            ax.plot(series[c], color=colour, lw=1.2)
            ax.fill_between(np.arange(len(series[c])), series[c],
                            color=colour, alpha=0.16)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_ylim(-0.05, 1.35)
            for sp in ax.spines.values():
                sp.set_edgecolor(HAIRLINE); sp.set_linewidth(1.2)
    ov.text(0.5, 0.965,
            "four different spectra, one trained autoencoder",
            ha="center", va="center", color=INK, fontsize=13.5,
            fontweight="bold")
    ov.text(0.5, 0.045,
            "the peaks survive, the noise does not \u2014 the bottleneck can only "
            "carry what the training spectra had in common",
            ha="center", va="center", color=INK, fontsize=12.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45", fc=TEAL_SOFT, ec=TEAL, lw=1.6))
    _save(fig, name)


def _lecture7_figures():
    seq_order_matters()
    tokenize_sentence()
    embedding_space()
    selfattn_all_to_all()
    qkv_lookup(reveal=True, name="fig_qkv_ido.png")
    attention_heatmap("sentence", "fig_attn_heatmap_ido.png")
    attention_heatmap("youdo", "fig_attn_heatmap_youdo.png")
    attention_formula()
    qkv_projection()
    # I-do = ONE full head: q=(1,1); keys (1,1),(1,0),(2,-2) -> scores (2,1,0)
    #  softmax(2,1,0)=(0.67,0.24,0.09); values (10,4,1)
    #  output = 0.67·10 + 0.24·4 + 0.09·1 = 6.7 + 0.96 + 0.09 = 7.75
    attention_full_head((1, 1), [(1, 1), (1, 0), (2, -2)], [10, 4, 1],
                        "fig_attn_head_ido.png", reveal=True,
                        tokens=["token 1", "token 2", "token 3"])
    # you-do = Quiz 7 Q2: q=(1,1); keys (-1,1),(1,1),(1,0) -> scores (0,2,1)
    #  softmax(0,2,1)=(0.09,0.67,0.24); values (4,9,1)
    #  output = 0.09·4 + 0.67·9 + 0.24·1 = 0.36 + 6.03 + 0.24 = 6.63
    attention_full_head((1, 1), [(-1, 1), (1, 1), (1, 0)], [4, 9, 1],
                        "fig_attn_head_youdo.png", reveal=False,
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
    ax.text(3, 0.18, "MS: Casanovo — a spectrum→peptide model (de novo); deep-dive soon",
            ha="center", va="top", fontsize=9, color=RED, fontweight="bold")
    fig.subplots_adjust(wspace=0.12)
    _save(fig, name)


def positional_encoding_heatmap(name="fig_posenc_heatmap.png"):
    """Where the per-position signals come from (rule 6/9): sine/cosine waves at
    geometrically-spaced frequencies. Left: three example dimensions are 'clocks'
    at different speeds. Right: the full absolute positional-encoding matrix
    (position x dimension) as a heat map — each ROW is one position's unique
    fingerprint, and neighbouring rows look alike, so nearby positions get
    similar codes. This is the classic sinusoidal PE picture (Vaswani 2017)."""
    from matplotlib.colors import LinearSegmentedColormap
    N, D = 16, 16
    posv = np.arange(N)[:, None]
    iv = np.arange(D)[None, :]
    div = np.power(10000.0, (2 * (iv // 2)) / D)
    ang = posv / div
    PE = np.where(iv % 2 == 0, np.sin(ang), np.cos(ang))
    cmap = LinearSegmentedColormap.from_list("teal_amber", [TEAL, WHITE, AMBER])
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9),
                             gridspec_kw={"width_ratios": [1.02, 1.0]})
    # ---- left: three clocks at different speeds ----
    ax = axes[0]
    p = np.linspace(0, N - 1, 400)
    for dim, lab, col in [(0, "dim 0 · fast", TEAL),
                          (4, "dim 4 · medium", ROI_INK),
                          (8, "dim 8 · slow", RED)]:
        d = np.power(10000.0, (2 * (dim // 2)) / D)
        ax.plot(p, np.sin(p / d), color=col, lw=2.4)
        ax.text(N - 1 + 0.2, np.sin((N - 1) / d), lab, color=col,
                fontsize=10.5, fontweight="bold", va="center")
    ax.set_xlim(0, N - 1 + 3.0); ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel("position along the sequence", fontsize=11, color=INK_SOFT)
    ax.set_title("each dimension is a clock at a different speed",
                 fontsize=12.5, color=INK, fontweight="bold", pad=8)
    ax.axhline(0, color=HAIRLINE, lw=1.0, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(HAIRLINE); ax.spines["bottom"].set_color(HAIRLINE)
    ax.tick_params(colors=INK_SOFT, labelsize=9); ax.set_yticks([-1, 0, 1])
    # ---- right: the PE matrix as a heat map ----
    ax = axes[1]
    im = ax.imshow(PE, aspect="auto", cmap=cmap, vmin=-1, vmax=1)
    ax.set_xlabel("dimension   (fast clocks → slow clocks)", fontsize=11,
                  color=INK_SOFT)
    ax.set_ylabel("position", fontsize=11, color=INK_SOFT)
    ax.set_title("positional-encoding matrix · each row = one position",
                 fontsize=12.5, color=INK, fontweight="bold", pad=8)
    ax.set_xticks([0, 4, 8, 12]); ax.set_yticks([0, 4, 8, 12])
    ax.tick_params(colors=INK_SOFT, labelsize=9)
    # subtle highlight: two neighbouring positions look nearly identical
    ax.add_patch(plt.Rectangle((-0.5, 1.5), D, 2.0, fill=False,
                 edgecolor=INK, lw=2.2, zorder=5))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_ticks([-1, 0, 1]); cbar.ax.tick_params(labelsize=8, colors=INK_SOFT)
    fig.subplots_adjust(wspace=0.32)
    _save(fig, name)


def rope_relative(name="fig_rope.png"):
    """Absolute vs. relative position — the RoPE intuition (rule 9, no matrices).
    Absolute sinusoidal PE tags each seat; RoPE instead ROTATES each token's
    query/key by an angle proportional to its position, so the dot product ends
    up depending only on the GAP (m - n) between the two tokens — which is why it
    generalizes to sequences longer than it trained on (LLaMA, GPT-NeoX)."""
    fig, ax = plt.subplots(figsize=(11.8, 5.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.2); ax.axis("off")
    ax.set_aspect("equal")
    ax.text(6, 6.0, "Absolute tags the SEAT · RoPE encodes the GAP",
            ha="center", fontsize=15, color=INK, fontweight="bold")
    # left reminder: absolute = add a per-seat signal (last slide)
    ax.text(2.0, 5.15, "Absolute (sinusoidal)", ha="center", fontsize=12,
            color=TEAL, fontweight="bold")
    ax.text(2.0, 3.7,
            "add a fixed\nposition-7 signal\nto the token\n→ 'you are in\nseat 7'",
            ha="center", va="center", fontsize=11, color=INK_SOFT,
            linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.6", fc=TEAL_SOFT, ec=TEAL, lw=1.8))
    ax.plot([3.7, 3.7], [1.2, 5.0], color=HAIRLINE, lw=1.4)
    # right: RoPE = rotate q and k by position angle
    ax.text(8.0, 5.15, "Relative (RoPE) — rotate by the position angle",
            ha="center", fontsize=12, color=RED, fontweight="bold")
    th = np.deg2rad(30)

    def clock(cx, cy, r, ang_deg, col, lab):
        ax.add_patch(plt.Circle((cx, cy), r, fill=False, edgecolor=HAIRLINE,
                     lw=1.6))
        a = np.deg2rad(ang_deg)
        ax.annotate("", xy=(cx + r * np.cos(a), cy + r * np.sin(a)),
                    xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.6))
        ax.text(cx, cy - r - 0.32, lab, ha="center", va="top", fontsize=10.5,
                color=col, fontweight="bold")
    clock(6.4, 3.4, 0.95, 150, TEAL, "query · position m=5\nrotate by 5θ = 150°")
    clock(9.6, 3.4, 0.95, 60, AMBER, "key · position n=2\nrotate by 2θ = 60°")
    ax.text(8.0, 1.35, "angle between them = (5−2)θ = 3θ  —  only the GAP survives",
            ha="center", fontsize=11, color=INK, fontweight="bold")
    ax.text(6, 0.42,
            "rotate q and k by position×θ; the dot product then depends on the relative distance (m−n), "
            "not the seat — so it extrapolates to longer sequences (LLaMA, GPT-NeoX)",
            ha="center", va="center", fontsize=10.5, color=MUTED, style="italic")
    _save(fig, name)


def cross_attention(name="fig_cross_attention.png"):
    """How an encoder-decoder actually connects (rule 9): the encoder reads the
    whole input and exposes a set of KEY/VALUE vectors (its 'memory'); the
    decoder, writing one token at a time, sends a QUERY from the current step
    into that encoder K/V via CROSS-attention, on top of masked self-attention
    over what it has written so far. Anchored on Casanovo (spectrum -> peptide)."""
    fig, ax = plt.subplots(figsize=(12.0, 6.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    # ---- encoder (left) ----
    for i, c in enumerate([2.6, 6.2, 4.4]):
        _chip(ax, 0.9 + i * 0.9, 0.9, 0.7, 0.5, "", TEAL_SOFT, edge=TEAL, lw=1.2)
    ax.text(1.95, 0.35, "spectrum peaks", ha="center", fontsize=9.5,
            color=MUTED)
    ax.text(2.4, 4.55, "Encoder\nreads all peaks\n(self-attention)", ha="center",
            va="center", fontsize=11.5, color=INK, fontweight="bold",
            linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.6", fc=TEAL_SOFT, ec=TEAL, lw=2.2))
    ax.annotate("", xy=(2.4, 3.55), xytext=(2.4, 1.25),
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2.0))
    ax.text(2.4, 6.05, "Encoder  K, V\n(the read spectrum — its memory)",
            ha="center", va="center", fontsize=11, color=ROI_INK,
            fontweight="bold", linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.5", fc=AMBER_SOFT, ec=AMBER, lw=2.2))
    ax.annotate("", xy=(2.4, 5.5), xytext=(2.4, 5.05),
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=2.0))
    # ---- decoder (right) ----
    ax.text(8.7, 6.4, "Decoder · writing residue 3", ha="center",
            fontsize=12, color=RED, fontweight="bold")
    ax.text(8.7, 5.0, "② cross-attention\nquery from decoder · K,V from ENCODER",
            ha="center", va="center", fontsize=11, color=INK, fontweight="bold",
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.55", fc=AMBER_SOFT, ec=AMBER, lw=2.4))
    ax.text(8.7, 3.2, "① masked self-attention\nlook only at residues written so far",
            ha="center", va="center", fontsize=11, color=INK_SOFT,
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.55", fc="#F7E4E3", ec=RED, lw=2.0))
    for i, t in enumerate(["A", "C"]):
        _chip(ax, 7.9 + i * 0.9, 1.35, 0.7, 0.55, t, "#F7E4E3", txt=INK,
              fs=13, edge=RED, lw=1.3)
    _chip(ax, 9.7, 1.35, 0.7, 0.55, "D", WHITE, txt=INK, fs=13, edge=AMBER, lw=1.6)
    ax.text(9.0, 0.7, "→ next residue: D", ha="center", fontsize=10.5,
            color=ROI_INK, fontweight="bold")
    ax.annotate("", xy=(8.7, 4.25), xytext=(8.7, 3.9),
                arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, lw=1.8))
    ax.annotate("", xy=(8.7, 2.55), xytext=(8.7, 1.75),
                arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, lw=1.8))
    # ---- the cross-attention bridge ----
    ax.annotate("", xy=(6.65, 5.0), xytext=(3.75, 6.05),
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=3.0,
                                connectionstyle="arc3,rad=-0.15"))
    ax.text(5.2, 6.15, "cross-attention", ha="center", fontsize=11,
            color=ROI_INK, fontweight="bold", rotation=-12)
    ax.text(6, 0.12,
            "every residue the decoder writes sends a query into the encoder's K,V — so each output token "
            "can look back at the whole spectrum; repeat until the peptide ends",
            ha="center", va="center", fontsize=10, color=MUTED, style="italic")
    _save(fig, name)


def arch_families(name="fig_arch_families.png"):
    """The three transformer families side by side (rule 9), each built from the
    same block: encoder-only (reads -> label/embedding), encoder-decoder (reads
    -> writes a different sequence via cross-attention), decoder-only (continues
    text, next-token). Real model names + a 'good for' line so the room can pick
    a family for a task (decide beat)."""
    fig, ax = plt.subplots(figsize=(12.2, 5.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.4); ax.axis("off")
    cols = [
        (2.05, TEAL, TEAL_SOFT, "Encoder-only", "reads ↔ the whole input",
         "BERT · DistilBERT · ESM-2",
         "classify / embed\ne.g. spectrum → resistant / susceptible"),
        (6.1, AMBER, AMBER_SOFT, "Encoder–decoder", "reads → writes (cross-attn)",
         "T5 · Flan-T5 · Casanovo",
         "seq → different seq\ne.g. spectrum → peptide, translate, summarize"),
        (10.15, RED, "#F7E4E3", "Decoder-only", "writes →→ (next-token)",
         "GPT · LLaMA",
         "generate / chat\neverything as next-token text"),
    ]
    for cx, col, soft, name_, wiring, examples, goodfor in cols:
        ax.text(cx, 5.75, name_, ha="center", va="center", fontsize=13.5,
                color=WHITE, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", fc=col, ec=col))
        ax.text(cx, 4.75, wiring, ha="center", va="center", fontsize=10.5,
                color=INK, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.45", fc=soft, ec=col, lw=1.8))
        ax.text(cx, 3.5, "Examples", ha="center", fontsize=9.5, color=MUTED,
                fontweight="bold")
        ax.text(cx, 3.05, examples, ha="center", va="center", fontsize=10.5,
                color=INK, fontweight="bold")
        ax.text(cx, 2.2, "Good for", ha="center", fontsize=9.5, color=MUTED,
                fontweight="bold")
        ax.text(cx, 1.45, goodfor, ha="center", va="center", fontsize=10,
                color=INK_SOFT, linespacing=1.4)
    ax.text(6, 0.35,
            "reads only → encoder-only    ·    reads then writes a NEW sequence → encoder–decoder    ·    "
            "continues text → decoder-only",
            ha="center", va="center", fontsize=10.5, color=INK, fontweight="bold")
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


# ============================================================================
#  Lecture 10 · Your Data, Your Metrics: Representation, Evaluation, Trust
#  Hand-authored decision charts + numeric mechanics (confusion matrix, rule 6)
#  + bespoke MS-anchored schematics (rules 7 exception / 9). ROC & PR good-vs-
#  bad comparisons are teaching curve galleries in the same category as this
#  file's loss_curves / fit_trio / trainval_curves (matplotlib, not photos).
# ============================================================================

def _confusion_cells(ax, tp, fn, fp, tn, x0, ytop, cell, blank):
    """Draw a 2x2 confusion matrix (rows = model call, cols = truth)."""
    def val(v):
        return "?" if blank else f"{v:g}"
    cells = [  # (row, col, count, face, edge, tag)
        (0, 0, tp, TEAL_SOFT, TEAL, "TP"),
        (0, 1, fp, AMBER_SOFT, AMBER, "FP"),
        (1, 0, fn, "#F7E4E3", RED, "FN"),
        (1, 1, tn, TEAL_SOFT, TEAL, "TN"),
    ]
    for r, c, count, face, edge, tag in cells:
        cx = x0 + c * cell
        cy = ytop - r * cell
        ax.add_patch(FancyBboxPatch((cx, cy - cell * 0.92), cell * 0.92,
                    cell * 0.92, boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor=face, edgecolor=edge, lw=2.4, zorder=2))
        ax.text(cx + cell * 0.46, cy - cell * 0.36, tag, ha="center",
                va="center", color=edge, fontsize=13, fontweight="bold",
                zorder=3)
        ax.text(cx + cell * 0.46, cy - cell * 0.62, val(count), ha="center",
                va="center", color=INK, fontsize=20, fontweight="bold",
                zorder=3)
    # truth (column) headers
    ax.text(x0 + cell * 0.46, ytop + 0.22, "truth: +", ha="center",
            color=INK_SOFT, fontsize=12, fontweight="bold")
    ax.text(x0 + cell * 1.46, ytop + 0.22, "truth: −", ha="center",
            color=INK_SOFT, fontsize=12, fontweight="bold")
    # model-call (row) headers
    ax.text(x0 - 0.15, ytop - cell * 0.46, "call: +", ha="right",
            va="center", color=INK_SOFT, fontsize=12, fontweight="bold")
    ax.text(x0 - 0.15, ytop - cell * 1.46, "call: −", ha="right",
            va="center", color=INK_SOFT, fontsize=12, fontweight="bold")


def confusion_matrix(tp, fn, fp, tn, mode="ido", name="fig_confusion_ido.png",
                     positive="resistant", negative="susceptible"):
    """A worked confusion matrix with the THREE rates the room needs under
    imbalance (rule 6): recall/sensitivity = TP/(TP+FN), specificity =
    TN/(TN+FP), and precision = TP/(TP+FP). mode='ido' fills every number and
    shows the accuracy trap; mode='youdo' shows the counts but leaves the three
    rates as '?' for the worksheet Part-A question (rules 3/8)."""
    total = tp + fn + fp + tn
    pos = tp + fn
    neg = tn + fp
    flagged = tp + fp
    sens = tp / pos
    spec = tn / neg
    prec = tp / flagged
    acc = (tp + tn) / total
    always_neg = neg / total
    blank = mode == "youdo"
    fig, ax = plt.subplots(figsize=(11.8, 5.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    _confusion_cells(ax, tp, fn, fp, tn, 1.7, 5.05, 1.4, blank=False)
    ax.text(2.68, 5.62, f"+ = {positive}   ·   − = {negative}", ha="center",
            color=MUTED, fontsize=11, style="italic")
    # right column: three formulas + worked (or ?) values
    fx = 6.15
    rows = [
        ("recall (sensitivity) = TP / (TP + FN)",
         f"= {tp:g} / {pos:g} = {sens:.2f}  ({sens*100:.0f}%)", TEAL),
        ("specificity = TN / (TN + FP)",
         f"= {tn:g} / {neg:g} = {spec:.2f}  ({spec*100:.0f}%)", TEAL),
        ("precision = TP / (TP + FP)",
         f"= {tp:g} / {flagged:g} = {prec:.2f}  ({prec*100:.0f}%)", ROI_INK),
    ]
    y = 5.35
    for label, worked, col in rows:
        ax.text(fx, y, label, ha="left", color=INK, fontsize=13.5,
                fontweight="bold")
        ax.text(fx, y - 0.5, ("= ?" if blank else worked), ha="left",
                color=col, fontsize=13.5, fontweight="bold")
        y -= 1.28
    # the accuracy trap box -- auto-fitting bbox so long lines never overflow.
    if blank:
        trap = ("accuracy = ?  — but a model that ALWAYS\n"
                f"says “{negative}” scores {neg:g}/{total:g} = "
                f"{always_neg*100:.0f}%,\ncatching 0 of {pos:g}. "
                "Which rate do you trust?")
    else:
        trap = (f"accuracy = {tp+tn:g}/{total:g} = {acc*100:.0f}%  — yet ALWAYS\n"
                f"“{negative}” scores {neg:g}/{total:g} = "
                f"{always_neg*100:.0f}% and catches 0 of {pos:g}.\n"
                "Precision 0.33: 2 of 3 flags are FALSE.")
    ax.text(2.55, 1.15, trap, ha="center", va="center", color=INK,
            fontsize=10.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F7E4E3",
                      edgecolor=RED, linewidth=2.0))
    _save(fig, name)


# ---- AUROC / AUPRC, introduced from a real threshold sweep -----------------
# DRIAMS S. aureus + oxacillin, the same 41 R / 697 S split the whole hour uses.
# Sweeping the decision threshold gives one confusion matrix — and so one ROC
# point and one PR point — per threshold. The middle row IS the deck's I-do
# matrix (TP 30, FP 60), so every slide of Part A describes one classifier.
DRIAMS_POS, DRIAMS_NEG = 41, 697
#            threshold,  TP,  FP,  label
SWEEP = [(0.9, 18, 10, "strict"),
         (0.5, 30, 60, "default"),
         (0.2, 38, 210, "loose")]


def _sweep_points():
    """(threshold, TP, FP, recall, fpr, precision) rows plus the two endpoints
    a curve always has: flag nothing, and flag everything."""
    rows = []
    for t, tp, fp, tag in SWEEP:
        rows.append(dict(t=t, tp=tp, fp=fp, tag=tag,
                         rec=tp / DRIAMS_POS, fpr=fp / DRIAMS_NEG,
                         prec=tp / (tp + fp)))
    return rows


def _auc(xs, ys):
    return float(np.trapezoid(ys, xs)) if hasattr(np, "trapezoid") \
        else float(np.trapz(ys, xs))


def _roc_xy():
    rows = _sweep_points()
    xs = [0.0] + [r["fpr"] for r in rows] + [1.0]
    ys = [0.0] + [r["rec"] for r in rows] + [1.0]
    return xs, ys


def _pr_xy():
    rows = _sweep_points()
    xs = [0.0] + [r["rec"] for r in rows] + [1.0]
    ys = [rows[0]["prec"]] + [r["prec"] for r in rows] \
        + [DRIAMS_POS / (DRIAMS_POS + DRIAMS_NEG)]
    return xs, ys


def score_threshold_sweep(name="fig_threshold_sweep.png"):
    """One model, many operating points: the network outputs a SCORE, you pick
    a threshold, and every threshold gives a different confusion matrix. Score
    histograms (built so the counts match the table exactly) plus the table the
    room fills in for the strict threshold."""
    rng = np.random.default_rng(4)

    def scores(bands):
        out = []
        for lo, hi, k in bands:
            out.append(lo + (hi - lo) * rng.beta(2.0, 2.0, k))
        return np.concatenate(out)

    # bands chosen so the counts above each threshold are exactly SWEEP's
    pos = scores([(0.0, 0.2, 3), (0.2, 0.5, 8), (0.5, 0.9, 12), (0.9, 1.0, 18)])
    neg = scores([(0.0, 0.2, 487), (0.2, 0.5, 150), (0.5, 0.9, 50),
                  (0.9, 1.0, 10)])
    rows = _sweep_points()

    fig = plt.figure(figsize=(12.6, 4.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1.0],
                          height_ratios=[1.0, 1.15], hspace=0.12, wspace=0.22,
                          left=0.06, right=0.985, top=0.86, bottom=0.13)
    axp = fig.add_subplot(gs[0, 0])
    axn = fig.add_subplot(gs[1, 0], sharex=axp)
    axt = fig.add_subplot(gs[:, 1])
    axt.set_xlim(0, 1); axt.set_ylim(0, 1); axt.axis("off")

    bins = np.linspace(0, 1, 21)
    axp.hist(pos, bins=bins, color=RED, alpha=0.85)
    axn.hist(neg, bins=bins, color=TEAL, alpha=0.85)
    axp.set_ylabel(f"resistant\n({DRIAMS_POS})", fontsize=11.5, color=RED,
                   fontweight="bold")
    axn.set_ylabel(f"susceptible\n({DRIAMS_NEG})", fontsize=11.5, color=TEAL,
                   fontweight="bold")
    axn.set_xlabel("the model's score for “resistant”  →", fontsize=12)
    axp.tick_params(labelbottom=False)
    for ax in (axp, axn):
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        for r in rows:
            ax.axvline(r["t"], color=AMBER, lw=2.0, ls="--", zorder=5)
    for r in rows:
        axp.text(r["t"], axp.get_ylim()[1] * 1.06, f"t = {r['t']:g}",
                 ha="center", va="bottom", color=ROI_INK, fontsize=11.5,
                 fontweight="bold")
    axp.text(0.5, 1.30, "flag everything to the RIGHT of the threshold",
             transform=axp.transAxes, ha="center", va="bottom", color=INK,
             fontsize=13, fontweight="bold")

    # ---- the table: one confusion matrix, and one curve point, per threshold
    cols = ["threshold", "TP", "FP", "recall\n(TPR)", "FPR", "precision"]
    xcol = [0.10, 0.28, 0.40, 0.56, 0.72, 0.89]
    ytop = 0.86
    axt.text(0.5, 0.99, "each threshold → its own confusion matrix",
             ha="center", va="top", color=INK, fontsize=13, fontweight="bold")
    for x, c in zip(xcol, cols):
        axt.text(x, ytop, c, ha="center", va="top", color=INK_SOFT,
                 fontsize=11.5, fontweight="bold", linespacing=1.2)
    axt.plot([0.03, 0.97], [ytop - 0.115] * 2, color=HAIRLINE, lw=1.4)
    for i, r in enumerate(rows):
        y = ytop - 0.22 - i * 0.18
        blank = (i == 0)          # the do-it-together row
        vals = [f"{r['t']:g}  ({r['tag']})", f"{r['tp']}", f"{r['fp']}",
                "?" if blank else f"{r['rec']:.2f}",
                "?" if blank else f"{r['fpr']:.3f}",
                "?" if blank else f"{r['prec']:.2f}"]
        for j, (x, v) in enumerate(zip(xcol, vals)):
            axt.text(x, y, v, ha="center", va="center",
                     color=AMBER if v == "?" else (
                         ROI_INK if r["t"] == 0.5 and j > 0 else INK),
                     fontsize=13 if v != "?" else 16,
                     fontweight="bold" if (v == "?" or r["t"] == 0.5)
                     else "normal",
                     family="monospace" if j else "sans-serif")
    axt.text(0.5, 0.11,
             "recall = TP/41   ·   FPR = FP/697   ·   precision = TP/(TP+FP)",
             ha="center", va="center", color=MUTED, fontsize=11.5,
             style="italic")
    axt.text(0.5, 0.01,
             "the t = 0.5 row is the matrix we just worked by hand",
             ha="center", va="center", color=ROI_INK, fontsize=11.5,
             fontweight="bold")
    _save(fig, name)


def roc_curve_intro(name="fig_roc_intro.png"):
    """AUROC introduced properly: plot every threshold's (FPR, TPR), join them,
    and the area under the join IS the AUROC. Chance is the diagonal (0.5), the
    top-left corner is perfect. Curve and area come from the sweep table, so
    the number on the slide is reproducible from the counts."""
    rows = _sweep_points()
    xs, ys = _roc_xy()
    auc = _auc(xs, ys)

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ax.fill_between(xs, ys, color=TEAL, alpha=0.13, zorder=1)
    ax.plot(xs, ys, "-o", color=TEAL, lw=3.0, ms=8, zorder=4)
    ax.plot([0, 1], [0, 1], color=RED, lw=2.0, ls="--", zorder=2)
    ax.text(0.62, 0.55, "chance: AUROC = 0.5", color=RED, fontsize=11.5,
            rotation=32, ha="center", va="center", fontweight="bold")
    ax.scatter([0], [1], s=110, marker="*", color=ROI_INK, zorder=5)
    ax.text(0.02, 0.965, "perfect", color=ROI_INK, fontsize=11.5,
            fontweight="bold", va="top")
    for r, dx, dy, ha in zip(rows, (0.055, 0.055, 0.04), (-0.10, -0.10, -0.12),
                             ("left", "left", "left")):
        ax.annotate(f"t = {r['t']:g}   ({r['fpr']:.3f}, {r['rec']:.2f})",
                    xy=(r["fpr"], r["rec"]),
                    xytext=(r["fpr"] + dx, r["rec"] + dy), ha=ha,
                    fontsize=11.5, color=INK, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.2))
    ax.text(0.52, 0.30, f"AUROC ≈ {auc:.2f}", ha="center", va="center",
            color=TEAL, fontsize=22, fontweight="bold")
    ax.text(0.52, 0.20, "the shaded area under the curve", ha="center",
            va="center", color=MUTED, fontsize=11.5, style="italic")
    ax.set_xlabel("false-positive rate,  FPR = FP/(FP+TN)   →", fontsize=12)
    ax.set_ylabel("true-positive rate  =  recall  →", fontsize=12)
    ax.set_title("ROC: one dot per threshold, joined up",
                 fontsize=14, color=INK, fontweight="bold")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.04)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.subplots_adjust(bottom=0.14, left=0.10, right=0.98, top=0.90)
    _save(fig, name)


def pr_curve_intro(name="fig_pr_intro.png"):
    """AUPRC on the SAME thresholds: precision against recall. The chance line
    is not 0.5 — it is the prevalence (41/738 = 0.056), which is where a
    coin-flip model lands, and it is why AUPRC stays honest under imbalance."""
    rows = _sweep_points()
    xs, ys = _pr_xy()
    auc = _auc(xs, ys)
    prev = DRIAMS_POS / (DRIAMS_POS + DRIAMS_NEG)

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ax.fill_between(xs, ys, color=AMBER, alpha=0.15, zorder=1)
    ax.plot(xs, ys, "-o", color=AMBER, lw=3.0, ms=8, zorder=4)
    ax.axhline(prev, color=RED, lw=2.0, ls="--", zorder=2)
    ax.text(0.02, prev + 0.035,
            f"chance = prevalence = 41/738 = {prev:.3f}", color=RED,
            fontsize=11.5, ha="left", va="bottom", fontweight="bold")
    for r, dx, dy in zip(rows, (0.03, 0.02, -0.07), (0.12, 0.20, 0.10)):
        ax.annotate(f"t = {r['t']:g}   ({r['rec']:.2f}, {r['prec']:.2f})",
                    xy=(r["rec"], r["prec"]),
                    xytext=(r["rec"] + dx, r["prec"] + dy),
                    ha="right" if dx < 0 else "left",
                    fontsize=11.5, color=INK, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.2))
    ax.text(0.30, 0.30, f"AUPRC ≈ {auc:.2f}", ha="center", va="center",
            color=ROI_INK, fontsize=22, fontweight="bold")
    ax.text(0.30, 0.20, "area under the PR curve", ha="center", va="center",
            color=MUTED, fontsize=11.5, style="italic")
    ax.set_xlabel("recall  =  TP/(TP+FN)   →", fontsize=12)
    ax.set_ylabel("precision  =  TP/(TP+FP)   →", fontsize=12)
    ax.set_title("PR: the same thresholds, precision against recall",
                 fontsize=14, color=INK, fontweight="bold")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.04)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.subplots_adjust(bottom=0.14, left=0.10, right=0.98, top=0.90)
    _save(fig, name)


def roc_pr_curves(name="fig_roc_pr.png"):
    """Same classifier, two pictures, with the I-do operating point marked on
    BOTH (rule 3): ROC point (FPR 0.086, TPR 0.73) hugs the top-left and LOOKS
    great, while the PR point (recall 0.73, precision 0.33) exposes that only 1
    flag in 3 is real. ROC's baseline is the diagonal (0.5); PR's baseline is
    the prevalence (0.056) — the honest floor under imbalance (Saito &
    Rehmsmeier 2015)."""
    fig, (axr, axp) = plt.subplots(1, 2, figsize=(11.8, 4.7))
    # ---- ROC panel ---- (the same threshold sweep the two intro slides plot)
    rxs, rys = _roc_xy()
    axr.plot(rxs, rys, "-o", color=TEAL, lw=3.0, ms=6, zorder=4,
             label=f"our model  (AUROC ≈ {_auc(rxs, rys):.2f})")
    axr.plot([0, 1], [0, 1], color=RED, lw=2.2, ls="--", zorder=2,
             label="chance  (AUC = 0.5)")
    axr.scatter([0.086], [0.73], s=95, color=INK, zorder=6)
    axr.annotate("our point: FPR 0.086, TPR 0.73\n— hugs the corner, LOOKS great",
                 xy=(0.086, 0.73), xytext=(0.24, 0.42), fontsize=10,
                 color=INK, fontweight="bold",
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    axr.set_title("ROC — sensitivity vs. 1−specificity", fontsize=13,
                  color=INK, fontweight="bold")
    axr.set_xlabel("false-positive rate  (1 − specificity)", fontsize=11)
    axr.set_ylabel("true-positive rate  (sensitivity)", fontsize=11)
    axr.set_xlim(0, 1)
    axr.set_ylim(0, 1.02)
    axr.legend(loc="upper center", bbox_to_anchor=(0.5, -0.155), ncol=2,
               frameon=False, fontsize=10)
    for sp in ("top", "right"):
        axr.spines[sp].set_visible(False)
    # ---- PR panel ----
    pxs, pys = _pr_xy()
    axp.plot(pxs, pys, "-o", color=TEAL, lw=3.0, ms=6, zorder=4,
             label=f"our model  (AUPRC ≈ {_auc(pxs, pys):.2f})")
    axp.axhline(0.056, color=RED, lw=2.2, ls="--", zorder=2,
                label="baseline = prevalence (0.056)")
    axp.scatter([0.73], [0.33], s=95, color=INK, zorder=6)
    axp.annotate("same model: recall 0.73, precision 0.33\n— only 1 flag in 3 is real",
                 xy=(0.73, 0.33), xytext=(0.02, 0.72), ha="left", fontsize=10,
                 color=INK, fontweight="bold",
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    axp.set_title("PR — precision vs. recall (honest under imbalance)",
                  fontsize=12, color=INK, fontweight="bold")
    axp.set_xlabel("recall  (= sensitivity)", fontsize=11)
    axp.set_ylabel("precision", fontsize=11)
    axp.set_xlim(0, 1)
    axp.set_ylim(0, 1.02)
    axp.legend(loc="upper right", frameon=False, fontsize=10)
    for sp in ("top", "right"):
        axp.spines[sp].set_visible(False)
    fig.subplots_adjust(wspace=0.28)
    _save(fig, name)


def focal_loss_volume(name="fig_focal_volume.png"):
    """Focal loss as the WHOLE CURVE plus the numbers (rule 6).
    LEFT: FL = (1-p)^gamma * (-ln p) drawn for gamma = 0 (plain cross-entropy),
    1, 2 and 5, so the room can SEE that the focusing factor collapses the loss
    on the confident right-hand side while barely touching the boundary at
    p = 0.5. The two worked points are marked on the gamma = 0 and gamma = 2
    curves. RIGHT: the same two cases computed exactly — easy p=0.90 -> FL 0.001
    (turned DOWN ~100x), hard p=0.50 -> FL 0.173 (barely moved) — and the ratio
    that is the whole point: 6.6x under plain CE, 165x under focal loss."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.8, 5.2),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # ---------------- LEFT: the loss curves ----------------
    p = np.linspace(0.02, 1.0, 500)
    ce = -np.log(p)
    curves = [
        (0, "γ = 0   (plain cross-entropy)", INK, 2.8, "-"),
        (1, "γ = 1", MUTED, 2.0, "--"),
        (2, "γ = 2   (the usual default)", RED, 3.0, "-"),
        (5, "γ = 5", TEAL, 2.0, ":"),
    ]
    # the "already easy" zone
    axl.axvspan(0.8, 1.0, color=TEAL_SOFT, zorder=0)
    axl.text(0.824, 2.15, "already easy", ha="center", va="center",
             rotation=90, color=TEAL, fontsize=9.8, fontweight="bold",
             style="italic")
    for g, lab, col, lw, ls in curves:
        axl.plot(p, (1 - p) ** g * ce, color=col, lw=lw, ls=ls, label=lab,
                 zorder=3)

    for x in (0.5, 0.9):
        axl.axvline(x, color=HAIRLINE, lw=1.2, ls=(0, (2, 3)), zorder=1)

    # the two worked points, on plain CE and on gamma = 2
    pts = [(0.5, 0.693, INK, "CE 0.693"), (0.5, 0.173, RED, "FL 0.173"),
           (0.9, 0.105, INK, "CE 0.105"), (0.9, 0.001, RED, "FL 0.001")]
    for x, y, col, lab in pts:
        axl.scatter([x], [y], s=62, color=col, edgecolor=WHITE, lw=1.4,
                    zorder=6)
    axl.annotate("hard, on the boundary\np = 0.50:  0.693 → 0.173\n(barely moved)",
                 xy=(0.505, 0.19), xytext=(0.792, 2.52), ha="right",
                 fontsize=10.5, color=RED, fontweight="bold", linespacing=1.4,
                 arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.7,
                                 connectionstyle="arc3,rad=0.12"))
    axl.annotate("easy, confident\np = 0.90:  0.105 → 0.001\n(turned DOWN ~100×)",
                 xy=(0.898, 0.075), xytext=(0.575, 1.32), fontsize=10.5,
                 color=TEAL, fontweight="bold", linespacing=1.4,
                 arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.7))
    axl.set_xlabel("p — the probability the model gives the TRUE class",
                   fontsize=11)
    axl.set_ylabel("loss on that one example", fontsize=11)
    axl.set_title("FL = (1 − p)$^γ$ · (−ln p)", fontsize=14, color=INK,
                  fontweight="bold", loc="left")
    axl.set_xlim(0, 1.0)
    axl.set_ylim(0, 4.35)
    axl.set_xticks([0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0])
    axl.legend(loc="upper right", frameon=False, fontsize=10.5)
    for sp in ("top", "right"):
        axl.spines[sp].set_visible(False)

    # ---------------- RIGHT: the numbers, worked ----------------
    axr.set_xlim(0, 8)
    axr.set_ylim(0, 8)
    axr.axis("off")
    lines = [
        ("easy  p = 0.90", "(0.10)$^2$ = 0.01   ·   −ln 0.90 = 0.105",
         "FL = 0.001", TEAL,
         "a clean, obviously susceptible\nisolate — and there are 697"),
        ("hard  p = 0.50", "(0.50)$^2$ = 0.25   ·   −ln 0.50 = 0.693",
         "FL = 0.173", RED,
         "an MIC on the breakpoint, a heteroresistant\nsubpopulation, a low-biomass spectrum"),
    ]
    y = 7.7
    for head, mid, res, col, example in lines:
        axr.add_patch(FancyBboxPatch((0.1, y - 2.92), 7.8, 2.82,
                      boxstyle="round,pad=0.05,rounding_size=0.1",
                      facecolor=(TEAL_SOFT if col is TEAL else "#F9EDEC"),
                      edgecolor=col, lw=2.2))
        axr.text(0.45, y - 0.32, head, fontsize=12.5, color=INK, va="top",
                 fontweight="bold")
        axr.text(0.45, y - 1.02, mid, fontsize=11, color=INK_SOFT, va="top")
        axr.text(0.45, y - 1.66, res, fontsize=13, color=col, va="top",
                 fontweight="bold")
        axr.text(0.45, y - 2.18, example, fontsize=9.6, color=MUTED,
                 va="top", style="italic", linespacing=1.4)
        y -= 3.12

    axr.add_patch(FancyBboxPatch((0.1, 0.15), 7.8, 1.32,
                  boxstyle="round,pad=0.05,rounding_size=0.1",
                  facecolor=AMBER_SOFT, edgecolor=AMBER, lw=2.2))
    axr.text(4.0, 1.06, "hard vs easy:   plain CE 6.6×   →   focal 165×",
             ha="center", va="center", color=ROI_INK, fontsize=13,
             fontweight="bold")
    axr.text(4.0, 0.52, "0.693 / 0.105 = 6.6        0.173 / 0.00105 = 165",
             ha="center", va="center", color=INK, fontsize=10.5,
             family="monospace")
    fig.subplots_adjust(wspace=0.16)
    _save(fig, name)


# SMOTE's two m/z-bin "features" are made CONCRETE (instructor, 2026-09-10):
# the same A = (2, 6) and B = (4, 10) are two real MALDI-TOF spectra, with
# bin 1 = m/z 2,700 and bin 2 = m/z 5,300. Interpolating them at lambda = 0.5
# reproduces the toy answer (3, 8) AND shows what the caveat means: A's private
# peak at m/z 4,100 and B's at m/z 6,800 survive at HALF height in the blend —
# two ghost peaks no instrument ever produced.
SMOTE_MZ = [2700, 4100, 5300, 6800]
SMOTE_A = [2, 7, 6, 0]      # A = (2, 6) in bins 1 and 2, plus a private peak
SMOTE_B = [4, 0, 10, 5]     # B = (4, 10) in bins 1 and 2, plus a private peak


def smote_interpolation(mode="ido", name="fig_smote.png"):
    """SMOTE = synthetic minority over-sampling by interpolation (rule 6), now
    with the CONCRETE spectra behind the toy coordinates. LEFT: feature space —
    the majority (susceptible) cloud is dense, the minority (resistant) points
    are few, and a new synthetic point is made ON the segment between two real
    minority points A=(2,6), B=(4,10): synthetic = A + lambda*(B-A). I-do
    lambda=0.5 -> (3,8). RIGHT: the same three points AS SPECTRA — and the
    blend carries two half-height GHOST peaks (m/z 4,100 from A, m/z 6,800 from
    B) that no real isolate has, which is exactly why raw-spectrum SMOTE is
    risky in 6,000 dimensions."""
    rng = np.random.default_rng(7)
    fig = plt.figure(figsize=(13.4, 5.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.14, 1], hspace=0.42,
                          wspace=0.17)
    ax = fig.add_subplot(gs[:, 0])

    # ---------------- LEFT: feature space ----------------
    mx = rng.normal(7.0, 1.4, 90)
    my = rng.normal(4.0, 1.5, 90)
    ax.scatter(mx, my, s=34, color=TEAL, alpha=0.35, edgecolor="none",
               zorder=2, label="susceptible (majority)")
    minx = [2, 4, 3.1, 1.4, 4.8]
    miny = [6, 10, 8.7, 8.2, 6.7]
    ax.scatter(minx, miny, s=70, color=RED, edgecolor=INK, lw=0.8, zorder=4,
               label="resistant (minority)")
    A = (2, 6)
    B = (4, 10)
    lam = 0.5 if mode == "ido" else 0.25
    syn = (A[0] + lam * (B[0] - A[0]), A[1] + lam * (B[1] - A[1]))
    ax.plot([A[0], B[0]], [A[1], B[1]], color=AMBER, lw=2.4, ls="--", zorder=3)
    ax.text(A[0] - 0.15, A[1] - 0.6, "A (2, 6)", ha="right", color=RED,
            fontsize=12, fontweight="bold")
    ax.text(B[0] + 0.2, B[1] + 0.1, "B (4, 10)", ha="left", color=RED,
            fontsize=12, fontweight="bold")
    if mode == "ido":
        ax.scatter([syn[0]], [syn[1]], s=170, marker="*", color=AMBER,
                   edgecolor=INK, lw=1.0, zorder=5)
        ax.annotate(f"synthetic = A + 0.5·(B−A)\n= ({syn[0]:g}, {syn[1]:g})",
                    xy=syn, xytext=(4.7, 7.2), fontsize=11.5, color=ROI_INK,
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.8))
    else:
        ax.scatter([syn[0]], [syn[1]], s=170, marker="*",
                   facecolor="none", edgecolor=ROI_INK, lw=1.6, zorder=5)
        ax.annotate("synthetic = A + 0.25·(B−A)\n= ( ? , ? )",
                    xy=syn, xytext=(4.7, 6.5), fontsize=11.5, color=ROI_INK,
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", color=ROI_INK, lw=1.6))
    ax.set_xlabel("intensity, bin 1  (m/z 2,700)", fontsize=11)
    ax.set_ylabel("intensity, bin 2  (m/z 5,300)", fontsize=11)
    ax.set_title("feature space: interpolate between two real points",
                 fontsize=12.5, color=INK, fontweight="bold")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 12.4)
    ax.legend(loc="upper right", frameon=False, fontsize=10.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ---------------- RIGHT: the same three points AS SPECTRA -------------
    blend = [a + lam * (b - a) for a, b in zip(SMOTE_A, SMOTE_B)]
    panels = [
        ("A — a real resistant spectrum", SMOTE_A, RED, None),
        ("B — a real resistant spectrum", SMOTE_B, RED, None),
        (f"synthetic = A + {lam:g}·(B−A)", blend, AMBER, "ghost"),
    ]
    axes = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    for k, (ax2, (head, vals, col, tag)) in enumerate(zip(axes, panels)):
        for mz, v in zip(SMOTE_MZ, vals):
            if v <= 0:
                continue
            shared = mz in (2700, 5300)
            if tag and not shared:          # a GHOST peak on the blend
                ax2.bar(mz, v, width=300, facecolor="none", edgecolor=RED,
                        hatch="////", lw=1.8, zorder=3)
                lab_col = RED
            else:
                ax2.vlines(mz, 0, v, color=col, lw=4.0, zorder=3)
                lab_col = col
            ax2.text(mz, v + 0.7, f"{v:g}", ha="center", color=lab_col,
                     fontsize=9.5, fontweight="bold")
        ax2.set_xlim(2000, 7600)
        ax2.set_ylim(0, 13.5)
        ax2.set_yticks([])
        for sp in ("top", "right", "left"):
            ax2.spines[sp].set_visible(False)
        ax2.set_title(head, fontsize=10.5, color=col, fontweight="bold",
                      loc="left", pad=3)
        if k < 2:
            ax2.set_xticks(SMOTE_MZ)
            ax2.set_xticklabels([])
        else:
            ax2.set_xticks(SMOTE_MZ)
            ax2.set_xticklabels([f"{m:,}" for m in SMOTE_MZ], fontsize=9)
            ax2.set_xlabel("m/z", fontsize=10)
    # mark the two ghost peaks on the synthetic spectrum
    axes[2].annotate("GHOST peaks — in NEITHER real isolate",
                     xy=(3880, blend[1] / 2 + 0.4), xytext=(2300, 11.8),
                     fontsize=9.8, color=RED, fontweight="bold",
                     arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6,
                                     connectionstyle="arc3,rad=-0.15"))
    axes[2].annotate("", xy=(6570, blend[3] / 2 + 0.3), xytext=(5800, 10.4),
                     arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6,
                                     connectionstyle="arc3,rad=0.15"))
    _save(fig, name)


def curriculum_learning(name="fig_curriculum.png"):
    """Curriculum learning made CONCRETE, with the two curves that define it
    (instructor, 2026-09-10). LEFT — the SCHEDULE you actually write: over 20
    epochs the sampler anneals from class-balanced (50% resistant per batch)
    down to the true prevalence (6%), while the hard, borderline spectra are
    admitted gradually from 0% to 100%. Three stage bands name what is in the
    diet at each point, in spectra. RIGHT — what it buys, drawn as an
    ILLUSTRATIVE shape (labelled as such on the figure, not measured data):
    the curriculum run climbs smoothly, the shuffled run lurches, and both
    runs see exactly the same 41 resistant spectra — curriculum reorders the
    diet, it never adds signal."""
    ep = np.arange(1, 21)

    # ---- the schedule (exact, piecewise — this is what you would code) ----
    def pct_resistant(e):
        if e <= 4:
            return 50.0
        if e >= 12:
            return 6.0
        return 50.0 + (6.0 - 50.0) * (e - 4) / 8.0

    def pct_hard(e):
        if e <= 4:
            return 0.0
        if e >= 14:
            return 100.0
        return 100.0 * (e - 4) / 10.0

    bal = np.array([pct_resistant(e) for e in ep])
    hard = np.array([pct_hard(e) for e in ep])

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13.2, 5.0),
                                   gridspec_kw={"width_ratios": [1.18, 1]})

    bands = [(0.5, 4.5, TEAL_SOFT, TEAL, "START",
              "clean, high-biomass\nspectra · ONE site"),
             (4.5, 12.5, AMBER_SOFT, ROI_INK, "THEN",
              "mix in low-biomass runs,\nMICs on the breakpoint"),
             (12.5, 20.5, "#F9EDEC", RED, "FINALLY",
              "every site, the real\n~6% resistant mix")]
    for x0, x1, face, col, head, body in bands:
        axl.axvspan(x0, x1, color=face, zorder=0)
        axl.text((x0 + x1) / 2, 96, head, ha="center", va="center", color=col,
                 fontsize=11.5, fontweight="bold")
        axl.text((x0 + x1) / 2, 84, body, ha="center", va="center",
                 color=col, fontsize=8.8, style="italic", linespacing=1.35)
    for x in (4.5, 12.5):
        axl.axvline(x, color=WHITE, lw=2.2, zorder=1)

    axl.plot(ep, bal, color=TEAL, lw=3.0, marker="o", ms=4.5, zorder=4,
             label="% resistant in each batch")
    axl.plot(ep, hard, color=RED, lw=3.0, ls="--", marker="s", ms=4.0,
             zorder=4, label="% of hard / borderline spectra admitted")
    axl.annotate("50% — class-balanced batches",
                 xy=(2.0, 50), xytext=(1.2, 63), fontsize=9.8, color=TEAL,
                 fontweight="bold",
                 arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.5))
    axl.annotate("6% — the TRUE prevalence,\nso the model ends calibrated",
                 xy=(16.5, 6), xytext=(12.9, 26), fontsize=9.8, color=TEAL,
                 fontweight="bold", linespacing=1.35,
                 arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.5))
    axl.set_xlabel("epoch", fontsize=11)
    axl.set_ylabel("percent", fontsize=11)
    axl.set_title("the schedule: what goes in each batch, week by week",
                  fontsize=12.5, color=INK, fontweight="bold", loc="left")
    axl.set_xlim(0.5, 20.5)
    axl.set_ylim(0, 104)
    axl.set_xticks([1, 4, 8, 12, 16, 20])
    axl.set_yticks([0, 6, 25, 50, 75, 100])
    axl.legend(loc="upper center", bbox_to_anchor=(0.5, -0.155), ncol=2,
               frameon=False, fontsize=10)
    for sp in ("top", "right"):
        axl.spines[sp].set_visible(False)

    # ---- what it buys: an ILLUSTRATIVE shape, labelled as such ----
    curr = 0.18 + 0.45 * (1 - np.exp(-(ep - 1) / 4.2))
    shuf = (0.15 + 0.33 * (1 - np.exp(-(ep - 1) / 6.0))
            + 0.055 * np.sin(ep * 1.6))
    axr.plot(ep, curr, color=TEAL, lw=3.0, marker="o", ms=4.5,
             label="with curriculum")
    axr.plot(ep, shuf, color=MUTED, lw=2.4, ls="--", marker="s", ms=4.0,
             label="shuffled from day one")
    axr.annotate("steadier climb — the model\nlearns the easy signal first",
                 xy=(7.0, curr[6]), xytext=(2.2, 0.645), fontsize=9.6,
                 color=TEAL, fontweight="bold", linespacing=1.35,
                 arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.5))
    axr.annotate("lurches when a batch happens\nto be all-easy or all-hard",
                 xy=(13.0, shuf[12] - 0.012), xytext=(12.3, 0.285),
                 fontsize=9.8, color=INK_SOFT, fontweight="bold",
                 linespacing=1.35,
                 arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.5))
    axr.set_xlabel("epoch", fontsize=11)
    axr.set_ylabel("validation AUPRC", fontsize=11)
    axr.set_title("what it buys", fontsize=12.5, color=INK,
                  fontweight="bold", loc="left")
    axr.set_xlim(0.5, 20.5)
    axr.set_ylim(0.035, 0.78)
    axr.set_xticks([1, 4, 8, 12, 16, 20])
    axr.legend(loc="upper center", bbox_to_anchor=(0.5, -0.155), ncol=2,
               frameon=False, fontsize=10)
    for sp in ("top", "right"):
        axr.spines[sp].set_visible(False)
    axr.text(0.5, 0.735, "ILLUSTRATIVE — the SHAPE, not measured numbers",
             fontsize=9.5, color=ROI_INK, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=AMBER_SOFT,
                       edgecolor=AMBER, lw=1.6))
    axr.text(10.5, 0.062,
             "both runs see the SAME 41 resistant spectra — "
             "curriculum reorders, it never adds signal",
             ha="center", va="center", fontsize=8.8, color=INK,
             fontweight="bold")
    fig.subplots_adjust(wspace=0.20)
    _save(fig, name)


def curation_pass(name="fig_curation.png"):
    """CURATION with the examples that make it concrete: what a curation pass
    on the DRIAMS split actually removes, and what each action looks like on a
    real plate. Counts are exact and internally consistent, so the punchline —
    you lose 6 of your 41 resistant spectra and the model gets BETTER — is
    reproducible from the figure."""
    # (label, ΔR, ΔS) — each step's cost, applied in order
    steps = [("−22 failed /\nlow-biomass runs", 1, 21),
             ("−8 discordant\nAST labels", 3, 5),
             ("−12 replicate\nspectra grouped", 2, 10)]
    r, s = 41, 697
    nodes = [(r, s)]
    for _, dr, ds in steps:
        r, s = r - dr, s - ds
        nodes.append((r, s))

    fig, ax = plt.subplots(figsize=(13.2, 5.4))
    ax.set_xlim(0, 14.6)
    ax.set_ylim(0, 5.8)
    ax.axis("off")

    # ---- the pipeline of counts
    bw, gap = 2.45, 1.5
    x = 0.3
    for i, (rr, ss) in enumerate(nodes):
        last = i == len(nodes) - 1
        col = TEAL if last else INK_SOFT
        ax.add_patch(FancyBboxPatch((x, 4.10), bw, 0.95,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor=TEAL_SOFT if last else WHITE,
                    edgecolor=col, lw=2.2))
        ax.text(x + bw / 2, 4.75, f"{rr + ss} spectra", ha="center",
                va="center", color=col, fontsize=13, fontweight="bold")
        ax.text(x + bw / 2, 4.39, f"{rr} R  ·  {ss} S", ha="center",
                va="center", color=INK_SOFT, fontsize=11.5,
                family="monospace")
        if not last:
            ax.annotate("", xy=(x + bw + gap - 0.12, 4.58),
                        xytext=(x + bw + 0.12, 4.58),
                        arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.0))
            ax.text(x + bw + gap / 2, 5.16, steps[i][0], ha="center",
                    va="bottom", color=RED, fontsize=9.5, fontweight="bold")
        x += bw + gap

    # ---- what each action looks like on a real plate
    cards = [
        (INK_SOFT, "failed runs",
         "TIC under the acceptance cut,\nfewer than ~20 peaks, no crystals —\n"
         "no biology in the spectrum"),
        (RED, "discordant labels",
         "broth microdilution says R,\ndisk diffusion says S — resolve it\n"
         "or drop it; 3 of them are R"),
        (AMBER, "replicate spectra",
         "three shots of the SAME isolate\nsplit across train and test = leakage;\n"
         "group by isolate, not by spectrum"),
        (TEAL, "wrong target",
         "a mixed culture, or the well was\nmis-ID'd as another species —\n"
         "the label means nothing"),
    ]
    cw, cgap = 3.35, 0.3
    x = 0.15
    for col, head, body in cards:
        ax.add_patch(FancyBboxPatch((x, 1.05), cw, 2.35,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor=WHITE, edgecolor=col, lw=2.0))
        ax.add_patch(FancyBboxPatch((x, 2.78), cw, 0.62,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor=col, edgecolor=col, lw=2.0))
        ax.text(x + cw / 2, 3.09, head, ha="center", va="center", color=WHITE,
                fontsize=12, fontweight="bold")
        ax.text(x + cw / 2, 1.90, body, ha="center", va="center", color=INK,
                fontsize=9.5, linespacing=1.45)
        x += cw + cgap

    ax.add_patch(FancyBboxPatch((0.15, 0.22), 14.3, 0.62,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=AMBER_SOFT, edgecolor=AMBER, lw=2.0))
    ax.text(7.3, 0.53,
            "you end up with 6 FEWER resistant spectra — and a better model, "
            "because 3 of the 41 were labelled wrong",
            ha="center", va="center", color=INK, fontsize=12.5,
            fontweight="bold")
    _save(fig, name)


def embedding_dedup(name="fig_embed_dedup.png"):
    """CURATION, part two — how you actually FIND the 12 replicate spectra:
    deduplicate in EMBEDDING space. Each spectrum is pushed through the trained
    CNN and the penultimate layer's activations are its embedding (a vector of
    'coordinates', Lecture 7); two shots of the same isolate land on top of each
    other. Cosine similarity is the ruler, and the do-it-together computes two
    of them exactly on 3-bin toy spectra: a=(3,4,0) vs b=(6,8,0) -> 50/(5*10) =
    1.00 (a near-duplicate: same peak PATTERN, twice the total ion current), and
    a vs c=(0,4,3) -> 16/(5*5) = 0.64 (a genuinely different isolate)."""
    rng = np.random.default_rng(11)
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.8, 5.0),
                                   gridspec_kw={"width_ratios": [1.05, 1]})

    # ---------- LEFT: the embedding cloud, with duplicate clusters ringed ----
    mx = rng.normal(0.0, 1.05, 120)
    my = rng.normal(0.0, 0.95, 120)
    axl.scatter(mx, my, s=26, color=TEAL, alpha=0.28, edgecolor="none",
                zorder=2, label="susceptible (majority)")
    minx = rng.normal(2.4, 0.7, 9)
    miny = rng.normal(1.9, 0.6, 9)
    axl.scatter(minx, miny, s=52, color=RED, alpha=0.85, edgecolor="none",
                zorder=3, label="resistant (minority)")
    dup_centres = [(-1.6, 1.8), (1.15, -1.7), (3.0, 0.5)]
    for k, (cx, cy) in enumerate(dup_centres):
        pts = np.array([[cx, cy], [cx + 0.055, cy + 0.05],
                        [cx - 0.05, cy + 0.045], [cx + 0.02, cy - 0.055]])
        col = RED if k == 2 else TEAL
        axl.scatter(pts[:, 0], pts[:, 1], s=52, color=col, edgecolor=INK,
                    lw=0.7, zorder=5)
        axl.add_patch(Circle((cx, cy), 0.34, facecolor="none",
                             edgecolor=AMBER, lw=2.4, zorder=6))
    axl.annotate("4 shots of the SAME isolate\n→ one point, four times",
                 xy=(dup_centres[0][0] + 0.30, dup_centres[0][1] + 0.18),
                 xytext=(-3.35, 2.95), fontsize=10.5, color=ROI_INK,
                 fontweight="bold", linespacing=1.35,
                 arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.8))
    axl.text(0.02, 0.03, "keep ONE per ring — drop the rest",
             transform=axl.transAxes, fontsize=10.5, color=ROI_INK,
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.35", facecolor=AMBER_SOFT,
                       edgecolor=AMBER, lw=1.8))
    axl.set_title("embedding space  (each spectrum = one point)",
                  fontsize=12.5, color=INK, fontweight="bold")
    axl.set_xlabel("embedding dim 1", fontsize=10.5)
    axl.set_ylabel("embedding dim 2", fontsize=10.5)
    axl.set_xticks([])
    axl.set_yticks([])
    axl.set_xlim(-3.8, 4.5)
    axl.set_ylim(-3.1, 3.7)
    for sp in ("top", "right"):
        axl.spines[sp].set_visible(False)
    axl.legend(loc="lower right", frameon=False, fontsize=10)

    # ---------- RIGHT: the ruler, computed exactly ----------
    axr.set_xlim(0, 10)
    axr.set_ylim(0, 10)
    axr.axis("off")
    axr.text(5.0, 9.6, "the ruler:  cos(a, b) = (a · b) / (|a| · |b|)",
             ha="center", color=INK, fontsize=14, fontweight="bold")
    axr.text(5.0, 8.95,
             "1.00 = identical pattern   ·   0 = nothing in common",
             ha="center", color=MUTED, fontsize=10.5, style="italic")

    rows = [
        (AMBER, "a = (3, 4, 0)      b = (6, 8, 0)",
         "a·b = 18 + 32 + 0 = 50      |a| = 5,   |b| = 10",
         "cos = 50 / 50 = 1.00   →  NEAR-DUPLICATE",
         "same peak pattern, twice the total ion current\n"
         "— a second shot of the same colony"),
        (TEAL, "a = (3, 4, 0)      c = (0, 4, 3)",
         "a·c = 0 + 16 + 0 = 16       |a| = 5,   |c| = 5",
         "cos = 16 / 25 = 0.64   →  KEEP BOTH",
         "a different peak pattern — a different isolate"),
    ]
    y = 8.30
    for col, head, work, verdict, gloss in rows:
        axr.add_patch(FancyBboxPatch((0.15, y - 3.12), 9.7, 3.02,
                      boxstyle="round,pad=0.05,rounding_size=0.12",
                      facecolor=WHITE, edgecolor=col, lw=2.2))
        axr.text(0.5, y - 0.30, head, fontsize=12, color=INK, va="top",
                 fontweight="bold", family="monospace")
        axr.text(0.5, y - 1.05, work, fontsize=10.5, color=INK_SOFT,
                 va="top", family="monospace")
        axr.text(0.5, y - 1.72, verdict, fontsize=12, color=col, va="top",
                 fontweight="bold")
        axr.text(0.5, y - 2.30, gloss, fontsize=9.8, color=MUTED, va="top",
                 style="italic", linespacing=1.4)
        y -= 3.35

    axr.add_patch(FancyBboxPatch((0.15, 0.10), 9.7, 1.58,
                  boxstyle="round,pad=0.05,rounding_size=0.12",
                  facecolor=TEAL_SOFT, edgecolor=TEAL, lw=2.2))
    axr.text(4.925, 1.30, "rule: cos ≥ 0.99  →  same isolate, keep one",
             ha="center", va="center", color=TEAL, fontsize=12.5,
             fontweight="bold")
    axr.text(4.925, 0.60,
             "on our 738 this finds the 12 replicate spectra (2 R, 10 S)\n"
             "— the leak that inflates every score",
             ha="center", va="center", color=INK, fontsize=10,
             linespacing=1.35)
    fig.subplots_adjust(wspace=0.14)
    _save(fig, name)


def class_weight_family(name="fig_class_weights.png"):
    """Make the rare class count, worked: balanced class weights on the real
    DRIAMS split (w = N/(K·n_c) → 9.00 and 0.53, a ratio of exactly 17), then
    its three siblings — oversample, undersample, augment — each with what it
    actually means on spectra."""
    N, K = 738, 2
    nr, ns = 41, 697
    wr, ws = N / (K * nr), N / (K * ns)

    fig, ax = plt.subplots(figsize=(13.2, 5.0))
    ax.set_xlim(0, 14.6)
    ax.set_ylim(0, 5.3)
    ax.axis("off")

    # ---- LEFT: the weights, computed
    ax.add_patch(FancyBboxPatch((0.3, 1.30), 6.4, 3.65,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                facecolor=WHITE, edgecolor=TEAL, lw=2.2))
    ax.text(3.5, 4.62, "class weights, computed", ha="center", va="center",
            color=TEAL, fontsize=13, fontweight="bold")
    ax.text(3.5, 4.14, "w_c  =  N / (K · n_c)", ha="center", va="center",
            color=INK, fontsize=14, fontweight="bold", family="monospace")
    ax.text(3.5, 3.80, "N = 738 spectra  ·  K = 2 classes", ha="center",
            va="center", color=MUTED, fontsize=10.5, style="italic")
    ax.text(3.5, 3.28, f"resistant:    738 / (2 × {nr})  =  {wr:.2f}",
            ha="center", va="center", color=RED, fontsize=12.5,
            fontweight="bold", family="monospace")
    ax.text(3.5, 2.86, f"susceptible:  738 / (2 × {ns})  =  {ws:.2f}",
            ha="center", va="center", color=TEAL, fontsize=12.5,
            fontweight="bold", family="monospace")
    ax.add_patch(FancyBboxPatch((0.45, 2.02), 6.1, 0.56,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                facecolor=AMBER_SOFT, edgecolor=AMBER, lw=1.8))
    ax.text(3.5, 2.30,
            f"ratio {wr:.2f} / {ws:.2f} = {round(ns / nr)}× — one resistant "
            f"counts as {round(ns / nr)} susceptibles",
            ha="center", va="center", color=INK, fontsize=10,
            fontweight="bold")
    ax.text(3.5, 1.60,
            "nn.CrossEntropyLoss(weight=torch.tensor([0.53, 9.00]))",
            ha="center", va="center", color=TEAL, fontsize=9.5,
            family="monospace", fontweight="bold")

    # ---- RIGHT: the siblings, each with what it means on spectra
    sibs = [
        (AMBER, "oversample",
         "draw each of the 41 resistant ~17× per epoch\n"
         "(WeightedRandomSampler) — same effect, more\n"
         "compute, and the model can memorise them"),
        (RED, "undersample",
         "keep only 41 susceptible to match — you throw\n"
         "away 656 real spectra. Almost never worth it."),
        (TEAL, "augment",
         "new VIEWS of the same 41: m/z jitter ±0.1%,\n"
         "intensity ±10%, baseline shift, added noise —\n"
         "the Lab 2 move, and the honest one for spectra"),
    ]
    x0, w = 7.1, 7.2
    y = 4.95
    for col, head, body in sibs:
        h = 1.02
        ax.add_patch(FancyBboxPatch((x0, y - h), w, h,
                    boxstyle="round,pad=0.02,rounding_size=0.05",
                    facecolor=WHITE, edgecolor=col, lw=2.0))
        ax.add_patch(FancyBboxPatch((x0, y - 0.42), 1.95, 0.42,
                    boxstyle="round,pad=0.01,rounding_size=0.05",
                    facecolor=col, edgecolor=col, lw=2.0))
        ax.text(x0 + 0.98, y - 0.21, head, ha="center", va="center",
                color=WHITE, fontsize=11.5, fontweight="bold")
        ax.text(x0 + (1.95 + w) / 2, y - 0.62, body, ha="center", va="center",
                color=INK, fontsize=9.5, linespacing=1.4)
        y -= h + 0.22
    ax.text(x0 + w / 2, 1.02,
            "SMOTE is the fourth sibling — it INVENTS points between two real "
            "ones (next slide)",
            ha="center", va="center", color=MUTED, fontsize=10.5,
            style="italic")
    ax.text(x0 + w / 2, 0.52,
            "all four re-use the same 41 isolates — none of them adds a new one",
            ha="center", va="center", color=ROI_INK, fontsize=11.5,
            fontweight="bold")
    _save(fig, name)


def treatment_menu(name="fig_treatment_menu.png"):
    """The imbalanced-data treatment menu as four cards mapped to a SITUATION
    (rule 9, no text wall), with the durable footer: none of these adds NEW
    minority signal — only collecting more real resistant spectra does."""
    fig, ax = plt.subplots(figsize=(12.2, 4.8))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.text(6.2, 4.72, "imbalanced data? match the move to the problem",
            ha="center", color=INK, fontsize=13.5, fontweight="bold")
    cards = [
        (TEAL, "1 · curate", "junk / mislabeled\nruns", "clean labels beat\nmore noisy ones",
         "drop 22 failed runs;\nresolve 8 discordant labels"),
        (AMBER, "2 · reweight /\nSMOTE", "rare class\nignored",
         "class weights,\noversample, SMOTE",
         "w = 9.0 vs 0.53;\noversample ×17; jitter m/z"),
        (RED, "3 · focal loss", "borderline cases\nmissed", "up-weight the\nhard, low-conf.",
         "MICs on the breakpoint;\nheteroresistant isolates"),
        (INK_SOFT, "4 · curriculum", "unstable across\neasy/hard", "order easy→hard /\nbalanced→true",
         "one site → all sites;\nbalanced → the true 6%"),
    ]
    w = 2.65
    gap = 0.3
    x = 0.5
    for col, head, situ, body, example in cards:
        ax.add_patch(FancyBboxPatch((x, 1.25), w, 2.85,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor=WHITE, edgecolor=col, lw=2.4))
        ax.add_patch(FancyBboxPatch((x, 3.35), w, 0.75,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor=col, edgecolor=col, lw=2.4))
        ax.text(x + w / 2, 3.72, head, ha="center", va="center", color=WHITE,
                fontsize=11.5, fontweight="bold")
        ax.text(x + w / 2, 3.02, "if: " + situ, ha="center", va="center",
                color=RED, fontsize=9.5, fontweight="bold", style="italic")
        ax.text(x + w / 2, 2.34, body, ha="center", va="center", color=INK,
                fontsize=10)
        ax.plot([x + 0.25, x + w - 0.25], [1.94, 1.94], color=HAIRLINE, lw=1.1)
        ax.text(x + w / 2, 1.62, example, ha="center", va="center",
                color=MUTED, fontsize=9, style="italic", linespacing=1.4)
        x += w + gap
    ax.add_patch(FancyBboxPatch((0.5, 0.35), 11.4, 0.72,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=AMBER_SOFT, edgecolor=AMBER, lw=2.2))
    ax.text(6.2, 0.71,
            "none of these adds NEW resistant signal — only more real resistant "
            "spectra does.", ha="center", va="center", color=INK, fontsize=12,
            fontweight="bold")
    _save(fig, name)


def distribution_shift_drop(name="fig_shift_drop.png"):
    """Distribution shift, two panels: LEFT a batch effect (the same assay on a
    new instrument/site/lot shifts the data cloud); RIGHT the DRIAMS cautionary
    tale — AUROC drops from internal to a new site and decays over time.
    Source: Wiesmann et al., J Clin Microbiol 2025 (DRIAMS AMR models)."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.8, 4.4),
                                   gridspec_kw={"width_ratios": [1, 1]})
    # LEFT: two offset bell curves (train vs. new site)
    xs = np.linspace(-4, 6, 300)
    train = np.exp(-0.5 * xs ** 2)
    shifted = np.exp(-0.5 * (xs - 2.2) ** 2)
    axl.plot(xs, train, color=TEAL, lw=2.8, zorder=3)
    axl.fill_between(xs, train, color=TEAL, alpha=0.12)
    axl.plot(xs, shifted, color=RED, lw=2.8, ls="--", zorder=3)
    axl.fill_between(xs, shifted, color=RED, alpha=0.10)
    axl.text(-1.3, 1.12, "training\nsite", ha="center", color=TEAL,
             fontsize=11.5, fontweight="bold")
    axl.text(3.0, 1.12, "new instrument /\nsite / reagent lot", ha="center",
             color=RED, fontsize=11.5, fontweight="bold")
    axl.annotate("", xy=(2.0, 0.5), xytext=(0.2, 0.5),
                 arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, lw=1.8))
    axl.text(1.1, 0.4, "batch effect\nshifts the data", ha="center", va="top",
             color=INK_SOFT, fontsize=10.5)
    axl.set_title("the data moves under you", fontsize=12.5, color=INK,
                  fontweight="bold")
    axl.set_xticks([])
    axl.set_yticks([])
    axl.set_ylim(0, 1.35)
    for sp in axl.spines.values():
        sp.set_visible(False)
    # RIGHT: AUROC bars internal vs external + temporal note
    bars = axr.bar([0, 1.6], [0.90, 0.71], width=0.55,
                   color=[TEAL, RED], zorder=3)
    for b, v in zip(bars, [0.90, 0.71]):
        axr.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                 ha="center", color=INK, fontsize=14, fontweight="bold")
    axr.set_xticks([0, 1.6])
    axr.set_xticklabels(["same site\n(internal)", "new site\n(external)"],
                        fontsize=11)
    axr.set_xlim(-0.6, 2.2)
    axr.set_ylim(0, 1.05)
    axr.set_ylabel("AUROC", fontsize=11)
    axr.set_yticks([0, 0.5, 1.0])
    for sp in ("top", "right"):
        axr.spines[sp].set_visible(False)
    axr.annotate("−0.07 to −0.23 AUROC\nacross sites; decays\nwithin 18 months",
                 xy=(1.35, 0.71), xytext=(0.8, 0.34), ha="center", fontsize=9.5,
                 color=RED, fontweight="bold",
                 arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6))
    axr.set_title("DRIAMS AMR models across sites", fontsize=12, color=INK,
                  fontweight="bold")
    fig.subplots_adjust(wspace=0.32)
    _save(fig, name)


def _lecture10_figures():
    # Part A -- evaluation under imbalance (I-do 30/11/60/637, you-do 16/4/32/448)
    confusion_matrix(30, 11, 60, 637, "ido", "fig_confusion_ido.png")
    confusion_matrix(16, 4, 32, 448, "youdo", "fig_confusion_youdo.png")
    score_threshold_sweep()
    roc_curve_intro()
    pr_curve_intro()
    roc_pr_curves()
    distribution_shift_drop()
    # Part B -- treating imbalance
    treatment_menu()
    curation_pass()
    embedding_dedup()
    class_weight_family()
    smote_interpolation("ido", "fig_smote.png")
    focal_loss_volume()
    curriculum_learning()


# ---- diffusion primer (Lecture 8, foreshadowing Lab 3) ----------------------
# Lab 3 fine-tunes a real pretrained DDPM digit-drawer, so the primer uses the
# lab's own schedule (linear betas 1e-4 → 0.02 over 1000 steps) and a real
# MNIST digit. The hand-worked numbers below are the ones on Quiz 8 Q5.
MNIST_RAW = (Path(__file__).resolve().parent.parent / "labs" / "solutions"
             / "torchvision_data" / "MNIST" / "raw" / "train-images-idx3-ubyte")
# I-do patch: x0, the noise actually added, and the model's guess of that noise
DIFF_IDO = dict(x0=[1.0, 0.0, -1.0, 0.5], eps=[0.5, -0.5, 1.0, 0.0],
                pred=[0.3, -0.4, 0.8, 0.1])
# you-do patch = Quiz 8 Q5 (same shape, new numbers): x_t = (0.8, −0.5, 0.1, 0.6),
# errors (0.2, −0.1, 0.2, −0.3) → MSE = 0.18/4 = 0.045
DIFF_QUIZ = dict(x0=[1.0, -1.0, 0.5, 0.0], eps=[0.0, 0.5, -0.5, 1.0],
                 pred=[0.2, 0.4, -0.3, 0.7])
SQRT_AB, SQRT_1MAB = 0.8, 0.6          # √ᾱ and √(1−ᾱ) at the worked noise level


def _mnist_digit(index=7):
    """One 28×28 MNIST image, read straight from the idx file the labs download
    (public domain). Returns floats in [-1, 1] like the lab's tensors, or None
    when the file is absent — it is gitignored and only appears once Lab 3/4 has
    been run, so a clean checkout must still build every other figure."""
    if not MNIST_RAW.exists():
        return None
    with open(MNIST_RAW, "rb") as fh:
        fh.read(16)
        fh.read(28 * 28 * index)
        buf = fh.read(28 * 28)
    img = np.frombuffer(buf, dtype=np.uint8).reshape(28, 28).astype(float) / 255.0
    return img * 2 - 1


def _ddpm_alpha_bar(n=1000, beta_start=1e-4, beta_end=0.02):
    """ᾱ_t for the linear beta schedule Lab 3's DDPMScheduler uses."""
    betas = np.linspace(beta_start, beta_end, n)
    return np.cumprod(1.0 - betas)


def diffusion_strip(name="fig_diffusion_idea.png"):
    """The one idea, in the canonical DDPM layout: one row of frames, forward
    arrows on top (add noise — a fixed recipe, nothing learned) and reverse
    arrows underneath (a network peels a little noise off at each step — the
    only learned part). Real MNIST digit, real linear schedule: the same one
    Lab 3's DDPMScheduler loads."""
    x0 = _mnist_digit()
    if x0 is None:
        print(f"  skipped {name}: MNIST not downloaded "
              f"({MNIST_RAW.relative_to(ROOT)}) — run Lab 3/4 to rebuild it")
        return
    ab = _ddpm_alpha_bar()
    steps = [0, 100, 300, 600, 999]
    rng = np.random.default_rng(0)
    eps = rng.standard_normal(x0.shape)
    frames = [np.sqrt(ab[t]) * x0 + np.sqrt(1 - ab[t]) * eps for t in steps]

    n = len(frames)
    fig = plt.figure(figsize=(12.4, 4.5))
    ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, 1); ov.set_ylim(0, 1)
    ov.axis("off")

    w = 0.155
    h = w * 12.4 / 4.5
    gap = (0.94 - n * w) / (n - 1)
    y0 = 0.315
    xs = [0.03 + i * (w + gap) for i in range(n)]
    for i, (f, t) in enumerate(zip(frames, steps)):
        ax = fig.add_axes([xs[i], y0, w, h])
        ax.imshow((f + 1) / 2, cmap="Greys_r", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(HAIRLINE); sp.set_linewidth(1.4)
        ov.text(xs[i] + w / 2, y0 - 0.055, f"t = {t}", ha="center", va="top",
                color=INK_SOFT, fontsize=12, fontweight="bold")
    ov.text(xs[0] + w / 2, y0 + h + 0.035, "a real image", ha="center",
            va="bottom", color=INK_SOFT, fontsize=11.5, style="italic")
    ov.text(xs[-1] + w / 2, y0 + h + 0.035, "pure noise", ha="center",
            va="bottom", color=INK_SOFT, fontsize=11.5, style="italic")

    ytop, ybot = y0 + h + 0.115, y0 - 0.135
    for i in range(n - 1):
        xa, xb = xs[i] + w, xs[i + 1]
        ov.annotate("", xy=(xb - 0.004, ytop), xytext=(xa + 0.004, ytop),
                    arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2.2))
        ov.annotate("", xy=(xa + 0.004, ybot), xytext=(xb - 0.004, ybot),
                    arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=2.2))
    ov.text(0.5, ytop + 0.075,
            "FORWARD · add a little noise, again and again — a FIXED recipe, "
            "nothing is learned", ha="center", va="center", color=TEAL,
            fontsize=13.5, fontweight="bold")
    ov.text(0.5, ytop + 0.155,
            "x_t = √ᾱ · x₀ + √(1−ᾱ) · ε    —    ᾱ shrinks as t grows, so the "
            "image fades and the noise takes over",
            ha="center", va="center", color=MUTED, fontsize=11.5,
            style="italic")
    ov.text(0.5, ybot - 0.075,
            "REVERSE · a network takes a little noise OFF at each step — pure "
            "static becomes a new image. This is the only learned part.",
            ha="center", va="center", color=ROI_INK, fontsize=13.5,
            fontweight="bold")
    _save(fig, name)


def diffusion_objective(spec=None, reveal=True, name="fig_diffusion_train.png",
                        note=None):
    """The training objective, worked on a 4-pixel patch (rule 6): noise the
    patch with a known ε, let the model guess ε, score the guess with MSE.
    reveal=False blanks the two answers for the you-do (Quiz 8 Q5)."""
    spec = spec or DIFF_IDO
    x0, eps, pred = spec["x0"], spec["eps"], spec["pred"]
    xt = [SQRT_AB * a + SQRT_1MAB * b for a, b in zip(x0, eps)]
    errs = [p - e for p, e in zip(pred, eps)]
    mse = sum(d * d for d in errs) / len(errs)

    fig, ax = plt.subplots(figsize=(12.6, 5.1))
    ax.set_xlim(0, 14.0)
    ax.set_ylim(0, 5.65)
    ax.axis("off")
    cell = 0.72

    def patch(vals, x, ytop, colour, label, blank=False):
        data = [[f"{v:g}" if not blank else "?" for v in vals]]
        _cellgrid(ax, data, x, ytop, cell, edge=colour, txt=colour,
                  fontsize=15, na="?")
        ax.text(x + len(vals) * cell / 2, ytop + 0.26, label, ha="center",
                color=colour, fontsize=12.5, fontweight="bold")
        return x + len(vals) * cell

    # ---- step 1: make the noisy patch
    ax.text(0.4, 5.30, "1 · add a KNOWN amount of noise", ha="left",
            color=INK, fontsize=14.5, fontweight="bold")
    row1 = 4.55
    mid1 = row1 - cell / 2
    x = 0.4
    x = patch(x0, x, row1, INK_SOFT, f"clean patch  x₀   (×{SQRT_AB:g})")
    ax.text(x + 0.45, mid1, "+", ha="center", va="center", fontsize=20,
            color=INK_SOFT)
    x += 0.9
    x = patch(eps, x, row1, TEAL, f"noise  ε   (×{SQRT_1MAB:g})")
    ax.text(x + 0.45, mid1, "=", ha="center", va="center", fontsize=20,
            color=INK_SOFT)
    x += 0.9
    x = patch(xt, x, row1, AMBER, "noisy patch  x_t", blank=not reveal)
    ax.text(x + 0.5, mid1,
            f"x_t = {SQRT_AB:g}·x₀ + {SQRT_1MAB:g}·ε\nelement by element",
            ha="left", va="center", color=MUTED, fontsize=11.5,
            style="italic", linespacing=1.35)

    # ---- step 2: the model guesses the noise; score it with MSE
    ax.text(0.4, 3.15, "2 · the model guesses that noise — score the guess",
            ha="left", color=INK, fontsize=14.5, fontweight="bold")
    row2 = 2.40
    mid2 = row2 - cell / 2
    x = 0.4
    x = patch(pred, x, row2, RED, "predicted noise  ε̂")
    ax.text(x + 0.45, mid2, "−", ha="center", va="center", fontsize=20,
            color=INK_SOFT)
    x += 0.9
    x = patch(eps, x, row2, TEAL, "actual noise  ε")
    ax.text(x + 0.55, mid2, "→", ha="center", va="center", fontsize=18,
            color=INK_SOFT)
    x += 1.1
    if reveal:
        sq = " + ".join(f"{d * d:.2f}" for d in errs)
        ax.text(x, mid2, f"MSE = ({sq}) / 4 = {mse:g}", ha="left",
                va="center", color=INK, fontsize=14, fontweight="bold",
                family="monospace")
    else:
        ax.text(x, mid2, "MSE = ?", ha="left", va="center", color=AMBER,
                fontsize=19, fontweight="bold", family="monospace")

    # ---- the one line that is the objective (and Lab 3's blank)
    ax.add_patch(FancyBboxPatch((0.4, 0.35), 13.2, 0.8,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=TEAL_SOFT, edgecolor=TEAL, lw=1.6))
    ax.text(7.0, 0.75, "loss = F.mse_loss(noise_pred, noise)",
            ha="center", va="center", color=TEAL, fontsize=15,
            fontweight="bold", family="monospace")
    ax.text(7.0, 0.14,
            "the whole training objective — and Lab 3's Blank 2",
            ha="center", va="center", color=MUTED, fontsize=12,
            style="italic")
    if note:
        ax.text(9.9, 5.30, note, ha="center", va="center", color=MUTED,
                fontsize=12, style="italic")
    _save(fig, name)


def _lecture8_figures():
    positional_encoding()
    positional_encoding_heatmap()
    rope_relative()
    norm_axes("ido", "fig_norm_axes.png")
    norm_axes("youdo", "fig_norm_youdo.png")
    transformer_block(True, "fig_transformer_block.png")
    bert_vs_gpt()
    cross_attention()
    arch_families()
    tokenization_strategies()
    tokenization_decision("ido", "fig_tokenization_decision.png")
    tokenization_decision("youdo", "fig_tokenization_youdo.png")
    transfer_learning()
    # the diffusion primer that foreshadows Lab 3
    diffusion_strip()
    diffusion_objective(DIFF_IDO, reveal=True, name="fig_diffusion_train.png")
    diffusion_objective(DIFF_QUIZ, reveal=False, name="fig_diffusion_youdo.png",
                        note="your turn — same two steps, new numbers (Quiz 8 Q5)")


# ============================================================================
#  Lecture 11 · Learning Without (Many) Labels
#  Autoencoders, VAEs (intuition, no derivation), self-supervised learning
#  (masking / contrastive / DINO) and semi-supervised pseudo-labelling.
#  The one numeric mechanic of the hour is reconstruction error / MSE as an
#  anomaly score (rule 6); everything else is a bespoke MS-anchored schematic
#  (rule 9). The canonical autoencoder schematic is a REAL licensed image
#  (Michela Massi, Wikimedia Commons, CC BY-SA 4.0) placed directly in the
#  deck (rule 7); the rest are drawn here to stay MS-anchored, offline, and
#  editable.
# ============================================================================

def _spectrum_curve(ax, peaks, color=TEAL, x_hi=10.0, lw=1.7, fill=True,
                    ls="-", alpha_fill=0.10, zorder=3, width=0.16):
    """A smooth spectrum = a sum of Gaussians at (centre, height)."""
    mz = np.linspace(0, x_hi, 600)
    y = np.zeros_like(mz)
    for c, h in peaks:
        y += h * np.exp(-((mz - c) / width) ** 2)
    ax.plot(mz, y, color=color, lw=lw, zorder=zorder, ls=ls,
            solid_capstyle="round")
    if fill:
        ax.fill_between(mz, y, color=color, alpha=alpha_fill, zorder=zorder - 1)
    return mz, y


def autoencoder_bottleneck(name="fig_ae_bottleneck.png"):
    """MS-anchored autoencoder as an hourglass of nodes: an input spectrum is
    SQUEEZED through a narrow latent code and then REBUILT. The bottleneck is
    the teachable idea — the latent code is a compact 'map' coordinate for a
    spectrum, and reconstruction is what makes anomaly scoring possible."""
    fig, ax = plt.subplots(figsize=(11.8, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.4, 6.2)
    ax.axis("off")
    counts = [5, 3, 2, 3, 5]
    xs = [1.5, 3.7, 6.0, 8.3, 10.5]
    yc = 3.3
    sp = 0.5
    positions = []
    for cnt, x in zip(counts, xs):
        ys = np.linspace(yc - sp * (cnt - 1) / 1.0, yc + sp * (cnt - 1) / 1.0, cnt)
        positions.append([(x, y) for y in ys])
    for li in range(len(positions) - 1):
        for (x0, y0) in positions[li]:
            for (x1, y1) in positions[li + 1]:
                ax.plot([x0, x1], [y0, y1], color=TEAL, lw=0.8, alpha=0.22,
                        zorder=1)
    for li, (layer, cnt) in enumerate(zip(positions, counts)):
        face = AMBER_SOFT if li == 2 else TEAL_SOFT
        edge = AMBER if li == 2 else TEAL
        for (x, y) in layer:
            ax.add_patch(Circle((x, y), 0.24, facecolor=face, edgecolor=edge,
                                lw=2.0, zorder=3))
    # top banner
    ax.text(6.0, 5.9, "compress  \u2192  latent code  \u2192  reconstruct",
            ha="center", color=INK, fontsize=15, fontweight="bold")
    # bottleneck callout above the narrow column
    ax.text(6.0, 4.85, "latent code z\n(2\u20133 numbers)", ha="center",
            va="bottom", color=ROI_INK, fontsize=12, fontweight="bold")
    # end labels
    ax.text(1.5, 0.82, "input spectrum x", ha="center", va="center", color=TEAL,
            fontsize=12, fontweight="bold")
    ax.text(10.5, 0.82, "reconstruction x\u0302", ha="center", va="center",
            color=TEAL, fontsize=12, fontweight="bold")
    # encoder / decoder brackets
    ax.annotate("", xy=(5.2, 0.4), xytext=(1.5, 0.4),
                arrowprops=dict(arrowstyle="-", color=INK_SOFT, lw=1.4))
    ax.annotate("", xy=(10.5, 0.4), xytext=(6.8, 0.4),
                arrowprops=dict(arrowstyle="-", color=INK_SOFT, lw=1.4))
    ax.text(3.35, 0.05, "ENCODER \u2014 squeeze", ha="center", va="top",
            color=INK_SOFT, fontsize=11.5, fontweight="bold")
    ax.text(8.65, 0.05, "DECODER \u2014 rebuild", ha="center", va="top",
            color=INK_SOFT, fontsize=11.5, fontweight="bold")
    ax.text(6.0, -0.28,
            "trained so output \u2248 input \u2014 the squeeze forces it to keep only what matters",
            ha="center", va="top", color=MUTED, fontsize=11.5, style="italic")
    _save(fig, name)


def recon_error_worked(mode="ido", name="fig_recon_error_ido.png",
                       x=(0.2, 0.5, 0.1), xhat=(0.2, 0.4, 0.1), thr=0.01):
    """The one numeric mechanic of the hour (rule 6): reconstruction error.
    Left = a 3-bin spectrum vs. its reconstruction as bars; right = the worked
    table (elementwise squared error \u2192 MSE) and the anomaly decision against a
    stated threshold. mode='ido' fills every number; mode='youdo' leaves the
    differences, squared errors, MSE and the decision as '?' for Quiz 11 Q3."""
    blank = mode == "youdo"
    diffs = [round(a - b, 4) for a, b in zip(x, xhat)]
    sq = [round(d * d, 4) for d in diffs]
    sse = round(sum(sq), 4)
    mse = sse / len(x)
    normal = mse <= thr
    fig, (axb, axt) = plt.subplots(1, 2, figsize=(11.8, 4.7),
                                   gridspec_kw={"width_ratios": [1, 1.3]})
    idx = np.arange(len(x))
    w = 0.36
    axb.bar(idx - w / 2, x, width=w, color=TEAL, label="x  (measured)",
            zorder=3)
    axb.bar(idx + w / 2, xhat, width=w, facecolor="none", edgecolor=AMBER,
            lw=2.6, label="x\u0302  (reconstruction)", zorder=3)
    axb.set_xticks(idx)
    axb.set_xticklabels([f"bin {i + 1}" for i in idx], fontsize=11)
    axb.set_ylim(0, max(max(x), max(xhat)) * 1.4)
    axb.set_yticks([])
    for s in ("top", "right", "left"):
        axb.spines[s].set_visible(False)
    axb.legend(loc="upper right", frameon=False, fontsize=10.5)
    axb.set_title("a 3-bin spectrum vs. its reconstruction", fontsize=12,
                  color=INK, fontweight="bold")
    # right: formula + worked table
    axt.set_xlim(0, 1)
    axt.set_ylim(0, 1)
    axt.axis("off")
    axt.text(0.0, 0.97, "MSE = (1/n) \u03a3 (x\u1d62 \u2212 x\u0302\u1d62)\u00b2",
             fontsize=15, color=INK, fontweight="bold")
    axt.text(0.0, 0.86, f"flag ANOMALY if MSE > {thr:g}", fontsize=12.5,
             color=RED, fontweight="bold")
    cols_x = [0.02, 0.24, 0.44, 0.66, 0.90]
    hdr = ["bin", "x\u1d62", "x\u0302\u1d62", "x\u1d62\u2212x\u0302\u1d62",
           "(x\u1d62\u2212x\u0302\u1d62)\u00b2"]
    y0 = 0.70
    for cx, h in zip(cols_x, hdr):
        axt.text(cx, y0, h, fontsize=11.5, color=MUTED, fontweight="bold")
    for i in range(len(x)):
        yy = y0 - 0.11 * (i + 1)
        axt.text(cols_x[0], yy, str(i + 1), fontsize=12, color=INK_SOFT)
        axt.text(cols_x[1], yy, f"{x[i]:g}", fontsize=12, color=TEAL,
                 fontweight="bold")
        axt.text(cols_x[2], yy, f"{xhat[i]:g}", fontsize=12, color=ROI_INK,
                 fontweight="bold")
        axt.text(cols_x[3], yy, "?" if blank else f"{diffs[i]:g}", fontsize=12,
                 color=INK)
        axt.text(cols_x[4], yy, "?" if blank else f"{sq[i]:g}", fontsize=12,
                 color=INK)
    axt.plot([0.0, 0.99], [0.30, 0.30], color=HAIRLINE, lw=1.2)
    mse_txt = ("MSE = ?" if blank else
               f"MSE = {sse:g} / {len(x)} = {mse:.4f}")
    axt.text(0.0, 0.21, mse_txt, fontsize=14, color=INK, fontweight="bold")
    # decision box
    if blank:
        dec = "decision:  ?  (normal or anomaly)"
        dcol, dface = ROI_INK, AMBER_SOFT
    elif normal:
        dec = f"{mse:.4f} < {thr:g}  \u2192  NORMAL (pass QC)"
        dcol, dface = TEAL, TEAL_SOFT
    else:
        dec = f"{mse:.4f} > {thr:g}  \u2192  ANOMALY (flag the run)"
        dcol, dface = RED, "#F7E4E3"
    axt.add_patch(FancyBboxPatch((0.0, 0.0), 0.99, 0.12,
                  boxstyle="round,pad=0.01,rounding_size=0.03",
                  facecolor=dface, edgecolor=dcol, lw=2.0))
    axt.text(0.495, 0.06, dec, ha="center", va="center", color=dcol,
             fontsize=12.5, fontweight="bold")
    fig.subplots_adjust(wspace=0.22)
    _save(fig, name)


def anomaly_overlay(name="fig_anomaly_overlay.png"):
    """Reconstruction error as an anomaly / QC score. The autoencoder is trained
    ONLY on good runs, so a normal run reconstructs well (small gap, LOW error,
    pass) but an abnormal run \u2014 e.g. a contaminant peak the model never learned
    \u2014 reconstructs poorly (large red gap, HIGH error, flag)."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.8, 4.5))
    normal = [(2.0, 1.0), (4.0, 0.6), (6.5, 0.85), (8.5, 0.4)]
    recon = [(2.0, 0.94), (4.0, 0.56), (6.5, 0.80), (8.5, 0.37)]
    contaminant = normal + [(5.2, 0.9)]
    for ax, title, inp, col in [
            (axl, "normal QC run", normal, TEAL),
            (axr, "abnormal run (contaminant)", contaminant, TEAL)]:
        mz, yin = _spectrum_curve(ax, inp, color=col, fill=True, alpha_fill=0.12)
        _, yrec = _spectrum_curve(ax, recon, color=AMBER, ls="--", fill=False,
                                  lw=2.0, zorder=4)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("m/z \u2192", fontsize=11, color=MUTED)
        ax.set_title(title, fontsize=12.5, color=INK, fontweight="bold")
    # legend proxies on left
    axl.plot([], [], color=TEAL, lw=2.0, label="measured x")
    axl.plot([], [], color=AMBER, lw=2.0, ls="--", label="reconstruction x\u0302")
    axl.legend(loc="upper right", frameon=False, fontsize=10)
    # shade the mismatch on the abnormal panel
    mz = np.linspace(0, 10, 600)
    yin = np.zeros_like(mz)
    for c, h in contaminant:
        yin += h * np.exp(-((mz - c) / 0.16) ** 2)
    yrec = np.zeros_like(mz)
    for c, h in recon:
        yrec += h * np.exp(-((mz - c) / 0.16) ** 2)
    axr.fill_between(mz, yrec, yin, where=(yin > yrec), color=RED, alpha=0.30,
                     zorder=2)
    axr.annotate("model never learned\nthis peak \u2192 big gap", xy=(5.2, 0.9),
                 xytext=(6.2, 1.28), ha="left", fontsize=10, color=RED,
                 fontweight="bold",
                 arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6))
    axl.text(0.5, -0.14, "recon error LOW  \u2192  pass", transform=axl.transAxes,
             ha="center", color=TEAL, fontsize=12.5, fontweight="bold")
    axr.text(0.5, -0.14, "recon error HIGH  \u2192  FLAG", transform=axr.transAxes,
             ha="center", color=RED, fontsize=12.5, fontweight="bold")
    fig.subplots_adjust(wspace=0.14, bottom=0.16)
    _save(fig, name)


def _latent_scatter(ax, kind, seed=7, annotate=True):
    """Draw one latent-space panel. kind='ae' = scattered islands with big empty
    gaps (points that fall between clusters decode to garbage); kind='vae' = one
    smooth, organised, gap-free cloud you can sample anywhere. annotate=False
    drops the answer-giving callouts for the Quiz 11 Q1 you-do variant so the
    room must read the shape itself."""
    rng = np.random.default_rng(seed)
    class_cols = [TEAL, AMBER, RED, INK_SOFT]
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(HAIRLINE)
    if kind == "ae":
        centers = [(-1.5, 1.4), (1.6, 1.5), (-1.5, -1.4), (1.5, -1.4)]
        for (cx, cy), col in zip(centers, class_cols):
            pts = rng.normal(0, 0.20, size=(26, 2)) + np.array([cx, cy])
            ax.scatter(pts[:, 0], pts[:, 1], s=22, color=col, alpha=0.85,
                       edgecolor="none", zorder=3)
        ax.text(0.0, 0.0, "\u2715", ha="center", va="center", color=RED,
                fontsize=20, fontweight="bold", zorder=4)
        if annotate:
            ax.annotate("empty gap \u2192\ndecodes to garbage", xy=(0.0, 0.0),
                        xytext=(0.15, -2.6), ha="center", fontsize=10, color=RED,
                        fontweight="bold",
                        arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4))
    else:
        # one broad blob; colour by angle so classes blend continuously
        n = 150
        r = rng.normal(0, 0.85, size=n)
        th = rng.uniform(0, 2 * np.pi, size=n)
        px = r * np.cos(th) + rng.normal(0, 0.35, n)
        py = r * np.sin(th) + rng.normal(0, 0.35, n)
        ang = (np.arctan2(py, px) + np.pi) / (2 * np.pi)
        cols = [class_cols[int(a * 4) % 4] for a in ang]
        ax.scatter(px, py, s=22, color=cols, alpha=0.8, edgecolor="none",
                   zorder=3)
        if annotate:
            ax.annotate("sample anywhere \u2192\na plausible spectrum", xy=(0.4, 0.3),
                        xytext=(0.1, -2.6), ha="center", fontsize=10, color=TEAL,
                        fontweight="bold",
                        arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.4))


def ae_vs_vae_latent(labels=("autoencoder latent", "VAE latent"),
                     name="fig_ae_vs_vae_latent.png", describe=True):
    """Side-by-side latent maps of the SAME spectra: an autoencoder's scattered
    islands with gaps vs. a VAE's smooth, sampleable cloud. Reused with A/B
    titles for the Quiz 11 Q1 you-do so the room reads it straight off the
    slide. describe=False strips the answer-giving descriptor captions and
    in-panel callouts so the you-do variant does not telegraph which is the
    VAE."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.4, 5.0))
    _latent_scatter(axl, "ae", annotate=describe)
    _latent_scatter(axr, "vae", annotate=describe)
    axl.set_title(labels[0], fontsize=14, color=INK, fontweight="bold", pad=8)
    axr.set_title(labels[1], fontsize=14, color=INK, fontweight="bold", pad=8)
    if describe:
        for ax, cap in [(axl, "scattered islands, big empty gaps"),
                        (axr, "one smooth, organised, gap-free cloud")]:
            ax.text(0.5, -0.08, cap, transform=ax.transAxes, ha="center",
                    va="top", color=MUTED, fontsize=11, style="italic")
    fig.subplots_adjust(wspace=0.14, bottom=0.12)
    _save(fig, name)


def vae_recipe(name="fig_vae_recipe.png"):
    """The VAE recipe in one picture, NO derivation: reconstruction (output \u2248
    input) PLUS a 'keep the cloud tidy' penalty (pull the codes toward one neat,
    centred blob) = a smooth, sampleable latent."""
    fig, ax = plt.subplots(figsize=(12.0, 4.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.text(6.0, 4.72, "the VAE recipe \u2014 two ingredients, no derivation",
            ha="center", color=INK, fontsize=14, fontweight="bold")
    # card 1: reconstruction
    ax.add_patch(FancyBboxPatch((0.4, 1.1), 3.4, 2.9,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=WHITE, edgecolor=TEAL, lw=2.4))
    ax.text(2.1, 3.6, "reconstruction", ha="center", color=TEAL, fontsize=12.5,
            fontweight="bold")
    # mini spectrum drawn directly in ax coordinates so it stays inside the card
    mzc = np.linspace(0.75, 3.45, 320)
    base = 1.75
    yin = base + sum(h * np.exp(-((mzc - cx) / 0.10) ** 2)
                     for cx, h in [(1.3, 1.2), (2.1, 0.7), (2.9, 0.95)])
    yrec = base + sum(h * np.exp(-((mzc - cx) / 0.10) ** 2)
                      for cx, h in [(1.3, 1.1), (2.1, 0.64), (2.9, 0.88)])
    ax.plot(mzc, yin, color=TEAL, lw=1.7, zorder=3)
    ax.fill_between(mzc, base, yin, color=TEAL, alpha=0.12, zorder=2)
    ax.plot(mzc, yrec, color=AMBER, lw=1.5, ls="--", zorder=4)
    ax.text(2.1, 1.4, "output \u2248 input", ha="center", color=INK_SOFT,
            fontsize=10.5, style="italic")
    ax.text(4.25, 2.55, "+", ha="center", va="center", color=INK, fontsize=30,
            fontweight="bold")
    # card 2: keep the cloud tidy
    ax.add_patch(FancyBboxPatch((4.7, 1.1), 3.4, 2.9,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=WHITE, edgecolor=AMBER, lw=2.4))
    ax.text(6.4, 3.6, "keep the cloud tidy", ha="center", color=ROI_INK,
            fontsize=12.5, fontweight="bold")
    rng = np.random.default_rng(3)
    scatter = rng.normal(0, 1, size=(18, 2)) * np.array([1.6, 1.0])
    for (dx, dy) in scatter:
        px, py = 6.4 + 0.28 * dx, 2.45 + 0.28 * dy
        ax.plot([px], [py], marker="o", ms=4, color=MUTED, zorder=3)
        ax.annotate("", xy=(6.4, 2.45), xytext=(px, py),
                    arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=0.8,
                                    alpha=0.55), zorder=2)
    ax.add_patch(Circle((6.4, 2.45), 0.28, facecolor=AMBER_SOFT,
                        edgecolor=AMBER, lw=1.6, zorder=4))
    ax.text(6.4, 1.35, "pull codes toward\none neat, centred blob", ha="center",
            color=INK_SOFT, fontsize=10, style="italic")
    ax.text(8.55, 2.55, "=", ha="center", va="center", color=INK, fontsize=30,
            fontweight="bold")
    # result card
    ax.add_patch(FancyBboxPatch((9.0, 1.1), 2.8, 2.9,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=TEAL_SOFT, edgecolor=TEAL, lw=2.4))
    ax.text(10.4, 3.6, "smooth latent", ha="center", color=TEAL, fontsize=12.5,
            fontweight="bold")
    rng2 = np.random.default_rng(5)
    blob = rng2.normal(0, 0.7, size=(40, 2))
    ax.scatter(10.4 + 0.32 * blob[:, 0], 2.45 + 0.32 * blob[:, 1], s=14,
               color=TEAL, alpha=0.7, zorder=3)
    ax.text(10.4, 1.35, "sampleable +\ninterpolatable", ha="center",
            color=INK_SOFT, fontsize=10, style="italic")
    _save(fig, name)


def latent_interpolation(name="fig_latent_interp.png"):
    """Walking a straight line through a VAE's latent space = one spectrum
    morphing into another. Top: the latent cloud with a line A\u2192B and 5 marked
    stops; bottom: the 5 decoded spectra, peaks sliding and swapping height."""
    fig = plt.figure(figsize=(11.8, 5.4))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.05, 1.0], hspace=0.42,
                          wspace=0.18)
    axtop = fig.add_subplot(gs[0, :])
    rng = np.random.default_rng(11)
    blob = rng.normal(0, 0.8, size=(120, 2))
    axtop.scatter(blob[:, 0] * 2.2, blob[:, 1], s=16, color=TEAL_SOFT,
                  edgecolor="none", zorder=1)
    ax_pts = np.linspace(-3.4, 3.4, 5)
    ay = np.linspace(-0.7, 0.7, 5)
    axtop.plot(ax_pts, ay, color=ROI_INK, lw=2.2, ls="--", zorder=3)
    for i, (px, py) in enumerate(zip(ax_pts, ay)):
        axtop.plot([px], [py], marker="o", ms=11, color=AMBER,
                   markeredgecolor=ROI_INK, zorder=4)
        axtop.text(px, py + 0.42, f"{i + 1}", ha="center", color=ROI_INK,
                   fontsize=11, fontweight="bold")
    axtop.text(ax_pts[0], ay[0] - 0.5, "A", ha="center", color=TEAL,
               fontsize=14, fontweight="bold")
    axtop.text(ax_pts[-1], ay[-1] - 0.5, "B", ha="center", color=RED,
               fontsize=14, fontweight="bold")
    axtop.set_xlim(-4.2, 4.2)
    axtop.set_ylim(-2.4, 2.0)
    axtop.set_xticks([])
    axtop.set_yticks([])
    for s in axtop.spines.values():
        s.set_edgecolor(HAIRLINE)
    axtop.set_title("walk a straight line through the VAE latent space",
                    fontsize=12.5, color=INK, fontweight="bold")
    A = [(2.0, 1.0), (4.0, 0.3), (7.0, 0.7)]
    B = [(3.0, 0.4), (5.5, 1.0), (8.5, 0.5)]
    for i in range(5):
        t = i / 4.0
        peaks = [((1 - t) * a[0] + t * b[0], (1 - t) * a[1] + t * b[1])
                 for a, b in zip(A, B)]
        axb = fig.add_subplot(gs[1, i])
        col = TEAL if i == 0 else (RED if i == 4 else AMBER)
        _spectrum_curve(axb, peaks, color=col, fill=True, alpha_fill=0.15)
        axb.set_xlim(0, 10)
        axb.set_ylim(0, 1.25)
        axb.set_xticks([])
        axb.set_yticks([])
        for s in axb.spines.values():
            s.set_visible(False)
        axb.set_title(f"stop {i + 1}", fontsize=10.5, color=col,
                      fontweight="bold")
    fig.text(0.5, 0.005, "decoded spectrum morphs smoothly from A into B",
             ha="center", color=MUTED, fontsize=11.5, style="italic")
    _save(fig, name)


def masking_pretext(name="fig_masking.png"):
    """Self-supervised masking / pretext task (the BERT callback, Lecture 8):
    hide a piece of the input and predict it from context. The data labels
    itself \u2014 the hidden token IS the label. MS anchor: mask an amino acid in a
    peptide (or a peak in a spectrum)."""
    fig, ax = plt.subplots(figsize=(11.6, 4.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.text(6.0, 4.68, "masking: hide a piece, predict it from context  (BERT)",
            ha="center", color=INK, fontsize=14, fontweight="bold")
    toks = ["A", "C", "[MASK]", "E", "F", "G"]
    w = 1.5
    gap = 0.28
    x0 = 6.0 - (len(toks) * w + (len(toks) - 1) * gap) / 2
    mask_cx = None
    for i, t in enumerate(toks):
        cx = x0 + i * (w + gap)
        masked = t == "[MASK]"
        _chip(ax, cx, 3.1, w, 0.8, t, RED if masked else TEAL_SOFT,
              txt=WHITE if masked else INK, fs=15,
              edge=RED if masked else TEAL, lw=2.0)
        if masked:
            mask_cx = cx + w / 2
    ax.text(6.0, 3.95, "peptide  A\u2013C\u2013?\u2013E\u2013F\u2013G", ha="center",
            color=MUTED, fontsize=11, style="italic")
    # prediction
    ax.annotate("", xy=(mask_cx, 1.55), xytext=(mask_cx, 2.6),
                arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, lw=2.0))
    ax.add_patch(FancyBboxPatch((mask_cx - 1.3, 0.7), 2.6, 0.85,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=TEAL_SOFT, edgecolor=TEAL, lw=2.2))
    ax.text(mask_cx, 1.12, "predict \u2192  D", ha="center", va="center",
            color=TEAL, fontsize=14, fontweight="bold")
    ax.text(6.0, 0.28,
            "no human labels \u2014 the hidden token is its own answer (mask a residue, or a peak)",
            ha="center", color=MUTED, fontsize=11, style="italic")
    _save(fig, name)


def contrastive_views(name="fig_contrastive.png"):
    """Contrastive learning: two augmented VIEWS of the SAME spectrum should land
    together in embedding space (attract), while a different spectrum is pushed
    away (repel). The label comes from 'same source or not' \u2014 no human labels."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.8, 4.6),
                                   gridspec_kw={"width_ratios": [1.05, 1]})
    base = [(2.2, 1.0), (4.5, 0.55), (6.8, 0.8), (8.4, 0.35)]
    v1 = [(c, h * (0.85 + 0.1)) for c, h in base]  # + noise (heights jittered)
    v2 = [(c + 0.5, h) for c, h in base]           # m/z shift
    axl.set_xlim(0, 10)
    axl.set_ylim(-0.2, 3.4)
    axl.axis("off")
    axl.text(5.0, 3.25, "one spectrum \u2192 two augmented views", ha="center",
             color=INK, fontsize=12.5, fontweight="bold")
    ax1 = fig.add_axes([0.06, 0.50, 0.20, 0.28])
    _spectrum_curve(ax1, v1, color=TEAL)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 1.25)
    ax1.axis("off")
    ax1.set_title("view 1  (+ noise)", fontsize=10, color=TEAL)
    ax2 = fig.add_axes([0.06, 0.13, 0.20, 0.28])
    _spectrum_curve(ax2, v2, color=TEAL)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 1.25)
    ax2.axis("off")
    ax2.set_title("view 2  (m/z shift)", fontsize=10, color=TEAL)
    # right: embedding space
    axr.set_xlim(0, 10)
    axr.set_ylim(0, 10)
    axr.set_xticks([])
    axr.set_yticks([])
    for s in axr.spines.values():
        s.set_edgecolor(HAIRLINE)
    axr.set_title("embedding space", fontsize=12, color=INK, fontweight="bold")
    p1 = (4.2, 6.6)
    p2 = (5.6, 5.6)
    other = (8.4, 2.2)
    axr.plot(*p1, marker="o", ms=15, color=TEAL, zorder=4)
    axr.plot(*p2, marker="o", ms=15, color=TEAL, zorder=4)
    axr.text(p1[0] - 1.3, p1[1], "view 1", ha="right", va="center", color=TEAL,
             fontsize=10, fontweight="bold")
    axr.text(p2[0] - 1.3, p2[1], "view 2", ha="right", va="center", color=TEAL,
             fontsize=10, fontweight="bold")
    axr.annotate("", xy=p2, xytext=p1,
                 arrowprops=dict(arrowstyle="<|-|>", color=TEAL, lw=2.2))
    axr.text((p1[0] + p2[0]) / 2 + 0.7, (p1[1] + p2[1]) / 2 + 0.3, "pull\ntogether",
             ha="left", color=TEAL, fontsize=10.5, fontweight="bold")
    axr.plot(*other, marker="o", ms=15, color=RED, zorder=4)
    axr.text(other[0], other[1] - 1.2, "a different\nspectrum", ha="center",
             color=RED, fontsize=10, fontweight="bold")
    axr.annotate("", xy=(7.4, 3.0), xytext=(5.9, 4.9),
                 arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.0,
                                 ls="--"))
    axr.text(7.2, 4.4, "push\napart", ha="left", color=RED, fontsize=10.5,
             fontweight="bold")
    fig.text(0.5, 0.02,
             "same source \u2192 same point; the augmentation IS the label",
             ha="center", color=MUTED, fontsize=10.5, style="italic")
    _save(fig, name)


def dino_student_teacher(name="fig_dino.png"):
    """DINO: student\u2013teacher self-distillation. Two augmented views of one input
    go to a STUDENT and a TEACHER network; the student is trained so its output
    MATCHES the teacher's, and the teacher is a slow moving average of the
    student \u2014 all with NO labels. The famous payoff: attention learns to
    segment the object on its own."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.8, 4.6),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    axl.set_xlim(0, 12)
    axl.set_ylim(0, 6)
    axl.axis("off")
    axl.text(6.0, 5.7, "student\u2013teacher self-distillation  (no labels)",
             ha="center", color=INK, fontsize=13, fontweight="bold")
    # input + two views
    ax_in = fig.add_axes([0.045, 0.42, 0.13, 0.22])
    _spectrum_curve(ax_in, [(2.5, 1.0), (5.5, 0.6), (7.8, 0.8)], color=TEAL)
    ax_in.set_xlim(0, 10)
    ax_in.set_ylim(0, 1.3)
    ax_in.axis("off")
    ax_in.set_title("one input", fontsize=9.5, color=INK_SOFT)
    # student / teacher boxes
    axl.add_patch(FancyBboxPatch((4.0, 3.3), 3.0, 1.3,
                  boxstyle="round,pad=0.02,rounding_size=0.06",
                  facecolor=TEAL_SOFT, edgecolor=TEAL, lw=2.4))
    axl.text(5.5, 3.95, "STUDENT", ha="center", va="center", color=TEAL,
             fontsize=13, fontweight="bold")
    axl.add_patch(FancyBboxPatch((4.0, 1.1), 3.0, 1.3,
                  boxstyle="round,pad=0.02,rounding_size=0.06",
                  facecolor=AMBER_SOFT, edgecolor=AMBER, lw=2.4))
    axl.text(5.5, 1.75, "TEACHER", ha="center", va="center", color=ROI_INK,
             fontsize=13, fontweight="bold")
    axl.annotate("", xy=(4.0, 3.95), xytext=(2.4, 3.7),
                 arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, lw=1.8))
    axl.annotate("", xy=(4.0, 1.75), xytext=(2.4, 2.0),
                 arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, lw=1.8))
    axl.text(2.9, 4.05, "view 1", ha="center", color=INK_SOFT, fontsize=9.5)
    axl.text(2.9, 1.65, "view 2", ha="center", color=INK_SOFT, fontsize=9.5)
    # match outputs (widened so the bold label clears both inner edges; the
    # box is centred on the text at x=9.7, arrow still lands on the left edge)
    axl.add_patch(FancyBboxPatch((7.6, 2.35), 4.2, 1.0,
                  boxstyle="round,pad=0.02,rounding_size=0.06",
                  facecolor=WHITE, edgecolor=INK_SOFT, lw=2.0))
    axl.text(9.7, 2.85, "make outputs MATCH", ha="center", va="center",
             color=INK, fontsize=11.5, fontweight="bold")
    axl.annotate("", xy=(7.6, 2.95), xytext=(6.9, 3.7),
                 arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.8))
    axl.annotate("", xy=(8.0, 2.75), xytext=(7.0, 1.75),
                 arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.8))
    # EMA arrow points STUDENT -> TEACHER: the teacher's weights are a slow
    # moving average of the student's, so the update flows down into the teacher.
    axl.annotate("", xy=(5.5, 2.4), xytext=(5.5, 3.3),
                 arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6,
                                 ls="--"))
    axl.text(6.4, 2.85, "EMA", ha="center", va="center", color=MUTED,
             fontsize=9, style="italic")
    axl.text(5.5, 0.35,
             "teacher = a slow moving average of the student", ha="center",
             va="center", color=MUTED, fontsize=9.5, style="italic")
    # right: unsupervised attention segmentation schematic
    axr.set_xlim(0, 10)
    axr.set_ylim(0, 10)
    axr.set_xticks([])
    axr.set_yticks([])
    for s in axr.spines.values():
        s.set_edgecolor(HAIRLINE)
    axr.set_title("attention finds the object\n(unsupervised segmentation)",
                  fontsize=11, color=INK, fontweight="bold")
    # background grid (the 'image')
    axr.add_patch(FancyBboxPatch((0.6, 0.8), 8.8, 7.8,
                  boxstyle="round,pad=0.02,rounding_size=0.04",
                  facecolor="#EFEFEA", edgecolor=HAIRLINE, lw=1.0))
    # the object, highlighted by amber attention
    axr.add_patch(FancyBboxPatch((3.2, 2.6), 3.8, 4.4,
                  boxstyle="round,pad=0.02,rounding_size=0.6",
                  facecolor=AMBER, edgecolor=ROI_INK, lw=2.2, alpha=0.75))
    axr.text(5.1, 4.8, "object", ha="center", va="center", color=WHITE,
             fontsize=12, fontweight="bold")
    axr.text(5.0, 1.5, "no labels, no boxes \u2014 learned", ha="center",
             color=MUTED, fontsize=9.5, style="italic")
    axr.text(5.0, 0.4, "(illustrative schematic)", ha="center",
             color=MUTED, fontsize=8.5, style="italic")
    fig.subplots_adjust(wspace=0.12)
    _save(fig, name)


def pseudo_labeling(name="fig_pseudolabel.png"):
    """Semi-supervised pseudo-labelling, the '10,000 unlabeled + 100 labeled'
    middle path (named, not derived): train on the few labels, predict on the
    many unlabeled, keep the confident guesses as pseudo-labels, retrain on
    both."""
    fig, ax = plt.subplots(figsize=(12.0, 4.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.text(6.0, 4.7, "semi-supervised: 100 labeled + 10,000 unlabeled",
            ha="center", color=INK, fontsize=14, fontweight="bold")
    boxes = [
        (1.25, TEAL, "100 labeled\nspectra", "the few you have"),
        (3.6, TEAL, "train a first\nmodel", "on the 100"),
        (5.95, AMBER, "predict on\n10,000 unlabeled", "guess their labels"),
        (8.3, AMBER, "keep CONFIDENT\nguesses", "= pseudo-labels"),
        (10.65, TEAL, "retrain on\n100 + pseudo", "far more data"),
    ]
    w = 1.95
    for i, (x, col, head, sub) in enumerate(boxes):
        cx = x
        ax.add_patch(FancyBboxPatch((cx - w / 2, 1.7), w, 1.6,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor=WHITE, edgecolor=col, lw=2.4))
        ax.text(cx, 2.78, head, ha="center", va="center", color=col,
                fontsize=10.5, fontweight="bold")
        ax.text(cx, 2.05, sub, ha="center", va="center", color=MUTED,
                fontsize=8.5, style="italic")
        if i < len(boxes) - 1:
            nx = boxes[i + 1][0]
            ax.annotate("", xy=(nx - w / 2 - 0.04, 2.5),
                        xytext=(cx + w / 2 + 0.04, 2.5),
                        arrowprops=dict(arrowstyle="-|>", color=INK_SOFT,
                                        lw=1.8))
    # loop-back arrow from retrain to predict, routed BELOW the boxes (negative
    # rad dips it down) so it never strikes through the box text.
    ax.annotate("", xy=(5.95, 1.62), xytext=(10.65, 1.62),
                arrowprops=dict(arrowstyle="-|>", color=ROI_INK, lw=1.6,
                                connectionstyle="arc3,rad=-0.22", ls="--"))
    ax.text(8.3, 0.5, "repeat \u2014 each round labels more", ha="center",
            color=ROI_INK, fontsize=10, style="italic")
    _save(fig, name)


def paradigm_chart(mode="ido", name="fig_paradigm_chart.png"):
    """The four learning paradigms as a decision/matching chart (rules 3/8): the
    single source the Quiz 11 Q2 four-way match is answerable from. Each row =
    paradigm | what you have | what it does | an MS example. mode='ido' is the
    teaching chart; the same PNG is reused for the you-do so the room reads the
    worksheet scenario straight off it."""
    rows = [
        ("SUPERVISED", TEAL, "many LABELED examples",
         "learn input \u2192 label", "labeled spectra \u2192 R/S classifier"),
        ("UNSUPERVISED", AMBER, "NO labels",
         "find structure / score anomalies",
         "autoencoder flags bad QC runs"),
        ("SELF-SUPERVISED", RED, "no labels \u2014 data labels itself",
         "pretext task pretrains a body",
         "mask peaks & predict, then fine-tune"),
        ("SEMI-SUPERVISED", INK_SOFT, "FEW labeled + MANY unlabeled",
         "pseudo-label the rest, retrain",
         "100 labeled + 10,000 unlabeled"),
    ]
    fig, ax = plt.subplots(figsize=(12.6, 6.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")
    header = ("match the data you have to a paradigm"
              if mode == "ido" else
              "your turn: read each quiz scenario's row off the chart")
    hcol = INK if mode == "ido" else ROI_INK
    ax.text(6.5, 5.72, header, ha="center", va="center", fontsize=15,
            color=hcol, fontweight="bold")
    ax.text(1.75, 5.15, "paradigm", ha="center", color=MUTED, fontsize=12)
    ax.text(5.15, 5.15, "what you have", ha="center", color=MUTED, fontsize=12)
    ax.text(8.15, 5.15, "what it does", ha="center", color=MUTED, fontsize=12)
    ax.text(11.3, 5.15, "MS example", ha="center", color=MUTED, fontsize=12)
    ys = [4.4, 3.35, 2.3, 1.25]
    for (name_, col, have, does, ex), y in zip(rows, ys):
        ax.add_patch(FancyBboxPatch((0.25, y - 0.45), 3.0, 0.9,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor=col, edgecolor=col, lw=2.0))
        ax.text(1.75, y, name_, ha="center", va="center", fontsize=11.5,
                color=WHITE, fontweight="bold")
        ax.add_patch(FancyBboxPatch((3.55, y - 0.45), 3.2, 0.9,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor=WHITE, edgecolor=col, lw=1.8))
        ax.text(5.15, y, have, ha="center", va="center", fontsize=10,
                color=INK, fontweight="bold")
        ax.add_patch(FancyBboxPatch((6.75, y - 0.45), 2.8, 0.9,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor="#F1F1EC", edgecolor=col, lw=1.8))
        ax.text(8.15, y, does, ha="center", va="center", fontsize=9.5,
                color=INK_SOFT)
        ax.add_patch(FancyBboxPatch((9.6, y - 0.45), 3.15, 0.9,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor=WHITE, edgecolor=col, lw=1.8))
        ax.text(11.18, y, ex, ha="center", va="center", fontsize=9,
                color=INK, style="italic")
    _save(fig, name)


def recon_aggregation_trap(name="fig_recon_trap_youdo.png"):
    """Quiz 11 Q3 (upgraded): the aggregation trap. A contaminant spikes ONE m/z
    bin out of 20; that bin's squared error is 0.40, all others 0. Left = the
    20-bin spectrum vs. its reconstruction with the single big gap flagged; right
    = compute the MEAN MSE (blank) and decide against threshold 0.05, then reason
    which single number catches a localized spike. Answer (key/notes):
    MSE = 0.40/20 = 0.02 < 0.05 -> the MEAN PASSES it (misses the contaminant);
    max per-bin error 0.40 > 0.05 -> FLAG."""
    n = 20
    meas = np.array([0.34, 0.52, 0.40, 0.58, 0.36, 0.50, 0.44, 0.31, 0.54, 0.42,
                     0.48, 0.90, 0.35, 0.51, 0.45, 0.60, 0.39, 0.33, 0.55, 0.46])
    recon = meas.copy()
    spike = 11
    recon[spike] = 0.27
    fig, (axb, axt) = plt.subplots(1, 2, figsize=(12.2, 4.7),
                                   gridspec_kw={"width_ratios": [1.25, 1.0]})
    idx = np.arange(n)
    axb.bar(idx, meas, width=0.82, color=TEAL, zorder=3, label="x  (measured)")
    axb.step(idx, recon, where="mid", color=AMBER, lw=2.4, zorder=4,
             label="x\u0302  (reconstruction)")
    axb.annotate("", xy=(spike, meas[spike]), xytext=(spike, recon[spike]),
                 arrowprops=dict(arrowstyle="<->", color=RED, lw=2.2))
    axb.text(spike + 0.5, 1.18,
             "contaminant:  (x\u1d62\u2212x\u0302\u1d62)\u00b2 = 0.40",
             ha="left", va="center", color=RED, fontsize=10.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", fc="#F7E4E3", ec=RED, lw=1.6))
    axb.set_xlim(-0.7, n - 0.3)
    axb.set_ylim(0, 1.42)
    axb.set_xticks([])
    axb.set_yticks([])
    for s in ("top", "right", "left"):
        axb.spines[s].set_visible(False)
    axb.set_xlabel("20 m/z bins  \u2014  19 rebuilt perfectly, 1 contaminant",
                   fontsize=11, color=MUTED)
    axb.legend(loc="upper left", frameon=False, fontsize=10.5)
    axt.set_xlim(0, 1)
    axt.set_ylim(0, 1)
    axt.axis("off")
    axt.text(0.0, 0.95, "error in 19 bins = 0", fontsize=12.5, color=INK_SOFT)
    axt.text(0.0, 0.85, "error in 1 bin    = 0.40", fontsize=12.5, color=RED,
             fontweight="bold")
    axt.text(0.0, 0.66, "MEAN over all 20 bins", fontsize=12.5, color=INK,
             fontweight="bold")
    axt.text(0.0, 0.55, "MSE = 0.40 / 20 = ?", fontsize=15, color=INK,
             fontweight="bold")
    axt.text(0.0, 0.40, "threshold: flag if error > 0.05", fontsize=12, color=RED)
    axt.text(0.49, 0.28, "mean says:  ?   (pass or flag)", ha="center",
             va="center", color=ROI_INK, fontsize=12, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", fc=AMBER_SOFT, ec=ROI_INK, lw=2.0))
    axt.text(0.49, 0.075,
             "Does the MEAN catch a 1-bin spike?\nIf not \u2014 which single number would?",
             ha="center", va="center", color=TEAL, fontsize=10.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", fc=TEAL_SOFT, ec=TEAL, lw=1.6))
    fig.subplots_adjust(wspace=0.18)
    _save(fig, name)


def semisup_scenario(name="fig_semisup_youdo.png"):
    """Quiz 11 Q2 (upgraded): one messy, realistic lab scenario instead of a 4-way
    lookup. 300 labeled spectra + 40,000 unlabeled QC runs -> decide the paradigm
    that uses ALL the data, name the technique, and name the risk. Answer
    (key/notes): SEMI-SUPERVISED / pseudo-labeling / confident-but-wrong
    pseudo-labels reinforce the model's own mistakes."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.2, 4.6),
                                   gridspec_kw={"width_ratios": [1.0, 1.15]})
    axl.set_xlim(0, 1)
    axl.set_ylim(0, 1)
    axl.axis("off")
    axl.text(0.5, 0.95, "Your data", ha="center", color=INK, fontsize=13,
             fontweight="bold")
    axl.text(0.5, 0.74, "300 labeled spectra  (S / R known)", ha="center",
             va="center", color=WHITE, fontsize=11.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.6", fc=TEAL, ec=TEAL))
    axl.text(0.5, 0.40, "40,000 unlabeled QC runs", ha="center", va="center",
             color=INK, fontsize=14, fontweight="bold",
             bbox=dict(boxstyle="round,pad=1.5", fc="#EDEDE7", ec=MUTED, lw=1.8))
    axl.text(0.5, 0.05, "labels are < 1% of your data", ha="center", color=MUTED,
             fontsize=11, style="italic")
    axr.set_xlim(0, 1)
    axr.set_ylim(0, 1)
    axr.axis("off")
    axr.text(0.0, 0.95, "Use EVERY spectrum \u2014 decide", fontsize=13,
             color=ROI_INK, fontweight="bold")
    for y, txt in [(0.72, "(a) which paradigm uses ALL of it?"),
                   (0.47, "(b) the specific technique?"),
                   (0.22, "(c) the failure mode to watch?")]:
        axr.text(0.0, y, txt, fontsize=12.5, color=INK, fontweight="bold")
        axr.text(0.06, y - 0.10, "\u2192  " + "_" * 26, fontsize=12, color=MUTED)
    _save(fig, name)


def _lecture11_figures():
    autoencoder_bottleneck()
    denoising_autoencoder()
    denoising_results()
    anomaly_overlay()
    recon_error_worked("ido", "fig_recon_error_ido.png",
                       x=(0.2, 0.5, 0.1), xhat=(0.2, 0.4, 0.1), thr=0.01)
    # (was a trivial 3-bin MSE plug-in; upgraded to the aggregation trap below)
    #  the aggregation trap: one contaminant bin (sq err 0.40) among 20 -> mean
    #  MSE 0.40/20 = 0.02 < 0.05 PASSES, but max per-bin error 0.40 > 0.05 FLAGS
    recon_aggregation_trap()
    ae_vs_vae_latent(("autoencoder latent", "VAE latent"),
                     "fig_ae_vs_vae_latent.png")
    ae_vs_vae_latent(("Picture A", "Picture B"), "fig_latent_ab_quiz.png",
                     describe=False)
    vae_recipe()
    latent_interpolation()
    masking_pretext()
    contrastive_views()
    dino_student_teacher()
    pseudo_labeling()
    paradigm_chart("ido", "fig_paradigm_chart.png")
    # you-do = Quiz 11 Q2 (upgraded): one messy scenario -> semi-supervised
    semisup_scenario()


# ============================================================================
#  Lecture 13 · Large Language Models + the DL-in-MS Landscape
#  A survey hour, so the ONE numeric mechanic (rule 6) is a light next-token
#  SOFTMAX do-it-together (reuse of Lecture 4/7's softmax): three candidate
#  token logits → probabilities → argmax. Everything else is a bespoke,
#  MS-anchored schematic (rule 9): the 3-stage LLM training pipeline, the RL
#  agent–environment–reward loop, RLHF-as-RL, the prompting/RAG/fine-tuning
#  decision chart, and per-tool landscape cards (problem → architecture you now
#  know → impact). The two flagship photos (AlphaGo board, AlphaFold render)
#  are REAL licensed images placed directly in the deck (rule 7), credited in
#  their figcaptions — not redrawn here.
# ============================================================================

# Lecture 13's LLM section runs on REAL NATURAL LANGUAGE (instructor,
# 2026-09-10): an LLM is a model of TEXT, so the next-token demo uses an
# ordinary English lab sentence — "The internal standard was spiked into
# each ___" — instead of an amino-acid chain. Same MS anchor (it is a
# sentence from a lab report), but the mechanic is the real thing rather
# than an analogy. Same clause is reused by the softmax do-it-together, so
# the two slides tell one continuous story.
NEXT_TOKEN_SENTENCE = "The internal standard was spiked into each ___"
NEXT_TOKEN_CONTEXT = ["…", "was", "spiked", "into", "each", "?"]
NEXT_TOKEN_CANDS = [("sample", 0.55), ("vial", 0.20), ("tube", 0.15),
                    ("others", 0.10)]
# The do-it-together keeps logits 2 / 1 / 0 → 0.665 / 0.245 / 0.090.
SOFTMAX_TOKENS = ["sample", "vial", "tube"]


def next_token_predict(name="fig_next_token.png"):
    """An LLM predicts the NEXT token: a context of tokens in, a probability over
    the vocabulary out. The context is a real English sentence from a lab report
    ("The internal standard was spiked into each ___") with a bar chart of
    candidate next WORDS; the argmax is the prediction."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.8, 4.2),
                                   gridspec_kw={"width_ratios": [1.05, 1]})
    axl.set_xlim(0, 10)
    axl.set_ylim(0, 6)
    axl.axis("off")
    axl.text(5.0, 5.5, "predict the next token from the context", ha="center",
             color=INK, fontsize=13.5, fontweight="bold")
    toks = NEXT_TOKEN_CONTEXT
    w = 1.58
    gap = 0.12
    x0 = 5.0 - (len(toks) * w + (len(toks) - 1) * gap) / 2
    qcx = None
    for i, t in enumerate(toks):
        cx = x0 + i * (w + gap)
        unknown = t == "?"
        _chip(axl, cx, 3.4, w, 0.85, t, AMBER_SOFT if unknown else TEAL_SOFT,
              txt=ROI_INK if unknown else INK, fs=17 if unknown else 12.5,
              edge=AMBER if unknown else TEAL, lw=2.0)
        if unknown:
            qcx = cx + w / 2
    axl.text(5.0, 4.3, f'text so far:  "{NEXT_TOKEN_SENTENCE}"', ha="center",
             color=MUTED, fontsize=11, style="italic")
    axl.annotate("", xy=(qcx, 1.7), xytext=(qcx, 2.95),
                 arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, lw=2.0))
    axl.text(5.0, 1.2,
             "the model scores EVERY word in the vocabulary → a probability",
             ha="center", color=INK_SOFT, fontsize=11)
    # right: probability bars over candidate next words
    cands = NEXT_TOKEN_CANDS
    ypos = np.arange(len(cands))[::-1]
    for (lab, p), yy in zip(cands, ypos):
        top = lab == "sample"
        axr.barh(yy, p, color=TEAL if top else "#CFE3E2", height=0.62,
                 zorder=3)
        axr.text(p + 0.02, yy, f"{p:.2f}", va="center", ha="left",
                 color=INK if top else MUTED, fontsize=12,
                 fontweight="bold" if top else "normal")
        axr.text(-0.03, yy, lab, va="center", ha="right", color=INK,
                 fontsize=12, fontweight="bold" if top else "normal")
    axr.set_xlim(-0.28, 0.72)
    axr.set_ylim(-0.6, len(cands) - 0.4)
    axr.set_xticks([])
    axr.set_yticks([])
    for s in axr.spines.values():
        s.set_visible(False)
    axr.set_title("P(next word)", fontsize=12, color=INK, fontweight="bold")
    axr.text(0.5, -0.12, 'pick the argmax → "sample"', transform=axr.transAxes,
             ha="center", color=TEAL, fontsize=12, fontweight="bold")
    fig.subplots_adjust(wspace=0.16, bottom=0.14)
    _save(fig, name)


def softmax_next_token(name="fig_softmax_next.png"):
    """The one numeric beat (rule 6), a do-it-together: three candidate next
    WORDS carry logits (2, 1, 0); softmax turns them into probabilities that
    sum to 1, and the argmax is the predicted next word. Formula shown on the
    figure (softmax was taught in Lecture 4 and used in Lecture 7) so it is
    self-contained."""
    tokens = SOFTMAX_TOKENS
    logits = [2, 1, 0]
    exps = ["e²=7.39", "e¹=2.72", "e⁰=1.00"]
    probs = [0.665, 0.245, 0.090]
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.8, 4.6),
                                   gridspec_kw={"width_ratios": [1, 1.15]})
    # left: candidate chips with logits, then a probability bar chart
    axl.set_xlim(0, 10)
    axl.set_ylim(0, 10)
    axl.axis("off")
    axl.text(5.0, 9.4, '"… spiked into each ___" — 3 candidates, with logits',
             ha="center", color=INK, fontsize=12, fontweight="bold")
    for i, (t, lg) in enumerate(zip(tokens, logits)):
        cx = 1.6 + i * 3.0
        top = i == 0
        _chip(axl, cx - 1.05, 8.1, 2.1, 0.8, t,
              TEAL_SOFT if top else "#EFEFEA", txt=INK, fs=13,
              edge=TEAL if top else MUTED, lw=2.2 if top else 1.4)
        axl.text(cx, 7.15, f"logit = {lg}", ha="center", color=INK_SOFT,
                 fontsize=12, fontweight="bold")
    # probability bars
    for i, (t, p) in enumerate(zip(tokens, probs)):
        cx = 1.6 + i * 3.0
        top = i == 0
        bh = p * 5.0
        axl.add_patch(FancyBboxPatch((cx - 0.75, 1.2), 1.5, bh,
                      boxstyle="square,pad=0", facecolor=TEAL if top else "#CFE3E2",
                      edgecolor="none", zorder=3))
        axl.text(cx, 1.2 + bh + 0.25, f"{p:.3f}", ha="center", color=INK,
                 fontsize=12.5, fontweight="bold" if top else "normal")
        axl.text(cx, 0.7, t, ha="center", color=INK, fontsize=12,
                 fontweight="bold" if top else "normal")
    axl.text(5.0, 6.15, "softmax", ha="center", color=TEAL, fontsize=13,
             fontweight="bold")
    axl.annotate("", xy=(5.0, 5.4), xytext=(5.0, 5.95),
                 arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2.2))
    # right: the worked computation
    axr.set_xlim(0, 1)
    axr.set_ylim(0, 1)
    axr.axis("off")
    axr.text(0.0, 0.95, "softmax(z)ᵢ = e^{zᵢ} / Σⱼ e^{zⱼ}", fontsize=15,
             color=INK, fontweight="bold")
    axr.text(0.0, 0.83, "turns logits into probabilities that sum to 1",
             fontsize=11, color=MUTED, style="italic")
    cols_x = [0.02, 0.34, 0.66]
    hdr = ["word", "e^{logit}", "probability"]
    y0 = 0.66
    for cx, h in zip(cols_x, hdr):
        axr.text(cx, y0, h, fontsize=11.5, color=MUTED, fontweight="bold")
    for i, (t, e, p) in enumerate(zip(tokens, exps, probs)):
        yy = y0 - 0.12 * (i + 1)
        top = i == 0
        axr.text(cols_x[0], yy, t, fontsize=12.5, color=INK,
                 fontweight="bold")
        axr.text(cols_x[1], yy, e, fontsize=12, color=INK_SOFT)
        axr.text(cols_x[2], yy, f"{p:.3f}", fontsize=12.5,
                 color=TEAL if top else INK,
                 fontweight="bold" if top else "normal")
    axr.plot([0.0, 0.99], [0.235, 0.235], color=HAIRLINE, lw=1.2)
    axr.text(0.0, 0.15, "sum = 7.39 + 2.72 + 1.00 = 11.11", fontsize=11.5,
             color=INK_SOFT)
    axr.add_patch(FancyBboxPatch((0.0, 0.0), 0.99, 0.10,
                  boxstyle="round,pad=0.01,rounding_size=0.03",
                  facecolor=TEAL_SOFT, edgecolor=TEAL, lw=2.0))
    axr.text(0.495, 0.05, 'argmax → predicted next word = "sample"',
             ha="center", va="center", color=TEAL, fontsize=12.5,
             fontweight="bold")
    fig.subplots_adjust(wspace=0.2)
    _save(fig, name)


def llm_pipeline(mode="intro", name="fig_llm_pipeline.png"):
    """The 3-stage LLM training pipeline, each stage in one sentence:
    next-token PRETRAINING → supervised fine-tuning (SFT) → preference tuning
    (RLHF). mode='intro' introduces all three; mode='rlhf' highlights the RLHF
    stage and tags its reward as a human preference (completed after the RL
    section)."""
    fig, ax = plt.subplots(figsize=(12.2, 4.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    title = ("how an LLM is trained — three stages" if mode == "intro"
             else "…and RLHF is the third stage: reward = human preference")
    ax.text(6.0, 4.72, title, ha="center", color=INK, fontsize=14,
            fontweight="bold")
    stages = [
        ("1. PRETRAINING", TEAL,
         "predict the next token\non a huge text corpus",
         "learns language & facts"),
        ("2. SUPERVISED\nFINE-TUNING (SFT)", AMBER,
         "train on curated\ninstruction → answer pairs",
         "learns to follow instructions"),
        ("3. PREFERENCE\nTUNING (RLHF)", RED,
         "humans rank answers;\ntune toward the preferred",
         "learns to be helpful & safe"),
    ]
    w = 3.5
    xs = [0.35, 4.25, 8.15]
    for i, ((head, col, body, foot), x) in enumerate(zip(stages, xs)):
        hot = (mode == "rlhf" and i == 2)
        ax.add_patch(FancyBboxPatch((x, 1.0), w, 2.9,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor="#F7E4E3" if hot else WHITE,
                    edgecolor=col, lw=3.2 if hot else 2.4))
        ax.text(x + w / 2, 3.45, head, ha="center", va="center", color=col,
                fontsize=12.5, fontweight="bold")
        ax.text(x + w / 2, 2.5, body, ha="center", va="center",
                color=INK, fontsize=11)
        ax.text(x + w / 2, 1.45, foot, ha="center", va="center",
                color=MUTED, fontsize=10, style="italic")
        if i < len(stages) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.05, 2.45),
                        xytext=(x + w + 0.05, 2.45),
                        arrowprops=dict(arrowstyle="-|>", color=INK_SOFT,
                                        lw=2.0))
    if mode == "rlhf":
        ax.text(8.15 + w / 2, 0.55,
                "the 'reward' comes from human preferences — that's RL",
                ha="center", color=RED, fontsize=10.5, fontweight="bold")
    else:
        ax.text(6.0, 0.45,
                "base model → instruction-follower → helpful assistant",
                ha="center", color=MUTED, fontsize=10.5, style="italic")
    _save(fig, name)


def llm_scale(name="fig_llm_scale.png"):
    """Scale is the engine: more data + more parameters + more compute → more
    capability. Three escalating model bars with rough parameter counts."""
    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(3.4, 5.6, "scale is the engine", ha="center", color=INK,
            fontsize=14, fontweight="bold")
    bars = [
        ("small\n~100M params", 1.3, "#CFE3E2", "grammar, short answers"),
        ("medium\n~10B params", 2.7, "#7FBFBE", "reasoning, coding"),
        ("large\n~1T params", 4.1, TEAL, "broad, emergent skills"),
    ]
    for i, (lab, h, col, cap) in enumerate(bars):
        cx = 1.3 + i * 2.0
        ax.add_patch(FancyBboxPatch((cx - 0.7, 0.8), 1.4, h,
                    boxstyle="round,pad=0.02,rounding_size=0.05",
                    facecolor=col, edgecolor="none", zorder=3))
        ax.text(cx, 0.8 + h + 0.28, lab, ha="center", va="bottom", color=INK,
                fontsize=10.5, fontweight="bold")
        ax.text(cx, 0.45, cap, ha="center", va="top", color=MUTED,
                fontsize=9, style="italic")
    ax.annotate("", xy=(6.1, 0.8), xytext=(0.4, 0.8),
                arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, lw=2.0))
    ax.text(10.7, 3.0, "more data +\nparameters +\ncompute\n→ more capability",
            ha="right", va="center", color=INK_SOFT, fontsize=12,
            fontweight="bold")
    _save(fig, name)


def hallucination(mode="ido", name="fig_hallucination_ido.png"):
    """Hallucination as a failure mode to design AROUND: the model answers
    fluently and CONFIDENTLY even when it is wrong — here inventing a citation.
    mode='ido' shows the guardrail; mode='youdo' leaves the guardrail blank for
    Quiz 13 Q3."""
    blank = mode == "youdo"
    fig, ax = plt.subplots(figsize=(11.6, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(6.0, 5.7, "hallucination: fluent, confident — and sometimes wrong",
            ha="center", color=INK, fontsize=13.5, fontweight="bold")
    # user prompt bubble
    ax.add_patch(FancyBboxPatch((0.6, 4.05), 6.4, 1.0,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                facecolor=TEAL_SOFT, edgecolor=TEAL, lw=2.0))
    # you-do uses a DIFFERENT same-shape fabrication than the I-do (transfer,
    # not recall); Quiz 13 Q4 quotes the you-do reply verbatim.
    if blank:
        _prompt = "“What's the upper reference limit for this analyte?”"
        _reply = "“It's 48 nmol/L — see Nguyen et al., Clin. Chem. 2021; 67:1043–1051.”"
        _note = "(fluent and confident — but is any of it grounded in a source?)"
    else:
        _prompt = "“Cite a reference for this method's LOD.”"
        _reply = "“See Smith et al., J. Clin. Mass Spectrom. 2019; 14:221–230.”"
        _note = "✗ that paper does not exist — invented, but plausible"
    ax.text(0.85, 4.55, _prompt,
            ha="left", va="center", color=INK, fontsize=12)
    # model reply bubble (confident but fabricated)
    ax.add_patch(FancyBboxPatch((2.4, 2.35), 8.9, 1.35,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                facecolor="#F7E4E3", edgecolor=RED, lw=2.2))
    ax.text(2.7, 3.25, _reply,
            ha="left", va="center", color=INK, fontsize=11.5)
    ax.text(2.7, 2.7, _note,
            ha="left", va="center", color=(MUTED if blank else RED),
            fontsize=11, fontweight=("normal" if blank else "bold"),
            style=("italic" if blank else "normal"))
    # bottom box: I-do shows the guardrail; you-do asks the room to SPOT + reason
    if blank:
        ax.add_patch(FancyBboxPatch((0.6, 0.35), 10.8, 1.55,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor=AMBER_SOFT, edgecolor=ROI_INK, lw=2.4))
        ax.text(6.2, 1.45,
                "(a) which parts can you NOT trust without checking a source?",
                ha="center", va="center", color=ROI_INK, fontsize=11.5,
                fontweight="bold")
        ax.text(6.2, 0.8,
                "(b) why did the model state them so confidently?",
                ha="center", va="center", color=ROI_INK, fontsize=11.5,
                fontweight="bold")
    else:
        ax.add_patch(FancyBboxPatch((0.6, 0.5), 10.8, 1.35,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor=TEAL_SOFT, edgecolor=TEAL, lw=2.4))
        ax.text(1.0, 1.5, "GUARDRAIL", ha="left", va="center", color=TEAL,
                fontsize=11, fontweight="bold")
        ax.text(6.2, 1.15,
                "ground answers in RAG'd source documents; require a human to",
                ha="center", va="center", color=INK, fontsize=11.5)
        ax.text(6.2, 0.75,
                "verify every citation/number before it reaches a report",
                ha="center", va="center", color=INK, fontsize=11.5)
    _save(fig, name)


def rl_loop(mode="ido", name="fig_rl_loop_ido.png"):
    """The reinforcement-learning loop in ONE diagram: an AGENT takes an ACTION
    in an ENVIRONMENT, which returns a REWARD and a new state; the agent learns
    by trial and feedback to earn more reward. mode='ido' labels the generic
    loop; mode='youdo' is a BLANK template the room fills in for a problem of
    their own — the open-ended Quiz 13 Q1."""
    blank = mode == "youdo"
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis("off")
    if blank:
        ax.text(5.5, 5.72,
                "your turn: formulate something from YOUR lab as RL",
                ha="center", color=ROI_INK, fontsize=13.5, fontweight="bold")
        # A blank template, not a pre-filled scenario: the room supplies every
        # role for a problem of their own (Quiz 13 Q1, open-ended).
        agent_lab, env_lab = "AGENT = ?", "ENVIRONMENT = ?"
        agent_sub = "who or what decides"
        env_sub = "what it acts on"
        act_lab, rew_lab = "ACTION = ?", "STATE = ?    REWARD = ?"
        act_sub = "what can it change?"
        rew_sub = "what does it see?        what counts as better?"
    else:
        ax.text(5.5, 5.7, "reinforcement learning: learn by trial & feedback",
                ha="center", color=INK, fontsize=13.5, fontweight="bold")
        agent_lab, env_lab = "AGENT", "ENVIRONMENT"
        agent_sub = "the learner / policy"
        env_sub = "the world it acts in"
        act_lab, rew_lab = "ACTION", "REWARD (+ new state)"
        act_sub = "what it does"
        rew_sub = "feedback: better or worse?"
    # agent (left) and environment (right) boxes
    ax.add_patch(FancyBboxPatch((0.7, 2.3), 3.2, 1.6,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=TEAL_SOFT, edgecolor=TEAL, lw=2.6))
    hdr_fs = 12 if blank else 14
    ax.text(2.3, 3.35, agent_lab, ha="center", va="center", color=TEAL,
            fontsize=hdr_fs, fontweight="bold")
    ax.text(2.3, 2.75, agent_sub, ha="center", va="center", color=INK_SOFT,
            fontsize=10.5, style="italic")
    ax.add_patch(FancyBboxPatch((7.1, 2.3), 3.2, 1.6,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=AMBER_SOFT, edgecolor=AMBER, lw=2.6))
    ax.text(8.7, 3.35, env_lab, ha="center", va="center", color=ROI_INK,
            fontsize=hdr_fs, fontweight="bold")
    ax.text(8.7, 2.75, env_sub, ha="center", va="center", color=INK_SOFT,
            fontsize=10.5, style="italic")
    # action arrow (top, agent -> environment)
    ax.annotate("", xy=(7.05, 4.35), xytext=(3.95, 4.35),
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2.4,
                                connectionstyle="arc3,rad=-0.26"))
    ax.text(5.5, 5.32, act_lab, ha="center", color=TEAL, fontsize=12.5,
            fontweight="bold")
    ax.text(5.5, 4.98, act_sub, ha="center", color=MUTED, fontsize=10,
            style="italic")
    # reward arrow (bottom, environment -> agent); bulge UP into the empty
    # middle so it never strikes through the labels below.
    ax.annotate("", xy=(3.95, 2.0), xytext=(7.05, 2.0),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.4,
                                connectionstyle="arc3,rad=0.32"))
    ax.text(5.5, 0.95, rew_lab, ha="center", color=RED, fontsize=12.5,
            fontweight="bold")
    ax.text(5.5, 0.55, rew_sub, ha="center", color=MUTED, fontsize=10,
            style="italic")
    ax.text(5.5, 0.12,
            ("and ask: is your STATE Markov — is it enough on its own?"
             if blank else
             "repeat — keep the actions that earn more reward"),
            ha="center", color=ROI_INK if blank else INK_SOFT, fontsize=10.5,
            style="italic", fontweight="bold" if blank else "normal")
    _save(fig, name)


# ---- RL mechanics: CLASSIC examples (instructor, 2026-09-11) ---------------
# Rewritten away from the collision-energy tuning story: RL is taught on the
# canonical textbook examples (a gridworld, a multi-armed bandit, cart-pole)
# so the mechanics are not entangled with chemistry. The MS anchor moves to
# where it belongs — the closing you-do, where the room formulates an RL problem
# from their OWN lab (Quiz 13 Q1).
#
# GRIDWORLD (3x3, sparse reward): start at middle-left, goal at top-right,
# reward 0 on every step and +1 on arriving at the goal. The 3-move path
# UP, RIGHT, RIGHT pays 0, 0, 1 — which is exactly the episode the discounted
# return is worked on (γ = 0.9 → 0.9² = 0.81).
GRID_N = 3
GRID_START = (0, 1)          # (col, row), row 0 at the bottom
GRID_GOAL = (2, 2)
GRID_PATH = [(0, 1), (0, 2), (1, 2), (2, 2)]
GRID_EPISODE = [("1", "start cell", "↑  up", "0"),
                ("2", "top-left", "→  right", "0"),
                ("3", "top-middle", "→  right", "+1")]


def _gridworld(ax, path=None, show_goal=True, cell=1.0, x0=0.0, y0=0.0,
               label_states=False, fs=13):
    """Draw the 3x3 gridworld; optionally trace a path of (col, row) cells."""
    for c in range(GRID_N):
        for r in range(GRID_N):
            goal = (c, r) == GRID_GOAL
            ax.add_patch(FancyBboxPatch(
                (x0 + c * cell, y0 + r * cell), cell * 0.94, cell * 0.94,
                boxstyle="round,pad=0.01,rounding_size=0.04",
                facecolor=TEAL_SOFT if goal and show_goal else WHITE,
                edgecolor=TEAL if goal and show_goal else HAIRLINE,
                lw=2.4 if goal and show_goal else 1.6, zorder=2))
            if goal and show_goal:
                ax.text(x0 + (c + 0.47) * cell, y0 + (r + 0.60) * cell, "GOAL",
                        ha="center", va="center", color=TEAL, fontsize=fs - 3.5,
                        fontweight="bold", zorder=4)
                ax.text(x0 + (c + 0.47) * cell, y0 + (r + 0.30) * cell, "+1",
                        ha="center", va="center", color=TEAL, fontsize=fs,
                        fontweight="bold", zorder=4)
            if label_states:
                ax.text(x0 + (c + 0.85) * cell, y0 + (r + 0.12) * cell,
                        f"s{r * GRID_N + c + 1}", ha="right", va="bottom",
                        color=MUTED, fontsize=fs - 5, zorder=4)
    sc, sr = GRID_START
    ax.add_patch(Circle((x0 + (sc + 0.47) * cell, y0 + (sr + 0.47) * cell),
                        cell * 0.20, facecolor=AMBER, edgecolor=ROI_INK,
                        lw=1.6, zorder=5))
    if path:
        for (c1, r1), (c2, r2) in zip(path[:-1], path[1:]):
            ax.annotate("", xy=(x0 + (c2 + 0.47) * cell, y0 + (r2 + 0.47) * cell),
                        xytext=(x0 + (c1 + 0.47) * cell,
                                y0 + (r1 + 0.47) * cell),
                        arrowprops=dict(arrowstyle="-|>", color=ROI_INK, lw=2.6,
                                        shrinkA=13, shrinkB=13), zorder=6)


def rl_anatomy(name="fig_rl_anatomy.png"):
    """The RL loop's remaining vocabulary on the canonical GRIDWORLD: STATE
    (which cell you are in), POLICY (the rule that maps a state to a move, and
    the thing you are actually training), beside the ACTION / REWARD the loop
    slide already gave. The right panel walks the three-move episode that pays
    0, 0, +1 — the same episode the discounted-return slide reuses."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13.0, 5.0),
                                   gridspec_kw={"width_ratios": [1, 1.12]})

    # ---------------- LEFT: the gridworld itself ----------------
    axl.set_xlim(-0.35, 3.35)
    axl.set_ylim(-1.55, 4.0)
    axl.axis("off")
    axl.set_aspect("equal")
    axl.text(1.5, 3.72, "the classic gridworld", ha="center", color=INK,
             fontsize=13.5, fontweight="bold")
    _gridworld(axl, path=GRID_PATH, label_states=True)
    axl.text(1.5, -0.42,
             "STATE = which cell you are in   ·   ACTIONS = ↑ ↓ ← →",
             ha="center", color=INK_SOFT, fontsize=11)
    axl.text(1.5, -0.80,
             "REWARD = 0 every step, +1 on reaching the goal",
             ha="center", color=RED, fontsize=11, fontweight="bold")
    axl.text(1.5, -1.20,
             "POLICY = the arrow you would draw in every cell",
             ha="center", color=TEAL, fontsize=11, fontweight="bold")

    # ---------------- RIGHT: the vocabulary, then one episode --------------
    axr.set_xlim(0, 10)
    axr.set_ylim(0, 10)
    axr.axis("off")
    axr.text(5.0, 9.55, "the two words the loop left out", ha="center",
             color=INK, fontsize=13, fontweight="bold")
    for k, (word, gloss, col) in enumerate((
            ("STATE  s", "where you are — the cell, nothing else", INK),
            ("POLICY  π", "state → action; THIS is the network you train", TEAL))):
        y = 8.80 - k * 1.32
        axr.add_patch(FancyBboxPatch((0.2, y - 0.95), 9.6, 1.05,
                      boxstyle="round,pad=0.04,rounding_size=0.09",
                      facecolor=TEAL_SOFT if col is TEAL else WHITE,
                      edgecolor=col if col is TEAL else HAIRLINE,
                      lw=2.2 if col is TEAL else 1.6))
        axr.text(0.6, y - 0.42, word, fontsize=12.5, color=col, va="center",
                 fontweight="bold")
        axr.text(3.1, y - 0.42, gloss, fontsize=10.5, color=INK_SOFT,
                 va="center")
    axr.text(5.0, 5.95, "one episode — the path drawn on the grid",
             ha="center", color=INK, fontsize=12, fontweight="bold")
    cols = [0.8, 3.0, 5.8, 8.3]
    for cx, h, c in zip(cols, ["step", "STATE", "ACTION", "REWARD"],
                        [MUTED, INK, TEAL, RED]):
        axr.text(cx, 5.30, h, ha="center", color=c, fontsize=11,
                 fontweight="bold")
    axr.plot([0.2, 9.8], [5.02, 5.02], color=HAIRLINE, lw=1.4)
    y = 4.48
    for step, st, act, rew in GRID_EPISODE:
        pays = rew == "+1"
        if pays:
            axr.add_patch(FancyBboxPatch((0.2, y - 0.44), 9.6, 0.88,
                          boxstyle="round,pad=0.02,rounding_size=0.08",
                          facecolor=TEAL_SOFT, edgecolor="none", zorder=1))
        for cx, txt, c in zip(cols, (step, st, act, rew),
                              (MUTED, INK, TEAL, RED)):
            axr.text(cx, y, txt, ha="center", va="center", color=c,
                     fontsize=11.5, zorder=3,
                     fontweight="normal" if c is MUTED else "bold")
        y -= 0.95
    axr.add_patch(FancyBboxPatch((0.2, 0.15), 9.6, 1.55,
                  boxstyle="round,pad=0.05,rounding_size=0.1",
                  facecolor=AMBER_SOFT, edgecolor=AMBER, lw=2.2))
    axr.text(5.0, 0.92,
             "nobody ever said which move was “correct” — there is no label,\n"
             "only a reward, and it arrives at the very end.",
             ha="center", va="center", color=INK, fontsize=10.2,
             fontweight="bold", linespacing=1.6)
    fig.subplots_adjust(wspace=0.14)
    _save(fig, name)


def markov_property(name="fig_markov.png"):
    """THE requirement RL is built on, stated correctly: MARKOV is a property of
    the STATE YOU HAND THE AGENT, not of the world.  Left, the good case — the
    gridworld cell is enough, so two agents arriving by different routes face
    identical futures.  Right, the classic failure, PERCEPTUAL ALIASING: in a
    five-cell corridor an agent that senses only 'wall to my left? wall to my
    right?' gets the SAME reading in cell 2 and cell 4, yet must go right from
    one and left from the other.  The underlying cell is Markov; the observation
    is not.  (Deliberately NOT cart-pole — instructor, 2026-09-12: cart-pole's
    physics state is perfectly Markov, so it is a bad example of failure.)"""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13.2, 5.2),
                                   gridspec_kw={"width_ratios": [1, 1.12]})

    # ---------------- LEFT: gridworld — two paths, same future -------------
    axl.set_xlim(-0.3, 7.4)
    axl.set_ylim(-2.05, 4.3)
    axl.axis("off")
    axl.text(3.55, 4.05, "✓  MARKOV — the cell is enough", ha="center",
             color=TEAL, fontsize=13, fontweight="bold")
    for x0, pth, lab in ((0.0, [(0, 0), (0, 1), (1, 1)], "arrived the short way"),
                         (4.1, [(2, 0), (1, 0), (1, 1)], "arrived the long way")):
        for c in range(GRID_N):
            for r in range(GRID_N):
                here = (c, r) == (1, 1)
                axl.add_patch(FancyBboxPatch(
                    (x0 + c * 1.0, r * 1.0), 0.94, 0.94,
                    boxstyle="round,pad=0.01,rounding_size=0.04",
                    facecolor=AMBER_SOFT if here else WHITE,
                    edgecolor=AMBER if here else HAIRLINE,
                    lw=2.4 if here else 1.4, zorder=2))
        for (c1, r1), (c2, r2) in zip(pth[:-1], pth[1:]):
            axl.annotate("", xy=(x0 + c2 + 0.47, r2 + 0.47),
                         xytext=(x0 + c1 + 0.47, r1 + 0.47),
                         arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2.2,
                                         shrinkA=10, shrinkB=10), zorder=5)
        axl.add_patch(Circle((x0 + 1.47, 1.47), 0.20, facecolor=AMBER,
                             edgecolor=ROI_INK, lw=1.6, zorder=6))
        axl.text(x0 + 1.5, -0.45, lab, ha="center", color=MUTED, fontsize=10,
                 style="italic")
    axl.text(3.55, -1.05, "same cell → same best move, same expected future.",
             ha="center", color=INK, fontsize=11, fontweight="bold")
    axl.text(3.55, -1.55, "how you got there carries NO extra information.",
             ha="center", color=MUTED, fontsize=10.5, style="italic")

    # ------- RIGHT: perceptual aliasing — the observation is not enough ----
    axr.set_xlim(0, 10.6)
    axr.set_ylim(-2.30, 4.3)
    axr.axis("off")
    axr.text(5.3, 4.05, "✗  NOT MARKOV — what the agent can SEE", ha="center",
             color=RED, fontsize=13, fontweight="bold")
    axr.text(5.3, 3.58,
             "a corridor; the agent senses only “wall left? wall right?”",
             ha="center", color=MUTED, fontsize=10.3, style="italic")
    cw, x0, y0 = 1.72, 0.9, 1.55
    cells = ["1", "2", "3", "4", "5"]
    for k, lab in enumerate(cells):
        goal = k == 2
        alias = k in (1, 3)
        axr.add_patch(FancyBboxPatch((x0 + k * cw, y0), cw * 0.94, 1.25,
                      boxstyle="round,pad=0.01,rounding_size=0.05",
                      facecolor=TEAL_SOFT if goal else
                      ("#F9EDEC" if alias else WHITE),
                      edgecolor=TEAL if goal else (RED if alias else HAIRLINE),
                      lw=2.6 if (goal or alias) else 1.5, zorder=3))
        axr.text(x0 + k * cw + cw * 0.47, y0 + 0.86,
                 "GOAL" if goal else f"cell {lab}", ha="center", va="center",
                 color=TEAL if goal else (RED if alias else MUTED),
                 fontsize=10 if goal else 9.5, fontweight="bold")
        if alias:
            axr.add_patch(Circle((x0 + k * cw + cw * 0.47, y0 + 0.40), 0.19,
                                 facecolor=AMBER, edgecolor=ROI_INK, lw=1.5,
                                 zorder=5))
    for wx in (x0 - 0.16, x0 + len(cells) * cw - 0.10):
        axr.plot([wx, wx], [y0 - 0.05, y0 + 1.30], color=INK_SOFT, lw=5.0,
                 solid_capstyle="butt", zorder=4)
    axr.text(x0 - 0.16, y0 - 0.42, "wall", ha="center", color=INK_SOFT,
             fontsize=9)
    axr.text(x0 + len(cells) * cw - 0.10, y0 - 0.42, "wall", ha="center",
             color=INK_SOFT, fontsize=9)
    for k, arrow in ((1, "must go  →"), (3, "←  must go")):
        axr.text(x0 + k * cw + cw * 0.47, y0 + 1.52, arrow, ha="center",
                 color=RED, fontsize=10.5, fontweight="bold")
    axr.add_patch(FancyBboxPatch((0.25, -1.18), 10.1, 1.42,
                  boxstyle="round,pad=0.04,rounding_size=0.09",
                  facecolor="#F9EDEC", edgecolor=RED, lw=2.2))
    axr.text(5.3, -0.47,
             "cells 2 and 4 give the IDENTICAL reading “open | open”\n"
             "— yet the correct moves are opposite",
             ha="center", va="center", color=RED, fontsize=10.4,
             fontweight="bold", linespacing=1.6)
    axr.add_patch(FancyBboxPatch((0.25, -2.25), 10.1, 0.92,
                  boxstyle="round,pad=0.04,rounding_size=0.09",
                  facecolor=TEAL_SOFT, edgecolor=TEAL, lw=2.2))
    axr.text(5.3, -1.80,
             "the CELL is Markov; the OBSERVATION is not — enrich the state",
             ha="center", va="center", color=TEAL, fontsize=10.8,
             fontweight="bold")
    fig.text(0.5, 0.012,
             "Markov is a property of the STATE YOU CHOOSE, not of the world.",
             ha="center", va="bottom", color=INK, fontsize=11.5,
             fontweight="bold")
    fig.subplots_adjust(wspace=0.14, bottom=0.09)
    _save(fig, name)


# Explore vs exploit, with exact probabilities.  I-do: the classic two-armed
# bandit, eps=0.1 -> P(the under-sampled arm) = eps/K = 0.05.  You-do (Quiz 13
# Q4): two routes to work, eps=0.2 -> 0.10.  Means are exact: 140/20 = 7.0,
# 5/1 = 5.0; you-do 180/30 = 6.0, 8/2 = 4.0.
EXPLORE_IDO = dict(
    title="the two-armed bandit — which machine do you play?",
    a=("A   the machine you know", "played 20 times", "140 coins total",
       "mean = 140/20 = 7.0"),
    b=("B   the machine you don't", "played 1 time", "5 coins total",
       "mean = 5/1 = 5.0"),
    eps=0.1, truth="B's true mean is 8.0 — better than A, and you would never know")
EXPLORE_YOUDO = dict(
    title="two routes to work — which one do you drive?",
    a=("A   the usual route", "driven 30 times", "score 180 total",
       "mean = 180/30 = 6.0"),
    b=("B   the new route", "driven 2 times", "score 8 total",
       "mean = 8/2 = 4.0"),
    eps=0.2, truth=None)


def explore_exploit(mode="ido", name="fig_explore_ido.png"):
    """EXPLORE vs EXPLOIT, the choice every RL agent makes every cycle, with the
    arithmetic done exactly.  epsilon-greedy: with probability eps pick UNIFORMLY
    at random among the K actions, otherwise pick the best average so far — so a
    specific non-greedy action is taken with probability eps/K.  I-do works
    eps = 0.1, K = 2 -> 0.05; the you-do (Quiz 13 Q4) is the same shape on two
    routes to work with eps = 0.2 -> 0.10, left blank."""
    d = EXPLORE_IDO if mode == "ido" else EXPLORE_YOUDO
    blank = mode == "youdo"
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13.0, 5.2),
                                   gridspec_kw={"width_ratios": [1, 1.1]})

    # ---------------- LEFT: what we know so far ----------------
    axl.set_xlim(0, 10)
    axl.set_ylim(0, 10)
    axl.axis("off")
    axl.text(5.0, 9.55, f"what we know so far — {d['title']}", ha="center",
             color=INK, fontsize=12.5, fontweight="bold")
    for k, (key, col) in enumerate((("a", TEAL), ("b", AMBER))):
        head, tries, total, mean = d[key]
        y0 = 5.85 - k * 3.70
        axl.add_patch(FancyBboxPatch((0.3, y0), 9.4, 3.10,
                      boxstyle="round,pad=0.05,rounding_size=0.12",
                      facecolor=WHITE, edgecolor=col, lw=2.4))
        axl.add_patch(FancyBboxPatch((0.3, y0 + 2.25), 9.4, 0.85,
                      boxstyle="round,pad=0.05,rounding_size=0.12",
                      facecolor=col, edgecolor=col, lw=2.4))
        axl.text(5.0, y0 + 2.67, head, ha="center", va="center", color=WHITE,
                 fontsize=13, fontweight="bold")
        axl.text(2.6, y0 + 1.55, tries, ha="center", va="center",
                 color=INK_SOFT, fontsize=11)
        axl.text(7.2, y0 + 1.55, total, ha="center", va="center",
                 color=INK_SOFT, fontsize=11)
        axl.text(5.0, y0 + 0.68, mean, ha="center", va="center", color=col
                 if col is not AMBER else ROI_INK, fontsize=13,
                 fontweight="bold")
    axl.text(5.0, 1.15,
             "B looks worse — but it has barely been tried.",
             ha="center", color=MUTED, fontsize=10.5, style="italic")

    # ---------------- RIGHT: the rule, and the arithmetic ----------------
    axr.set_xlim(0, 10)
    axr.set_ylim(0, 10)
    axr.axis("off")
    axr.text(5.0, 9.6, "ε-greedy: the rule", ha="center", color=INK,
             fontsize=13.5, fontweight="bold")
    axr.add_patch(FancyBboxPatch((0.2, 7.55), 9.6, 1.65,
                  boxstyle="round,pad=0.05,rounding_size=0.1",
                  facecolor="#F1F1EC", edgecolor=MUTED, lw=1.8))
    axr.text(5.0, 8.72,
             "with probability ε — pick a random action (uniform over K)",
             ha="center", va="center", color=INK, fontsize=10.3)
    axr.text(5.0, 8.28, "otherwise — pick the highest average so far (greedy)",
             ha="center", va="center", color=INK, fontsize=10.3)
    axr.text(5.0, 7.85, "so  P(a specific non-greedy action)  =  ε / K",
             ha="center", va="center", color=TEAL, fontsize=11.5,
             fontweight="bold")

    eps = d["eps"]
    q = "?" if blank else None
    rows = [
        ("EXPLOIT — the greedy pick", "A" if not blank else "?",
         f"its average, {d['a'][3].split('= ')[-1]}, is the best we have seen"
         if not blank else "which action has the best average?"),
        (f"EXPLORE — with ε = {eps:g}, K = 2",
         "?" if blank else f"P(B) = {eps:g} / 2 = {eps / 2:.2f}",
         "how often do we try the under-sampled action?"
         if blank else f"about 1 cycle in {int(round(1 / (eps / 2)))}"),
    ]
    y = 6.55
    for head, val, gloss in rows:
        col = TEAL if head.startswith("EXPLOIT") else AMBER
        txt_col = TEAL if col is TEAL else ROI_INK
        axr.add_patch(FancyBboxPatch((0.2, y - 1.95), 9.6, 1.85,
                      boxstyle="round,pad=0.05,rounding_size=0.1",
                      facecolor=WHITE, edgecolor=col, lw=2.2))
        axr.text(0.6, y - 0.42, head, fontsize=11, color=txt_col, va="top",
                 fontweight="bold")
        axr.text(9.4, y - 0.42, val, fontsize=14 if val == "?" else 12.5,
                 color=ROI_INK if val == "?" else txt_col, va="top",
                 ha="right", fontweight="bold")
        axr.text(0.6, y - 1.28, gloss, fontsize=9.8, color=MUTED, va="top",
                 style="italic")
        y -= 2.20
    foot_face, foot_edge = ((AMBER_SOFT, AMBER) if blank
                           else (TEAL_SOFT, TEAL))
    axr.add_patch(FancyBboxPatch((0.2, 0.15), 9.6, 1.95,
                  boxstyle="round,pad=0.05,rounding_size=0.1",
                  facecolor=foot_face, edgecolor=foot_edge, lw=2.2))
    if blank:
        axr.text(5.0, 1.12,
                 "and in one sentence: why would ALWAYS-greedy\n"
                 "be a mistake here?",
                 ha="center", va="center", color=ROI_INK, fontsize=11.5,
                 fontweight="bold", linespacing=1.6)
    else:
        axr.text(5.0, 1.58, "why bother exploring at all?", ha="center",
                 va="center", color=TEAL, fontsize=11.5, fontweight="bold")
        axr.text(5.0, 0.78,
                 "over 200 cycles B gets tried ~10 times — enough to find out\n"
                 f"{d['truth']}",
                 ha="center", va="center", color=INK, fontsize=10.0,
                 linespacing=1.6)
    fig.subplots_adjust(wspace=0.14)
    _save(fig, name)


# The delayed-reward do-it-together, on the GRIDWORLD episode from the anatomy
# slide: three moves paying 0, 0, 1 (only arriving at the goal pays).
# Return G = r1 + γ·r2 + γ²·r3.
# At γ = 0.9 the first step still earns 0.81 credit — exact, since 0.9² = 0.81.
# γ = 0.5 is deliberately NOT shown: it is the Quiz 13 Q5 answer (0.25).
def discounted_return(name="fig_return_discount.png"):
    """DELAYED REWARD and the fix.  AlphaGo's win arrives 200 moves after the
    move that earned it — that is the credit-assignment problem.  RL answers it
    with the RETURN: each step is credited with the rewards that FOLLOW it,
    discounted by γ per step.  Worked here on a three-cycle method-development
    episode paying 0, 0, 1, so G from step 1 is 0.81 at γ = 0.9."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13.2, 5.0),
                                   gridspec_kw={"width_ratios": [1.1, 1]})

    # ---------------- LEFT: the episode, and the problem ----------------
    axl.set_xlim(0, 11)
    axl.set_ylim(0, 10)
    axl.axis("off")
    axl.text(5.5, 9.55, "the reward arrives LATE", ha="center", color=INK,
             fontsize=13.5, fontweight="bold")
    axl.text(5.5, 8.9,
             "the same three gridworld moves — only the last one pays",
             ha="center", color=MUTED, fontsize=10.5, style="italic")
    steps = [("move 1", "↑  up\nout of the start cell", "0", TEAL),
             ("move 2", "→  right\nalong the top row", "0", TEAL),
             ("move 3", "→  right\nonto the GOAL", "1", RED)]
    w, gap = 2.85, 0.75
    x = 0.55
    cxs = []
    for head, body, r, col in steps:
        pays = r == "1"
        axl.add_patch(FancyBboxPatch((x, 4.35), w, 3.05,
                      boxstyle="round,pad=0.03,rounding_size=0.1",
                      facecolor=AMBER_SOFT if pays else WHITE,
                      edgecolor=AMBER if pays else MUTED, lw=2.4))
        axl.text(x + w / 2, 6.85, head, ha="center", va="center",
                 color=ROI_INK if pays else INK_SOFT, fontsize=11.5,
                 fontweight="bold")
        axl.text(x + w / 2, 5.85, body, ha="center", va="center", color=INK,
                 fontsize=10.5, linespacing=1.4)
        axl.text(x + w / 2, 4.85, f"reward  r = {r}", ha="center",
                 va="center", color=RED if pays else MUTED, fontsize=11.5,
                 fontweight="bold")
        cxs.append(x + w / 2)
        x += w + gap
    for x0, x1 in zip(cxs[:-1], cxs[1:]):
        axl.annotate("", xy=(x1 - w / 2 - 0.1, 5.9),
                     xytext=(x0 + w / 2 + 0.1, 5.9),
                     arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, lw=2.2))
    axl.text(cxs[-1], 3.95, "the agent finally reaches the goal",
             ha="center", va="top", color=ROI_INK, fontsize=9.8,
             style="italic")
    axl.add_patch(FancyBboxPatch((0.55, 1.40), 9.9, 1.90,
                  boxstyle="round,pad=0.05,rounding_size=0.1",
                  facecolor="#F9EDEC", edgecolor=RED, lw=2.2))
    axl.text(5.5, 2.92, "the credit-assignment problem", ha="center",
             va="center", color=RED, fontsize=12, fontweight="bold")
    axl.text(5.5, 2.05,
             "moves 1 and 2 scored ZERO — but they set up the win.\n"
             "Which move earned it? AlphaGo\u2019s win lands 200 moves later.",
             ha="center", va="center", color=INK, fontsize=9.6,
             linespacing=1.6)
    axl.text(5.5, 0.75,
             "score a step by the rewards that FOLLOW it, not just its own",
             ha="center", va="center", color=TEAL, fontsize=11.5,
             fontweight="bold")

    # ---------------- RIGHT: the return, computed ----------------
    axr.set_xlim(0, 10)
    axr.set_ylim(0, 10)
    axr.axis("off")
    axr.text(5.0, 9.55, "the RETURN, with a discount γ", ha="center",
             color=INK, fontsize=13.5, fontweight="bold")
    axr.add_patch(FancyBboxPatch((0.2, 8.05), 9.6, 1.05,
                  boxstyle="round,pad=0.05,rounding_size=0.1",
                  facecolor="#F1F1EC", edgecolor=MUTED, lw=1.8))
    axr.text(5.0, 8.57, "G  =  r₁  +  γ·r₂  +  γ²·r₃", ha="center",
             va="center", color=INK, fontsize=14, fontweight="bold")
    axr.text(5.0, 7.55, "the episode: r₁ = 0, r₂ = 0, r₃ = 1,   with γ = 0.9",
             ha="center", va="center", color=MUTED, fontsize=10.8,
             style="italic")
    lines = [("credit for move 1", "0 + 0.9×0 + 0.81×1", "0.81", TEAL),
             ("credit for move 2", "0 + 0.9×1", "0.90", TEAL),
             ("credit for move 3", "1", "1.00", RED)]
    y = 6.75
    for head, work, val, col in lines:
        axr.add_patch(FancyBboxPatch((0.2, y - 1.02), 9.6, 1.0,
                      boxstyle="round,pad=0.04,rounding_size=0.08",
                      facecolor=WHITE, edgecolor=col, lw=2.0))
        axr.text(0.6, y - 0.52, head, fontsize=11, color=INK, va="center",
                 fontweight="bold")
        axr.text(5.6, y - 0.52, work, fontsize=10.8, color=INK_SOFT,
                 va="center", ha="center", family="monospace")
        axr.text(9.4, y - 0.52, val, fontsize=13, color=col, va="center",
                 ha="right", fontweight="bold")
        y -= 1.18
    axr.text(5.0, 2.86, "γ is a dial: how much does a LATER reward count?",
             ha="center", color=INK, fontsize=11, fontweight="bold")
    dial = [("γ = 0", "0.00", "only the reward\nin front of you"),
            ("γ = 0.9", "0.81", "our setting"),
            ("γ = 0.99", "0.98", "the distant win\ncounts almost fully")]
    w2 = 3.05
    x = 0.35
    for head, val, gloss in dial:
        on = head == "γ = 0.9"
        axr.add_patch(FancyBboxPatch((x, 0.35), w2, 2.15,
                      boxstyle="round,pad=0.04,rounding_size=0.08",
                      facecolor=TEAL_SOFT if on else WHITE,
                      edgecolor=TEAL if on else HAIRLINE, lw=2.2 if on else 1.6))
        axr.text(x + w2 / 2, 2.12, head, ha="center", va="center",
                 color=TEAL if on else INK_SOFT, fontsize=11.5,
                 fontweight="bold")
        axr.text(x + w2 / 2, 1.55, val, ha="center", va="center",
                 color=TEAL if on else INK, fontsize=14, fontweight="bold")
        axr.text(x + w2 / 2, 0.82, gloss, ha="center", va="center",
                 color=MUTED, fontsize=9.2, style="italic", linespacing=1.4)
        x += w2 + 0.30
    fig.subplots_adjust(wspace=0.14)
    _save(fig, name)


def rlhf_as_rl(name="fig_rlhf.png"):
    """RLHF = RL where the reward comes from human preferences. Map each RL role
    onto the LLM-training setting, and note it is the third pipeline stage."""
    fig, ax = plt.subplots(figsize=(11.6, 4.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.text(6.0, 4.7, "RLHF = RL where the reward is a human preference",
            ha="center", color=INK, fontsize=14, fontweight="bold")
    rows = [
        ("AGENT", TEAL, "the LLM being tuned"),
        ("ACTION", TEAL, "write an answer to a prompt"),
        ("ENVIRONMENT", AMBER, "the prompt + a human rater"),
        ("REWARD", RED, "how much a human PREFERS the answer"),
    ]
    ys = [3.55, 2.75, 1.95, 1.15]
    for (role, col, meaning), y in zip(rows, ys):
        ax.add_patch(FancyBboxPatch((1.1, y - 0.33), 2.9, 0.66,
                    boxstyle="round,pad=0.02,rounding_size=0.08",
                    facecolor=col, edgecolor=col, lw=2.0))
        ax.text(2.55, y, role, ha="center", va="center", color=WHITE,
                fontsize=12, fontweight="bold")
        ax.annotate("", xy=(4.55, y), xytext=(4.05, y),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0))
        ax.add_patch(FancyBboxPatch((4.65, y - 0.33), 6.2, 0.66,
                    boxstyle="round,pad=0.02,rounding_size=0.08",
                    facecolor=WHITE, edgecolor=col, lw=1.8))
        ax.text(7.75, y, meaning, ha="center", va="center", color=INK,
                fontsize=11.5, fontweight="bold")
    ax.text(6.0, 0.45,
            "this is stage 3 of the pipeline — no new machinery, just RL",
            ha="center", color=MUTED, fontsize=10.5, style="italic")
    _save(fig, name)


# ---- RLHF opened up (instructor, 2026-09-14) -------------------------------
# The one-slide "RLHF = RL with a human reward" mapping now gets two slides of
# depth: HOW it is actually built (preference pairs -> reward model -> tune the
# policy, with the KL leash) and WHAT it buys and breaks (instruction-following
# vs reward hacking, sycophancy, and the fact that it does NOT fix
# hallucination).  Both are anchored on one clinical prompt so the room sees the
# same question travel through the whole pipeline.
RLHF_PROMPT = "“QC failed on this run — report the cortisol result?”"
RLHF_GOOD = ("No — a failed QC invalidates the run.\n"
             "Repeat the QC and re-run before\nreporting anything.")
RLHF_BAD = ("Yes, as long as the patient value\nlooks clinically reasonable.")


def rlhf_pipeline(name="fig_rlhf_pipeline.png"):
    """RLHF in the three steps it is actually built from: (1) humans compare two
    answers to the same prompt; (2) a REWARD MODEL is trained on those
    comparisons to score answers the way the raters did; (3) the LLM — the
    POLICY from the anatomy slide — is tuned to earn higher scores, on a leash
    that stops it drifting away from the model it started as."""
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.0))
    steps = [
        ("1 · COLLECT PREFERENCES", TEAL,
         "same prompt, two answers,\na human says which is better"),
        ("2 · TRAIN A REWARD MODEL", AMBER,
         "a second network learns to score\nanswers the way the raters did"),
        ("3 · TUNE THE POLICY", RED,
         "the LLM is nudged toward answers\nthe reward model scores higher"),
    ]
    for ax, (head, col, gloss) in zip(axes, steps):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.2, 8.75), 9.6, 1.05,
                     boxstyle="round,pad=0.04,rounding_size=0.09",
                     facecolor=col, edgecolor=col, lw=2.0))
        ax.text(5.0, 9.28, head, ha="center", va="center", color=WHITE,
                fontsize=11.5, fontweight="bold")
        ax.text(5.0, 7.95, gloss, ha="center", va="center", color=INK_SOFT,
                fontsize=9.8, linespacing=1.5)

    # ---------------- step 1: the preference pair ----------------
    ax = axes[0]
    ax.add_patch(FancyBboxPatch((0.2, 6.45), 9.6, 1.10,
                 boxstyle="round,pad=0.04,rounding_size=0.09",
                 facecolor="#F1F1EC", edgecolor=MUTED, lw=1.6))
    ax.text(5.0, 7.26, "prompt", ha="center", va="center", color=MUTED,
            fontsize=8.6, style="italic")
    ax.text(5.0, 6.82, RLHF_PROMPT, ha="center", va="center", color=INK,
            fontsize=8.0)
    for k, (lab, txt, col, mark) in enumerate((
            ("answer A", RLHF_GOOD, TEAL, "✓ preferred"),
            ("answer B", RLHF_BAD, MUTED, ""))):
        y0 = 3.95 - k * 2.50
        ax.add_patch(FancyBboxPatch((0.2, y0), 9.6, 2.20,
                     boxstyle="round,pad=0.04,rounding_size=0.09",
                     facecolor=TEAL_SOFT if col is TEAL else WHITE,
                     edgecolor=col, lw=2.2 if col is TEAL else 1.5))
        ax.text(0.65, y0 + 1.86, lab, ha="left", va="center", color=col,
                fontsize=9.5, fontweight="bold")
        if mark:
            ax.text(9.35, y0 + 1.86, mark, ha="right", va="center", color=TEAL,
                    fontsize=9.5, fontweight="bold")
        ax.text(5.0, y0 + 0.92, txt, ha="center", va="center", color=INK,
                fontsize=8.5, linespacing=1.5)
    ax.text(5.0, 0.52, "→ thousands of (prompt, better, worse) triples",
            ha="center", color=TEAL, fontsize=9.4, fontweight="bold")

    # ---------------- step 2: the reward model ----------------
    ax = axes[1]
    ax.add_patch(FancyBboxPatch((1.3, 5.15), 7.4, 1.85,
                 boxstyle="round,pad=0.04,rounding_size=0.1",
                 facecolor=AMBER_SOFT, edgecolor=AMBER, lw=2.4))
    ax.text(5.0, 6.42, "REWARD MODEL", ha="center", va="center", color=ROI_INK,
            fontsize=12, fontweight="bold")
    ax.text(5.0, 5.70, "(prompt, answer)  →  one number",
            ha="center", va="center", color=INK, fontsize=9.6)
    ax.annotate("", xy=(5.0, 7.30), xytext=(5.0, 7.00),
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=2.2))
    ax.text(5.0, 4.55, "what it learns from the pairs",
            ha="center", color=INK, fontsize=9.8, fontweight="bold")
    ax.add_patch(FancyBboxPatch((1.0, 3.05), 8.0, 1.15,
                 boxstyle="round,pad=0.04,rounding_size=0.09",
                 facecolor=WHITE, edgecolor=TEAL, lw=2.2))
    ax.text(5.0, 3.62, "score(A)   >   score(B)", ha="center", va="center",
            color=TEAL, fontsize=13, fontweight="bold")
    ax.text(5.0, 2.45, "it never sees an absolute “correct” score —\n"
            "only which of two answers people preferred",
            ha="center", va="center", color=MUTED, fontsize=9.0,
            style="italic", linespacing=1.5)
    ax.add_patch(FancyBboxPatch((0.5, 0.25), 9.0, 1.35,
                 boxstyle="round,pad=0.04,rounding_size=0.09",
                 facecolor="#F1F1EC", edgecolor=MUTED, lw=1.6))
    ax.text(5.0, 0.92, "now the reward is automatic — no human\n"
            "in the loop for every single answer",
            ha="center", va="center", color=INK, fontsize=9.2,
            linespacing=1.5)

    # ---------------- step 3: tune the policy ----------------
    ax = axes[2]
    ax.add_patch(FancyBboxPatch((0.6, 5.55), 8.8, 1.45,
                 boxstyle="round,pad=0.04,rounding_size=0.1",
                 facecolor=TEAL_SOFT, edgecolor=TEAL, lw=2.4))
    ax.text(5.0, 6.28, "the LLM = the POLICY", ha="center", va="center",
            color=TEAL, fontsize=11.5, fontweight="bold")
    ax.annotate("", xy=(9.05, 4.40), xytext=(9.05, 5.45),
                arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, lw=2.0))
    ax.annotate("", xy=(0.95, 5.45), xytext=(0.95, 4.40),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.0))
    ax.text(5.0, 4.92, "write  →  score  →  nudge", ha="center", va="center",
            color=INK_SOFT, fontsize=9.2, fontweight="bold")
    ax.add_patch(FancyBboxPatch((0.6, 3.15), 8.8, 1.15,
                 boxstyle="round,pad=0.04,rounding_size=0.09",
                 facecolor=AMBER_SOFT, edgecolor=AMBER, lw=2.2))
    ax.text(5.0, 3.72, "reward model scores it", ha="center", va="center",
            color=ROI_INK, fontsize=10.5, fontweight="bold")
    ax.text(5.0, 2.62, "repeat over millions of answers", ha="center",
            color=MUTED, fontsize=9.2, style="italic")
    ax.add_patch(FancyBboxPatch((0.35, 0.25), 9.3, 1.85,
                 boxstyle="round,pad=0.04,rounding_size=0.09",
                 facecolor=WHITE, edgecolor=RED, lw=2.2))
    ax.text(5.0, 1.72, "and keep it on a leash", ha="center", va="center",
            color=RED, fontsize=10, fontweight="bold")
    ax.text(5.0, 0.90, "penalise drifting too far from the model it\n"
            "started as — otherwise it games the score\nand stops writing English",
            ha="center", va="center", color=INK, fontsize=8.6,
            linespacing=1.5)
    fig.subplots_adjust(wspace=0.16, top=0.97, bottom=0.03)
    _save(fig, name)


def rlhf_limits(name="fig_rlhf_limits.png"):
    """What RLHF buys and what it breaks. LEFT: the same clinical question put
    to a base model (which merely CONTINUES text) and to the tuned assistant.
    RIGHT: the three failure modes a lab must plan for — reward hacking,
    sycophancy, and the fact that preference tuning does NOT fix hallucination,
    because raters reward answers that LOOK good."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13.2, 5.2),
                                   gridspec_kw={"width_ratios": [1, 1.02]})
    axl.set_xlim(0, 10); axl.set_ylim(0, 10); axl.axis("off")
    axl.text(5.0, 9.6, "what it buys", ha="center", color=TEAL, fontsize=13,
             fontweight="bold")
    axl.add_patch(FancyBboxPatch((0.2, 8.15), 9.6, 0.95,
                  boxstyle="round,pad=0.04,rounding_size=0.09",
                  facecolor="#F1F1EC", edgecolor=MUTED, lw=1.6))
    axl.text(5.0, 8.62, "you ask:  “What is the reference interval for "
             "serum cortisol?”", ha="center", va="center", color=INK,
             fontsize=9.2)
    panels = [
        ("BASE MODEL — pretraining only", MUTED,
         "“…and what is the reference interval\n"
         "for serum ACTH? What is the reference\n"
         "interval for serum aldosterone?”",
         "it CONTINUES your text — it was never\ntaught to answer you"),
        ("AFTER SFT + RLHF", TEAL,
         "“It depends on collection time and on\n"
         "the assay — a morning draw runs higher.\n"
         "Use your own laboratory’s interval.”",
         "it answers, hedges honestly, and defers\nto your local range"),
    ]
    y = 7.15
    for head, col, quote, gloss in panels:
        axl.add_patch(FancyBboxPatch((0.2, y - 3.35), 9.6, 3.25,
                      boxstyle="round,pad=0.05,rounding_size=0.11",
                      facecolor=TEAL_SOFT if col is TEAL else WHITE,
                      edgecolor=col, lw=2.4 if col is TEAL else 1.6))
        axl.text(5.0, y - 0.48, head, ha="center", va="center", color=col,
                 fontsize=10.2, fontweight="bold")
        axl.text(5.0, y - 1.60, quote, ha="center", va="center", color=INK,
                 fontsize=9.0, linespacing=1.55)
        axl.text(5.0, y - 2.85, gloss, ha="center", va="center", color=MUTED,
                 fontsize=8.6, style="italic", linespacing=1.45)
        y -= 3.62

    axr.set_xlim(0, 10); axr.set_ylim(0, 10); axr.axis("off")
    axr.text(5.0, 9.6, "what it breaks", ha="center", color=RED, fontsize=13,
             fontweight="bold")
    cards = [
        ("REWARD HACKING", "it optimises what SCORES well, not what is true —\n"
         "long, confident, agreeable answers score well"),
        ("SYCOPHANCY", "“Actually I think the cutoff is 50.”  →  “You’re right,\n"
         "my mistake.”  It agrees with whoever is talking."),
        ("IT DOES NOT FIX HALLUCINATION",
         "raters reward answers that LOOK good, so a confident\n"
         "fabrication can out-score an honest “I don’t know”"),
    ]
    y = 8.70
    for head, gloss in cards:
        axr.add_patch(FancyBboxPatch((0.2, y - 2.15), 9.6, 2.05,
                      boxstyle="round,pad=0.05,rounding_size=0.11",
                      facecolor="#F9EDEC", edgecolor=RED, lw=2.0))
        axr.text(0.65, y - 0.55, head, ha="left", va="center", color=RED,
                 fontsize=10.5, fontweight="bold")
        axr.text(0.65, y - 1.52, gloss, ha="left", va="center", color=INK,
                 fontsize=8.8, linespacing=1.55)
        y -= 2.35
    axr.add_patch(FancyBboxPatch((0.2, 0.20), 9.6, 1.35,
                  boxstyle="round,pad=0.04,rounding_size=0.09",
                  facecolor=TEAL_SOFT, edgecolor=TEAL, lw=2.2))
    axr.text(5.0, 0.87, "so the guardrails do not change:\n"
             "ground it in your documents, and sign it off",
             ha="center", va="center", color=TEAL, fontsize=9.4,
             fontweight="bold", linespacing=1.6)
    fig.subplots_adjust(wspace=0.14, top=0.97, bottom=0.03)
    _save(fig, name)

def use_llm_chart(mode="ido", name="fig_use_llm_ido.png"):
    """Using an LLM WITHOUT training one: prompting vs RAG vs fine-tuning, as a
    decision chart anchored to one clinical report-drafting task. mode='ido'
    shows all three options + when to use each; mode='youdo' presents the
    Quiz 13 Q2 scenario with a word bank of the three options."""
    fig, ax = plt.subplots(figsize=(12.4, 5.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")
    if mode == "ido":
        ax.text(6.5, 5.65, "three ways to use an LLM — no training a new one",
                ha="center", color=INK, fontsize=14, fontweight="bold")
        # Instructor, 2026-09-12: every option carries a CONCRETE example, so
        # the chart is three worked cases rather than three definitions.  The
        # fine-tuning example is deliberately NOT the house-report-style one —
        # that is the Quiz 13 Q2 answer and has to stay unseen.
        rows = [
            ("PROMPTING", TEAL,
             "just ask — instructions and\na few examples in the prompt",
             "general task · no private data\n· instant to set up",
             "“Rewrite this method section\nin plain English for patients.”",
             "nothing to build — paste and go"),
            ("RAG", AMBER,
             "retrieve YOUR documents at\nquery time → into the prompt",
             "answers must cite current\nor private documents",
             "“What is today’s acceptance\ncriterion for the cortisol assay?”",
             "pulls the current SOP, and cites it"),
            ("FINE-TUNING", RED,
             "further-train the model on\nyour labelled examples",
             "a consistent new skill or style\n+ many examples",
             "“Turn this error log into a\nplain-language cause.”",
             "from 4,000 past logs — baked into the weights"),
        ]
        ys = [4.30, 2.85, 1.40]
        for (opt, col, what, when, ex, why), y in zip(rows, ys):
            ax.add_patch(FancyBboxPatch((0.25, y - 0.62), 2.35, 1.24,
                        boxstyle="round,pad=0.02,rounding_size=0.06",
                        facecolor=col, edgecolor=col, lw=2.0))
            ax.text(1.425, y, opt, ha="center", va="center", color=WHITE,
                    fontsize=11.5, fontweight="bold")
            ax.add_patch(FancyBboxPatch((2.85, y - 0.62), 3.15, 1.24,
                        boxstyle="round,pad=0.02,rounding_size=0.06",
                        facecolor=WHITE, edgecolor=col, lw=1.8))
            ax.text(4.425, y, what, ha="center", va="center", color=INK,
                    fontsize=9.6, fontweight="bold", linespacing=1.45)
            ax.add_patch(FancyBboxPatch((6.25, y - 0.62), 3.05, 1.24,
                        boxstyle="round,pad=0.02,rounding_size=0.06",
                        facecolor="#F1F1EC", edgecolor=col, lw=1.8))
            ax.text(7.775, y, when, ha="center", va="center", color=INK_SOFT,
                    fontsize=9.4, style="italic", linespacing=1.45)
            ax.add_patch(FancyBboxPatch((9.55, y - 0.62), 3.20, 1.24,
                        boxstyle="round,pad=0.02,rounding_size=0.06",
                        facecolor=TEAL_SOFT if col is TEAL else
                        (AMBER_SOFT if col is AMBER else "#F9EDEC"),
                        edgecolor=col, lw=1.8))
            ax.text(11.15, y + 0.19, ex, ha="center", va="center", color=INK,
                    fontsize=8.8, linespacing=1.4)
            ax.text(11.15, y - 0.45, why, ha="center", va="center",
                    color=MUTED, fontsize=7.6, style="italic")
        ax.text(1.425, 5.12, "option", ha="center", color=MUTED, fontsize=10.5)
        ax.text(4.425, 5.12, "what it does", ha="center", color=MUTED,
                fontsize=10.5)
        ax.text(7.775, 5.12, "use when", ha="center", color=MUTED,
                fontsize=10.5)
        ax.text(11.15, 5.12, "one concrete example", ha="center", color=INK,
                fontsize=10.5, fontweight="bold")
        ax.text(6.5, 0.42,
                "cheapest first: try prompting, reach for RAG when the answer "
                "must be current, fine-tune last",
                ha="center", color=TEAL, fontsize=10.8, fontweight="bold")
    else:
        ax.text(6.5, 5.5, "your turn — which approach?", ha="center",
                color=ROI_INK, fontsize=14, fontweight="bold")
        ax.add_patch(FancyBboxPatch((1.2, 2.6), 10.6, 2.3,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor=WHITE, edgecolor=INK_SOFT, lw=2.2))
        ax.text(6.5, 4.35, "scenario", ha="center", color=MUTED, fontsize=11,
                style="italic")
        ax.text(6.5, 3.6,
                "You want the assistant to ALWAYS format results in your lab's",
                ha="center", color=INK, fontsize=12.5)
        ax.text(6.5, 3.15,
                "exact report style — and you have 2,000 past reports as examples.",
                ha="center", color=INK, fontsize=12.5)
        ax.text(6.5, 1.9,
                "word bank:   PROMPTING   ·   RAG   ·   FINE-TUNING",
                ha="center", color=INK_SOFT, fontsize=12, fontweight="bold")
    _save(fig, name)


# ---- Landscape-tour result panels (instructor, 2026-09-11) -----------------
# Every tour slide now carries the PAPER: a full citation with DOI, and that
# paper's headline experimental result.  Licensing reality (checked 2026-09-11
# against Europe PMC): only Casanovo (Nat Commun) and AlphaFold (Nature) are
# CC BY; Prosit and DRIAMS are subscription, and the DIA-NN deposit is
# free-to-read with no reuse licence.  So rather than mix two reusable journal
# figures with three we may not redistribute — and rather than put 6-pt
# multi-panel journal art on a projector — every panel here is REDRAWN by the
# course from numbers the paper itself states.  Data are not copyrightable; the
# numbers below are quoted verbatim from each paper and the citation is on the
# slide.  Swap in the real CC BY figures for Casanovo/AlphaFold if desired.
TOUR_CITE = {
 "casanovo": ("Yilmaz M, Fondrie WE, Bittremieux W, et al. “Sequence-to-sequence "
              "translation from mass spectra to peptides with a transformer model.” "
              "Nature Communications 15:6427 (2024) · doi:10.1038/s41467-024-49731-x "
              "· CC BY 4.0 — chart redrawn by the course from the paper’s Fig. 2a values"),
 "prosit":   ("Gessulat S, Schmidt T, Zolg DP, et al. “Prosit: proteome-wide prediction "
              "of peptide tandem mass spectra by deep learning.” Nature Methods "
              "16:509–518 (2019) · doi:10.1038/s41592-019-0426-7 "
              "— figures © the publisher; numbers quoted, chart drawn for this course"),
 "diann":    ("Demichev V, Messner CB, Vernardis SI, Lilley KS, Ralser M. “DIA-NN: neural "
              "networks and interference correction enable deep proteome coverage in high "
              "throughput.” Nature Methods 17:41–44 (2020) · doi:10.1038/s41592-019-0638-x "
              "— figures © the publisher; numbers quoted, chart drawn for this course"),
 "driams":   ("Weis C, Cuénod A, Rieck B, et al. “Direct antimicrobial resistance prediction "
              "from clinical MALDI-TOF mass spectra using machine learning.” Nature Medicine "
              "28:164–174 (2022) · doi:10.1038/s41591-021-01619-9 "
              "— figures © the publisher; numbers quoted, chart drawn for this course"),
 "alphafold":("Jumper J, Evans R, Pritzel A, et al. “Highly accurate protein structure "
              "prediction with AlphaFold.” Nature 596:583–589 (2021) · "
              "doi:10.1038/s41586-021-03819-2 · CC BY 4.0 "
              "— chart redrawn by the course from the paper’s reported CASP14 accuracy"),
}


def _cite_strip(fig, key, y=0.012, fs=7.2):
    """The citation line every tour panel carries, wrapped to the figure width."""
    import textwrap
    txt = "\n".join(textwrap.wrap(TOUR_CITE[key], 155))
    fig.text(0.5, y, txt, ha="center", va="bottom", color=MUTED, fontsize=fs,
             linespacing=1.5)


def _result_bars(ax, labels, values, hero, ylabel, ylim, fmt="{:.2f}",
                 baseline=None, baseline_lab=""):
    cols = [TEAL if l == hero else "#CFE3E2" for l in labels]
    bars = ax.bar(labels, values, color=cols, width=0.62, zorder=3)
    for b, v, l in zip(bars, values, labels):
        ax.text(b.get_x() + b.get_width() / 2, v + (ylim[1] - ylim[0]) * 0.025,
                fmt.format(v), ha="center", color=INK if l == hero else MUTED,
                fontsize=13 if l == hero else 11.5,
                fontweight="bold" if l == hero else "normal")
    if baseline is not None:
        ax.axhline(baseline, color=RED, lw=1.6, ls="--", zorder=2)
        ax.text(len(labels) - 0.42, baseline + (ylim[1] - ylim[0]) * 0.015,
                baseline_lab, ha="right", color=RED, fontsize=9.5,
                style="italic")
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.tick_params(axis="x", labelsize=10.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def tour_casanovo(name="fig_tour_casanovo.png"):
    """Casanovo's headline benchmark: average peptide-level precision on the
    nine-species benchmark — Casanovo 0.81 vs PointNovo 0.74, DeepNovo 0.70,
    Novor 0.58 (paper, Fig. 2a)."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.6, 4.6),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    _result_bars(axl, ["Casanovo", "PointNovo", "DeepNovo", "Novor"],
                 [0.81, 0.74, 0.70, 0.58], "Casanovo",
                 "average peptide-level precision", (0, 0.95))
    axl.set_title("nine-species de novo benchmark", fontsize=12.5, color=INK,
                  fontweight="bold")
    axr.set_xlim(0, 10); axr.set_ylim(0, 10); axr.axis("off")
    facts = [("30 million", "labelled spectra (MassIVE-KB) used to train it"),
             ("0.81 → 0.95", "average precision when trained on that larger set"),
             ("no database", "needed — it reads the peptide off the spectrum")]
    y = 8.6
    for big, gloss in facts:
        axr.add_patch(FancyBboxPatch((0.2, y - 2.35), 9.6, 2.25,
                      boxstyle="round,pad=0.05,rounding_size=0.1",
                      facecolor=WHITE, edgecolor=TEAL, lw=2.0))
        axr.text(0.7, y - 0.85, big, fontsize=17, color=TEAL, va="center",
                 fontweight="bold")
        axr.text(0.7, y - 1.80, gloss, fontsize=10.3, color=INK_SOFT,
                 va="center")
        y -= 2.75
    fig.subplots_adjust(wspace=0.28, bottom=0.24, top=0.88)
    _cite_strip(fig, "casanovo")
    _save(fig, name)


def tour_prosit(name="fig_tour_prosit.png"):
    """Prosit's headline claim, in the paper's own numbers: a 550,000-peptide /
    21-million-spectrum synthetic training set, and rescoring that yields more
    identifications at >10x lower FDR."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.6, 4.6),
                                   gridspec_kw={"width_ratios": [1, 1.05]})
    axl.set_xlim(0, 10); axl.set_ylim(0, 10); axl.axis("off")
    axl.text(5.0, 9.4, "what it was trained on", ha="center", color=INK,
             fontsize=12.5, fontweight="bold")
    chain = [("550,000", "synthetic tryptic peptides\n(the ProteomeTools library)"),
             ("21 million", "high-quality tandem mass spectra"),
             ("Prosit", "predicts fragment intensities + retention time")]
    y = 7.9
    for k, (big, gloss) in enumerate(chain):
        last = k == len(chain) - 1
        axl.add_patch(FancyBboxPatch((0.4, y - 1.85), 9.2, 1.80,
                      boxstyle="round,pad=0.05,rounding_size=0.1",
                      facecolor=TEAL_SOFT if last else WHITE,
                      edgecolor=TEAL if last else HAIRLINE, lw=2.2 if last else 1.6))
        axl.text(5.0, y - 0.62, big, ha="center", fontsize=15.5,
                 color=TEAL if last else INK, fontweight="bold")
        axl.text(5.0, y - 1.35, gloss, ha="center", fontsize=9.8, color=MUTED,
                 va="center", linespacing=1.4)
        if not last:
            axl.annotate("", xy=(5.0, y - 2.30), xytext=(5.0, y - 1.95),
                         arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, lw=2.2))
        y -= 2.65
    axr.set_xlim(0, 10); axr.set_ylim(0, 10); axr.axis("off")
    axr.text(5.0, 9.4, "what it bought", ha="center", color=INK, fontsize=12.5,
             fontweight="bold")
    axr.add_patch(FancyBboxPatch((0.3, 5.3), 9.4, 3.3,
                  boxstyle="round,pad=0.05,rounding_size=0.12",
                  facecolor=TEAL_SOFT, edgecolor=TEAL, lw=2.4))
    axr.text(5.0, 7.55, "more identifications", ha="center", color=INK,
             fontsize=14, fontweight="bold")
    axr.text(5.0, 6.65, "at  > 10×  lower FDR", ha="center", color=TEAL,
             fontsize=19, fontweight="bold")
    axr.text(5.0, 5.85, "when Prosit predictions are folded into the "
             "database search", ha="center", color=MUTED, fontsize=9.8,
             style="italic")
    axr.text(5.0, 4.55, "the paper’s words: predictions that “exceed the quality\n"
             "of the experimental data”", ha="center", va="center", color=INK,
             fontsize=10.5, linespacing=1.5)
    axr.add_patch(FancyBboxPatch((0.3, 1.35), 9.4, 2.4,
                  boxstyle="round,pad=0.05,rounding_size=0.12",
                  facecolor=WHITE, edgecolor=AMBER, lw=2.0))
    axr.text(5.0, 3.15, "and the same trick builds DIA libraries",
             ha="center", color=ROI_INK, fontsize=11.5, fontweight="bold")
    axr.text(5.0, 2.15, "predict the library instead of measuring it —\n"
             "which is what the next tool consumes",
             ha="center", va="center", color=INK_SOFT, fontsize=10,
             linespacing=1.5)
    fig.subplots_adjust(wspace=0.20, bottom=0.115, top=0.95)
    _cite_strip(fig, "prosit")
    _save(fig, name)


def tour_diann(name="fig_tour_diann.png"):
    """DIA-NN's headline experiment (paper Fig. 1b), described honestly: the
    paper plots precursor identifications against FDR for 0.5-4 h gradients,
    DIA-NN vs Spectronaut vs OpenSWATH. The one hard number quoted in the text
    is >35,000 precursors from a half-hour gradient."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.6, 4.6),
                                   gridspec_kw={"width_ratios": [1.1, 1]})
    axl.set_xlim(0, 10); axl.set_ylim(0, 10); axl.axis("off")
    axl.text(5.0, 9.4, "the experiment in the paper", ha="center", color=INK,
             fontsize=12.5, fontweight="bold")
    rows = [("sample", "technical repeat injections of a HeLa tryptic digest"),
            ("gradients", "0.5 h  ·  1 h  ·  2 h  ·  4 h  (Q Exactive HF)"),
            ("measured", "precursor IDs plotted against FDR"),
            ("compared", "DIA-NN  vs  Spectronaut  vs  OpenSWATH")]
    y = 8.0
    for k, (head, gloss) in enumerate(rows):
        axl.add_patch(FancyBboxPatch((0.3, y - 1.55), 9.4, 1.50,
                      boxstyle="round,pad=0.04,rounding_size=0.09",
                      facecolor=WHITE, edgecolor=HAIRLINE, lw=1.6))
        axl.text(0.75, y - 0.80, head, fontsize=11, color=TEAL, va="center",
                 fontweight="bold")
        axl.text(3.15, y - 0.80, gloss, fontsize=10.2, color=INK, va="center")
        y -= 1.85
    axr.set_xlim(0, 10); axr.set_ylim(0, 10); axr.axis("off")
    axr.text(5.0, 9.4, "the headline number", ha="center", color=INK,
             fontsize=12.5, fontweight="bold")
    axr.add_patch(FancyBboxPatch((0.4, 5.0), 9.2, 3.6,
                  boxstyle="round,pad=0.05,rounding_size=0.12",
                  facecolor=TEAL_SOFT, edgecolor=TEAL, lw=2.4))
    axr.text(5.0, 7.35, "> 35,000", ha="center", color=TEAL, fontsize=30,
             fontweight="bold")
    axr.text(5.0, 6.15, "precursors from a HALF-HOUR gradient",
             ha="center", color=INK, fontsize=11.5, fontweight="bold")
    axr.text(5.0, 5.50, "“more than what was achieved only a few years ago\n"
             "with 2 h nanoflow gradients”", ha="center", va="center",
             color=MUTED, fontsize=9.5, style="italic", linespacing=1.5)
    axr.add_patch(FancyBboxPatch((0.4, 1.3), 9.2, 3.0,
                  boxstyle="round,pad=0.05,rounding_size=0.12",
                  facecolor=AMBER_SOFT, edgecolor=AMBER, lw=2.2))
    axr.text(5.0, 3.55, "the teaching point", ha="center", color=ROI_INK,
             fontsize=11.5, fontweight="bold")
    axr.text(5.0, 2.35, "the architecture doing this is an ensemble of\n"
             "plain feed-forward nets — the MLP from Lectures 1–4.\n"
             "Match the model to the job, not to the hype.",
             ha="center", va="center", color=INK, fontsize=10,
             linespacing=1.55)
    fig.subplots_adjust(wspace=0.20, bottom=0.115, top=0.95)
    _cite_strip(fig, "diann")
    _save(fig, name)


def tour_driams(name="fig_tour_driams.png"):
    """DRIAMS: the AUROC bars the paper reports for its three validation
    species, plus the dataset scale and the retrospective clinical case study
    (63 patients; treatment would have changed in 9, beneficially in 8)."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.6, 4.6),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    _result_bars(axl, ["S. aureus\noxacillin", "E. coli\nceftriaxone",
                       "K. pneumoniae\nceftriaxone"],
                 [0.80, 0.74, 0.74], "S. aureus\noxacillin", "AUROC",
                 (0, 1.0), baseline=0.5, baseline_lab="coin flip")
    axl.set_title("resistance called from the routine MALDI-TOF spectrum",
                  fontsize=12, color=INK, fontweight="bold")
    axr.set_xlim(0, 10); axr.set_ylim(0, 10); axr.axis("off")
    axr.add_patch(FancyBboxPatch((0.3, 6.55), 9.4, 3.05,
                  boxstyle="round,pad=0.05,rounding_size=0.12",
                  facecolor=WHITE, edgecolor=TEAL, lw=2.2))
    axr.text(5.0, 8.95, "the database they built (DRIAMS)", ha="center",
             color=TEAL, fontsize=11.5, fontweight="bold")
    for k, (big, gloss) in enumerate((("300,000+", "mass spectra"),
                                      ("750,000+", "AMR phenotypes"),
                                      ("4", "institutions"))):
        cx = 1.9 + k * 3.1
        axr.text(cx, 8.05, big, ha="center", color=INK, fontsize=14,
                 fontweight="bold")
        axr.text(cx, 7.20, gloss, ha="center", color=MUTED, fontsize=9.8)
    axr.add_patch(FancyBboxPatch((0.3, 1.15), 9.4, 4.65,
                  boxstyle="round,pad=0.05,rounding_size=0.12",
                  facecolor=AMBER_SOFT, edgecolor=AMBER, lw=2.4))
    axr.text(5.0, 5.15, "and the part that matters to a clinician",
             ha="center", color=ROI_INK, fontsize=11.5, fontweight="bold")
    axr.text(5.0, 4.25, "retrospective study of 63 patients", ha="center",
             color=INK, fontsize=11)
    for k, (big, gloss, col) in enumerate((
            ("9", "treatments would\nhave CHANGED", INK),
            ("8", "of those would have\nbeen BENEFICIAL (89%)", TEAL))):
        cx = 2.55 + k * 4.9
        axr.text(cx, 3.30, big, ha="center", color=col, fontsize=24,
                 fontweight="bold")
        axr.text(cx, 2.05, gloss, ha="center", va="center", color=INK_SOFT,
                 fontsize=9.3, linespacing=1.5)
    fig.subplots_adjust(wspace=0.28, bottom=0.22, top=0.88)
    _cite_strip(fig, "driams")
    _save(fig, name)


def tour_alphafold(name="fig_tour_alphafold.png"):
    """AlphaFold's CASP14 result in one bar pair: median backbone accuracy
    0.96 A r.m.s.d.95 (95% CI 0.85-1.16) against 2.8 A for the next best
    method — the number that ended the assessment."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.6, 4.6),
                                   gridspec_kw={"width_ratios": [1, 1.1]})
    labels = ["AlphaFold", "next best\nmethod"]
    vals = [0.96, 2.8]
    bars = axl.bar(labels, vals, color=[TEAL, "#CFE3E2"], width=0.55, zorder=3)
    axl.errorbar(0, 0.96, yerr=[[0.96 - 0.85], [1.16 - 0.96]], color=INK,
                 lw=1.8, capsize=7, zorder=5)
    for b, v in zip(bars, vals):
        axl.text(b.get_x() + b.get_width() / 2, v + (0.46 if v < 1 else 0.13),
                 f"{v:g} Å", ha="center", color=INK, fontsize=14,
                 fontweight="bold")
    axl.set_ylim(0, 3.5)
    axl.set_ylabel("median backbone error\n(Å r.m.s.d.₉₅ — lower is better)",
                   fontsize=10)
    axl.set_title("CASP14 — the blind structure-prediction assessment",
                  fontsize=11.5, color=INK, fontweight="bold")
    axl.tick_params(axis="x", labelsize=10.5)
    for sp in ("top", "right"):
        axl.spines[sp].set_visible(False)
    axl.text(0, 2.10, "95% CI\n0.85–1.16", ha="center", color=MUTED,
             fontsize=8.8, style="italic", linespacing=1.4)
    axr.set_xlim(0, 10); axr.set_ylim(0, 10); axr.axis("off")
    axr.add_patch(FancyBboxPatch((0.3, 6.2), 9.4, 3.4,
                  boxstyle="round,pad=0.05,rounding_size=0.12",
                  facecolor=TEAL_SOFT, edgecolor=TEAL, lw=2.4))
    axr.text(5.0, 8.55, "0.96 Å is about the width of an atom", ha="center",
             color=TEAL, fontsize=13.5, fontweight="bold")
    axr.text(5.0, 7.25, "near-experimental accuracy, from sequence alone —\n"
             "a problem that had stood for fifty years",
             ha="center", va="center", color=INK, fontsize=10.5,
             linespacing=1.55)
    rows = [("architecture", "attention — the Evoformer (Lectures 7–8)"),
            ("since then", "200 M+ predicted structures released"),
            ("recognition", "shared the 2024 Nobel Prize in Chemistry")]
    y = 5.3
    for head, gloss in rows:
        axr.add_patch(FancyBboxPatch((0.3, y - 1.55), 9.4, 1.50,
                      boxstyle="round,pad=0.04,rounding_size=0.09",
                      facecolor=WHITE, edgecolor=HAIRLINE, lw=1.6))
        axr.text(0.75, y - 0.80, head, fontsize=10.5, color=TEAL, va="center",
                 fontweight="bold")
        axr.text(3.5, y - 0.80, gloss, fontsize=10.2, color=INK, va="center")
        y -= 1.75
    fig.subplots_adjust(wspace=0.26, bottom=0.22, top=0.88)
    _cite_strip(fig, "alphafold")
    _save(fig, name)

def landscape_card(tool, color, problem, arch, impact, name):
    """One landmark MS/biology DL tool in the shared 'problem → architecture you
    now know → impact' frame (rule 2/8). A titled card with three labelled
    bands so every tour slide reads identically."""
    fig, ax = plt.subplots(figsize=(11.4, 4.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    # title bar
    ax.add_patch(FancyBboxPatch((0.5, 4.05), 11.0, 0.8,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=color, edgecolor=color, lw=2.0))
    ax.text(6.0, 4.45, tool, ha="center", va="center", color=WHITE,
            fontsize=16, fontweight="bold")
    bands = [
        ("PROBLEM", problem, WHITE),
        ("ARCHITECTURE", arch, TEAL_SOFT),
        ("IMPACT", impact, AMBER_SOFT),
    ]
    ys = [3.05, 1.95, 0.85]
    edges = [INK_SOFT, TEAL, AMBER]
    for (lab, body, face), y, ec in zip(bands, ys, edges):
        ax.add_patch(FancyBboxPatch((0.5, y - 0.48), 11.0, 0.96,
                    boxstyle="round,pad=0.02,rounding_size=0.05",
                    facecolor=face, edgecolor=ec, lw=1.8))
        ax.text(0.8, y, lab, ha="left", va="center", color=ec,
                fontsize=10, fontweight="bold")
        ax.text(3.55, y, body, ha="left", va="center", color=INK,
                fontsize=10.5)
    _save(fig, name)


def landscape_match(mode="ido", name="fig_landscape_match_ido.png"):
    """The landscape tour as one matching grid: tool | problem | architecture
    family | impact. mode='ido' is the full teaching grid (all five systems
    shown before the match); mode='youdo' blanks the problem + architecture
    columns and gives word banks for the spoken match (the old Quiz 13 Q1
    transfer question was removed from the sheet)."""
    rows = [
        ("Casanovo", TEAL, "de novo peptide sequencing",
         "Transformer (translation)", "reads peptides not in any database"),
        ("Prosit", AMBER, "predict fragment spectrum + RT",
         "deep sequence model (RNN/attn)", "rescores IDs → more confident hits"),
        ("DIA-NN", RED, "ID + quantify in DIA runs",
         "feed-forward neural net (MLP)", "robust high-throughput proteomics"),
        ("DRIAMS", INK_SOFT, "resistance (R/S) from MALDI-TOF",
         "1D CNN (spectrum classifier)", "faster antibiotic calls, routine data"),
        ("AlphaFold", TEAL, "3D protein structure from sequence",
         "Transformer / attention", "near-experimental accuracy — flagship"),
    ]
    fig, ax = plt.subplots(figsize=(12.8, 6.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    header = ("the tour on one page: tool → problem → architecture → impact"
              if mode == "ido" else
              "your turn: match each tool to its problem & architecture")
    hcol = INK if mode == "ido" else ROI_INK
    ax.text(6.5, 6.05, header, ha="center", color=hcol, fontsize=14.5,
            fontweight="bold")
    ax.text(1.6, 5.5, "tool", ha="center", color=MUTED, fontsize=11)
    ax.text(4.7, 5.5, "problem", ha="center", color=MUTED, fontsize=11)
    ax.text(8.0, 5.5, "architecture family", ha="center", color=MUTED,
            fontsize=11)
    ax.text(11.1, 5.5, "impact", ha="center", color=MUTED, fontsize=11)
    ys = [4.75, 3.85, 2.95, 2.05, 1.15]
    for (tool, col, prob, arch, imp), y in zip(rows, ys):
        ax.add_patch(FancyBboxPatch((0.25, y - 0.4), 2.7, 0.8,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor=col, edgecolor=col, lw=2.0))
        ax.text(1.6, y, tool, ha="center", va="center", color=WHITE,
                fontsize=11.5, fontweight="bold")
        ax.add_patch(FancyBboxPatch((3.05, y - 0.4), 3.3, 0.8,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor=WHITE, edgecolor=col, lw=1.6))
        ax.text(4.7, y, "?" if mode == "youdo" else prob, ha="center",
                va="center", color=INK if mode == "ido" else ROI_INK,
                fontsize=9.5 if mode == "ido" else 15, fontweight="bold")
        ax.add_patch(FancyBboxPatch((6.45, y - 0.4), 3.1, 0.8,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor="#F1F1EC", edgecolor=col, lw=1.6))
        ax.text(8.0, y, "?" if mode == "youdo" else arch, ha="center",
                va="center", color=INK_SOFT if mode == "ido" else ROI_INK,
                fontsize=9.5 if mode == "ido" else 15, fontweight="bold")
        ax.add_patch(FancyBboxPatch((9.65, y - 0.4), 3.1, 0.8,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor=WHITE, edgecolor=col, lw=1.6))
        ax.text(11.2, y, imp, ha="center", va="center", color=INK,
                fontsize=9, style="italic")
    if mode == "youdo":
        ax.text(6.5, 0.45,
                "problem bank: de novo sequencing · spectrum+RT prediction · "
                "DIA ID+quant · MALDI-TOF resistance · protein structure",
                ha="center", color=INK_SOFT, fontsize=8.6)
        ax.text(6.5, 0.1,
                "architecture bank: Transformer (translation) · sequence model "
                "(RNN/attn) · MLP · 1D CNN · Transformer/attention",
                ha="center", color=INK_SOFT, fontsize=8.6)
    _save(fig, name)


def rag_pipeline(name="fig_rag_pipeline.png"):
    """Open the RAG black box (the five components): TOP row builds the knowledge
    base offline (documents -> CHUNK -> EMBED -> vector store); BOTTOM row answers
    online (question -> embed -> RETRIEVE nearest chunks -> GENERATE grounded
    answer). Tiny anchor: 'today's cortisol calibration?' pulls the cortisol-log
    chunk because its embedding is nearest."""
    fig, ax = plt.subplots(figsize=(12.9, 5.7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def tb(x, y, label, fc, ec, tc=INK, fs=9.8):
        ax.text(x, y, label, ha="center", va="center", color=tc, fontsize=fs,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.42", fc=fc, ec=ec, lw=1.8))

    def ar(x0, x1, y, lab, col=INK_SOFT, ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                     mutation_scale=15, color=col, lw=1.8, linestyle=ls))
        ax.text((x0 + x1) / 2, y + 0.26, lab, ha="center", va="bottom",
                color=TEAL, fontsize=8.8, fontweight="bold")

    ax.text(0.15, 5.62, "BUILD THE KNOWLEDGE BASE  (once, offline)", ha="left",
            color=MUTED, fontsize=10.5, fontweight="bold")
    ax.text(0.15, 2.72, "ANSWER A QUESTION  (each query, online)", ha="left",
            color=MUTED, fontsize=10.5, fontweight="bold")

    yt = 4.35
    tb(1.35, yt, "your documents\n(SOPs · logs · papers)", "#F1F1EC", MUTED)
    ar(2.35, 3.35, yt, "1  chunk")
    tb(4.25, yt, "passages\n(chunks)", WHITE, INK_SOFT)
    ar(5.15, 6.35, yt, "2  embed")
    tb(7.35, yt, "chunk vectors\n■ ■ ■", TEAL_SOFT, TEAL)
    ar(8.45, 9.35, yt, "3  store")
    ax.text(11.2, 3.5, "Vector store\n(knowledge base)\nchunks + embeddings",
            ha="center", va="center", color=INK, fontsize=9.8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.8", fc=AMBER_SOFT, ec=AMBER, lw=2.2))

    yb = 1.75
    tb(1.25, yb, "question\n“today's cortisol\ncalibration?”", TEAL_SOFT, TEAL)
    ar(2.3, 3.3, yb, "embed")
    tb(4.15, yb, "query\nvector", WHITE, INK_SOFT)
    ar(5.05, 6.4, yb, "4  retrieve\nnearest")
    tb(7.4, yb, "top-k chunks\n(most similar)", WHITE, TEAL)
    ax.add_patch(FancyArrowPatch((11.2, 2.6), (7.9, yb + 0.5),
                 arrowstyle="-|>", mutation_scale=14, color=AMBER, lw=1.8,
                 linestyle="--"))
    ar(8.55, 9.5, yb, "5  generate")
    ax.text(11.2, yb, "LLM →\ngrounded answer\n+ citation", ha="center",
            va="center", color=INK, fontsize=9.8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.55", fc="#F1F1EC", ec=RED, lw=2.0))
    _save(fig, name)


def landscape_transfer(name="fig_landscape_transfer.png"):
    """The closing transfer exercise, run OUT LOUD (it is no longer a quiz
    item): transfer, not recall. Two NEW unnamed tools ->
    infer the architecture family + the most-similar landmark; plus a reasoning
    kicker on why two landmarks share an architecture. The full five-tool map is
    the I-do chart on the previous slide (the reference bank)."""
    fig, ax = plt.subplots(figsize=(12.6, 5.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(6.5, 5.6, "your turn — place tools you've never seen",
            ha="center", color=ROI_INK, fontsize=14, fontweight="bold")
    items = [
        ("(a)", "A tool reads a routine MALDI-TOF spectrum and calls the\nbacterial SPECIES (not resistance)."),
        ("(b)", "A tool reads a peptide SEQUENCE and predicts its fragment\nspectrum + retention time for a library."),
    ]
    ys = [4.35, 3.0]
    for (tag, prob), y in zip(items, ys):
        ax.text(0.5, y, tag, ha="left", va="center", color=TEAL, fontsize=13,
                fontweight="bold")
        ax.text(1.35, y + 0.2, prob, ha="left", va="center", color=INK,
                fontsize=11)
        ax.text(1.35, y - 0.44, "architecture family?  ______      most like which landmark?  ______",
                ha="left", va="center", color=MUTED, fontsize=10, fontweight="bold")
    ax.text(6.5, 1.45,
            "(c)  Casanovo and AlphaFold solve very different problems but share ONE\n"
            "architecture family — which, and what do their problems have in common\n"
            "that makes that architecture fit?",
            ha="center", va="center", color=INK, fontsize=10.8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.6", fc=TEAL_SOFT, ec=TEAL, lw=1.8))
    ax.text(6.5, 0.3, "reference: the five-tool map on the previous slide",
            ha="center", color=MUTED, fontsize=9.5, style="italic")
    _save(fig, name)


def _lecture13_figures():
    next_token_predict()
    softmax_next_token()
    llm_pipeline("intro", "fig_llm_pipeline.png")
    llm_scale()
    hallucination("ido", "fig_hallucination_ido.png")
    hallucination("youdo", "fig_hallucination_youdo.png")
    rl_loop("ido", "fig_rl_loop_ido.png")
    rl_anatomy()
    markov_property()
    explore_exploit("ido", "fig_explore_ido.png")
    explore_exploit("youdo", "fig_explore_youdo.png")
    discounted_return()
    rl_loop("youdo", "fig_rl_loop_youdo.png")
    rlhf_as_rl()
    rlhf_pipeline()
    rlhf_limits()
    use_llm_chart("ido", "fig_use_llm_ido.png")
    use_llm_chart("youdo", "fig_use_llm_youdo.png")
    tour_casanovo()
    tour_prosit()
    tour_diann()
    tour_driams()
    tour_alphafold()
    landscape_card("Casanovo / DeepNovo", TEAL,
                   "de novo peptide sequencing — read the peptide from an MS/MS spectrum, no database",
                   "Transformer encoder–decoder — spectrum → peptide (Lec 7–8)",
                   "identifies peptides missing from any database (novel, mutated, immunopeptides)",
                   "fig_land_casanovo.png")
    landscape_card("Prosit / AlphaPeptDeep", AMBER,
                   "predict a peptide's fragment spectrum + retention time from its sequence",
                   "deep sequence model — RNN/LSTM + attention (Lec 7–8)",
                   "predicted spectra rescore search hits → more confident IDs; power DIA libraries",
                   "fig_land_prosit.png")
    landscape_card("DIA-NN", RED,
                   "identify & quantify peptides in DIA runs where many signals overlap",
                   "feed-forward neural nets — MLP (Lec 1–4)",
                   "robust, high-throughput DIA proteomics — thousands of proteins per run",
                   "fig_land_diann.png")
    landscape_card("DRIAMS — MALDI-TOF resistance", INK_SOFT,
                   "predict antibiotic resistance (R/S) from a routine clinical MALDI-TOF spectrum",
                   "1D CNN — spectrum classifier (Lec 5)",
                   "faster resistance calls from data the lab already collects — earlier right antibiotic",
                   "fig_land_driams.png")
    landscape_match("ido", "fig_landscape_match_ido.png")
    landscape_transfer()
    rag_pipeline()


# =====================================================================
#  Lecture 14 · Agentic AI, Demo-Driven. An agent = LLM + tools + a
#  reason–act loop (ReAct). The spine is tracing a ReAct transcript and
#  LABELLING each step reason vs act, so the transcript figures are the
#  centrepiece (rule 6/9). Everything hand-authored, MS-anchored, offline.
# =====================================================================

def agent_schematic(name="fig_agent_schematic.png"):
    """From chatbot to agent. Left: a plain CHATBOT (LLM text-in → text-out,
    answers from what it already knows). Right: an AGENT = LLM + TOOLS + a
    reason→act LOOP (+ memory) that can read files, run code, plot, search."""
    fig, ax = plt.subplots(figsize=(12.2, 5.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # ---- left: chatbot ----
    ax.add_patch(FancyBboxPatch((0.4, 0.6), 4.6, 4.8,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor="#F1F1EC", edgecolor=HAIRLINE, lw=1.6))
    ax.text(2.7, 5.0, "CHATBOT", ha="center", color=INK_SOFT, fontsize=13,
            fontweight="bold")
    ax.add_patch(FancyBboxPatch((1.35, 2.55), 2.7, 1.4,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=TEAL_SOFT, edgecolor=TEAL, lw=2.4))
    ax.text(2.7, 3.25, "LLM", ha="center", va="center", color=TEAL,
            fontsize=17, fontweight="bold")
    ax.annotate("", xy=(1.3, 3.25), xytext=(0.55, 3.25),
                arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, lw=2.0))
    ax.text(0.9, 3.62, "prompt", ha="center", color=INK_SOFT, fontsize=10)
    ax.annotate("", xy=(4.85, 3.25), xytext=(4.1, 3.25),
                arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, lw=2.0))
    ax.text(4.5, 3.62, "reply", ha="center", color=INK_SOFT, fontsize=10)
    ax.text(2.7, 1.35, "answers only from what\nit already knows", ha="center",
            va="center", color=MUTED, fontsize=10.5, style="italic")

    # ---- right: agent ----
    ax.add_patch(FancyBboxPatch((5.6, 0.6), 7.0, 4.8,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=PAPER, edgecolor=TEAL, lw=2.2))
    ax.text(9.1, 5.0, "AGENT  =  LLM  +  TOOLS  +  LOOP", ha="center",
            color=TEAL, fontsize=13.5, fontweight="bold")
    # LLM (the reasoner) on the left of the agent panel
    ax.add_patch(FancyBboxPatch((6.0, 2.55), 2.3, 1.4,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=TEAL_SOFT, edgecolor=TEAL, lw=2.4))
    ax.text(7.15, 3.35, "LLM", ha="center", va="center", color=TEAL,
            fontsize=15, fontweight="bold")
    ax.text(7.15, 2.85, "reasons", ha="center", va="center", color=INK_SOFT,
            fontsize=9.5, style="italic")
    # TOOLS box on the right
    ax.add_patch(FancyBboxPatch((10.0, 2.35), 2.3, 1.8,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=AMBER_SOFT, edgecolor=AMBER, lw=2.4))
    ax.text(11.15, 3.85, "TOOLS", ha="center", color=ROI_INK, fontsize=12,
            fontweight="bold")
    for i, t in enumerate(["read files", "run code", "plot", "search"]):
        ax.text(11.15, 3.4 - i * 0.34, t, ha="center", color=INK,
                fontsize=9.5)
    # loop arrows: LLM -> tools (act) top, tools -> LLM (observe) bottom
    ax.annotate("", xy=(9.95, 3.75), xytext=(8.35, 3.75),
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=2.2,
                                connectionstyle="arc3,rad=-0.28"))
    ax.text(9.15, 4.4, "act (call a tool)", ha="center", color=ROI_INK,
            fontsize=9.5, fontweight="bold")
    ax.annotate("", xy=(8.35, 2.75), xytext=(9.95, 2.75),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.2,
                                connectionstyle="arc3,rad=-0.28"))
    ax.text(9.15, 2.05, "observe the result", ha="center", color=RED,
            fontsize=9.5, fontweight="bold")
    # memory strip along the bottom
    ax.add_patch(FancyBboxPatch((6.0, 0.95), 6.3, 0.55,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor="#F1F1EC", edgecolor=INK_SOFT, lw=1.4))
    ax.text(9.15, 1.22, "MEMORY — keeps what it has seen and done, and repeats",
            ha="center", va="center", color=INK_SOFT, fontsize=10)
    _save(fig, name)


# ---- the ReAct transcript (the centrepiece) --------------------------------
# One shared transcript body so the I-do (colour-coded, reason vs act named)
# and the you-do (labels blanked, room fills R/A) read identically. The
# you-do transcript is REPRODUCED VERBATIM on Quiz 14 Q1, and the correct
# labels are fixed here: steps 1/3/5 = reason (Thought), 2/4/6 = act (Action).

_REACT_IDO = [
    ("thought", "To find anomalies I first need to open the QC table and see its columns."),
    ("action", 'read_csv("qc_runs.csv")'),
    ("obs", "8 runs · columns: run_id, mass_error_ppm, resolution, TIC"),
    ("thought", "Mass error must stay within ±2 ppm — I'll check each run against that limit."),
    ("action", "filter |mass_error_ppm| > 2"),
    ("obs", "run R-07 → mass_error_ppm = 5.3"),
    ("thought", "R-07 is far outside ±2 ppm, so it is the anomaly to flag."),
    ("action", 'draft_summary(flag="R-07")'),
    ("answer", "R-07 is out of spec (5.3 ppm); recommend re-run. The other 7 runs pass."),
]

# you-do / quiz transcript: SAME ReAct shape, different task + numbers.
# (step_text, correct_label) where correct_label in {'R','A'}; observations
# carry label None and are shown as un-numbered context.
_REACT_YOUDO = [
    ("Before I can compare runs, I need to load results.csv and see its columns.", "R"),
    ('read_csv("results.csv")', "A"),
    ("6 runs · column: peptide_ids", None),
    ("I'll compute the average ID count so I can see which runs fall below it.", "R"),
    ("compute the mean of peptide_ids across the 6 runs", "A"),
    ("mean = 1,850 · run B-04 = 420", None),
    ("B-04 (420) is far below the mean of 1,850, so that run underperformed.", "R"),
    ('write_summary(flag="B-04")', "A"),
    ("summary drafted", None),
    ('Answer: "B-04 is low (420 vs mean 1,850). The cause is a clogged '
     'emitter — I recommend replacing it."', "ANS"),
]


def react_transcript(mode="ido", name="fig_react_transcript_ido.png"):
    """The ReAct pattern annotated on a real-looking MS-assistant transcript.
    mode='ido': colour-code Thought (reason, teal) → Action (act, amber) →
    Observation (grey) → … → Answer (red), with a reason/act legend.
    mode='youdo': the SAME-shape quiz transcript with an 'R or A?' blank beside
    each numbered agent step (Quiz 14 Q1)."""
    tagcol = {"thought": TEAL, "action": ROI_INK, "obs": MUTED, "answer": RED}
    tagface = {"thought": TEAL_SOFT, "action": AMBER_SOFT, "obs": "#F1F1EC",
               "answer": "#F7E4E3"}
    taglab = {"thought": "THOUGHT", "action": "ACTION", "obs": "OBSERVATION",
              "answer": "ANSWER"}
    if mode == "ido":
        rows = _REACT_IDO
        fig, ax = plt.subplots(figsize=(12.4, 6.2))
    else:
        rows = _REACT_YOUDO
        fig, ax = plt.subplots(figsize=(12.4, 5.6))
    ax.set_xlim(0, 13)
    n = len(rows)
    ax.set_ylim(0, n + (2.3 if mode == "ido" else 1.4))
    ax.axis("off")

    if mode == "ido":
        ax.text(6.5, n + 1.75,
                "ReAct loop:  Thought → Action → Observation → … → Answer",
                ha="center", color=INK, fontsize=14, fontweight="bold")
        # legend: reason vs act (well above the first row)
        ax.add_patch(FancyBboxPatch((0.4, n + 0.85), 0.34, 0.34,
                    boxstyle="round,pad=0.02", facecolor=TEAL_SOFT,
                    edgecolor=TEAL, lw=1.6))
        ax.text(0.85, n + 1.02, "Thought = REASON", ha="left", va="center",
                color=TEAL, fontsize=10.5, fontweight="bold")
        ax.add_patch(FancyBboxPatch((4.6, n + 0.85), 0.34, 0.34,
                    boxstyle="round,pad=0.02", facecolor=AMBER_SOFT,
                    edgecolor=AMBER, lw=1.6))
        ax.text(5.05, n + 1.02, "Action = ACT (call a tool: read / compute / write)",
                ha="left", va="center", color=ROI_INK, fontsize=10.5,
                fontweight="bold")
        # cue: this demo table is separate from the handout QC table
        ax.text(6.5, 0.28,
                "a short standalone demo table (qc_runs.csv) — separate from "
                "the QC table on your handout",
                ha="center", va="center", color=MUTED, fontsize=9.5,
                style="italic")
        for i, (kind, text) in enumerate(rows):
            y = n - i
            ax.add_patch(FancyBboxPatch((0.4, y - 0.34), 2.15, 0.68,
                        boxstyle="round,pad=0.02,rounding_size=0.06",
                        facecolor=tagface[kind], edgecolor=tagcol[kind],
                        lw=1.8))
            ax.text(1.47, y, taglab[kind], ha="center", va="center",
                    color=tagcol[kind], fontsize=9.5, fontweight="bold")
            fam = "monospace" if kind == "action" else None
            ax.text(2.8, y, text, ha="left", va="center", color=INK,
                    fontsize=11 if kind != "obs" else 10.2,
                    style="italic" if kind == "obs" else "normal",
                    family=fam)
    else:
        ax.text(6.5, n + 0.9,
                "your turn — (a) label each step  R (reason)  or  A (act)   ·   "
                "(b) spot the claim no tool showed",
                ha="center", color=ROI_INK, fontsize=12.5, fontweight="bold")
        step = 0
        for i, (text, lab) in enumerate(rows):
            y = n - i
            if lab is None:  # observation context, not a numbered step
                ax.text(3.0, y, "→ " + text, ha="left", va="center",
                        color=MUTED, fontsize=10, style="italic")
                continue
            if lab == "ANS":  # final answer — hides one unsupported claim
                ax.add_patch(FancyBboxPatch((0.4, y - 0.34), 12.3, 0.68,
                            boxstyle="round,pad=0.02,rounding_size=0.06",
                            facecolor="#F7E4E3", edgecolor=RED, lw=1.9))
                ax.text(0.62, y, "ANSWER", ha="left", va="center", color=RED,
                        fontsize=8.5, fontweight="bold")
                ax.text(2.15, y, text, ha="left", va="center", color=INK,
                        fontsize=10.2)
                continue
            step += 1
            ax.add_patch(FancyBboxPatch((0.4, y - 0.32), 0.6, 0.64,
                        boxstyle="round,pad=0.02,rounding_size=0.06",
                        facecolor=TEAL_SOFT, edgecolor=TEAL, lw=1.6))
            ax.text(0.7, y, str(step), ha="center", va="center", color=TEAL,
                    fontsize=12, fontweight="bold")
            fam = "monospace" if text.endswith(")") else None
            ax.text(1.25, y, text, ha="left", va="center", color=INK,
                    fontsize=11, family=fam)
            # blank for the label
            ax.add_patch(FancyBboxPatch((11.7, y - 0.3), 1.0, 0.6,
                        boxstyle="round,pad=0.02,rounding_size=0.06",
                        facecolor=AMBER_SOFT, edgecolor=AMBER, lw=1.6))
            ax.text(12.2, y, "R / A", ha="center", va="center", color=ROI_INK,
                    fontsize=10, fontweight="bold")
        ax.text(6.5, 0.28,
                "(b) the ANSWER states a CAUSE no tool ever established — which, "
                "and where should it have stopped for a human?",
                ha="center", va="center", color=RED, fontsize=9.5,
                style="italic")
    _save(fig, name)


def followalong_card(name="fig_followalong_card.png"):
    """The reason→act→check prompt template the room pastes into their own
    free-tier bot (ChatGPT / Claude / Gemini), with the handout QC table."""
    fig, ax = plt.subplots(figsize=(12.0, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(6.0, 5.65, "paste this into your own bot — ask for three labelled steps",
            ha="center", color=INK, fontsize=13.5, fontweight="bold")
    steps = [
        ("REASON", TEAL, TEAL_SOFT,
         "state what a QC anomaly looks like and\nwhich limit you will check"),
        ("ACT", AMBER, AMBER_SOFT,
         "check each run against that limit and\nname any out-of-range run"),
        ("CHECK", RED, "#F7E4E3",
         "say what a human must verify before\nthe answer is trusted"),
    ]
    ys = [4.35, 3.05, 1.75]
    for (lab, col, face, body), y in zip(steps, ys):
        ax.add_patch(FancyBboxPatch((0.6, y - 0.55), 2.5, 1.1,
                    boxstyle="round,pad=0.02,rounding_size=0.08",
                    facecolor=col, edgecolor=col, lw=2.0))
        ax.text(1.85, y, lab, ha="center", va="center", color=WHITE,
                fontsize=14, fontweight="bold")
        ax.annotate("", xy=(3.55, y), xytext=(3.15, y),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0))
        ax.add_patch(FancyBboxPatch((3.7, y - 0.55), 7.7, 1.1,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor=face, edgecolor=col, lw=1.6))
        ax.text(7.55, y, body, ha="center", va="center", color=INK,
                fontsize=11)
    ax.text(6.0, 0.75,
            "… then paste the QC table from your handout, and compare answers with a neighbour",
            ha="center", color=INK_SOFT, fontsize=10.5, style="italic")
    _save(fig, name)


def followalong_debrief(name="fig_followalong_debrief.png"):
    """What a GOOD reason→act→check answer shows — the debrief checklist for the
    do-it-together (a genuine 3-item check, MS-anchored to the QC table)."""
    fig, ax = plt.subplots(figsize=(11.6, 4.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.text(6.0, 4.6, "a good answer — what to look for", ha="center",
            color=INK, fontsize=13.5, fontweight="bold")
    items = [
        ("shows its REASON before it ACTS",
         "it states the ±2 ppm / resolution limit first, then checks the runs"),
        ("flags the right run and no other",
         "QC-04 — and it names WHICH metrics are out of range"),
        ("invents no number that isn't in the table",
         "every value it cites is in the pasted QC table (hallucination check)"),
        ("defers the final pass/fail to a human",
         "it drafts; a qualified person signs off before results are used"),
    ]
    ys = [3.7, 2.85, 2.0, 1.15]
    for (head, body), y in zip(items, ys):
        ax.text(0.7, y, "✓", ha="center", va="center", color=TEAL,
                fontsize=17, fontweight="bold")
        ax.text(1.15, y, head, ha="left", va="center", color=INK,
                fontsize=12, fontweight="bold")
        ax.text(6.6, y, body, ha="left", va="center", color=INK_SOFT,
                fontsize=10.5, style="italic")
    _save(fig, name)


def demo_storyboard(name="fig_demo_storyboard.png"):
    """Live-demo storyboard: the MS analysis assistant reads → analyzes → plots
    → flags → drafts, each panel a reason→act beat the instructor narrates."""
    fig, ax = plt.subplots(figsize=(12.8, 4.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.text(6.5, 4.65, "instructor live demo — the 'MS analysis assistant', narrated",
            ha="center", color=INK, fontsize=13.5, fontweight="bold")
    panels = [
        ("READ", TEAL, "open the\nQC / spectra CSV"),
        ("ANALYZE", AMBER, "run the checks\n(limits, stats)"),
        ("PLOT", TEAL, "draw the\nspectra / trends"),
        ("FLAG", RED, "mark the\nout-of-range run"),
        ("DRAFT", INK_SOFT, "write the QC\nsummary for review"),
    ]
    w = 2.15
    gap = 0.42
    x0 = 0.5
    for i, (lab, col, body) in enumerate(panels):
        cx = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((cx, 1.35), w, 2.5,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor=PAPER, edgecolor=col, lw=2.2))
        ax.add_patch(FancyBboxPatch((cx, 3.15), w, 0.7,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor=col, edgecolor=col, lw=2.0))
        ax.text(cx + w / 2, 3.5, lab, ha="center", va="center", color=WHITE,
                fontsize=12.5, fontweight="bold")
        ax.text(cx + w / 2, 2.25, body, ha="center", va="center", color=INK,
                fontsize=10.5)
        if i < len(panels) - 1:
            ax.annotate("", xy=(cx + w + gap - 0.06, 2.6),
                        xytext=(cx + w + 0.06, 2.6),
                        arrowprops=dict(arrowstyle="-|>", color=INK_SOFT,
                                        lw=2.0))
    ax.text(6.5, 0.7,
            "each panel = one reason→act step · primary: Claude Code  ·  alt: Bedrock API  ·  fallback: pre-recorded run",
            ha="center", color=INK_SOFT, fontsize=10.5, style="italic")
    _save(fig, name)


def demo_flag(name="fig_demo_flag.png"):
    """The flag+draft beat of the demo made concrete: a mini QC table with one
    clearly out-of-range row highlighted, and the agent's drafted summary —
    reason then act. Mirrors the handout table so the room recognises it."""
    fig, ax = plt.subplots(figsize=(12.2, 5.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(6.0, 5.65, "FLAG → DRAFT: the agent finds the bad run, then writes it up",
            ha="center", color=INK, fontsize=13, fontweight="bold")
    # mini table
    headers = ["run", "mass err (ppm)", "resolution", "TIC"]
    data = [
        ("QC-01", "+0.8", "42,000", "10.2", False),
        ("QC-04", "+5.6", "22,300", "3.1", True),
        ("QC-05", "-0.9", "40,100", "9.9", False),
    ]
    cxs = [1.1, 2.7, 4.5, 6.1]
    ytop = 4.7
    for cx, h in zip(cxs, headers):
        ax.text(cx, ytop, h, ha="center", color=MUTED, fontsize=10.5,
                fontweight="bold")
    for r, (run, me, res, tic, bad) in enumerate(data):
        y = ytop - 0.55 - r * 0.55
        if bad:
            ax.add_patch(FancyBboxPatch((0.7, y - 0.24), 6.0, 0.48,
                        boxstyle="round,pad=0.01,rounding_size=0.04",
                        facecolor="#F7E4E3", edgecolor=RED, lw=1.8, zorder=1))
        col = RED if bad else INK
        for cx, v in zip(cxs, [run, me, res, tic]):
            ax.text(cx, y, v, ha="center", va="center", color=col,
                    fontsize=10.5, fontweight="bold" if bad else "normal",
                    zorder=2)
    ax.text(6.8, ytop - 0.55 - 1 * 0.55 - 0.34, "↑ out of spec", ha="left",
            va="center", color=RED, fontsize=9.5, fontweight="bold")
    # reason + act narration
    ax.add_patch(FancyBboxPatch((8.2, 3.1), 3.5, 1.5,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=TEAL_SOFT, edgecolor=TEAL, lw=1.8))
    ax.text(8.45, 4.3, "REASON", ha="left", color=TEAL, fontsize=10,
            fontweight="bold")
    ax.text(8.45, 3.75,
            "QC-04 breaks all three\nlimits (±2 ppm, >30k, TIC 8–12)",
            ha="left", va="center", color=INK, fontsize=10)
    # drafted summary bubble
    ax.add_patch(FancyBboxPatch((0.7, 0.6), 11.0, 1.9,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor="#F1F1EC", edgecolor=INK_SOFT, lw=1.8))
    ax.text(1.05, 2.2, "ACT — drafted summary (for human review)", ha="left",
            color=ROI_INK, fontsize=10.5, fontweight="bold")
    ax.text(1.05, 1.5,
            "“Run QC-04 failed QC: mass error +5.6 ppm (limit ±2), resolution 22,300\n"
            "(limit >30,000), TIC 3.1 (limit 8–12). Recommend recalibrate + re-run.\n"
            "Runs QC-01/02/03/05/06 pass. Please verify before releasing results.”",
            ha="left", va="center", color=INK, fontsize=10.5)
    _save(fig, name)


def agents_fit(name="fig_agents_fit.png"):
    """Where agents fit in the lab (good first fits, each with a check) and where
    they must NOT run unsupervised — plus the three guardrails."""
    fig, ax = plt.subplots(figsize=(12.6, 5.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    # left column: good fits
    ax.add_patch(FancyBboxPatch((0.4, 1.35), 6.0, 4.4,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                facecolor=TEAL_SOFT, edgecolor=TEAL, lw=2.0))
    ax.text(3.4, 5.45, "GOOD FIRST FITS  (with a check)", ha="center",
            color=TEAL, fontsize=12.5, fontweight="bold")
    fits = [
        ("triage", "flag runs for a human to look at"),
        ("report drafting", "human verifies every number/citation"),
        ("data wrangling", "work on copies; spot-check + log"),
        ("literature", "verify each citation against the source"),
    ]
    for i, (t, chk) in enumerate(fits):
        y = 4.75 - i * 0.85
        ax.text(0.75, y, "✓", ha="center", color=TEAL, fontsize=15,
                fontweight="bold")
        ax.text(1.15, y, t, ha="left", va="center", color=INK, fontsize=11.5,
                fontweight="bold")
        ax.text(1.15, y - 0.34, chk, ha="left", va="center", color=INK_SOFT,
                fontsize=9.5, style="italic")
    # right column: must NOT run unsupervised
    ax.add_patch(FancyBboxPatch((6.7, 1.35), 5.9, 4.4,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                facecolor="#F7E4E3", edgecolor=RED, lw=2.0))
    ax.text(9.65, 5.45, "MUST NOT RUN UNSUPERVISED", ha="center", color=RED,
            fontsize=12.5, fontweight="bold")
    nots = [
        "releasing a patient report / signing results",
        "the final clinical call (diagnosis, R/S)",
        "deleting or overwriting raw data",
        "ordering reagents / anything irreversible",
    ]
    for i, t in enumerate(nots):
        y = 4.75 - i * 0.85
        ax.text(7.05, y, "✗", ha="center", color=RED, fontsize=15,
                fontweight="bold")
        ax.text(7.45, y, t, ha="left", va="center", color=INK, fontsize=11)
    ax.text(6.5, 0.7,
            "guardrails on every fit:   human review   ·   restricted tools   ·   logging",
            ha="center", color=INK, fontsize=11.5, fontweight="bold")
    _save(fig, name)


# ---- guardrails: failure → guardrail (I-do) / match (you-do = Q2) ----------
_GUARDRAILS = [
    ("the agent writes a confident summary with a\nfabricated number or citation",
     "HUMAN REVIEW",
     "a person verifies every number and flag\nbefore anything is released"),
    ("the agent deletes or overwrites the raw data\nfile (or orders reagents) on its own",
     "RESTRICTED TOOLS",
     "give it read-only access; no write / delete /\npurchase without approval"),
    ("a wrong result ships and no one can tell what\nthe agent did or reproduce it",
     "LOGGING",
     "record every reason→act step so the run is\nauditable and reproducible"),
]

# you-do = Quiz 14 Q2: NEW, disguised failures (different wording from the I-do)
# so the room must reason from what each guardrail does, not pattern-match.
_GUARDRAILS_YOUDO = [
    ("the agent's draft cites a reference range\nthat appears in no SOP",
     "HUMAN REVIEW", ""),
    ("told to “tidy old files,” it overwrites\nlast month's raw .raw files",
     "RESTRICTED TOOLS", ""),
    ("a released result was wrong and the lab can't\nreconstruct which files it used",
     "LOGGING", ""),
]


def guardrails(mode="ido", name="fig_guardrails_ido.png"):
    """Three failure modes, each with the guardrail that prevents it. mode='ido'
    pairs failure → guardrail → what it does; mode='youdo' shows the three
    failures with the guardrail column blanked + a word bank (Quiz 14 Q2)."""
    fig, ax = plt.subplots(figsize=(12.8, 5.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")
    if mode == "ido":
        ax.text(6.5, 5.6, "each failure mode has a guardrail", ha="center",
                color=INK, fontsize=14, fontweight="bold")
    else:
        ax.text(6.5, 5.6, "your turn — which guardrail stops which failure?",
                ha="center", color=ROI_INK, fontsize=14, fontweight="bold")
    ax.text(2.4, 4.95, "failure mode", ha="center", color=MUTED, fontsize=11)
    ax.text(7.4, 4.95, "guardrail", ha="center", color=MUTED, fontsize=11)
    if mode == "ido":
        ax.text(11.0, 4.95, "what it does", ha="center", color=MUTED,
                fontsize=11)
    ys = [3.9, 2.55, 1.2]
    cols = [RED, AMBER, TEAL]
    _rows = _GUARDRAILS if mode == "ido" else _GUARDRAILS_YOUDO
    for (fail, guard, does), y, col in zip(_rows, ys, cols):
        ax.add_patch(FancyBboxPatch((0.4, y - 0.55), 4.0, 1.1,
                    boxstyle="round,pad=0.02,rounding_size=0.05",
                    facecolor="#F7E4E3", edgecolor=RED, lw=1.8))
        ax.text(2.4, y, fail, ha="center", va="center", color=INK,
                fontsize=10)
        ax.annotate("", xy=(5.5, y), xytext=(4.55, y),
                    arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, lw=1.8))
        if mode == "ido":
            ax.add_patch(FancyBboxPatch((5.6, y - 0.5), 3.6, 1.0,
                        boxstyle="round,pad=0.02,rounding_size=0.06",
                        facecolor=col, edgecolor=col, lw=2.0))
            ax.text(7.4, y, guard, ha="center", va="center", color=WHITE,
                    fontsize=12.5, fontweight="bold")
            ax.add_patch(FancyBboxPatch((9.25, y - 0.5), 3.6, 1.0,
                        boxstyle="round,pad=0.02,rounding_size=0.06",
                        facecolor=WHITE, edgecolor=col, lw=1.6))
            ax.text(11.05, y, does, ha="center", va="center", color=INK_SOFT,
                    fontsize=9.0)
        else:
            ax.add_patch(FancyBboxPatch((5.6, y - 0.5), 3.6, 1.0,
                        boxstyle="round,pad=0.02,rounding_size=0.06",
                        facecolor=AMBER_SOFT, edgecolor=AMBER, lw=1.8))
            ax.text(7.4, y, "?", ha="center", va="center", color=ROI_INK,
                    fontsize=16, fontweight="bold")
    if mode == "youdo":
        ax.text(6.5, 0.35,
                "word bank:   HUMAN REVIEW   ·   RESTRICTED TOOLS   ·   LOGGING",
                ha="center", color=INK_SOFT, fontsize=12, fontweight="bold")
    _save(fig, name)


def human_review(name="fig_human_review.png"):
    """Clinical workflow strip for Quiz 14 Q3: which steps may an agent do and
    which MUST stay human-reviewed before they reach a patient. The room marks
    the human gate."""
    fig, ax = plt.subplots(figsize=(12.6, 4.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    ax.text(6.5, 4.85,
            "your turn — which steps must a qualified human review & sign?",
            ha="center", color=ROI_INK, fontsize=13.5, fontweight="bold")
    steps = [
        ("(a)", "HOLD a suspect run\nfor human review", "agent"),
        ("(b)", "AUTO-RELEASE runs\nit judges in-range", "human"),
        ("(c)", "file the final R/S\ncall in the report", "human"),
        ("(d)", "reformat QC table\nto house template", "agent"),
        ("(e)", "draft a plain note\nexplaining a result", "agent"),
    ]
    w = 2.2
    gap = 0.35
    x0 = 0.45
    for i, (tag, body, who) in enumerate(steps):
        cx = x0 + i * (w + gap)
        # neutral cards on the you-do (the room decides who); tag only
        ax.add_patch(FancyBboxPatch((cx, 1.7), w, 2.2,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor=PAPER, edgecolor=INK_SOFT, lw=1.8))
        ax.text(cx + w / 2, 3.55, tag, ha="center", va="center",
                color=INK_SOFT, fontsize=12, fontweight="bold")
        ax.text(cx + w / 2, 2.6, body, ha="center", va="center", color=INK,
                fontsize=10.5)
        # a blank 'human gate?' checkbox under each
        ax.add_patch(FancyBboxPatch((cx + w / 2 - 0.3, 1.1, ), 0.6, 0.45,
                    boxstyle="round,pad=0.02,rounding_size=0.05",
                    facecolor=AMBER_SOFT, edgecolor=AMBER, lw=1.6))
        ax.text(cx + w / 2, 1.32, "human?", ha="center", va="center",
                color=ROI_INK, fontsize=9, fontweight="bold")
    ax.text(6.5, 0.5,
            "rule of thumb: anything that becomes a released result or a "
            "patient-affecting decision needs the human gate",
            ha="center", color=INK_SOFT, fontsize=10.5, style="italic")
    _save(fig, name)


def _lecture14_figures():
    agent_schematic()
    react_transcript("ido", "fig_react_transcript_ido.png")
    react_transcript("youdo", "fig_react_transcript_youdo.png")
    followalong_card()
    followalong_debrief()
    demo_storyboard()
    demo_flag()
    agents_fit()
    guardrails("ido", "fig_guardrails_ido.png")
    guardrails("youdo", "fig_guardrails_youdo.png")
    human_review()


FUNCS = {
    "maldi": maldi_real,
    "chatgpt": chatgpt_panel,
    "driams": driams_mrsa_spectrum,
    "activations": lambda: (
        activations(True, "fig_activations.png", ("relu", "sigmoid", "linear")),
        activations(False, "fig_activations_quiz.png", ("sigmoid", "relu", "linear")),
        activation_apply("fig_activation_apply.png"),
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
    "loop_bug": loop_bug,
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
    "treatments": lambda: (treatment_menu(), curation_pass(),
                           embedding_dedup(),
                           class_weight_family(), smote_interpolation(
                               "ido", "fig_smote.png"),
                           focal_loss_volume(), curriculum_learning()),
    "embed_dedup": embedding_dedup,
    "roc_intro": lambda: (score_threshold_sweep(), roc_curve_intro(),
                          pr_curve_intro(), roc_pr_curves()),
    # ---- Lecture 5 ----
    "cnn": _lecture5_figures,
    "dense_explosion": dense_explosion,
    "conv_youdo": lambda: conv_walk_two_strides(
        CONV_IMG_QUIZ, CONV_FILT_QUIZ, name="fig_conv_youdo.png",
        filt_note="corners + centre",
        note="your turn — both whole maps, new image and filter (Quiz 5 Q1)"),
    "conv_real_filters": conv_real_filters,
    "channels": lambda: (
        channels_in(name="fig_channels_in.png",
                    credit="worked example after Zhang et al., Dive into Deep "
                           "Learning \u00a77.4 (CC BY-SA 4.0)"),
        channels_out(name="fig_channels_out.png",
                     credit="filters K, K+1, K+2 and every map value after d2l "
                            "\u00a77.4 (CC BY-SA 4.0)"),
        channels_in(CHAN_QUIZ_X, CHAN_QUIZ_K, reveal=False,
                    name="fig_channels_youdo.png",
                    credit="your turn \u2014 same drill, new numbers (Quiz 5 Q5)"),
    ),
    "conv1d": lambda: (conv1d_spectrum(), conv1d_code()),
    "detection": lambda: (
        detection_grid(),
        peak_detection(reveal=True, name="fig_peak_detection.png"),
        peak_detection_decide(name="fig_peak_detection_youdo.png"),
    ),
    # ---- Lecture 7 ----
    "attention": _lecture7_figures,
    "seq_order": seq_order_matters,
    "tokenize": tokenize_sentence,
    "embedding": embedding_space,
    "selfattn": selfattn_all_to_all,
    "qkv_projection": qkv_projection,
    "denoising": lambda: (denoising_autoencoder(), denoising_results()),
    "qkv": lambda: (
        qkv_lookup(reveal=True, name="fig_qkv_ido.png"),
    ),
    "attn_heatmap": lambda: (
        attention_heatmap("sentence", "fig_attn_heatmap_ido.png"),
        attention_heatmap("youdo", "fig_attn_heatmap_youdo.png"),
    ),
    "attn_formula": attention_formula,
    "attn_head": lambda: (
        attention_full_head((1, 1), [(1, 1), (1, 0), (2, -2)], [10, 4, 1],
                            "fig_attn_head_ido.png", reveal=True,
                            tokens=["token 1", "token 2", "token 3"]),
        attention_full_head((1, 1), [(-1, 1), (1, 1), (1, 0)], [4, 9, 1],
                            "fig_attn_head_youdo.png", reveal=False,
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
    ),
    "bert_vs_gpt": bert_vs_gpt,
    "tokenization": lambda: (
        tokenization_strategies(),
        tokenization_decision("ido", "fig_tokenization_decision.png"),
        tokenization_decision("youdo", "fig_tokenization_youdo.png"),
    ),
    "transfer": transfer_learning,
    "diffusion": lambda: (
        diffusion_strip(),
        diffusion_objective(DIFF_IDO, reveal=True,
                            name="fig_diffusion_train.png"),
        diffusion_objective(DIFF_QUIZ, reveal=False,
                            name="fig_diffusion_youdo.png",
                            note="your turn — same two steps, new numbers (Quiz 8 Q5)"),
    ),
    # ---- Lecture 10 (imbalanced data: evaluation + treatment) ----
    "metrics": _lecture10_figures,
    "imbalanced": _lecture10_figures,
    "confusion": lambda: (
        confusion_matrix(30, 11, 60, 637, "ido", "fig_confusion_ido.png"),
        confusion_matrix(16, 4, 32, 448, "youdo", "fig_confusion_youdo.png"),
    ),
    "roc_pr": roc_pr_curves,
    "focal_volume": focal_loss_volume,
    "treatment_menu": treatment_menu,
    "smote": lambda: smote_interpolation("ido", "fig_smote.png"),
    "curriculum": curriculum_learning,
    "shift_drop": distribution_shift_drop,
    # ---- Lecture 11 ----
    "unsupervised": _lecture11_figures,
    "ae_bottleneck": autoencoder_bottleneck,
    "anomaly_overlay": anomaly_overlay,
    "recon_error": lambda: (
        recon_error_worked("ido", "fig_recon_error_ido.png",
                           x=(0.2, 0.5, 0.1), xhat=(0.2, 0.4, 0.1), thr=0.01),
        recon_aggregation_trap(),
    ),
    "recon_trap": recon_aggregation_trap,
    "ae_vs_vae": lambda: (
        ae_vs_vae_latent(("autoencoder latent", "VAE latent"),
                         "fig_ae_vs_vae_latent.png"),
        ae_vs_vae_latent(("Picture A", "Picture B"), "fig_latent_ab_quiz.png",
                         describe=False),
    ),
    "vae_recipe": vae_recipe,
    "latent_interp": latent_interpolation,
    "masking": masking_pretext,
    "contrastive": contrastive_views,
    "dino": dino_student_teacher,
    "pseudolabel": pseudo_labeling,
    "paradigm": lambda: (
        paradigm_chart("ido", "fig_paradigm_chart.png"),
        semisup_scenario(),
    ),
    "semisup": semisup_scenario,
    # ---- Lecture 14 ----
    "agents": _lecture14_figures,
    "agent_schematic": agent_schematic,
    "react": lambda: (
        react_transcript("ido", "fig_react_transcript_ido.png"),
        react_transcript("youdo", "fig_react_transcript_youdo.png"),
    ),
    "followalong": lambda: (followalong_card(), followalong_debrief()),
    "demo_storyboard": demo_storyboard,
    "demo_flag": demo_flag,
    "agents_fit": agents_fit,
    "guardrails": lambda: (
        guardrails("ido", "fig_guardrails_ido.png"),
        guardrails("youdo", "fig_guardrails_youdo.png"),
    ),
    "human_review": human_review,
    # ---- Lecture 13 ----
    "llms": _lecture13_figures,
    "next_token": next_token_predict,
    "softmax_next": softmax_next_token,
    "llm_pipeline": lambda: llm_pipeline("intro", "fig_llm_pipeline.png"),
    "llm_scale": llm_scale,
    "hallucination": lambda: (
        hallucination("ido", "fig_hallucination_ido.png"),
        hallucination("youdo", "fig_hallucination_youdo.png"),
    ),
    "rl_anatomy": rl_anatomy,
    "markov": markov_property,
    "tour_papers": lambda: (tour_casanovo(), tour_prosit(), tour_diann(),
                            tour_driams(), tour_alphafold()),
    "explore": lambda: (explore_exploit("ido", "fig_explore_ido.png"),
                        explore_exploit("youdo", "fig_explore_youdo.png")),
    "return_discount": discounted_return,
    "rl_loop": lambda: (
        rl_loop("ido", "fig_rl_loop_ido.png"),
        rl_loop("youdo", "fig_rl_loop_youdo.png"),
    ),
    "rlhf": rlhf_as_rl,
    "rlhf_deep": lambda: (rlhf_pipeline(), rlhf_limits()),
    "use_llm": lambda: (
        use_llm_chart("ido", "fig_use_llm_ido.png"),
        use_llm_chart("youdo", "fig_use_llm_youdo.png"),
    ),
    "landscape": lambda: (
        landscape_card("Casanovo / DeepNovo", TEAL,
                       "de novo peptide sequencing — read the peptide from an MS/MS spectrum, no database",
                       "Transformer encoder–decoder — spectrum → peptide (Lec 7–8)",
                       "identifies peptides missing from any database (novel, mutated, immunopeptides)",
                       "fig_land_casanovo.png"),
        landscape_card("Prosit / AlphaPeptDeep", AMBER,
                       "predict a peptide's fragment spectrum + retention time from its sequence",
                       "deep sequence model — RNN/LSTM + attention (Lec 7–8)",
                       "predicted spectra rescore search hits → more confident IDs; power DIA libraries",
                       "fig_land_prosit.png"),
        landscape_card("DIA-NN", RED,
                       "identify & quantify peptides in DIA runs where many signals overlap",
                       "feed-forward neural nets — MLP (Lec 1–4)",
                       "robust, high-throughput DIA proteomics — thousands of proteins per run",
                       "fig_land_diann.png"),
        landscape_card("DRIAMS — MALDI-TOF resistance", INK_SOFT,
                       "predict antibiotic resistance (R/S) from a routine clinical MALDI-TOF spectrum",
                       "1D CNN — spectrum classifier (Lec 5)",
                       "faster resistance calls from data the lab already collects — earlier right antibiotic",
                       "fig_land_driams.png"),
        landscape_match("ido", "fig_landscape_match_ido.png"),
        landscape_transfer(),
    ),
    "landscape_transfer": landscape_transfer,
    "rag": rag_pipeline,
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
