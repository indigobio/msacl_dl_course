#!/usr/bin/env python3
"""Lecture 8's sampling slide: a REAL diffusion model turning pure noise into a new image.

Runs the pretrained generator Lab 3 downloads (balakrish181/ddpm-class-mnist-28,
MIT licence, ~15 MB, no login) through the plain DDPM sampler it was trained
with - 1,000 small steps, each predicting the noise in the current image and
removing a little of it - and snapshots three independent runs at a few steps
between pure static (t = 1000) and the finished digit (t = 0). Nothing here is
drawn by hand: every panel is the model's actual intermediate image.

Needs torch + diffusers (the student_pack uv environment has both):

    cd student_pack && uv run python ../tools/make_diffusion_sampling_fig.py
    # -> slides/assets/img/fig_diffusion_sampling.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from diffusers import DDPMScheduler, UNet2DModel

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "slides" / "assets" / "img" / "fig_diffusion_sampling.png"
MODEL_ID = "balakrish181/ddpm-class-mnist-28"
SEEDS = [3, 5, 2]                           # three starting noises -> three clean new digits (5, 6, 0)
SNAPS = [999, 700, 500, 300, 150, 50, 0]     # timesteps to show (999 = the pure noise we start from)

# course palette (tools/make_slide_figures.py / the PowerPoint template)
PAPER, INK, INK_SOFT, TEAL, ROI_INK, MUTED = "#FBFBF8", "#1B2025", "#3D444C", "#0E7C7B", "#A9721A", "#8A9099"


@torch.no_grad()
def trajectory(unet, scheduler, seed):
    """Run the full DDPM reverse process; return {timestep: image in [0, 1]}."""
    g = torch.Generator("cpu").manual_seed(seed)
    x = torch.randn(1, 1, 28, 28, generator=g)
    snaps = {999: x.clone()}
    for t in scheduler.timesteps:                       # 999, 998, ..., 0
        eps_hat = unet(x, t).sample                     # the network predicts the noise in x_t
        x = scheduler.step(eps_hat, t, x, generator=g).prev_sample   # remove a little of it
        if int(t) in SNAPS and int(t) != 999:
            snaps[int(t)] = x.clone()
    return {t: ((im[0, 0].clamp(-1, 1) + 1) / 2).numpy() for t, im in snaps.items()}


def main():
    torch.manual_seed(0)
    unet = UNet2DModel.from_pretrained(MODEL_ID, use_safetensors=True).eval()
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear",
                              beta_start=1e-4, beta_end=0.02, clip_sample=True)
    scheduler.set_timesteps(1000)
    runs = [trajectory(unet, scheduler, s) for s in SEEDS]

    fig, axes = plt.subplots(len(SEEDS), len(SNAPS), figsize=(11.6, 5.0),
                             gridspec_kw=dict(wspace=0.12, hspace=0.12))
    fig.patch.set_facecolor(PAPER)
    for r, run in enumerate(runs):
        for c, t in enumerate(SNAPS):
            ax = axes[r, c]
            ax.imshow(run[t], cmap="Greys_r", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            last = c == len(SNAPS) - 1
            for sp in ax.spines.values():
                sp.set_edgecolor(TEAL if last else MUTED); sp.set_linewidth(2.6 if last else 0.8)
            if r == 0:
                label = "t = 1000\npure noise" if t == 999 else ("t = 0\nnew image" if t == 0 else f"t = {t}")
                ax.set_title(label, fontsize=12, color=TEAL if last else INK,
                             fontweight="bold", pad=6)
        axes[r, 0].set_ylabel(f"run {r + 1}", fontsize=11, color=INK_SOFT,
                              fontweight="bold", rotation=0, labelpad=26, va="center")
    fig.text(0.5, 0.035,
             "each step: the network predicts the noise in the image and a little of it is removed "
             "— 1,000 small steps, left to right",
             ha="center", fontsize=11.5, color=ROI_INK, fontweight="bold")
    fig.text(0.5, 0.0,
             f"real samples from the Lab 3 generator, {MODEL_ID} (MIT licence); DDPM sampler, seeds {SEEDS}",
             ha="center", fontsize=9, color=MUTED, style="italic")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches="tight", facecolor=PAPER)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
