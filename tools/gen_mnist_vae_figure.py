#!/usr/bin/env python3
"""One-off: train a tiny 2-D-latent VAE on MNIST and compose a course-styled
latent-space figure for Lecture 11 slide 10 (the canonical VAE "walk a line ->
smooth morph" demo). MNIST is public-domain; the figure is self-generated, so it
is license-clean. Output: slides/assets/img/fig_mnist_latent.png

Run once:  python3 tools/gen_mnist_vae_figure.py
It is NOT wired into make_slide_figures.py (a stored asset, like resnet_block.png).
"""
import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

SEED = 7
np.random.seed(SEED); torch.manual_seed(SEED)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "slides", "assets", "img", "fig_mnist_latent.png")

PAPER = "#FBFBF8"; INK = "#1B2025"; TEAL = "#0E7C7B"; AMBER = "#E09E2F"; INK_SOFT = "#3D444C"

# ---- data (sklearn fetch_openml, cached) --------------------------------
from sklearn.datasets import fetch_openml
print("loading MNIST ...", flush=True)
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
X = (mnist.data.astype("float32") / 255.0)
y = mnist.target.astype(int)
# subsample for speed
idx = np.random.permutation(len(X))
tr = idx[:20000]; te = idx[20000:24000]
Xtr = torch.tensor(X[tr]); Xte = torch.tensor(X[te]); yte = y[te]

# ---- tiny MLP VAE, 2-D latent -------------------------------------------
class VAE(nn.Module):
    def __init__(self, d=2):
        super().__init__()
        self.e1 = nn.Linear(784, 400); self.e2 = nn.Linear(400, 200)
        self.mu = nn.Linear(200, d); self.lv = nn.Linear(200, d)
        self.d1 = nn.Linear(d, 200); self.d2 = nn.Linear(200, 400); self.d3 = nn.Linear(400, 784)
    def enc(self, x):
        h = F.relu(self.e2(F.relu(self.e1(x)))); return self.mu(h), self.lv(h)
    def dec(self, z):
        h = F.relu(self.d2(F.relu(self.d1(z)))); return torch.sigmoid(self.d3(h))
    def forward(self, x):
        mu, lv = self.enc(x); std = torch.exp(0.5 * lv)
        z = mu + std * torch.randn_like(std)
        return self.dec(z), mu, lv

dev = "cpu"
m = VAE().to(dev)
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
EPOCHS, BS = 25, 256
print("training ...", flush=True)
for ep in range(EPOCHS):
    perm = torch.randperm(len(Xtr))
    tot = 0.0
    for i in range(0, len(Xtr), BS):
        xb = Xtr[perm[i:i+BS]].to(dev)
        xr, mu, lv = m(xb)
        recon = F.binary_cross_entropy(xr, xb, reduction="sum")
        kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
        loss = recon + kl
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    print(f"  epoch {ep+1}/{EPOCHS}  elbo/sample={tot/len(Xtr):.1f}", flush=True)

m.eval()
with torch.no_grad():
    mu_te, _ = m.enc(Xte.to(dev))
mu_te = mu_te.cpu().numpy()

# ---- panel 1: latent manifold grid --------------------------------------
n = 13
gx = norm.ppf(np.linspace(0.06, 0.94, n))
gy = norm.ppf(np.linspace(0.06, 0.94, n))
canvas = np.zeros((n * 28, n * 28), dtype="float32")
with torch.no_grad():
    for i, yi in enumerate(gy[::-1]):        # flip so +z2 is up
        for j, xi in enumerate(gx):
            d = m.dec(torch.tensor([[xi, yi]], dtype=torch.float32)).numpy().reshape(28, 28)
            canvas[i*28:(i+1)*28, j*28:(j+1)*28] = d

# ---- panels 2-3: interpolations between two real digits -----------------
def first_of(cls):
    return int(np.where(yte == cls)[0][0])

def interp_row(a_cls, b_cls, k=8):
    a = Xte[first_of(a_cls)].unsqueeze(0); b = Xte[first_of(b_cls)].unsqueeze(0)
    with torch.no_grad():
        ma, _ = m.enc(a); mb, _ = m.enc(b)
        out = []
        for al in np.linspace(0, 1, k):
            z = (1 - al) * ma + al * mb
            out.append(m.dec(z).numpy().reshape(28, 28))
    return np.concatenate(out, axis=1)         # 28 x (k*28)

row1 = interp_row(3, 8)
row2 = interp_row(7, 1)

# ---- compose ------------------------------------------------------------
fig = plt.figure(figsize=(12.6, 5.4), facecolor=PAPER)
gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.25], height_ratios=[1, 1],
                      wspace=0.13, hspace=0.30, left=0.02, right=0.985, top=0.90, bottom=0.055)

# left: manifold (spans both rows)
axm = fig.add_subplot(gs[:, 0]); axm.set_facecolor(PAPER)
axm.imshow(canvas, cmap="gray_r", interpolation="bilinear")
axm.set_xticks([]); axm.set_yticks([])
for s in axm.spines.values(): s.set_color(TEAL); s.set_linewidth(2.2)
axm.set_title("One smooth latent plane — every point is a digit",
              color=INK, fontsize=12.5, fontweight="bold", pad=8)
# z-axis arrows
axm.annotate("", xy=(0.5*n*28, n*28+16), xytext=(0.5*n*28, n*28+2),
             arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, lw=1.6), annotation_clip=False)
axm.text(0.5*n*28, n*28+30, "z\u2081", ha="center", va="top", color=INK_SOFT, fontsize=12)
axm.annotate("", xy=(-16, 0.5*n*28), xytext=(-2, 0.5*n*28),
             arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, lw=1.6), annotation_clip=False)
axm.text(-30, 0.5*n*28, "z\u2082", ha="right", va="center", color=INK_SOFT, fontsize=12)

def strip(ax, row, a_lbl, b_lbl, title):
    ax.set_facecolor(PAPER)
    ax.imshow(row, cmap="gray_r", interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(AMBER); s.set_linewidth(2.0)
    ax.set_title(title, color=INK, fontsize=11.5, fontweight="bold", pad=5)
    w = row.shape[1]
    ax.text(0, 30, a_lbl, ha="left", va="top", color=TEAL, fontsize=11, fontweight="bold")
    ax.text(w, 30, b_lbl, ha="right", va="top", color=TEAL, fontsize=11, fontweight="bold")

ax1 = fig.add_subplot(gs[0, 1]); strip(ax1, row1, "A = 3", "B = 8", "Walk a straight line A \u2192 B: a smooth morph")
ax2 = fig.add_subplot(gs[1, 1]); strip(ax2, row2, "A = 7", "B = 1", "\u2026and again, a different pair")

fig.text(0.66, 0.012,
         "Trained VAE on MNIST (2-D latent). No gaps: intermediate points decode to real digits.",
         ha="center", va="bottom", color=INK_SOFT, fontsize=9.5)

fig.savefig(OUT, dpi=150, facecolor=PAPER, bbox_inches="tight")
print("saved", OUT, flush=True)

# downscale to keep the asset small
try:
    from PIL import Image
    im = Image.open(OUT)
    if im.width > 1700:
        im = im.resize((1700, round(im.height * 1700 / im.width)), Image.LANCZOS)
        im.save(OUT, optimize=True)
        print("downscaled to", im.size, flush=True)
except Exception as e:
    print("skip downscale:", e, flush=True)
