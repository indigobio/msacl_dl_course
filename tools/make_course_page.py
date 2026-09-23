#!/usr/bin/env python3
"""Build the one page students open to reach every lab: labs/handouts/msacl_ds301_labs.html.

Each lab is a single "Open in Colab" link; the notebook's own setup cell does the
rest (packages + data), so this page only has to say where to click and what
each lab will fetch. Download sizes are read from student_pack/datasets.json and
the notebook list from student_pack/labs/, so the page cannot drift from either.

The output is one self-contained file (no images, no scripts). It is written as
a page fragment - <title>, <style>, content - which browsers render as-is and
which the claude.ai Artifact publisher wraps in its own skeleton.

    python3 tools/make_course_page.py
"""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACK = ROOT / "student_pack"
OUT = ROOT / "labs" / "handouts" / "msacl_ds301_labs.html"

REPO = "indigobio/msacl_dl_course"
BRANCH = "main"
COLAB = f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/student_pack/labs/"
GITHUB = f"https://github.com/{REPO}/blob/{BRANCH}/student_pack/labs/"
HF_REPO = "https://huggingface.co/datasets/jaztsong88/msacl-ds301"

# One row per lab. `gpu` is what to pick under Runtime -> Change runtime type.
LABS = [
    dict(n=1, seg=1, nb="lab01_training_loop.ipynb",
         title="Your first network: the training loop",
         what="Predict biological sex from a 189-metabolite serum panel (MTBLS90), running "
              "the whole train → evaluate loop yourself.",
         gpu=False, extra=[]),
    dict(n=2, seg=2, nb="lab02_cnn_spectra.ipynb",
         title="A 1D CNN on real clinical spectra",
         what="Call MRSA — oxacillin resistance in S. aureus — straight from raw "
              "MALDI-TOF spectra (DRIAMS).",
         gpu=True, extra=[]),
    dict(n=3, seg=3, nb="lab03_generator_transfer.ipynb",
         title="Transfer learning: teach a generator your symbol",
         what="Download a pretrained digit-drawing diffusion model and fine-tune it, "
              "on a few examples, to draw a symbol you invent.",
         gpu=True, extra=[("pretrained model, from the Hugging Face Hub", 15_000_000)]),
    dict(n=4, seg=4, nb="lab04_vae_fashion.ipynb",
         title="The Fashion VAE: latent playground + impostor detector",
         what="Build a VAE that only knows clothes, walk its latent map, then use it "
              "to catch handwritten digits as impostors.",
         gpu=False, extra=[]),
    dict(n=5, seg=5, nb="lab05_capstone.ipynb",
         title="Capstone: an end-to-end MS mini-project",
         what="Pick one of three tracks, adapt a working pipeline, and evaluate it "
              "like a clinical lab — sensitivity, specificity, AUROC.",
         gpu=True, extra=[]),
]


def mb(n: int) -> str:
    return f"{n / 1e6:.1f} MB"


def lab_data(lab_id: str, datasets: dict, extra: list) -> tuple[str, int]:
    items = [(name, e["bytes"]) for name, e in datasets.items()
             if lab_id in e["labs"] and not e.get("optional")]
    items += extra
    total = sum(b for _, b in items)
    return ", ".join(html.escape(n) for n, _ in items), total


CSS = """
:root{
  --ground:#f6f8f7; --panel:#ffffff; --ink:#172023; --muted:#5b6a6c;
  --line:#d7e0de; --teal:#0e7c7b; --teal-ink:#ffffff; --teal-soft:#e4f1f0;
  --amber:#b8801f; --amber-soft:#fbf1dc;
  color-scheme:light;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#0f1516; --panel:#162022; --ink:#e3ecea; --muted:#93a5a3;
    --line:#29393b; --teal:#3fb3ad; --teal-ink:#08201f; --teal-soft:#16302f;
    --amber:#e2ad55; --amber-soft:#2f2616; color-scheme:dark;
  }
}
:root[data-theme="dark"]{
  --ground:#0f1516; --panel:#162022; --ink:#e3ecea; --muted:#93a5a3;
  --line:#29393b; --teal:#3fb3ad; --teal-ink:#08201f; --teal-soft:#16302f;
  --amber:#e2ad55; --amber-soft:#2f2616; color-scheme:dark;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font:16px/1.6 "Avenir Next","Nunito Sans",-apple-system,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{max-width:880px;margin:0 auto;padding-inline:20px;padding-block:36px 56px}
.mono{font-family:Menlo,"JetBrains Mono",ui-monospace,SFMono-Regular,monospace}
.eyebrow{font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:var(--teal);font-weight:600}
h1{font-size:clamp(28px,5vw,38px);line-height:1.15;margin:.25em 0 .2em;text-wrap:balance;font-weight:700}
.lede{font-size:18px;color:var(--muted);margin:0;max-width:60ch}
h2{font-size:13px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);
  margin:40px 0 12px;font-weight:600}
a{color:var(--teal)}
a:focus-visible,.go:focus-visible{outline:3px solid var(--amber);outline-offset:2px}

/* the four moves: a real sequence, so it is numbered */
.moves{list-style:none;margin:0;padding:0;display:grid;gap:10px;
  grid-template-columns:repeat(auto-fit,minmax(190px,1fr));counter-reset:m}
.moves li{counter-increment:m;border-top:3px solid var(--teal);padding-top:10px}
.moves li::before{content:counter(m);font:700 13px/1 Menlo,ui-monospace,monospace;color:var(--teal);
  display:block;margin-bottom:6px}
.moves b{display:block}
.moves span{color:var(--muted);font-size:14.5px}

/* labs */
.labs{display:grid;gap:14px}
.lab{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:18px 20px;display:grid;gap:14px;grid-template-columns:1fr auto;align-items:center}
.lab .slot{font-size:12px;color:var(--muted);letter-spacing:.06em;text-transform:uppercase}
.lab h3{font-size:19px;line-height:1.3;margin:2px 0 4px;text-wrap:balance}
.lab p{margin:0 0 10px;color:var(--muted);max-width:62ch}
.meta{display:flex;flex-wrap:wrap;gap:6px 8px;font-size:13px}
.chip{border-radius:999px;padding:2px 10px;background:var(--teal-soft);color:var(--ink);white-space:nowrap}
.chip.gpu{background:var(--amber-soft)}
.chip .k{color:var(--muted)}
.data{font-size:12.5px;color:var(--muted);margin-top:8px;overflow-wrap:anywhere}
.actions{display:flex;flex-direction:column;gap:8px;align-items:stretch;min-width:170px}
.go{display:block;text-align:center;background:var(--teal);color:var(--teal-ink);
  text-decoration:none;font-weight:700;border-radius:9px;padding:12px 16px}
.go:hover{filter:brightness(1.08)}
.src{text-align:center;font-size:13.5px}
@media (max-width:640px){
  .lab{grid-template-columns:1fr}
  .actions{min-width:0}
}

/* help */
.help{display:grid;gap:10px;grid-template-columns:repeat(auto-fit,minmax(250px,1fr))}
.help div{border-left:3px solid var(--line);padding:2px 0 2px 12px}
.help b{display:block;font-size:15px}
.help span{color:var(--muted);font-size:14.5px}
footer{margin-top:40px;padding-top:14px;border-top:1px solid var(--line);font-size:13px;color:var(--muted)}
code{font:13px/1.4 Menlo,ui-monospace,monospace;background:var(--teal-soft);border-radius:4px;padding:1px 5px}
"""


def build() -> str:
    datasets = json.loads((PACK / "datasets.json").read_text())["datasets"]
    shipped = {p.name for p in (PACK / "labs").glob("*.ipynb")}
    rows = []
    for lab in LABS:
        if lab["nb"] not in shipped:
            raise SystemExit(f"{lab['nb']} is not in student_pack/labs/")
        names, total = lab_data(f"lab{lab['n']:02d}", datasets, lab["extra"])
        runtime = ('<span class="chip gpu"><span class="k">Runtime</span> T4 GPU</span>'
                   if lab["gpu"] else
                   '<span class="chip"><span class="k">Runtime</span> CPU is fine</span>')
        rows.append(f"""
<article class="lab">
  <div>
    <div class="slot mono">Segment {lab['seg']} · Lab {lab['n']}</div>
    <h3>{html.escape(lab['title'])}</h3>
    <p>{html.escape(lab['what'])}</p>
    <div class="meta">
      {runtime}
      <span class="chip"><span class="k">Downloads</span> {mb(total)}</span>
    </div>
    <div class="data mono">{names}</div>
  </div>
  <div class="actions">
    <a class="go" href="{COLAB}{lab['nb']}" target="_blank" rel="noopener">Open Lab {lab['n']} in Colab</a>
    <a class="src" href="{GITHUB}{lab['nb']}" target="_blank" rel="noopener">view on GitHub</a>
  </div>
</article>""")

    return f"""<title>MSACL DS301 Labs</title>
<meta name="description" content="Open any of the five MSACL DS301 Deep Learning labs in Google Colab.">
<style>{CSS}</style>
<div class="wrap">
<header>
  <div class="eyebrow mono">MSACL · DS301 Deep Learning</div>
  <h1>The five labs</h1>
  <p class="lede">Each lab is one link. Nothing to install, clone or download — the
  first cell of every notebook sets up the software and fetches that lab's data for you.</p>
</header>

<h2>Every lab, the same four moves</h2>
<ol class="moves">
  <li><b>Open the lab's link</b><span>It opens in Google Colab. Sign in with any Google account.</span></li>
  <li><b>File → Save a copy in Drive</b><span>The link is read-only. Work in the tab named <i>Copy of …</i></span></li>
  <li><b>Run the ⚙️ Setup cell</b><span>About a minute. It ends with <code>✓ labNN is ready</code>.</span></li>
  <li><b>Fill in the ✏️ blanks</b><span>Run the cells top to bottom. The <code>assert</code> cells check your answers.</span></li>
</ol>

<h2>Labs</h2>
<div class="labs">{''.join(rows)}
</div>

<h2>If something goes wrong</h2>
<div class="help">
  <div><b>"Restart session, then run this setup cell again"</b>
    <span>Expected once per session. <b>Runtime → Restart session</b>, then run ⚙️ Setup again.</span></div>
  <div><b>Colab asks for a GPU, or runs slowly</b>
    <span><b>Runtime → Change runtime type → T4 GPU</b>, then run ⚙️ Setup again.</span></div>
  <div><b>"Could not get …" during setup</b>
    <span>A download failed. Check your connection and run the setup cell again — it resumes.</span></div>
  <div><b>Colab will not sign you in</b>
    <span>Some hospital and university accounts block Colab. Use a personal Google account.</span></div>
</div>

<footer>
  Course data is hosted at <a href="{HF_REPO}" target="_blank" rel="noopener">huggingface.co/datasets/jaztsong88/msacl-ds301</a>
  (DRIAMS CC0 · FashionMNIST MIT · MNIST CC BY-SA 3.0); every file is checked against its
  SHA-256 fingerprint as it downloads. Notebooks:
  <a href="https://github.com/{REPO}/tree/{BRANCH}/student_pack" target="_blank" rel="noopener">github.com/{REPO}</a>.
</footer>
</div>
"""


def main():
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
