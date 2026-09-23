#!/usr/bin/env python3
"""Build the clickable HTML version of the Lab 1 Colab setup sheet.

The printed PDF (labs/handouts/lab01_colab_setup.tex) is the paper artifact; this
is the one participants open on the laptop they are about to use, where the
point is that the Colab link is a real link they can click rather than a URL
they have to retype.

The screenshots are inlined as base64 data URIs, so the result is a SINGLE file
that can be emailed, dropped on a USB stick or served from anywhere and still
render offline — and so the images can never drift from labs/handouts/img/.

    python3 tools/make_colab_setup_html.py   # -> labs/handouts/lab01_colab_setup.html

Re-run it after tools/make_colab_shots.py refreshes the screenshots.
"""
import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "labs" / "handouts" / "img"
OUT = ROOT / "labs" / "handouts" / "lab01_colab_setup.html"

NOTEBOOK_URL = ("https://colab.research.google.com/github/indigobio/msacl_dl_course/"
                "blob/main/student_pack/labs/lab01_training_loop.ipynb")
COLAB_URL = "https://colab.research.google.com/"
CAPTURED = "22 September 2026"


def img(name, alt, width="100%"):
    """Inline a screenshot as a data URI so the page stays self-contained."""
    data = base64.b64encode((IMG / name).read_bytes()).decode()
    return (f'<figure><img src="data:image/png;base64,{data}" alt="{alt}" '
            f'style="width:{width}">' f"</figure>")


def mk(n, colour="teal"):
    return f'<span class="mk {colour}">{n}</span>'


CSS = """
:root{--ink:#1b2025;--muted:#6b7280;--teal:#0e7c7b;--amber:#b8801f;
      --line:#dcdcd6;--paper:#fbfbf7;--soft:#eef5f4;--warn:#fdf1f0;--red:#b4413d}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
     font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{max-width:860px;margin:0 auto;padding:32px 22px 64px}
header{border-bottom:2px solid var(--teal);padding-bottom:14px;margin-bottom:22px}
.eyebrow{font:600 12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;
         letter-spacing:.09em;text-transform:uppercase;color:var(--teal)}
h1{font-size:31px;margin:.24em 0 .12em;line-height:1.2}
.sub{color:var(--muted);font-size:17px;margin:0}
h2{font-size:21px;margin:34px 0 8px;padding-top:4px}
h2 .num{color:var(--teal);font-variant-numeric:tabular-nums}
p{margin:.55em 0}
a{color:var(--teal)}
.box{border:1px solid var(--line);background:#fff;border-radius:10px;
     padding:14px 18px;margin:18px 0}
.box.soft{background:var(--soft);border-color:var(--teal)}
.cta{display:block;margin:18px 0;padding:16px 20px;border-radius:10px;
     background:var(--teal);color:#fff;text-decoration:none;font-weight:600;
     font-size:17px}
.cta:hover{background:#0b6564}
.cta .url{display:block;font:400 12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
          word-break:break-all;opacity:.85;margin-top:5px}
figure{margin:14px 0}
img{display:block;border:1px solid var(--line);border-radius:7px}
ul{margin:.5em 0;padding-left:1.25em}
li{margin:.3em 0}
.mk{display:inline-block;min-width:21px;height:21px;line-height:21px;
    border-radius:50%;text-align:center;color:#fff;font-size:12.5px;
    font-weight:700;margin-right:5px;vertical-align:1px}
.mk.teal{background:var(--teal)}
.mk.amber{background:var(--amber)}
kbd{border:1px solid var(--line);border-bottom-width:2px;border-radius:5px;
    background:#fff;padding:1px 6px;font:600 13px/1.5 ui-monospace,Menlo,monospace}
code{background:#f1f1ec;border-radius:4px;padding:1px 5px;
     font:13.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
.menu{white-space:nowrap;font-weight:600}
table{border-collapse:collapse;width:100%;margin-top:10px;font-size:15px}
th{text-align:left;border-bottom:2px solid var(--line);padding:7px 10px 7px 0}
td{border-bottom:1px solid var(--line);padding:9px 10px 9px 0;vertical-align:top}
td:first-child{width:38%;color:var(--red)}
footer{margin-top:34px;padding-top:12px;border-top:1px solid var(--line);
       color:var(--muted);font-size:12.5px;font-style:italic}
@media print{
  body{background:#fff}
  .wrap{max-width:none;padding:0}
  .cta{background:#fff;color:var(--teal);border:1px solid var(--teal)}
  h2{page-break-after:avoid} figure{page-break-inside:avoid}
  tr{page-break-inside:avoid}
}
"""

HTML = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lab 1 — Setting up Google Colab · MSACL DS301</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<header>
  <div class="eyebrow">MSACL · DS301 Deep Learning · Lab 1</div>
  <h1>Setting up Google Colab</h1>
  <p class="sub">Everything runs in your browser. Nothing to install.</p>
</header>

<div class="box soft">
  <strong>What you need:</strong> a web browser and a <strong>free Google
  account</strong> (a personal <code>@gmail.com</code> is fine — some hospital and
  university accounts block Colab, so if yours does, use a personal one).<br>
  <strong>You do not need:</strong> Python, PyTorch, an install, or a GPU.<br>
  <em>Colab</em> is a free Google service that runs Python notebooks on Google's
  computers; your laptop only shows the page.
</div>

<h2><span class="num">Step 1</span> — Open the Lab 1 notebook</h2>
<p>Click the button. It opens Colab with the notebook already loaded — there is
nothing to download first.</p>

<a class="cta" href="{NOTEBOOK_URL}" target="_blank" rel="noopener">
  Open Lab 1 in Colab →
  <span class="url">{NOTEBOOK_URL}</span>
</a>

{img("colab_step1_chrome.png", "The top of the Colab window: notebook name, Sign in, Connect")}

<ul>
  <li>{mk(1)} Check the file name reads <code>lab01_training_loop.ipynb</code> — that is the right notebook.</li>
  <li>{mk(2, "amber")} <strong>Sign in</strong> (top right) with your Google account.
      Until you do, you can read the notebook but not run it.</li>
  <li>{mk(3, "amber")} <strong>Connect</strong> (top right) starts your free machine.
      It usually connects on its own the first time you run a cell; if it says
      <em>Connect</em>, click it and wait for a green tick.</li>
</ul>

<h2><span class="num">Step 2</span> — Save your own copy <span style="font-weight:400;color:var(--muted)">(before you type anything)</span></h2>
<p>The link opens a <strong>read-only</strong> copy. Use
<span class="menu">File → Save a copy in Drive</span> and work in that copy, or your
edits will be lost when you close the tab. Colab opens the copy in a new tab named
<code>Copy of lab01_training_loop.ipynb</code> — <strong>that is the tab you want to
work in</strong>.</p>

{img("colab_step2_savecopy.png", "The File menu with Save a copy in Drive highlighted", "66%")}

<h2><span class="num">Step 3</span> — Run a cell</h2>
<p>A notebook is a list of <em>cells</em>. A cell holds either text or code. Click a
code cell, then press <kbd>Shift</kbd> + <kbd>Enter</kbd> — or hover over the
{mk(1)} bracket at its left, which turns into a ▶ button, and click it.</p>

{img("colab_step4_cell.png", "A code cell, with the run bracket at its left highlighted")}

<ul>
  <li>Run the cells <strong>in order, top to bottom</strong>. Later cells depend on earlier ones.</li>
  <li>While a cell runs the bracket shows a spinner; when it finishes you get a number, e.g. <code>[1]</code>.</li>
  <li><strong>Run the first code cell, “⚙️ Setup”, before anything else.</strong> It installs the course packages and downloads the lab’s data for you — about a minute, and it ends with <code>✓ lab01 is ready</code>. You never download or copy data files yourself.</li>
  <li>Only the cells marked <strong>YOUR TURN</strong> ✏️ need editing. Run all the others exactly as they are.</li>
</ul>

<h2><span class="num">Step 4</span> — If something gets tangled</h2>
<p>The <span class="menu">Runtime</span> menu is the fix for almost everything.</p>

{img("colab_step3_runtime.png", "The Runtime menu, with Run all, Restart session and run all, and Change runtime type highlighted", "72%")}

<ul>
  <li>{mk(1)} <strong>Run all</strong> — runs the whole notebook from the top. Good after you fix a cell.</li>
  <li>{mk(2, "amber")} <strong>Restart session and run all</strong> — throws away every
      variable and starts clean. This is the answer to “it worked a minute ago”. You
      lose nothing but the running state; your typed code stays.</li>
  <li>{mk(3, "amber")} <strong>Change runtime type</strong> — where you would pick a
      <strong>T4 GPU</strong>. <strong>Lab 1 does not need one</strong>; the default CPU
      is plenty and connects faster. You will use this in Lab 2.</li>
</ul>

<h2><span class="num">Step 5</span> — If the link does not open</h2>
<p>Go to <a href="{COLAB_URL}" target="_blank" rel="noopener">colab.research.google.com</a>
directly, sign in {mk(1, "amber")}, then use
<span class="menu">File → Open notebook → GitHub</span> and paste
<code>indigobio/msacl_dl_course</code>. Or use <strong>Upload notebook</strong>
{mk(2)} if an instructor hands you the <code>.ipynb</code> file on a USB stick.</p>

{img("colab_step5_landing.png", "The Colab landing page, with Sign in and Upload notebook highlighted")}

<h2>When things go wrong</h2>
<table>
<tr><th>What you see</th><th>What to do</th></tr>
<tr><td>“Sign in” keeps reappearing, or Colab will not load</td>
    <td>Your institutional Google account may block Colab. Sign in with a personal
        <code>@gmail.com</code> instead.</td></tr>
<tr><td><code>NameError</code> / <code>… is not defined</code></td>
    <td>You skipped a cell, or ran them out of order.
        <span class="menu">Runtime → Restart session and run all</span>.</td></tr>
<tr><td>A cell has been running for minutes</td>
    <td>Normal for training cells. If it is stuck,
        <span class="menu">Runtime → Interrupt execution</span>, then restart and run all.</td></tr>
<tr><td>“Cannot connect to runtime” / disconnected</td>
    <td>Free Colab disconnects after a while idle. Click <strong>Connect</strong> and
        <span class="menu">Runtime → Run all</span>; you lose only the running state.</td></tr>
<tr><td>An <code>assert</code> cell failed</td>
    <td>That is the notebook checking your answer — it is meant to catch you. Read the
        message, fix the marked line, re-run that cell. The <strong>Code Hint
        Sheet</strong> lists the options for every blank.</td></tr>
<tr><td>You edited the wrong tab and lost your work</td>
    <td>You were in the read-only original. Redo
        <span class="menu">File → Save a copy in Drive</span> and check the tab title
        starts with <code>Copy of</code>.</td></tr>
</table>

<footer>
Screenshots are of the Google Colab interface, captured {CAPTURED}, with callout
numbers added for this sheet. Colab is a Google product and its appearance changes
from time to time — if a button has moved, the menu path in the text is still the one
to follow. A printable version of this sheet is <code>lab01_colab_setup.pdf</code>.
</footer>

</div>
</body>
</html>
"""


def main():
    missing = [n for n in ("colab_step1_chrome.png", "colab_step2_savecopy.png",
                           "colab_step3_runtime.png", "colab_step4_cell.png",
                           "colab_step5_landing.png") if not (IMG / n).exists()]
    if missing:
        raise SystemExit("missing screenshots: " + ", ".join(missing)
                         + "\nrun tools/make_colab_shots.py first")
    OUT.write_text(HTML, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1024:.0f} KB, self-contained)")


if __name__ == "__main__":
    main()
