#!/usr/bin/env python3
"""Re-capture the annotated Google Colab screenshots for the Lab 1 setup sheet.

Colab's interface changes every few months, so the figures in
labs/handouts/lab01_colab_setup.tex are generated rather than pasted in. This
script drives headless Chrome over the DevTools protocol, opens the real course
notebook signed-OUT (no account, no credentials — the whole UI renders anyway),
clicks the File and Runtime menus so they are captured open, then crops each
view and draws the numbered callouts the handout refers to.

    python3 tools/make_colab_shots.py        # rewrites labs/handouts/img/*.png

Needs Google Chrome installed and the `websocket-client` package. If a callout
lands in the wrong place after Google restyles Colab, adjust the boxes in
ANNOTATIONS below — the coordinates are in the 1440x900 capture's own pixels.
"""
import base64
import json
import shutil
import socket
import subprocess
import tempfile
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import websocket

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "labs" / "handouts" / "img"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
NOTEBOOK = ("https://colab.research.google.com/github/indigobio/msacl_dl_course/"
            "blob/main/student_pack/labs/lab01_training_loop.ipynb")
LANDING = "https://colab.research.google.com/"
TEAL, AMBER = (14, 124, 123), (224, 158, 47)
def font(sz):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try: return ImageFont.truetype(p, sz)
        except OSError: pass
    return ImageFont.load_default()


def badge(d, x, y, n, col=TEAL, r=17):
    d.ellipse([x - r, y - r, x + r, y + r], fill=col, outline=(255, 255, 255), width=3)
    f = font(21)
    b = d.textbbox((0, 0), str(n), font=f)
    d.text((x - (b[2] - b[0]) / 2, y - (b[3] - b[1]) / 2 - b[1]), str(n),
           fill=(255, 255, 255), font=f)


def box(d, xy, col=TEAL, w=3, pad=5):
    x0, y0, x1, y1 = xy
    d.rounded_rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], radius=7,
                        outline=col, width=w)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def capture(url, out_png, clicks=()):
    """Load `url` in headless Chrome, click each (x, y), and save a screenshot.

    Each call gets its own debugging port and profile directory: sharing either
    makes a second capture attach to the previous, dying browser.
    """
    port = _free_port()
    profile = tempfile.mkdtemp(prefix="colabshot-")
    proc = subprocess.Popen(
        [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
         f"--remote-debugging-port={port}", "--remote-allow-origins=*",
         f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check",
         "--window-size=1440,900", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        page = None
        for _ in range(40):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
                page = [t for t in tabs if t["type"] == "page"][0]
                break
            except Exception:
                time.sleep(0.5)
        if page is None:
            sys.exit("could not reach headless Chrome on port %d" % port)
        ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=60)
        seq = [0]

        def send(method, **params):
            seq[0] += 1
            ws.send(json.dumps({"id": seq[0], "method": method, "params": params}))
            while True:
                msg = json.loads(ws.recv())
                if msg.get("id") == seq[0]:
                    return msg.get("result", {})

        send("Page.enable")
        send("Page.navigate", url=url)
        time.sleep(14)                      # Colab is a heavy single-page app
        for (x, y) in clicks:
            for kind in ("mousePressed", "mouseReleased"):
                send("Input.dispatchMouseEvent", type=kind, x=x, y=y,
                     button="left", clickCount=1)
            time.sleep(1.6)
        out_png.write_bytes(base64.b64decode(
            send("Page.captureScreenshot", format="png")["data"]))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(profile, ignore_errors=True)


# (source capture, crop box, [(box, colour, badge x, badge y, n)], output name)
ANNOTATIONS = [
    ("nb", (0, 52, 1440, 160), [
        ((86, 14, 322, 40), TEAL, 352, 27, 1),
        ((1330, 20, 1406, 48), AMBER, 1298, 34, 2),
        ((1292, 70, 1392, 96), AMBER, 1260, 83, 3)], "colab_step1_chrome.png"),
    ("file", (60, 96, 700, 460), [
        ((26, 259, 344, 286), TEAL, 386, 273, 1)], "colab_step2_savecopy.png"),
    ("runtime", (248, 96, 900, 500), [
        ((22, 28, 300, 54), TEAL, 340, 41, 1),
        ((22, 260, 372, 287), AMBER, 412, 274, 2),
        ((22, 340, 352, 367), AMBER, 392, 354, 3)], "colab_step3_runtime.png"),
    ("nb", (60, 735, 1180, 855), [
        ((18, 28, 52, 58), TEAL, 92, 43, 1)], "colab_step4_cell.png"),
    ("landing", (0, 10, 1440, 250), [
        ((1320, 12, 1410, 35), AMBER, 1288, 23, 1),
        ((387, 142, 570, 172), TEAL, 604, 157, 2)], "colab_step5_landing.png"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = {}
    shots = [("nb", NOTEBOOK, ()),                    # the notebook as it opens
             ("file", NOTEBOOK, ((82, 102),)),        # with the File menu open
             ("runtime", NOTEBOOK, ((280, 102),)),    # with the Runtime menu open
             ("landing", LANDING, ())]
    for key, url, clicks in shots:
        path = OUT / f".raw_{key}.png"
        print(f"  capturing {key} …")
        capture(url, path, clicks)
        raw[key] = path

    for key, cropbox, marks, name in ANNOTATIONS:
        im = Image.open(raw[key]).convert("RGB").crop(cropbox)
        d = ImageDraw.Draw(im)
        for xy, col, bx, by, n in marks:
            box(d, xy, col)
            badge(d, bx, by, n, col)
        d.rectangle([0, 0, im.width - 1, im.height - 1],
                    outline=(190, 190, 186), width=2)
        im.save(OUT / name, optimize=True)
        print(f"  wrote {name}")
    for p in raw.values():
        p.unlink(missing_ok=True)
    print("done — rebuild the sheet with: tools/build.sh handouts")


if __name__ == "__main__":
    main()
