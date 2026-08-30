#!/usr/bin/env python3
"""Print quiz HTML sources to letter-size PDFs for handout printing.

Quizzes live in quizzes/src/ as self-contained HTML (quizNN_topic.html plus
quizNN_topic_key.html for the answer key). PDFs land in quizzes/pdf/.

Usage:
    python tools/make_quiz_pdf.py quizzes/src/quiz01_fundamentals.html
    python tools/make_quiz_pdf.py --all

Setup: pip install -r tools/requirements.txt && playwright install chromium
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "quizzes" / "src"
PDF_DIR = ROOT / "quizzes" / "pdf"


def main(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[1] == "--all":
        targets = sorted(SRC_DIR.glob("*.html"))
        if not targets:
            print(f"No quiz sources found in {SRC_DIR}")
            return 1
    else:
        targets = [Path(a).resolve() for a in argv[1:]]

    from playwright.sync_api import sync_playwright

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    status = 0
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        for src in targets:
            try:
                page.goto(src.resolve().as_uri())
                page.wait_for_load_state("networkidle")
                dest = PDF_DIR / (src.stem + ".pdf")
                page.pdf(
                    path=str(dest),
                    format="Letter",
                    margin={t: "0.6in" for t in ("top", "bottom", "left", "right")},
                    print_background=True,
                )
                print(f"{src.name} -> {dest.relative_to(ROOT)}")
            except Exception as exc:
                print(f"ERROR in {src.name}: {exc}", file=sys.stderr)
                status = 1
        browser.close()
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv))
