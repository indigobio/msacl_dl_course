#!/usr/bin/env python3
"""Print an 'Open in Colab' link for every lab notebook in student_pack/labs/.

Requires the repo to be published to a public GitHub repo. Pass your repo path via
flags or env vars, then paste the links into your slides / handout / email.

Usage:
    python tools/colab_links.py --org myorg --repo mycourse --branch main
    ORG=myorg REPO=mycourse python tools/colab_links.py

No third-party dependencies.
"""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABS = ROOT / "student_pack" / "labs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", default=os.environ.get("ORG", "<ORG>"))
    ap.add_argument("--repo", default=os.environ.get("REPO", "<REPO>"))
    ap.add_argument("--branch", default=os.environ.get("BRANCH", "main"))
    a = ap.parse_args()

    notebooks = sorted(LABS.glob("*.ipynb"))
    if not notebooks:
        print(f"No notebooks found in {LABS}")
        return 1

    for nb in notebooks:
        rel = f"student_pack/labs/{nb.name}"
        url = (f"https://colab.research.google.com/github/"
               f"{a.org}/{a.repo}/blob/{a.branch}/{rel}")
        print(f"{nb.name}\n  {url}\n")

    if "<ORG>" in (a.org, a.repo) or "<REPO>" in (a.org, a.repo):
        print("Fill in --org/--repo (or set ORG/REPO) with your public GitHub path "
              "once the repo is published.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
