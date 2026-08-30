#!/usr/bin/env python3
"""Assemble the shareable student lab pack.

The uv env files (pyproject.toml, uv.lock, .python-version) and README.md live in
`student_pack/` and are HAND-MAINTAINED + committed. pyproject.toml (+ uv.lock) is
the single source of truth for dependencies — there is no exported requirements
file. The lab notebooks themselves are GENERATED (fill-in-the-blank copies produced
by tools/strip_solutions.py in labs/student/) and are gitignored both there and
inside the pack, so this script copies the current student notebooks into the pack.

Typical flow when a new lab is finished:
    python tools/strip_solutions.py --all       # regenerate labs/student/*.ipynb
    python tools/build_student_pack.py           # copy them into student_pack/labs/

If you change dependencies in student_pack/pyproject.toml, re-lock:
    cd student_pack
    uv lock

Then zip `student_pack/` (excluding .venv/) to share with students.
"""
from __future__ import annotations

import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STUDENT_LABS = REPO / "labs" / "student"
PACK_LABS = REPO / "student_pack" / "labs"


def main() -> None:
    if not STUDENT_LABS.is_dir():
        raise SystemExit(
            f"{STUDENT_LABS} not found — run `python tools/strip_solutions.py --all` first."
        )
    PACK_LABS.mkdir(parents=True, exist_ok=True)

    notebooks = sorted(STUDENT_LABS.glob("*.ipynb"))
    if not notebooks:
        raise SystemExit(f"No .ipynb files in {STUDENT_LABS}.")

    for nb in notebooks:
        dest = PACK_LABS / nb.name
        shutil.copy2(nb, dest)
        print(f"copied  {nb.name}  ->  student_pack/labs/{nb.name}")

    print(f"\nDone. {len(notebooks)} notebook(s) in student_pack/labs/.")
    print("Share the pack by zipping student_pack/ (the .venv/ folder is excluded by .gitignore).")


if __name__ == "__main__":
    main()
