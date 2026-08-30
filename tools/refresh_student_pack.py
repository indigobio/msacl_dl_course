#!/usr/bin/env python3
"""Regenerate the shareable student_pack from source, in one command.

Runs, in order:
  1. tools/strip_solutions.py --all   (labs/solutions/ -> labs/student/)
  2. tools/build_student_pack.py      (labs/student/   -> student_pack/labs/)

With --relock it also refreshes uv.lock (run `uv lock`). Do that only after
changing dependencies in student_pack/pyproject.toml — pyproject.toml (+ uv.lock)
is the single source of truth; there is no exported requirements file.

Usage:
    python tools/refresh_student_pack.py
    python tools/refresh_student_pack.py --relock

After running, review the changes and commit the env files + assembled notebooks.
No third-party dependencies (uv is only needed for --relock).
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
PACK = ROOT / "student_pack"


def run(cmd, cwd=None):
    print("»", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main(argv):
    run([sys.executable, str(TOOLS / "strip_solutions.py"), "--all"])
    run([sys.executable, str(TOOLS / "build_student_pack.py")])
    if "--relock" in argv:
        # Refresh uv.lock after a dependency change in pyproject.toml.
        run(["uv", "lock"], cwd=PACK)
    print("\nstudent_pack refreshed. Review, then commit the env files + notebooks.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
