#!/usr/bin/env python3
"""Generate fill-in-the-blank student notebooks from solution notebooks.

In labs/solutions/*.ipynb, wrap the code students must write between marker
lines (markers must sit on their own lines, indentation preserved):

    ### BEGIN SOLUTION
    hidden = code.here()
    ### END SOLUTION

Everything between the markers is replaced with a "YOUR CODE HERE" stub at the
same indentation. Keep `assert` self-check lines OUTSIDE the markers so student
notebooks retain them. All cell outputs and execution counts are cleared.

Usage:
    python tools/strip_solutions.py labs/solutions/lab01_mlp.ipynb
    python tools/strip_solutions.py --all

Output goes to labs/student/ under the same filename. No dependencies beyond
the standard library.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOLUTIONS_DIR = ROOT / "labs" / "solutions"
STUDENT_DIR = ROOT / "labs" / "student"

BEGIN_RE = re.compile(r"^(\s*)### BEGIN SOLUTION\s*$")
END_RE = re.compile(r"^\s*### END SOLUTION\s*$")


def strip_source(lines, cell_index):
    out, in_solution, found = [], False, 0
    for line in lines:
        begin = BEGIN_RE.match(line)
        if begin:
            if in_solution:
                raise ValueError(f"cell {cell_index}: nested BEGIN SOLUTION")
            in_solution = True
            found += 1
            indent = begin.group(1)
            out.append(f"{indent}# ========== YOUR CODE HERE ==========\n")
            out.append(f"{indent}...  # replace with your code\n")
            continue
        if END_RE.match(line):
            if not in_solution:
                raise ValueError(f"cell {cell_index}: END SOLUTION without BEGIN")
            in_solution = False
            continue
        if not in_solution:
            out.append(line)
    if in_solution:
        raise ValueError(f"cell {cell_index}: BEGIN SOLUTION never closed")
    return out, found


def strip_notebook(path: Path) -> Path:
    nb = json.loads(path.read_text(encoding="utf-8"))
    total_blanks = 0
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        if isinstance(src, str):
            src = src.splitlines(keepends=True)
        stripped, found = strip_source(src, i)
        total_blanks += found
        cell["source"] = stripped
        cell["outputs"] = []
        cell["execution_count"] = None
    STUDENT_DIR.mkdir(parents=True, exist_ok=True)
    dest = STUDENT_DIR / path.name
    dest.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{path.name}: {total_blanks} blank(s) -> {dest.relative_to(ROOT)}")
    if total_blanks == 0:
        print(f"  WARNING: no solution markers found in {path.name}")
    return dest


def main(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[1] == "--all":
        targets = sorted(SOLUTIONS_DIR.glob("*.ipynb"))
        if not targets:
            print(f"No notebooks found in {SOLUTIONS_DIR}")
            return 1
    else:
        targets = [Path(a).resolve() for a in argv[1:]]
    status = 0
    for nb_path in targets:
        try:
            strip_notebook(nb_path)
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"ERROR in {nb_path.name}: {exc}", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv))
