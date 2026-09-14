#!/usr/bin/env bash
# Build course PDFs (quizzes + lab handouts) and the shareable student_pack.
#
# Usage:
#   tools/build.sh              # everything: quizzes + handouts + student_pack
#   tools/build.sh pdfs         # quizzes + handouts only
#   tools/build.sh quizzes      # quizzes/src/*.tex  -> quizzes/pdf/  (student + answer key)
#   tools/build.sh handouts     # labs/handouts/*.tex -> labs/handouts/*.pdf
#   tools/build.sh pack         # regenerate student_pack (strip solutions + assemble)
#
# LaTeX quizzes are single-source: a file containing \showsolutions/\ifsolution
# also gets an answer-key PDF (…_key.pdf). Needs: latexmk + pdflatex, python3.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

need() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' not found on PATH" >&2; exit 1; }; }

# compile_tex <src.tex> <dest_dir> — student PDF, plus a _key.pdf if the source toggles solutions
compile_tex() {
  local src="$1" dest="$2" dir base
  dir="$(cd "$(dirname "$src")" && pwd)"; base="$(basename "$src" .tex)"
  echo "  · $base"
  ( cd "$dir" && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="$BUILD" "$base.tex" >/dev/null )
  cp "$BUILD/$base.pdf" "$dest/"
  if grep -q 'showsolutions' "$src"; then
    ( cd "$dir" && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="$BUILD" \
        -jobname="${base}_key" -usepretex='\def\showsolutions{}' "$base.tex" >/dev/null )
    cp "$BUILD/${base}_key.pdf" "$dest/"
  fi
}

build_quizzes() {
  need latexmk
  echo "Building quizzes -> quizzes/pdf/"
  mkdir -p "$ROOT/quizzes/pdf"
  for f in "$ROOT"/quizzes/src/*.tex; do compile_tex "$f" "$ROOT/quizzes/pdf"; done
}

build_handouts() {
  need latexmk
  echo "Building lab handouts -> labs/handouts/"
  for f in "$ROOT"/labs/handouts/*.tex; do compile_tex "$f" "$ROOT/labs/handouts"; done
}

build_pack() {
  need python3
  echo "Building student_pack"
  python3 "$ROOT/tools/refresh_student_pack.py"
}

case "${1:-all}" in
  quizzes)  build_quizzes ;;
  handouts) build_handouts ;;
  pdfs)     build_quizzes; build_handouts ;;
  pack)     build_pack ;;
  all)      build_quizzes; build_handouts; build_pack ;;
  *) echo "Usage: tools/build.sh [all|pdfs|quizzes|handouts|pack]" >&2; exit 2 ;;
esac
echo "Done."
