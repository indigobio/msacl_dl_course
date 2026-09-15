# MSACL DS301 Deep Learning

A 15-hour hands-on introduction to deep learning for lab technicians, clinical
chemists, and engineers, taught at the MSACL conference as the capstone of the
Data Science track. Intuition over math, mass-spec examples throughout, and a
Colab lab closing every half-day segment.

## What's here

| Path | Contents |
|---|---|
| `Learning_outcomes.md` | Master lecture-by-lecture outcomes and assessment map |
| `slides/` | PowerPoint decks (`pptx/`, the source of truth), their reviewable text dumps (`txt/`), figures and the course template |
| `labs/` | Colab notebooks — instructor solutions and generated student versions |
| `quizzes/` | Printable handout quizzes and answer keys |
| `data/` | Dataset registry and preparation scripts (raw data is not committed) |
| `references/` | Reading list of landmark deep-learning-in-MS papers |
| `tools/` | Build scripts (strip solutions, deck text dumps, quiz→PDF, slide figures, deck linters) |
| `course_plan/` | Design decisions log |

## Building course materials

One-time setup:

```bash
pip install -r tools/requirements.txt
```

Then:

```bash
python tools/strip_solutions.py --all   # solution notebooks → student notebooks
python tools/pptx2txt.py --all          # decks → reviewable text dumps
tools/build.sh quizzes                  # quiz LaTeX → printable PDFs + answer keys
```

The decks in `slides/pptx/` are the SOURCE OF TRUTH: edit them directly in
PowerPoint (or with python-pptx) and commit the regenerated `slides/txt/` dump
alongside each one. `slides/spec/` is scaffolding only — `tools/spec2pptx.py`
writes to `slides/scaffold/`, never to `slides/pptx/`.

Generated outputs (`labs/student/`, `quizzes/pdf/`) are never hand-edited —
always edit the source and rebuild. See `CLAUDE.md` for the full toolchain.
