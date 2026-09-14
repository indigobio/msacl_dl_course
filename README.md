# MSACL DS301 Deep Learning

A 15-hour hands-on introduction to deep learning for lab technicians, clinical
chemists, and engineers, taught at the MSACL conference as the capstone of the
Data Science track. Intuition over math, mass-spec examples throughout, and a
Colab lab closing every half-day segment.

## What's here

| Path | Contents |
|---|---|
| `Learning_outcomes.md` | Master lecture-by-lecture outcomes and assessment map |
| `slides/` | HTML slide decks (source) and generated PowerPoint files |
| `labs/` | Colab notebooks — instructor solutions and generated student versions |
| `quizzes/` | Printable handout quizzes and answer keys |
| `data/` | Dataset registry and preparation scripts (raw data is not committed) |
| `references/` | Reading list of landmark deep-learning-in-MS papers |
| `tools/` | Build scripts (strip solutions, HTML→PPTX, quiz→PDF) |
| `course_plan/` | Design decisions log |

## Building course materials

One-time setup:

```bash
pip install -r tools/requirements.txt
playwright install chromium
```

Then:

```bash
python tools/strip_solutions.py --all   # solution notebooks → student notebooks
python tools/pptx2txt.py --all          # decks → reviewable text dumps
tools/build.sh quizzes                  # quiz LaTeX → printable PDFs + answer keys
```

Generated outputs (`labs/student/`, `slides/pptx/`, `quizzes/pdf/`) are never
hand-edited — always edit the source and rebuild.
