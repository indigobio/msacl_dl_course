# MSACL DS301 Deep Learning

**MSACL DS301 Deep Learning** is a 15-hour hands-on deep learning short course at
MSACL (Mass Spectrometry: Applications to the Clinical Lab) — the highest-level
course in the conference's Data Science track. Use this course name in all
materials. Five 3-hour segments (Sunday–Tuesday), 15 one-hour sessions, each half-day
segment ending in a Colab lab.

## Audience — this shapes every design decision

- Participants are lab technicians, clinical chemists, and engineers with a
  chemistry background. Prerequisite is Data Science 203: basic Python, NO
  advanced math, NO computer-science training.
- The course goal: participants leave able to apply deep learning to their own
  assays/products, and to judge when deep learning is NOT the right tool.

## Non-negotiable style rules

1. **Intuition first, light math welcome.** Most participants hold PhDs — small
   worked computations deepen learning: matrix–vector multiplication by hand,
   one-step gradient updates, backpropagation walked numerically on a tiny
   two-weight network, feature-map cells computed on a grid, an attention
   weighted average. Always concrete numbers, never symbolic derivations, and
   no calculus beyond "derivative = slope."
2. **Mass spec first.** Every architecture and concept gets a mass-spectrometry
   or clinical-lab anchor example (spectra, chromatograms, MALDI-TOF, peptides,
   patient panels). Generic examples (MNIST, cats-vs-dogs) appear only as brief
   bridges, never as the main event.
3. **If we teach it, we assess it.** Every session maps to at least one
   assessment: a printable quiz, a lab checkpoint with self-checking `assert`s,
   or a structured discussion prompt. No orphan content.
4. **Code is read more than written.** Labs are fill-in-the-blank: participants
   complete small marked blanks and get instant feedback from `assert` cells.
   They never write code from scratch. Paper hint sheets offer multiple-choice
   code options for the least code-comfortable attendees.
5. Accessible language: define every piece of jargon at first use; prefer
   diagrams over text walls; slides carry one idea each.
6. **Slide style: minimalist, content-engaging, easy to edit.** Generous
   whitespace, few words, no decorative clutter; the content IS the visual.
   For mechanics (convolution, pooling, stride, backprop), use step-by-step
   numeric walkthrough diagrams in the style of Hung-yi Lee's tutorial deck
   (`Supplements/Deep_learning_in_1day.pptx`, CNN section ≈ slides 150–170):
   real numbers on grids, computed cell by cell across a slide sequence.
   Build decks from simple, editable elements (text, tables, basic shapes,
   SVG diagrams) rather than baked-in raster images wherever possible.
7. **Real figures before hand-drawn ones (strong requirement).** When a concept
   has a canonical real image or published diagram — network architectures
   (VGG/ResNet/transformer), photos, microscopy, real spectra/chromatograms,
   textbook figures — FIRST fetch a properly-licensed one (Wikimedia Commons /
   CC / public domain, or an instructor-provided file in `Supplements/`) and
   store it under `slides/assets/img/` so decks still render offline. Only
   hand-author an SVG when no real image captures the idea: the step-by-step
   numeric mechanics of rule 6, or a bespoke MS-anchored schematic. Always
   credit source + license in the figcaption; downscale to keep files small.

## Repo layout

```
Learning_outcomes.md      Master lecture-by-lecture spec (outcomes + assessments)
course_plan/DECISIONS.md  Dated log of every course-design decision
slides/html/              Slide decks, authored as HTML (design source of truth)
slides/assets/            Shared CSS/images for decks
slides/pptx/              Generated PowerPoint files (never hand-edit)
labs/solutions/           Authoritative notebooks WITH solutions (author here)
labs/student/             Generated fill-in-the-blank copies (never hand-edit)
quizzes/src/              Quiz + answer-key HTML sources
quizzes/pdf/              Generated printable PDFs (never hand-edit)
data/README.md            Dataset registry: source, license, size, prep notes
data/prep/                Scripts that slice raw datasets into Colab-sized files
references/               Reading list of landmark DL-in-MS papers
Supplements/              Instructor-provided reference material (e.g. the
                          Hung-yi Lee deck that sets the diagram style)
tools/                    Build scripts (see Toolchain)
```

## Toolchain

- **Labs:** PyTorch on Google Colab, free tier. Every notebook must run end to
  end in under ~10 minutes on a T4 (or CPU where feasible) and must not require
  sign-ups or credentials to fetch data.
- **Slides:** authored as HTML — one `<section class="slide">` per slide, fixed
  1280×720 design, speaker notes in a child `<aside class="notes">` — then
  converted to **hybrid PPTX**: slide titles and body text become native,
  editable PowerPoint text boxes; diagrams/figures are placed as images
  rendered from the HTML/SVG. (tools/html2pptx.py currently does image-only
  slides; rework to hybrid is a Phase 3 task.)
- **Quizzes:** HTML source printed to letter-size PDF.

### Commands

```
python tools/strip_solutions.py --all      # regenerate labs/student/ from labs/solutions/
python tools/html2pptx.py --all            # regenerate slides/pptx/ from slides/html/
python tools/make_quiz_pdf.py --all        # regenerate quizzes/pdf/ from quizzes/src/
pip install -r tools/requirements.txt && playwright install chromium   # one-time setup
```

## Conventions

- Notebooks: `labNN_topic.ipynb`. Author only in `labs/solutions/`; wrap hidden
  code between `### BEGIN SOLUTION` and `### END SOLUTION` lines; keep `assert`
  self-checks OUTSIDE those markers so students keep them. Regenerate student
  copies with the strip tool.
- Slides: `lectureNN_topic.html`. Keep decks self-contained (inline or
  `slides/assets/` CSS only) so headless rendering works offline. Title-slide
  eyebrow template: `MSACL · DS301 Deep Learning · Segment <n> · Lecture <n>`.
- Write "Lecture <n>", never the "L<n>" shorthand ("L2" the regularization
  term is the exception).
- Quizzes: `quizNN_topic.tex` (source of truth) plus compiled `quizzes/pdf/quizNN_topic.pdf` and `..._key.pdf`. **NN is the lecture number it belongs to, never a sequential count** — e.g. Lecture 7's quiz is `quiz07_...`, "Quiz 7" inside. Lab slots (3/6/9/12/15) and Lecture 10 (worksheet instead) have no quiz — numbers are skipped, not renumbered.
- Data: never commit raw datasets; register every dataset in `data/README.md`
  and keep its slicing script in `data/prep/`.
- Every course-design decision gets a dated entry in `course_plan/DECISIONS.md`.

## Project phases

- Phase 0 — repo scaffold, tooling, CLAUDE.md: **done** (2026-08-25)
- Phase 1 — final lecture-by-lecture outcomes + assessment map (rewrite
  `Learning_outcomes.md`): next
- Phase 2 — dataset & reference collection (verify availability/licenses, prep
  Colab-sized slices, reading list)
- Phase 3 — slide decks (HTML first, then PPTX)
- Phase 4 — the five lab notebooks (solutions first, then stripped student versions)
- Phase 5 — quizzes/handouts, answer keys, timing dry run

## Working with the instructor

The instructor (Lixing Song) actively steers this project. Pause at the end of
each phase for review before starting the next, and ask questions interactively
whenever a design decision is ambiguous rather than assuming.
