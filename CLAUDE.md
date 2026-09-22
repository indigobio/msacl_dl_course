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
   **Exercise accuracy is the TOP priority and non-negotiable.** Every you-do,
   quiz item, and worked example must be *solvable and exact*: (a) every fact,
   rule, and **formula** a student needs must be written explicitly on the deck
   *before* the slide that asks them to use it — if an exercise asks for a
   cross-entropy loss value, the formula `CE = -Σ yᵢ log ŷᵢ` (for one-hot truth,
   `-log p_true`) must already be on a prior slide; "read the number off a curve"
   is NOT acceptable when a numeric answer is requested; (b) recompute every
   answer by hand and confirm it is exact and reproducible from the given
   numbers; (c) the figure, the slide `notes:`, and the matching quiz `.tex`
   key must all agree on identical numbers and the identical final answer. An
   unsolvable or wrong exercise is a hard failure, not a style nit.
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
8. **Interactive throughout, not just at the end (strong requirement).** Do-able
   moments are woven across each lecture, never saved for a single block at the
   close. Every mechanic follows one of two patterns:
   - **I-do → you-do → quiz (the default spine):** the instructor works one
     concrete numeric example on a slide (I-do), then the VERY NEXT slide is a
     you-do that **points the room to the matching quiz question** to do now on
     their sheet (same shape, different numbers). The printed quiz is therefore
     a **distributed worksheet filled in across the hour**, not an end-of-class
     block. The you-do slide keeps a visual cue (the figure shape) and its
     `callout: {kind: check, ...}` names the question, e.g. "Your turn → Quiz 1,
     Q2"; the worked answer lives in `notes:`. The recap can say "you've now
     done all of Quiz N — keep the sheet." **Every I-do gets a you-do slide;**
     if a taught mechanic has no matching quiz item, add one (rule 3: if we
     teach it, we assess it) rather than dropping the you-do.
   - **Do-it-together:** a slide the room fills in live, guided cell by cell.
   Aim for a hands-on beat roughly every 8-10 minutes. 💬 pair prompts still
   count, but the spine is compute-it-yourself against the quiz.
9. **Illustrative examples over text walls (strong requirement).** Use a
   concrete, illustrative example — a worked number, a diagram, a picture, an
   MS-anchored scenario — to carry every idea, as much as possible, for
   engagement. A slide that is a wall of plain text is a failure: convert it to
   a figure, a numeric walkthrough, a labeled schematic, a small table, or a
   pipeline. If an idea seems to need paragraphs, it needs an example instead.
   Plain-text bullets are allowed only for genuine list content (agendas,
   recaps, "you can now…") — never to explain a mechanic.

## Repo layout

```
Learning_outcomes.md      Master lecture-by-lecture spec (outcomes + assessments)
course_plan/DECISIONS.md  Dated log of every course-design decision
slides/pptx/              SOURCE OF TRUTH: the decks, edited directly in PowerPoint
                          (or with python-pptx for scripted/bulk edits)
slides/template/          Course template msacl_ds301.pptx — the single home of
                          deck style (theme colors/fonts, master, 6 layouts);
                          layout_preview.pptx is a hand-kept visual reference of
                          all six layouts, not a build artifact
slides/txt/               Generated text dump of every deck (pptx2txt.py) so deck
                          edits stay git-diffable and grep-able; commit with the deck
slides/spec/              YAML specs for SCAFFOLDING new slide sequences;
                          archive/ holds the shipped decks' historical specs
slides/scaffold/          Generated scratch decks from specs (git-ignored) —
                          copy the slides you want into the real deck
slides/assets/            Shared images for decks (slides/assets/img/)
slides/html/              DEPRECATED legacy HTML decks; kept only until each
                          lecture is ported, then deleted
labs/solutions/           Authoritative notebooks WITH solutions (author here)
labs/student/             Generated fill-in-the-blank copies (never hand-edit)
quizzes/src/              Quiz LaTeX sources (one .tex per quiz; the answer key
                          is the same file built with \showsolutions)
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
- **Slides — PPTX-first (DECISIONS.md 2026-09-08):** the decks in `slides/pptx/`
  are the **source of truth**, edited directly in PowerPoint; the YAML→rebuild
  pipeline is retired as owner of the decks. The look lives in a real
  PowerPoint template, `slides/template/msacl_ds301.pptx` (course palette as
  theme colors, Avenir Next/Menlo theme fonts, styled master, six layouts:
  Title Slide, Title and Body, Two Column, Screenshot + Caption, Title Only,
  Blank — all with the mono eyebrow placeholder). New slides are added in
  PowerPoint via New Slide → layout and are on-style automatically. To change
  the style: edit the template (regenerate with `tools/make_template.py` or
  tweak its master in PowerPoint), then run `tools/adopt_template.py --all` to
  propagate the master/theme into every deck (slide content is untouched;
  decks open in PowerPoint are skipped). **After any deck edit, regenerate its
  text dump with `tools/pptx2txt.py` and commit the `.txt` with the `.pptx`** —
  that dump is what makes deck changes reviewable and grep-able (rule 3
  audits). Scripted/bulk edits to decks are done with python-pptx directly.
  `tools/spec2pptx.py` remains as a **scaffolding tool**: it builds a YAML spec
  (slide types `title`, `bullets`, `two-col`, `image`, `pipeline`, `grid`)
  into a scratch deck in `slides/scaffold/` on the same course master — use it
  to generate new rule-6 numeric-walkthrough sequences, then copy those slides
  into the real deck ("Use Destination Theme" keeps them exact). It never
  writes `slides/pptx/`. Real licensed images live under `slides/assets/img/`.
  Everything stays real, editable PowerPoint objects — never screenshots. The
  overlap/overflow checkers below are the lint for hand-edited decks.
- **Quizzes:** one LaTeX source per quiz, built to a letter-size student PDF and
  an answer key by `tools/build.sh`.

### Commands

```
python tools/strip_solutions.py --all      # regenerate labs/student/ from labs/solutions/
python tools/pptx2txt.py --all             # regenerate slides/txt/ dumps (run after ANY deck edit)
python tools/make_template.py              # rebuild slides/template/msacl_ds301.pptx (the style)
python tools/adopt_template.py --all       # propagate the template's master/theme into every deck
python tools/spec2pptx.py <spec.yaml>      # scaffold a spec into slides/scaffold/ (never slides/pptx/)
python tools/check_pptx_overlap.py --all   # sanity-check decks for box overlaps / off-slide content
python tools/check_text_overflow.py --all  # sanity-check decks for text flowing outside a pptx box
python tools/check_figure_overflow.py      # sanity-check figures for text flowing outside a drawn box
tools/build.sh quizzes                     # regenerate quizzes/pdf/ from quizzes/src/*.tex
tools/build.sh handouts                    # compile labs/handouts/*.tex
python tools/make_colab_shots.py           # re-capture the annotated Colab screenshots (Lab 1 setup sheet)
tools/build.sh pack                        # regenerate the student_pack
pip install -r tools/requirements.txt      # one-time setup
```

## Conventions

- Notebooks: `labNN_topic.ipynb`. Author only in `labs/solutions/`; wrap hidden
  code between `### BEGIN SOLUTION` and `### END SOLUTION` lines; keep `assert`
  self-checks OUTSIDE those markers so students keep them. Regenerate student
  copies with the strip tool.
- **Every lab ships a paper hint sheet** at `labs/handouts/labNN_hints.tex` (+
  compiled `.pdf`), because most attendees are clinical chemists / physicians
  with a weak coding background. Lab 1 additionally ships
  `lab01_colab_setup.tex` — a step-by-step Colab sheet with annotated
  screenshots of the real interface. Those figures live in `labs/handouts/img/`
  and are **generated, not pasted**: `tools/make_colab_shots.py` drives headless
  Chrome over the signed-out Colab UI, opens the File/Runtime menus, crops each
  view and draws the callout numbers, so the sheet can be refreshed whenever
  Google restyles Colab. The hint sheet gives multiple-choice code
  options (one correct) for each marked blank, in the style of
  `labs/handouts/lab02_hints.tex`, and explains why the wrong options fail. A
  lab is not done until its hint sheet exists and compiles. Options must match
  the notebook's actual blanks and its `assert` checks exactly.
- Slides: `lectureNN_topic.pptx` in `slides/pptx/` is the deck itself — edit it
  (PowerPoint or python-pptx), never regenerate it from a spec. Keep every
  slide on the course master's layouts; keep `slides/txt/` in sync
  (`tools/pptx2txt.py --all`) and run the overlap/overflow checkers after
  scripted edits. Title-slide eyebrow template:
  `MSACL · DS301 Deep Learning · Segment <n> · Lecture <n>`.
- Write "Lecture <n>", never the "L<n>" shorthand ("L2" the regularization
  term is the exception).
- Quizzes: `quizNN_topic.tex` (source of truth) plus compiled `quizzes/pdf/quizNN_topic.pdf` and `..._key.pdf`. **NN is the lecture number it belongs to, never a sequential count** — e.g. Lecture 7's quiz is `quiz07_...`, "Quiz 7" inside. Only the Lab slots (3/6/9/12/15) have no quiz — numbers are skipped, not renumbered. (Lecture 10 has `quiz10_imbalanced_data.tex`.)
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
