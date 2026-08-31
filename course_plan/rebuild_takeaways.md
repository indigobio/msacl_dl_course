# Slides rebuild — cross-lecture takeaways

Running handoff notes from the per-lecture rebuild (YAML spec → native PPTX,
interactive redesign). Each lecture's subagent appends a short section here on
approval so the next fresh agent inherits conventions, reusable assets, and
open decisions. Keep it terse.

## Conventions locked so far
- Pipeline: source of truth is `slides/spec/*.yaml`; build with
  `python tools/spec2pptx.py <spec>`; never hand-edit `slides/pptx/`.
- Interactivity: rule 8 in CLAUDE.md — I-do→you-do and do-it-together beats
  ~every 8-10 min; do-you slides use `callout: {kind: check}` with the answer
  in `notes:`.
- Type scale (builder defaults): title 34, headline 44, bullets 26, body 22,
  callout 17, grid cell 20.
- Legacy HTML: on approval, MOVE `slides/html/lectureNN_*.html` to
  `slides/html_archive/` (archive, not delete).

## Reusable assets (slides/assets/img/)
- sparrow.jpg, sparrows_two.jpg (+ edge/blur variants), cat.jpg, vgg16.png,
  alphafold_protein.png, alphago_go_board.jpg, selfdriving_waymo.jpg,
  maldi_spectrum.png, ai_statistics_meme.png, conv_worked_example.png (CC BY 4.0).

## Per-lecture log
<!-- newest first; one block per lecture on approval -->

### Lecture 1 · What Is Deep Learning? — APPROVED 2026-08-30
- **22 slides, 11 figure-driven.** Spec: `slides/spec/lecture01_intro.yaml`. HTML
  archived to `slides/html_archive/lecture01_intro.html`.
- **Interactive spine fully wired (rule 8):** each I-do → a you-do slide that
  points to its quiz item. Callout convention to replicate verbatim:
  "Your turn → Quiz N, Qk" (compute) and "Pair → Quiz N, Qk" (discuss);
  worked answer always in `notes:`. Recap says "you've done all of Quiz N — keep
  the sheet." Quiz is a distributed worksheet, never an end block.
- **Quiz coupling:** added Quiz 1 **Q6** (two-layer forward pass) so the
  two-layer I-do has a quiz-backed you-do. Every taught mechanic needs its quiz
  item at authoring time (rule 3), not retrofitted. `Learning_outcomes.md` Quiz 1
  line updated. Quiz `.tex` is single-source `\ifsolution`; do NOT rebuild PDFs
  (no LaTeX assumed) — just leave `.tex` correct.
- **NEW reusable figure toolkit `tools/make_slide_figures.py`** (matplotlib +
  numpy + PIL, course palette, ~150dpi, paper bg, auto-downscale). Functions:
  `activations(labeled, order)`, `neuron_schematic(...)`, `matvec(...)`,
  `two_layer_net(...)`, `universal_approx()`, `hook_4panel()`, `cat_hierarchy()`,
  `chatgpt_panel()` (synthetic English, license-safe), `driams_mrsa_spectrum()`.
  Run `python3 tools/make_slide_figures.py [names…]`. REUSE these for Lecture 2.
- **Mathy slides read best as an `image:` figure with numbers baked in + a short
  callout**, not bullet-arithmetic. I-do and you-do share the same figure shape,
  different numbers.
- **Real clinical anchors from `data/slices/` beat generic figures.** DRIAMS MRSA
  (`driams_c_saureus_oxacillin.npz`: keys X (738,6000) float16 TIC-normalized,
  y uint8 697 susc/41 res, code, year; bins map to ~2000–20000 Da) → CC0.
  Lecture 2's backprop MS anchor could similarly plot from a real slice.
- **Builder change:** `image` slide type gained optional `img_h` (default 3.5in),
  backward-compatible. For image slides with caption+callout keep figure aspect
  ≤ ~2.2:1 and skip `calc:`.
- **Sourcing:** synthetic course-palette panels are a clean fallback when a real
  image is language/rights-encumbered. Wikimedia license via
  `action=query&prop=imageinfo&iiprop=extmetadata` with a descriptive User-Agent.

## Layout sanity check (2026-08-30) — overlaps

`tools/check_pptx_overlap.py` opens each built .pptx and flags (A) any shape
running past the slide bottom and (B) the callout box colliding with the
figure/table/caption above it — the recurring "boxes overlapping" bug. It
excludes amber annotation boxes drawn on images (no false positives on the
approved L1/2/4/5/7 decks). **Every build must end with**
`python tools/check_pptx_overlap.py <pptx>` and show 0 FAIL.

ROOT CAUSE still open: `tools/spec2pptx.py::slide_image` places the callout at
`min(y, Inches(6.5))`, so when a figure is tall the callout is clamped ON TOP of
it instead of below. This produced real collisions on lecture08 slides 6/7/8/11.
FIX (do at the next safe window, then rebuild --all and re-check): never place a
callout above the true content bottom returned by `_place_image`; if the callout
would run off-slide, shrink the image (reduce `img_h`) instead of overlapping.
Apply the same "callout top >= content bottom" rule to slide_pipeline/grid/bullets.
