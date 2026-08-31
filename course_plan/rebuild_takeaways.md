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
  alphafold_protein.png (AlphaFold DB, CC0), alphago_go_board.jpg (Xchen27,
  Wikimedia Commons, CC BY-SA 3.0), selfdriving_waymo.jpg,
  maldi_spectrum.png, ai_statistics_meme.png, conv_worked_example.png (CC BY 4.0),
  autoencoder_schema.png (Michela Massi, Wikimedia Commons, CC BY-SA 4.0).

## Per-lecture log
<!-- newest first; one block per lecture on approval -->

### Lecture 13 · Large Language Models + the DL-in-MS Landscape — BUILT 2026-08-31 (pending review)
- **25 slides, 21 figure-driven.** Spec: `slides/spec/lecture13_llms_landscape.yaml`
  → `slides/pptx/lecture13_llms_landscape.pptx`. Survey hour, so the ONE numeric
  beat (rule 6) is a LIGHT next-token **softmax do-it-together** (reuse of
  Lecture 4/7 softmax): logits (2,1,0) → e (7.39,2.72,1.00), sum 11.11 →
  probs (0.665,0.245,0.090), argmax → next token K. Worked in `notes:`, not a
  quiz item.
- **Interactive spine (rule 8), all four Quiz 13 items worked live:** four
  I-do→you-do pairs, one per item — hallucination guardrail (Q4), RL
  agent/action/reward on an auto-tuned LC-gradient loop (Q2),
  prompting-vs-RAG-vs-fine-tuning (Q3 → RAG), and the tool→problem→architecture
  landscape match (Q1). Every landmark tool's problem+arch+impact appears on its
  own tour slide BEFORE the Q1 match (rule 3).
- **Quiz** `quizzes/src/quiz13_llms_landscape.tex` single-source `\ifsolution`,
  quiz08 conventions (`\rb`,`\keybox`,`\work`). Q1 is a 3-col match table
  (tool | problem | architecture) with two word banks; Q2 labels agent/action/
  reward; Q3 one RAG scenario + why; Q4 hallucination + guardrail. Student+key
  both compile (pdflatex, 2 pp each).
- **NEW toolkit funcs (extend, not fork) in `_lecture13_figures()`:**
  `next_token_predict`, `softmax_next_token`, `llm_pipeline(intro/rlhf)`,
  `llm_scale`, `hallucination(ido/youdo)`, `rl_loop(ido/youdo)`, `rlhf_as_rl`,
  `use_llm_chart(ido/youdo)`, `landscape_card` (× Casanovo/Prosit/DIA-NN/DRIAMS),
  `landscape_match(ido/youdo)`. Registered in build_all() and FUNCS (key `llms`
  + per-figure keys). Rebuild: `python3 tools/make_slide_figures.py llms`.
- **Real licensed images (rule 7), REUSED existing assets:** the two flagship
  photos are placed directly in the deck, credited in figcaptions —
  `alphago_go_board.jpg` (Xchen27, Wikimedia Commons, **CC BY-SA 3.0**) on the
  AlphaGo/RL slide, and `alphafold_protein.png` (**AlphaFold DB, CC0**) on the
  AlphaFold tour slide. No new images fetched; the pipeline/RL/RAG/landscape-card
  schematics are hand-authored (MS-anchored, offline, editable) as no clean
  licensed offline diagram exists for them.
- **Verify:** overlap checker 0 FAIL, fractional-EMU 0, all 25 notes non-empty;
  regression rebuild L7/8/10/11 all 0 FAIL / 0 fractional-EMU.

### Lecture 11 · Learning Without (Many) Labels — BUILT 2026-08-30 (pending review)
- **21 slides, 19 figure-driven.** Spec: `slides/spec/lecture11_unsupervised.yaml`
  → `slides/pptx/lecture11_unsupervised.pptx`. Conceptual topic, so the ONE
  numeric compute beat (rule 6) is reconstruction error / MSE → anomaly-vs-normal
  against a threshold (an explicit callback to Lecture 10's sens/spec choice).
- **Interactive spine (rule 8), all four Quiz 11 items worked live:** Q3 recon-error
  compute (I-do x=(.2,.5,.1),x̂=(.2,.4,.1)→MSE 0.01/3=0.0033<0.01 NORMAL; you-do
  x=(.1,.8,.2),x̂=(.1,.5,.2)→MSE 0.09/3=0.03>0.01 ANOMALY), Q1 pick-the-VAE picture,
  Q2 four-way paradigm match, Q4 "why must two views agree" (contrastive/DINO).
- **Quiz** `quizzes/src/quiz11_unsupervised.tex` single-source `\ifsolution`,
  quiz08 conventions (`\rb`,`\keybox`,`\work`). The TWO latent pictures are
  reproduced IN the quiz as self-contained TikZ (Picture A = AE scattered
  islands+gap; Picture B = VAE smooth cloud) so Q1 is answerable without an
  external image — student+key both compile (pdflatex, 2 pp each).
- **NEW toolkit funcs (extend, not fork) in `_lecture11_figures()`:**
  `autoencoder_bottleneck`, `recon_error_worked(ido/youdo)`, `anomaly_overlay`,
  `ae_vs_vae_latent` (labeled + A/B `fig_latent_ab_quiz`), `vae_recipe`,
  `latent_interpolation`, `masking_pretext`, `contrastive_views`,
  `dino_student_teacher`, `pseudo_labeling`, `paradigm_chart(ido/youdo)` +
  helpers `_spectrum_curve`,`_latent_scatter`. Registered in build_all() and FUNCS
  (key `unsupervised` + per-figure keys). Rebuild: `python3 tools/make_slide_figures.py unsupervised`.
- **Real licensed image (rule 7):** `slides/assets/img/autoencoder_schema.png`
  (Michela Massi, Wikimedia Commons, **CC BY-SA 4.0**) for the canonical AE
  schematic; everything else hand-authored (MS-anchored, offline, editable).
  The AE-vs-VAE latent comparison + DINO/contrastive schematics are the rule-6/9
  exception (no clean licensed offline image for those).
- **Verify:** overlap checker 0 FAIL, fractional-EMU 0, all 21 notes non-empty;
  regression rebuild L5/7/8/10 all 0 FAIL / 0 fractional-EMU.

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

ROOT CAUSE FIXED (2026-08-31): `tools/spec2pptx.py::slide_image` used to place
the callout at `min(y, Inches(6.5))`, clamping it ON TOP of a tall figure. Now
`slide_image` caps the image height so the whole stack (image + caption + calc +
callout) fits above the slide bottom, and places the callout at the TRUE content
bottom `y` (never above it). Tall figures shrink instead of colliding. After the
fix, `spec2pptx.py --all` + `check_pptx_overlap.py --all` reports 0 FAIL across
all 7 decks (lecture08 slides 6/7/8/11 and lecture10 slides 3/4 now clean).
Builds since keep 0 fractional-EMU and non-empty notes.
