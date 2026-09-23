# Course Design Decisions Log

Newest entries at the top. Every entry: date, decision, rationale.

## 2026-09-09 — Lecture 10 Part B: every treatment gets its own examples (instructor)

**Problem.** The treatment menu named four moves, but only two of them (SMOTE,
focal loss) had a slide. Curation — the highest-value move for a clinical lab —
was one card, and class weights, the thing everyone actually types, appeared
only as three words on that card. "Match the move to the symptom" (Quiz 10 Q3)
was being asked of moves the room had never seen worked.

**Decision — two new slides, and examples everywhere else:**

- **Part B · 1 · Curation** (new): a real pass on our own 738 spectra —
  −22 runs that failed acceptance, −8 whose S/R label the two AST methods
  disagree on, −12 replicate shots grouped by isolate (leakage) → 696 spectra,
  **35 R / 661 S**. The punchline is exact and counter-intuitive: six fewer
  resistant examples and a better model, because three of the 41 were labelled
  wrong. Four example cards make each action concrete (failed runs, discordant
  labels, replicate spectra, wrong target).
- **Part B · 2 · Make the rare class count** (new, do-it-together): balanced
  weights w = N/(K·n_c) worked on the real split — **9.00** vs **0.53**, a ratio
  of exactly 17 because 697/41 = 17 — beside its three siblings with what each
  means on spectra: oversample (draw the 41 ~17×/epoch), undersample (throws
  away 656 real spectra), augment (m/z jitter ±0.1%, intensity ±10% — Lab 2's
  move, and the physically honest one). Plus the calibration warning: weighting
  changes the effective prevalence, so recalibrate before quoting a probability.
- **Menu**: cards are numbered 1–4 and each carries an example.
- **Focal loss**: the hard cases are now named — MICs on the breakpoint,
  heteroresistant subpopulations, low-biomass spectra — on the figure and in
  the callout.
- **Curriculum**: each stage carries a lab example — clean high-biomass from
  one site → low-biomass and borderline MICs → every site at the true ~6%.

**Timing.** Part B 18 → 22 min, paid for from Part A's opening (13 → 12), the
curve block (17 → 16) and the debrief (7 → 5).

**Noticed while editing:** the instructor removed the two you-do slides (Part A
Q2, Part B Q3/Q4) from the deck earlier the same day. The notes that pointed
"next slide" at them now point at the quiz sheet instead. Quiz 10 still carries
all four items, but Q2–Q4 no longer have a slide that hands the room the pen —
worth a decision before the dry run.

**Propagated.** `slides/pptx/lecture10_imbalanced_data.pptx` (new slides 13–14,
re-embedded menu/focal/curriculum figures, focal + curriculum callouts, agenda,
notes); `curation_pass()` and `class_weight_family()` plus the enriched
`treatment_menu()`, `focal_loss_volume()` and `curriculum_learning()` in
`tools/make_slide_figures.py` (figure key `treatments`); `Learning_outcomes.md`
Lecture 10 outcomes and timing.

## 2026-09-09 — Lecture 10 BUILDS the ROC and PR curves before comparing them (instructor)

**Problem.** The hour went straight from one confusion matrix to "AUROC
flatters, trust AUPRC" — a verdict about two metrics the deck had never
introduced. Nowhere did it say that a classifier emits a *score*, that each
threshold gives its own confusion matrix, or that AUROC/AUPRC are the *areas*
under the resulting curves. Quiz 10 Q2 then asked the room to compute an ROC
point and a PR point and decide between the two areas.

**Decision — three new slides (7–9), built on one threshold sweep of the same
DRIAMS 41 R / 697 S data:**

1. **From one point to a curve (do-it-together)** — score histograms for the
   two classes with three thresholds marked, and a table: strict t = 0.9
   (TP 18, FP 10), default t = 0.5 (TP 30, FP 60 — *the matrix already worked
   by hand*), loose t = 0.2 (TP 38, FP 210). The room fills the t = 0.9 row:
   recall 0.44, FPR 0.014, precision 0.64.
2. **The ROC curve · I-do** — the three dots plus the two free endpoints,
   joined; **AUROC ≈ 0.90** is literally the shaded area; chance is the 0.5
   diagonal, perfect is the top-left corner.
3. **The PR curve · I-do** — the same dots as precision vs. recall;
   **AUPRC ≈ 0.48**; and the point that matters, its chance line is the
   **prevalence** (41/738 = 0.056), not 0.5, so the bar moves with your class
   balance.

The existing "why AUROC flatters" slide is now a comparison rather than an
introduction, and it plots the *same* sweep (its old figure used invented
curves that merely passed near the operating point; both areas are now computed
from the counts by trapezoid, so every number on the three slides is
reproducible).

**No new quiz item.** Quiz 10 Q2 already asks for the ROC point, the PR point,
the prevalence baseline and the AUROC-vs-AUPRC decision — the you-do that
follows these slides is unchanged and now assesses content the deck actually
taught.

**Timing.** +5 min: the curve block is 12 → 17 min on the agenda, paid for out
of the closing debrief (12 → 7), since Quiz 10 is filled across the hour and
the last block is only the pair debrief and collection.

**Propagated.** `slides/pptx/lecture10_imbalanced_data.pptx` (new slides 7–9,
agenda, the comparison slide's figure/callout/notes, the outcomes recap);
figures `fig_threshold_sweep / fig_roc_intro / fig_pr_intro` and a rebuilt
`fig_roc_pr` from `score_threshold_sweep()`, `roc_curve_intro()`,
`pr_curve_intro()` and the shared `SWEEP` table in
`tools/make_slide_figures.py` (figure key `roc_intro`); `Learning_outcomes.md`
Lecture 10 outcomes and timing.

## 2026-09-09 — Lecture 8 points at the NEW Lab 3, and gains a diffusion primer (instructor)

**Problem.** Lecture 8 still sold the retired Lab 3: DistilBERT text
classification, an ESM-2 bonus cell, and a three-way race (full fine-tune vs.
frozen head vs. from scratch). The shipped Lab 3 fine-tunes a pretrained
**diffusion generator** (`balakrish181/ddpm-class-mnist-28`) on ~32 copies of a
symbol the student draws, and two of its three blanks are the diffusion loss
and the fine-tune learning rate. A student who met "diffusion" for the first
time in the notebook would be typing a line they had never seen.

**Decision.** Every Lab 3 reference now describes the real lab, and the hour
ends with a **three-slide diffusion primer** (slides 21–23) so the lab's blank
is something they have already computed by hand:

1. *The idea* — the canonical two-arrow picture on a real MNIST digit and the
   lab's own linear schedule (β 1e-4 → 0.02, 1000 steps): forward adds noise by
   a FIXED recipe (nothing learned), reverse is a network peeling a little
   noise off at each step (the only learned part).
2. *How it trains (I-do)* — worked on a 4-pixel patch: x_t = 0.8·x₀ + 0.6·ε
   gives (1.1, −0.3, −0.2, 0.4), the model guesses the noise, and MSE =
   (0.04+0.01+0.04+0.01)/4 = **0.025**. The slide carries the one code line,
   `loss = F.mse_loss(noise_pred, noise)` — Lab 3's Blank 2 verbatim.
3. *You-do → new **Quiz 8 Q5*** — same two steps, new numbers: x_t = (0.8,
   −0.5, 0.1, 0.6), MSE = **0.045**, plus "which half is learned?" (removing
   the noise; the forward recipe has no weights).

**Scope guards.** No probability, no ELBO, no VAE vocabulary — Lecture 11 still
owns latents and the probabilistic story. The primer stays tied to the hour by
naming DiT (swap the UNet denoiser for a transformer) and is anchored for this
room in the notes only (generative molecule/spectrum design, topping up a rare
class).

**Timing (worth the instructor's eye).** The primer costs ~5 min: the agenda now
reads six stops with positional encoding 12 → 10, assembly 15 → 14 and transfer
learning 13 → 11. Six bullets also needed the agenda type at 21 pt.

**Propagated.** `slides/pptx/lecture08_transformer.pptx` — slide 2 (agenda +
notes), 10, 13, 17 and 19 (stale DistilBERT / ESM-2 / three-way-race
references), new slides 21–23, recap 24 and the Lab 3 bridge 25 rewritten;
figures `fig_diffusion_idea / _train / _youdo` from `diffusion_strip()` and
`diffusion_objective()` in `tools/make_slide_figures.py` (figure key
`diffusion`; the digit is read straight from the MNIST idx file already in the
repo, public domain); `quizzes/src/quiz08_transformer.tex` Q5 + rebuilt PDFs;
`Learning_outcomes.md` Lecture 8 outcomes, timing and assessment.

## 2026-09-09 — Lecture 7 runs on LANGUAGE examples; peptides become applications (instructor)

**Decision, and it is a deliberate exception to style rule 2 ("mass spec
first").** In Lecture 7 the examples we tokenize, embed, and read heat maps on
are now **sentences**. Peptides, spectra and QC traces stay in the hour, but
only as named applications — never as the thing being computed.

**Why the instructor called it.** Attention was invented for language, every
published attention picture is language, and coreference ("it" → "cat") is
checkable by everyone in the room in one second. The peptide-as-sentence
analogy asked the room to hold a chemistry claim and a new mechanic at the same
time, and the old Q1 answer ("K attends to E") rested on a plausible-but-
unverifiable salt-bridge story. Lecture 8 already carries the real MS payoff
(Casanovo, spectrum → tokens), so nothing is lost.

**What changed.**
- *Order matters* (slide 3): "the dog bit the man" vs. "the man bit the dog" —
  identical words, opposite meaning. The QC-drift trace stays as the
  application panel.
- *Tokens* (slide 6): "the cat drank the milk" → word tokens with IDs (the
  repeated "the" reuses id 2), plus an unknown word split into sub-words
  (chroma / ##to / ##graphy). Pair prompt is now "HbA1c", "New York", a
  hyphenated name — then "what is the vocabulary for YOUR sequences?"
- *Embeddings* (slide 7): word clusters (animals / royalty / drinks / verbs)
  instead of residue clusters; the residue version is one application line.
- *Self-attention* (slides 8, 10): the clause **cat · drank · milk · because ·
  it**, with the full 5×5 query×key heat map and the "it" row read as
  coreference (0.55 on "cat").
- *You-do → Quiz 7 Q1* (slide 11): the sentence "the farmer fed the chickens
  because they were hungry"; query "they" over farmer 0.10, fed 0.05, chickens
  0.60, because 0.10, they 0.15. (a) chickens; (b) coreference, and **number
  agreement** rules out the singular "farmer" — a checkable linguistic reason
  where the old item had a hand-wave.
- Multi-head, "why attention won", and the recap now speak of coreference and
  agreement, with charge complementarity as the "train it on residues instead"
  aside. Q2 (the full attention head) is pure arithmetic and is unchanged.

**Propagated.** `slides/pptx/lecture07_attention.pptx` (slides 1–3, 5–8, 10, 11,
15–17: figures, titles, callouts, notes); the six regenerated figures
`fig_seq_order / fig_tokenize / fig_embedding / fig_selfattn_alltoall /
fig_attn_heatmap_ido / fig_attn_heatmap_youdo` from `seq_order_matters()`,
`tokenize_sentence()` (was `tokenize_peptide`), `embedding_space()`,
`selfattn_all_to_all()` and `attention_heatmap()` in
`tools/make_slide_figures.py`, with the shared token/weight tables
`ATTN_SENTENCE`, `ATTN_SENTENCE_MAP`, `ATTN_QUIZ_TOKENS`, `ATTN_QUIZ_WEIGHTS`;
`quizzes/src/quiz07_attention.tex` Q1 + rebuilt PDFs; `Learning_outcomes.md`
Lecture 7 outcomes, timing and assessment lines.

## 2026-09-09 — Lecture 5 teaches channels properly: c_in and c_out, on d2l §7.4's numbers (instructor)

**Decision.** The deck had one qualitative channels slide ("every filter its
own map"). It now has a four-slide arc — **CHANNELS · 1–4 OF 4**, slides 17–20:

1. (existing) real filtered photos: many filters → many maps.
2. **Multiple INPUT channels, worked by hand.** One filter carries *one slice
   per input channel*; the same window is taken from every channel, each is
   multiplied by its own slice, and the partial sums **add** into one cell:
   19 + 37 = 56. Two channels in, still ONE map out.
3. **Multiple OUTPUT channels.** The same input through three filters gives
   three maps — [[56,72],[104,120]], [[76,100],[148,172]],
   [[96,128],[192,224]] — i.e. a 3-channel output stack, with the two rules
   (filter depth = c_in, filter count = c_out) and the weight count
   c_out × c_in × k × k + c_out.
4. **You-do → Quiz 5 Q5** (new item): add two per-channel partials into one
   cell (8 + 3 = 11), count a 2→5-channel 3×3 layer (90 weights + 5 biases,
   next layer in_channels = 5), and give in_channels for three MRM transitions.

**Numbers come from d2l §7.4** (Zhang et al., *Dive into Deep Learning*,
CC BY-SA 4.0, fig. 7.4.1 and the c_o × c_i × k_h × k_w section), redrawn in the
course grid style rather than copied, and credited on both figures — so the
canonical worked example a participant meets online is the one they did here.
1×1 convolutions (d2l §7.4.3) are deliberately NOT taught; slide 19's notes
carry a one-line answer if someone asks.

**Why it was missing.** in_channels/out_channels appear as Conv1d arguments in
the 1D slide and in Lab 2, and "a filter is one per channel" is the standard
misconception; the deck asserted the depth axis without ever computing it.

**MS anchor.** Channels are co-recorded traces on a shared axis: the three MRM
transitions of one analyte (in_channels=3), sample + internal standard, or one
channel per selected m/z image in imaging MS — while a single MALDI-TOF
spectrum, i.e. Lab 2, is in_channels=1.

**Timing (needs the instructor's eye).** The block costs ~5 min: the deck and
`Learning_outcomes.md` now show the convolution stop at 31 min (was 26) and the
closing stop at 11 min (was 16), which compresses the LeNet→VGG row to ~3 min
and detection to 8. If that trade is wrong, the cheapest reversal is to drop
slide 20 (the you-do) and set Quiz 5 Q5 as take-home.

**Propagated.** Deck slides 2 (agenda), 17–20 (new arc), 25 and 35 (recap now
lists Q5); new figures `fig_channels_in/out/youdo.png` from
`channels_in()`/`channels_out()` in `tools/make_slide_figures.py` (figure key
`channels`); `quizzes/src/quiz05_cnns.tex` Q5 + rebuilt PDFs;
`Learning_outcomes.md` outcomes, timing and assessment lines.

## 2026-09-09 — Quiz 5 Q1 asks for the WHOLE output map at both strides (instructor)

**Decision.** Quiz 5's convolution item no longer asks for a sample of cells
(three at stride 1, one at stride 2). It now asks the room to **derive the
entire output matrix twice**: all nine cells of the stride-1 map and all four
cells of the stride-2 map, each written into a printed answer grid.

**Rationale.** Filling a whole map is where the mechanic actually lands — the
window has to be re-placed nine times, so weight sharing stops being a claim
and becomes something they did. It also makes the stride insight self-evident:
the stride-2 map turns out to be the four corners of the stride-1 map, i.e. no
new arithmetic at all, which is exactly the "stride is a cost/resolution dial"
point from the I-do slides. Answer: stride 1 = [[4,3,1],[1,4,3],[5,0,4]],
stride 2 = [[4,1],[5,4]] (filter = corners + centre, so each cell counts how
many of those five window positions are 1).

**Propagated.** `quizzes/src/quiz05_cnns.tex` (+ rebuilt student/key PDFs);
Lecture 5 slide 13 (the you-do) — new figure `fig_conv_youdo.png` now shows
two empty maps, `3×3` and `2×2`, via a new `conv_walk_two_strides()` in
`tools/make_slide_figures.py` (figure key `conv_youdo`); slide 13 subtitle,
callout and notes; slide 12's notes; `Learning_outcomes.md` Lecture 5
assessment line.

## 2026-09-08 — Slides go PPTX-first: decks are hand-edited, style lives in a real template (instructor)

**Problem.** The YAML-spec → `spec2pptx.py` pipeline made every deck a build
artifact: adding one paragraph or pasting a screenshot meant a YAML round-trip,
and any direct PowerPoint edit was destroyed by the next `--all` rebuild. The
root cause: decks carried NO slide master/theme — all styling was Python
constants, so PowerPoint could offer no on-style way to add or edit a slide.

**Decision (instructor-approved): flip ownership.**

- **`slides/pptx/` decks are now the source of truth**, edited directly in
  PowerPoint. The old "never hand-edit slides/pptx" rule is retired.
- **Style lives in a real PowerPoint template** —
  `slides/template/msacl_ds301.pptx`, generated by `tools/make_template.py`:
  course palette as Office theme colors, Avenir Next/Menlo as theme fonts, a
  styled master, and six layouts (Title Slide, Title and Body, Two Column,
  Screenshot + Caption, Title Only, Blank), every one carrying the mono
  eyebrow placeholder. New Slide → pick a layout → already on-style; pasting a
  screenshot has a dedicated picture-placeholder layout.
- **`tools/adopt_template.py`** spliced that master/theme into all existing
  decks (slide content untouched — verified byte-identical text/notes/images;
  old Office master removed). Re-run it after editing the template to
  propagate a style change to every deck. Idempotent; skips decks PowerPoint
  has open (lock file).
- **`tools/spec2pptx.py` is demoted to a scaffolding tool.** It now builds
  from the course template into `slides/scaffold/` (git-ignored, never touches
  slides/pptx/) — used to generate new rule-6 numeric-walkthrough sequences,
  which are then copy-pasted into the real deck ("Use Destination Theme" is an
  exact match since both share the master). The ten shipped specs moved to
  `slides/spec/archive/` as historical record; rebuilding them does NOT
  reproduce hand edits made after this date.
- **`tools/pptx2txt.py`** dumps every deck's text/tables/notes to
  `slides/txt/lectureNN.txt` (committed alongside the .pptx) so deck edits
  stay git-diffable and exercise-accuracy audits stay grep-able. Regenerate
  after any deck edit.
- The overlap/overflow checkers are unchanged and now serve as the lint for
  hand edits.

**Rationale.** The generator's one guarantee (regenerate everything from text)
was costing more than it delivered once the decks existed; the natural
lifecycle of course slides is scaffold once, hand-tune for years. Persistent
cosmetics belong in the file format's own machinery (theme + master + layouts),
where a palette change is a master edit that PowerPoint propagates natively.

## 2026-08-28 — Reproducible uv lab environment + shareable student pack (pyproject-only)

- **Single source of truth = `pyproject.toml` (+ `uv.lock`).** No exported
  requirements file anywhere (an earlier `requirements-colab.txt` /
  `uv export` design was dropped per instructor). Install is `uv sync` locally;
  Colab reads `pyproject.toml` directly.
- **New folder `student_pack/`** is the single self-contained thing students
  receive. Committed: `pyproject.toml`, `uv.lock`, `.python-version`, `README.md`.
  Generated (gitignored): `student_pack/labs/*.ipynb` (assembled from
  `labs/student/` by `tools/build_student_pack.py`) and `student_pack/.venv/`.
  *(Folder name and committed-vs-assembled split are open to instructor review.)*
- **Dependency layout (the crux of the local↔Colab split):**
  - `[project.dependencies]` = the base stack shared by BOTH environments:
    numpy, pandas, scikit-learn, matplotlib, openpyxl, transformers, datasets,
    accelerate. **torch is intentionally NOT here.**
  - `[dependency-groups]`: `local = ["torch"]` (CPU torch, local only),
    `dev = ["jupyterlab", "ipykernel"]` (local Jupyter).
  - `[tool.uv] default-groups = ["local", "dev"]` so a **bare local `uv sync`**
    installs base + CPU torch + Jupyter.
- **Cross-platform CPU torch via uv.** `[tool.uv.sources]` sends torch to the
  explicit PyTorch CPU index (`download.pytorch.org/whl/cpu`) only on
  Linux/Windows (`marker = "sys_platform == 'linux' or sys_platform == 'win32'"`);
  macOS falls back to PyPI (already CPU-only). Lock pins **torch 2.13.0**
  (2.13.0+cpu on Linux/Win); universal resolution pulled in **zero nvidia/CUDA
  wheels**.
- **Colab keeps its own GPU torch** and its env is not pruned. Bootstrap cell:
  `pip install -q uv` then **`uv pip install --system -q -r pyproject.toml`**.
  This reads only `[project.dependencies]` (so torch and Jupyter — both in
  groups — are skipped), is *additive* (never removes Colab packages), and leaves
  the preinstalled GPU torch untouched (accelerate's transitive torch requirement
  is satisfied by Colab's existing torch, so nothing is reinstalled).
- **Why `uv pip install`, not `uv sync`, on Colab:** tested both against a
  scratch non-venv system prefix. `uv sync` (even `--inexact --no-group local
  --no-install-package torch`) works on a uv-*venv* but, pointed at a non-venv
  system prefix like Colab's, plans to uninstall ~50 / install ~64 to realign the
  WHOLE system env to the lock — too disruptive and risks breaking Colab's other
  preinstalled libs. Also confirmed the group alone can't exclude torch (accelerate
  pulls it transitively); only `--no-install-package torch` suppresses it under
  sync. `uv pip install -r pyproject.toml` is additive and system-native, so it is
  the robust choice. Tradeoff: it does not read `uv.lock`, so Colab gets
  compatible-but-not-lock-pinned versions (acceptable; lock-pinned parity is
  guaranteed for the local env only).
- **Env kept separate from `tools/requirements.txt`** (slide/quiz build chain).
- **`requires-python = ">=3.10,<3.13"`**, `.python-version` = 3.12.
- **Verified:** `uv lock` (162 pkgs); bare `uv sync` installs base + CPU torch
  2.13.0 + Jupyter, imports OK; Lab 1 solutions notebook runs headless in ~29 s,
  all asserts pass (loss 0.259→0.014, test acc 0.742). Colab command validated in
  a scratch venv with a preinstalled torch: base deps installed, **torch 2.13.0
  untouched**, nothing pruned (only fsspec version-realigned per datasets'
  constraint), Jupyter correctly skipped. **uv 0.7.2 sufficed** (features need
  ≥0.5.3; no newer uv required).

## 2026-08-28 — Lecture 2 locked; workflow rule; Lab 1 scoped (instructor)

- **Lecture 2 is locked** — deck `slides/html/lecture02_training.html` (25 slides,
  rendered/verified) and quiz `quizzes/src/quiz02_training.tex` (+ student and key
  PDFs in `quizzes/pdf/`). Key design choices, for continuity with later lectures:
  - Opens with a **catapult analogy** (fire → measure the miss → adjust the power →
    repeat) as the intuitive "guess → measure → correct" loop, then the four-box
    loop overview.
  - **Loss is honest MSE** = mean of (guess − truth)², *no* ½ factor (matches
    PyTorch `MSELoss`); the backward chain therefore starts from the local slope
    `2 × error`. Cross-entropy stays deferred to Lecture 4. (The MS-anchor
    resistant/susceptible loss slide was cut per instructor.)
  - "**Step size**" is introduced before naming the "learning rate"; a too-small
    rate is shown to crawl and can **stall in a local optimum**.
  - **Backprop worked on a two-weight net**: x=2, w₁=0.5, w₂=1.5, target y=1 →
    forward ŷ=1.5, loss=0.25; gradients **3.0 (w₁) / 1.0 (w₂)**; one step at
    lr=0.1 → w₁=0.2, w₂=1.4, loss 0.25→0.19. A companion **PyTorch/autograd slide**
    reproduces the same numbers. Partial-derivative notation is shown **above each
    chain box on all backprop slides** (reference only; audience still just
    multiplies numbers). Chain rule named once, no symbols.
  - **Newton's method / curvature (2nd derivative)** is a clearly-optional side
    slide ("OPTIONAL · SAFE TO SKIP").
  - **Quiz 2**: one-step backprop on a two-weight net (numbers given), one GD
    update by hand, a **four-curve** LR diagnosis that also asks for the fix
    (adds "far too high / diverging" vs. "a bit too high / noisy but converging"),
    order the five training-loop lines, and a short **overfitting discussion**
    that foreshadows Lecture 4. `\ifsolution` single-source toggle.

- **Workflow rule (instructor).** The instructor updates `Learning_outcomes.md`
  before requesting each new lecture/lab. Always **re-read the relevant outcomes
  section first**, build from it, then **pause for review** before moving on.

- **Lab 1 scope narrowed (instructor).** Lab 1 is a ~50-minute, coding-light
  session for lab techs/MDs with minimal coding experience. Focus is purely on
  getting comfortable with **Colab + PyTorch and the general train/eval loop**
  (the five-line loop from Lecture 2 plus an evaluation pass). Data prep is
  provided as read-and-run; the only blanks are the loop + eval; keep the
  notebook small. Not a data-science lab. (Learning_outcomes.md Lab 1 section
  rewritten accordingly.)

- **Language cleanup.** Copyedited instructor-typed notes in Learning_outcomes.md
  (Lecture 1's "Changes" scratch block folded into the 0:10–0:20 content bullet;
  Lecture 4's learning-rate-schedule additions given a proper outcome + timed
  home). No pedagogical decisions changed.

## 2026-08-25 — Naming + Lecture 5 motivation (instructor)

- **Write "Lecture <n>", never "L<n>"** in all course materials — the shorthand
  is too vague. (Careful: "L2" as the *regularization* term stays "L2".)
- **Course branding on slides: "DS301 Deep Learning".** Every deck's title-slide
  eyebrow follows the template
  "MSACL · DS301 Deep Learning · Segment <n> · Lecture <n>".
- **Lecture 5's "Why CNN" section motivates with real 2D photographs**
  (Wikimedia CC BY-SA sparrow photos, stored in slides/assets/img/ with
  attribution in figcaptions) in the Hung-yi Lee beak-detector spirit; the
  mass-spec version of each property lands via the "MASS SPEC ANCHOR" callout
  on the same slide.

## 2026-08-25 — Phase 2 resolutions (research verified; see data/README.md and references/reading_list.md)

- **Lab 1 dataset: MTBLS90** serum LC-MS metabolomics (CIMCB tidy copy; 968×189
  named metabolites, balanced binary task, zero NaNs). HCV liver panel (UCI 571)
  kept in reserve as an Lecture 10 imbalance/missing-data example.
- **Lab 2/4/5 dataset: DRIAMS-A** via the frozen Nov-2021 Zenodo release (CC0;
  Dryad's 2025 update breaks loaders). Primary task S. aureus+oxacillin (MRSA,
  3,064 spectra, 24% R); secondary E. coli+ceftriaxone (3,875, 28% R). Prep
  script `data/prep/prepare_driams.py` (instructor one-time 86 GB pull;
  stream-extract binned_6000 + id only).
- **Lab 3: DistilBERT + HF `medical_abstracts`** (CC-BY-SA) as graded core
  (fine-tune vs. head-only vs. from-scratch, fp16, 6–9 min on T4; labels are
  1–5 → remap), plus **ESM-2 8M peptide bonus demo** (~30 s, demo-only due to
  peptide-dataset licensing and weak from-scratch contrast).
- **Lecture 5 segment / Lab 5 Track B: PeakOnly annotated ROIs** (5,365 windows,
  peak/noise + quality sub-labels; annotation zip has no explicit license —
  cite the paper). Prep script tested; slice built (3.8 MB npz in data/slices/).
- **Lecture 10 cautionary tale: Wiesmann et al., J Clin Microbiol 2025** — external
  validation of the DRIAMS paradigm: 0.065–0.225 AUROC loss cross-site, decay
  within 18 months. Same task as Lab 2. Zech et al. 2018 as optional second.
- **Participant handout review: Beck et al., ACS Meas Sci Au 2024** (CC-BY,
  free to print/distribute).
- Reading list fully verified with DOIs (references/reading_list.md).

## 2026-08-25 — PPTX pipeline: hybrid output (instructor)

- Slide decks convert to **hybrid PPTX**: titles and body text as native,
  editable PowerPoint text boxes; diagrams/figures placed as images rendered
  from the HTML/SVG source. tools/html2pptx.py to be reworked accordingly in
  Phase 3 (slide HTML will need a structural contract: title/body/figure
  regions).

## 2026-08-25 — Phase 1 review feedback (instructor), applied to Learning_outcomes.md

- **Math requirement relaxed** — participants are largely PhDs. Worked light math
  is in: matrix–vector multiplication (Lecture 1/Q1), gradient one-steps and a numeric
  backprop walk on a two-weight network (Lecture 2/Q2), feature-map grid computations
  (Lecture 5/Q4). Still no symbolic derivations.
- **Imbalanced/scarce-data strategies added to Lecture 10**: class weights + focal loss,
  data curation, augmentation, transfer learning.
- **Lecture 5 CNNs go 2D-first** with detailed conv/pooling/stride numeric-grid
  visualizations in the style of Supplements/Deep_learning_in_1day.pptx
  (Hung-yi Lee, CNN slides ≈150–170); Quiz 4 becomes diagram-based; 1D mapping
  to spectra follows the 2D mechanics.
- **Lecture 8 adds "spectrum → tokens" strategies**: patch/chunk + linear projection
  (ViT), CNN front-end, peak-as-token (Casanovo).
- **Lecture 11: VAE explained more intuitively** with side-by-side AE vs. VAE latent-space
  visuals; semi-supervised reduced to one slide; DINO added as the modern
  self-supervision exemplar.
- **Lab 4 uses a VAE** (not a plain AE): anomaly detection + latent-space
  exploration + sample/interpolate generation payoff.
- **RL touch-base moves to Lecture 13** (agent/env/reward, AlphaGo, RLHF); Lecture 14 only
  calls back to it.
- **Lecture 14 demo plan**: participants follow along on their own free-tier chatbots;
  instructor's advanced demo runs on paid Claude Code (Bedrock API alternative,
  pre-recorded fallback).
- **Slide style**: minimalist, content-engaging, easy to edit (CLAUDE.md rule 6).

## 2026-08-25 — Phase 1 kickoff decisions (instructor)

- **Object-detection segment lives at the end of Lecture 5 (CNNs)**, ~15–20 min,
  framed as "CNNs beyond classification: object detection for peak picking in MS"
  (chromatogram as image, peaks as detected objects; PeakOnly as the anchor).
- **Discussion format: both.** (1) A 10-minute structured "apply this to your lab"
  discussion closes every half-day segment, right after the lab, with prompts
  printed on the handouts; (2) short 2–3 minute think-pair-share prompts embedded
  inside lectures. The ~15 min/day cost comes out of lecture content, which is
  accounted for in the Phase 1 timing plans.
- Original instructor draft archived at course_plan/Learning_outcomes_draft_v0.md
  before the Phase 1 rewrite of Learning_outcomes.md.

## 2026-08-25 — Initial course-shape decisions (instructor review of critique)

- **Adopt the revised 15-slot outline**: labs stay at slots 3/6/9/12/15; Lecture 1
  splits math refresher from PyTorch; Lecture 2 merges backprop intuition with the
  PyTorch training loop.
- **Object detection: light and quick, not a full hour + lab.** Keep a short
  segment specifically showing object detection applied to *peak detection in MS*
  (chromatograms/spectra as the "image"). Placement to be finalized in Phase 1.
  The former object-detection lab (slot 12) becomes an autoencoder-on-spectra
  QC/anomaly-detection lab.
- **Lecture 10 becomes "Representing MS data for DL + evaluation & trust in the
  clinical lab"**: data representations (binned spectra, peak lists, 1D signals,
  imaging MS), metrics beyond accuracy, class imbalance, batch effects/instrument
  drift, external validation, when NOT to use deep learning, regulatory touchpoint.
- **Assessments test intuition, not symbol manipulation** — no backprop/attention
  calculation quizzes.
- **Datasets: no constraints** (public preferred but sign-ups acceptable).
- **Toolchain confirmed**: PyTorch + Google Colab labs; HTML slides as design
  source converted to PPTX; printable PDF handout quizzes.
- **Add structured discussion sessions** to the schedule; format and placement to
  be decided in Phase 1.
- **Process**: work proceeds in phases (0 scaffold → 1 outcomes → 2 datasets →
  3 slides → 4 labs → 5 quizzes/dry-run); pause for instructor review at the end
  of each phase; instructor actively steers via interactive questions.

## 2026-08-30 — Lab 2 restructure: threshold as the imbalance teaching moment

- **Plain BCE, no `pos_weight`.** Lab 2's main CNN now trains with vanilla
  `BCEWithLogitsLoss()`. Dropping the class weight removes a confusing moving
  part and makes the imbalance failure honest: at the default 0.5 threshold the
  model scores ~0.94 accuracy but ~0.00 resistant recall — the "always
  susceptible" trap of Lecture 10.
- **New Step 7 is a threshold sweep.** Replaces the old focal-loss-in-main-flow
  section. `evaluate_model` gained a `threshold=` argument; the step sweeps a
  ladder of cuts (0.5 plus data-driven quantiles) and shows recall climbing
  (0.00 → ~0.67) as precision/accuracy fall. Core lesson: recall is not a fixed
  model property — it depends on the decision threshold, and on rare-class data
  you pick the cut to hit a clinical target rather than trusting 0.5.
- **Focal loss demoted to Optional stretch**, replacing the old lr/batch grid
  search. Kept as the "train a model whose probabilities aren't so squashed"
  follow-up (focal @ 0.5 recovers recall ~0.89). The earlier bug fix (unused
  `alpha` term) is retained.

## 2026-08-30 — Slides pipeline v2: YAML spec → native editable PPTX; interactive redesign

- **Retired the HTML→screenshot pipeline.** `tools/html2pptx.py` (rendered each
  HTML slide to a PNG and pasted it full-bleed → flat, un-editable, web-styled
  decks, 700–900 lines of HTML per lecture) is **removed**. Reasons: costly to
  author, hard to maintain, too web-like, and illustrations were mostly
  self-plotted SVG.
- **New source of truth: `slides/spec/lectureNN_topic.yaml`** — a compact deck
  spec (~10-15 lines/slide) built to **native, editable PowerPoint** by
  `tools/spec2pptx.py` (python-pptx). Slide types: `title`, `bullets`,
  `two-col`, `image`, `pipeline`, `grid`. Native theme mirrors the course
  palette (no CSS). Real licensed images placed with `image:` + annotation
  `boxes:`; SVG→PNG only where numeric mechanics need richer visuals.
- **Interactivity is now a non-negotiable rule (rule 8).** Every lecture weaves
  do-it-together / I-do-then-you-do beats throughout (~every 8-10 min), not a
  single do-able block at the end. Do-you slides carry a `check` callout with the
  worked answer in speaker notes.
- **Rebuild process:** one fresh subagent per lecture builds the spec from the
  legacy HTML + Learning_outcomes, iterates with the instructor over multiple
  review rounds, then returns general takeaways to the main agent
  (`course_plan/rebuild_takeaways.md`) before a fresh agent starts the next
  lecture/lab. Each legacy HTML deck is **moved to `slides/html_archive/`**
  (not deleted) once its spec is approved, so it stays available for reference
  while later rebuilds still find their source in `slides/html/`.

## 2026-08-30 — Automated build campaign for remaining lectures & labs
Instructor authorized a **fully autonomous** build of every remaining artifact,
one subagent-orchestrated pipeline per artifact, no review pauses (instructor
reviews all at the end). Added **principle 9** to CLAUDE.md: illustrative
examples over walls of plain text.
- **Baselines (excluded):** Lecture 1, Lecture 2, Lab 1, Lab 2. (Lecture 5 was
  briefly mis-listed as a baseline — corrected 2026-08-30: its partial 11-slide
  POC is only a starting point; the campaign COMPLETES it into a full deck.)
- **Queue:** Lecture 4 → Lecture 5 (complete the partial POC) → Lecture 7 →
  Lecture 8 → Lab 3 → Lecture 10 (Architecture Matchmaker worksheet, no quiz) →
  Lecture 11 → Lab 4 → Lecture 13 → Lecture 14 → Lab 5 (capstone). Missing
  quizzes (quiz05/08/11/13/14) are authored inside their lecture pipeline;
  quiz04/07 exist (verify/upgrade). Labs get an end-to-end run.
- **Per-artifact pipeline (one async workflow each, sequential to protect the
  shared figure toolkit / assets / quizzes):** worker BUILD → reviewer CRITIQUE
  → worker REVISE (1 fixed round). Then a gate critique; if it still reports
  STRONG issues, run ONE more revise (1 optional round) and push. (Rounds were
  cut over 2026-08-30 to reduce wall-clock/cost: first 2+2 → 2+1 → finally 1+1.
  The 1+1 cap applies from Lecture 5 onward; Lecture 4 ran under an earlier cap.)
- **Critique focus:** compliance with all 9 CLAUDE.md principles, accuracy of
  quiz-question references, soundness of examples, effectiveness of assessments,
  engagement, and logic-flow streamlining.
- **Commit policy (added 2026-08-30):** each finished lecture/lab lands as ONE
  git commit, made by the parent in the handoff turn after verifying the
  pipeline output and before launching the next artifact. The pre-campaign work
  (Lectures 1-2 polish, figure toolkit, decisions) is committed as a baseline
  commit just before the first artifact (Lecture 4) commit.

## 2026-08-31 — Autonomous rebuild campaign complete

All lectures, labs, quizzes, worksheets, and handouts have been rebuilt under the
draft → critique → revise pipeline (fresh disk-reading subagent per artifact,
1-fixed-plus-1-optional review cap, exercise-accuracy as the #1 gate and
`check_pptx_overlap.py` 0-FAIL as a hard layout gate). Final state:

- **10 native-PPTX decks** (lectures 1,2,4,5,7,8,10,11,13,14) — all build with
  0 overlap failures, 0 fractional-EMU, non-empty notes; the I-do→you-do→quiz
  spine and illustrative-example rules applied throughout.
- **Assessments:** quizzes 1,2,4,5,7,8,11,13,14; the Lecture 10 Architecture
  Matchmaker worksheet; the Lecture 14 follow-along handout. (No quiz for lab
  slots or Lecture 10, by design.)
- **5 lab solutions** (lab01–05) with fill-in-the-blank blanks + structural
  asserts, each with a paper hint sheet in `labs/handouts/`; Lab 5 also ships a
  project one-pager. Labs 3/4/5 have persisted executed smoke evidence.
- **Tooling:** `check_pptx_overlap.py` added; `spec2pptx.py` callout-over-image
  overlap root cause fixed (cap image height, callout at true content bottom).

Pre-class human TODOs (out of scope for the build): one real-T4 dry run for Lab 3
(confirm the ~65% fine-tuned vs ~33% scratch gap and 6–9 min budget) and the
repo-wide `RAW_BASE '<ORG>/<REPO>'` fill-in across lab02/04/05 before Colab publish.

## 2026-08-31 — Casanovo: keep, but introduce before referencing

Reversed an in-progress wholesale removal of the "Casanovo" (de novo peptide
sequencing) reference. Decision: **keep** Casanovo as the concrete MS anchor for
encoder–decoder / peak-as-token, but never name it "silently" before it is
introduced. Concretely: (1) Lecture 7 (which precedes Lecture 8) no longer names
Casanovo — its two forward mentions are now generic ("peptide-sequencing
transformers we build in Lecture 8" / plain "peak-as-token"); (2) Lecture 8 now
introduces it on first use — the first student-visible mention (BERT-vs-GPT
figure) reads "Casanovo — a spectrum→peptide model (de novo); deep-dive soon,"
so the later three-families / tokenization references and the dedicated
"spectrum in, peptide out" intro slide all land after a real gloss. Downstream
references (Lecture 10/13, quizzes, worksheet, reading list) are unchanged since
they follow the Lecture 8 introduction. A fuller Casanovo mini-session remains a
possible future addition (candidate home: Lecture 8, where the encoder–decoder
machinery lives).

## 2026-08-31 — Lecture 10 refocused on imbalanced data; worksheet → Quiz 10

Lecture 10 was "Your Data, Your Metrics: Representation, Evaluation, and Trust,"
which grazed five topics (representation chart, metrics, imbalance, distribution
shift, when-not-DL) and landed none deeply. Instructor decision: **refocus the
whole hour on IMBALANCED DATA** — its evaluation and its treatment — done deeply.
New title: **"No Perfect Data: Evaluation and Treatment."**

- **Part A (measure it):** the accuracy trap (real DRIAMS 697 S / 41 R); confusion
  matrix by hand → precision (NEW this hour), recall (sensitivity), specificity;
  why AUROC flatters a rare positive while AUPRC / the PR curve is honest (with
  the numbers — same operating point on both curves, PR baseline = prevalence);
  one folded honesty slide on external + temporal validation.
- **Part B (treat it):** data curation; SMOTE (numeric do-it-together A=(2,6),
  B=(4,10), λ=0.5→(3,8), with a high-dim non-physical-spectrum caveat); class
  weights + focal loss (worked: easy p=0.9→FL 0.001, hard p=0.5→FL 0.173, 165× vs
  plain-CE 6.6×); curriculum learning (intuition); match-the-treatment decide beat
  + the durable "which adds NEW signal? → none, only more real minority data."
- **CUT:** the representation decision-chart and the when-NOT-deep-learning content
  (and their figures fig_repr_chart / fig_not_dl / fig_scarce_toolkit, git-removed).

**Assessment change:** the Architecture Matchmaker WORKSHEET is retired entirely;
Lecture 10 now has **Quiz 10 (`quiz10_imbalanced_data.tex`)** — a distributed quiz
filled across the hour (Q1 confusion-matrix rates, Q2 AUROC vs. AUPRC, Q3 treatment
match + "no new signal," Q4 a SMOTE point), matching the I-do → you-do → quiz spine
of the other lectures. Quiz-numbering updated: only the Lab slots (3/6/9/12/15) now
skip a quiz; Lecture 10 joins 1,2,4,5,7,8,11,13,14. Deck renamed
lecture10_data_metrics.yaml → lecture10_imbalanced_data.yaml (18 → 16 slides).
Learning_outcomes.md and CLAUDE.md updated accordingly.

## 2026-09-10 — Lecture 13: the LLM demo runs on real language, not peptides

Instructor decision, mirroring the 2026-09-09 call that put Lecture 7 on language
examples: the "LLMs demystified" opening of Lecture 13 was teaching next-token
prediction on an **amino-acid chain** (`A – C – D – E – ?`, candidates K / R / G).
That is an analogy standing in for the real thing — a large language model is a
model of **text** — so the two mechanics slides now use an ordinary English
sentence out of a lab report:

- **Slide 3 (the core idea, `fig_next_token.png`):** context
  `"The internal standard was spiked into each ___"` → P(next word)
  = sample 0.55, vial 0.20, tube 0.15, others 0.10; argmax → **"sample"**.
- **Slide 4 (the do-it-together, `fig_softmax_next.png`):** the same clause
  continued, three candidate next **words** "sample" / "vial" / "tube" with logits
  2 / 1 / 0. **The arithmetic is unchanged** — e²=7.39, e¹=2.72, e⁰=1.00,
  sum 11.11 → 0.665 / 0.245 / 0.090, argmax → "sample" — so figure, body text and
  `notes:` still agree exactly (rule 3), and Quiz 13 needed no change (it never
  referenced the token names).

The MS anchor is kept by the sentence itself (rule 2: it is lab-report prose, not
a generic corpus sentence), and slide 3's `notes:` now says explicitly that a
peptide extended one residue at a time is the same game in another vocabulary,
pointing forward to the landscape tour. **The landscape tour is untouched** —
Casanovo, Prosit, DIA-NN, DRIAMS and AlphaFold stay peptide/spectrum work, which
is the point of that block. Tokenization is a Lecture 7 callback, so the
word-piece framing is already on the deck before this slide.

## 2026-09-10 — Lecture 10 Part B: every treatment taught to depth

Instructor review: the four treatment moves were named and illustrated, but none
was explained *in depth* — curation asserted a replicate count without showing how
you find replicates, SMOTE stayed a 2-D toy, focal loss was two points with no
curve, and curriculum was three text cards. Four fixes, one per move.

1. **Curation → deduplication in embedding space (NEW slide 14).** The curation
   slide claims "−12 replicate spectra grouped"; this slide is the *mechanism*.
   A trained CNN's last hidden layer turns each spectrum into a vector (Lecture 7's
   embeddings, Lecture 5's feature maps), and repeat shots of one isolate land on
   top of each other. Cosine is the ruler, and it is a **do-it-together**, exact by
   hand: `cos(a,b) = (a·b)/(|a|·|b|)`; a = (3,4,0) vs b = (6,8,0) → 50/(5·10) =
   **1.00** (same peak pattern, twice the TIC — a second shot), a vs c = (0,4,3) →
   16/25 = **0.64** (keep both). Rule: cos ≥ 0.99 → keep one. Payoff: a duplicate
   split across train/test is leakage, and on a 41-member class it moves the score.
   Figure `fig_embed_dedup.png`.
2. **SMOTE → the same blend drawn as real spectra.** The feature-space panel is
   unchanged (A = (2,6), B = (4,10), λ = 0.5 → (3,8), so Quiz 10 Q4 is untouched),
   but bin 1 and bin 2 are now named as m/z 2,700 and m/z 5,300 and a second panel
   shows A, B and the blend AS SPECTRA. A carries a private peak of 7 at m/z 4,100,
   B one of 5 at m/z 6,800 — and the blend keeps both at **half height** (3.5 and
   2.5), hatched in red. The non-physical caveat stops being a warning and becomes
   something the room can see.
3. **Focal loss → the whole curve.** The left panel is now FL = (1−p)^γ·(−ln p)
   plotted for **γ = 0 (plain cross-entropy), 1, 2, 5**, with an "already easy"
   band over p > 0.8 and the two worked points marked on the γ = 0 and γ = 2
   curves. The exact numbers are unchanged (0.105 → 0.001; 0.693 → 0.173; 6.6× →
   165×), so figure, callout and notes still agree.
4. **Curriculum → an actual schedule, with curves.** The three text cards become a
   20-epoch **schedule**: the sampler anneals from 50% resistant per batch to the
   true 6% (reaching it at epoch 12, so the model finishes *calibrated* — the fix
   for the class-weights calibration warning), while hard/borderline spectra are
   admitted 0% → 100% between epochs 5 and 14, over three named stage bands. A
   second panel shows curriculum vs shuffled training, **labelled ILLUSTRATIVE on
   the figure itself** — it is a shape, not measured data — with the footer that
   both runs see the same 41 resistant spectra.

**Assessment (rule 3).** Deduplication is new taught content, so Quiz 10 **Q3(a)**
now reads "Junk runs and mislabeled spectra contaminate the training set — and the
same isolate appears three times," and its key names the cos ≥ 0.99 embedding pass.
The other three enrichments deepen mechanics already assessed (Q3 c/d, Q4); the
dedup slide is itself a do-it-together, which rule 8 accepts without a quiz item.

**Timing.** Part B 22 → 25 min for the extra slide and the two curve-reading beats;
paid for one minute each by the Part A opener (12 → 11), the validation slide
(5 → 4) and the closing debrief (5 → 4). Total still 60. Deck 20 → 21 slides.
If running long, the two curve reads compress first — the numbers beside them
carry the point alone.

## 2026-09-11 — Lecture 13: reinforcement learning, properly (was "touched base")

Instructor decision, overriding the original design (2026-08-30) that RL in
Lecture 13 is "context and vocabulary, not the destination": the RL block goes
from 3 slides + a you-do (12 min) to **8 slides (20 min)**, teaching the core
mechanics. The constraints are unchanged — **no Markov decision processes, no
policy-gradient math**; the only arithmetic added is one division and one square.

**The four new slides**, in block order:
- **2 OF 8 · THE ANATOMY** — the two words the loop omits. STATE = where you are;
  POLICY = the rule mapping state → action, *and the network you are actually
  training*. Walked on one collision-energy episode: 20 eV +5 → 7 confident ions;
  25 eV +5 → 5; 30 eV −5 → 7. Punchline: nobody ever said "25 eV is correct" —
  there is no label, only a reward, which is what separates RL from Lectures 1–11.
  Deliberately **not** the LC-gradient scenario, which belongs to the Q2 you-do.
  Figure `fig_rl_anatomy.png`.
- **4 OF 8 · DELAYED REWARD · DO-IT-TOGETHER** — placed right after AlphaGo, whose
  win lands 200 moves after the move that earned it. The credit-assignment problem
  on three method-development cycles paying 0, 0, 1, then the fix: the discounted
  return `G = r₁ + γ·r₂ + γ²·r₃`. At γ = 0.9, cycle 1 earns 0.9² = **0.81**; cycle
  2, 0.90; cycle 3, 1.00. γ shown as a dial (0 → 0.00, 0.9 → 0.81, 0.99 → 0.98);
  **γ = 0.5 is deliberately absent — it is the Q6 answer.** Figure
  `fig_return_discount.png`.
- **5 OF 8 · EXPLORE vs EXPLOIT · I-DO** — the choice made every cycle. Two
  collision energies: A (25 eV) 20 tries / 140 ions → mean **7.0**; B (35 eV) ONE
  try / 5 ions → mean **5.0**. ε-greedy stated on the slide (rule 3a): with
  probability ε pick uniformly among K actions, else take the best average, so
  `P(a specific non-greedy action) = ε/K`. With ε = 0.1, K = 2 → **0.05**, one
  cycle in 20; over 200 cycles B is tried ~10 times and its true mean turns out to
  be 8.0, which pure greed would never have found. Figure `fig_explore_ido.png`.
- **6 OF 8 · YOUR TURN (YOU-DO = Q5)** — same rule, MALDI laser power: A (45%) 30
  tries / 180 → 6.0, B (60%) 2 tries / 8 → 4.0, ε = 0.2 → P(B) = **0.10**.
  Figure `fig_explore_youdo.png`.

Existing eyebrows renumbered 1–8 OF 8; the Part Two divider, the agenda bullet,
the "what you can do now" recap and the Lecture 14 bridge all updated.

**Assessment (rule 3).** Quiz 13 grows from four items to **six**: **Q5** ε-greedy
(greedy pick, P(B) = 0.2/2 = 0.10, and one sentence on why always-greedy fails)
and **Q6** the return at γ = 0.5 → 0.25 plus what a small γ does. Both answers are
exact and reproducible from rules printed on the deck before they are asked.

**Timing — the hour deliberately runs long.** Instructor's call: rather than cut
another stop, Lecture 13's taught content now runs **~8 minutes past 60**, and the
instructor decides live what to drop. Every new slide's `notes:` opens with
**OPTIONAL DEPTH** and the cut order is stated in three places (slide 1, slide 2,
the divider): drop **explore/exploit + its you-do first** (~6 min, costs Q5), then
**delayed reward** (~3 min, costs Q6), and **keep the anatomy slide** (~2 min) —
the rest of the block leans on STATE and POLICY. Explicit instruction in the
notes: do **not** buy the time out of the landscape tour, which is the payoff of
the whole course. The quiz sheet itself says "Q5–Q6 go with the reinforcement-
learning block; if we skip it, leave them blank."

## 2026-09-11 (later) — Lecture 13: classic RL examples, the Markov requirement, cited tour papers, Quiz 13 reshaped

Four instructor decisions, all applied together.

**1 · RL is taught on the CLASSIC examples, not mass-spec ones.** The
collision-energy tuning story introduced earlier the same day is gone. The
mechanics now run on the canonical textbook examples, which keeps them from
being entangled with chemistry:
- **Gridworld** (3×3, sparse reward): agent starts middle-left, goal top-right,
  reward 0 every step and **+1** on arrival. The three-move path ↑ → → pays 0, 0, 1
  and is reused by the discounted-return slide, so one world carries two slides.
- **Two-armed bandit** for explore vs exploit (the origin of the terms): A played
  20 times, mean **7.0**; B played once, mean **5.0**; ε = 0.1, K = 2 →
  P(B) = **0.05**; B's true mean is 8.0. The you-do is **two routes to work**
  (A: 30 drives, mean 6.0; B: 2 drives, mean 4.0; ε = 0.2 → **0.10**).
- **An aliased corridor** for the non-Markov counter-example (see the correction below).
The MS anchor moves to where it belongs: the block's closing you-do, where the
room formulates a decision from their **own** lab as RL.

**2 · NEW slide — the Markov requirement (RL · 3 OF 9).** A state is Markov when
the future depends only on it, not on the path that reached it. Gridworld as the
good case (two routes into the same cell → identical futures); **perceptual
aliasing** as the broken one. The transferable rule: **fix the STATE, not the
algorithm**. Assessed in Quiz 13 Q1(f).
RL block is now **9 slides / ~24 min**; the hour runs ~10 min long by design with
the optional-depth cut order kept in the notes.

**3 · Every landscape-tour stop now carries its paper.** Full citation with DOI
on the slide and in the notes, plus that paper's **headline experimental result**
as the slide figure:

| Tool | Paper | Result on the slide |
|---|---|---|
| Casanovo | Yilmaz et al., *Nat Commun* **15**:6427 (2024), doi:10.1038/s41467-024-49731-x, **CC BY** | avg peptide-level precision on the nine-species benchmark: **0.81** vs PointNovo 0.74, DeepNovo 0.70, Novor 0.58; 30 M training spectra; 0.81 → 0.95 on MassIVE-KB |
| Prosit | Gessulat et al., *Nat Methods* **16**:509–518 (2019), doi:10.1038/s41592-019-0426-7 | 550,000 peptides / 21 M spectra → more IDs at **>10× lower FDR** |
| DIA-NN | Demichev et al., *Nat Methods* **17**:41–44 (2020), doi:10.1038/s41592-019-0638-x | the Fig. 1b experiment named, and **>35,000 precursors** from a 0.5 h gradient |
| DRIAMS | Weis et al., *Nat Med* **28**:164–174 (2022), doi:10.1038/s41591-021-01619-9 | AUROC **0.80 / 0.74 / 0.74**; >300 k spectra, >750 k phenotypes, 4 sites; 63 patients → 9 treatments changed, **8 beneficial (89%)** |
| AlphaFold | Jumper et al., *Nature* **596**:583–589 (2021), doi:10.1038/s41586-021-03819-2, **CC BY** | CASP14 median backbone **0.96 Å** r.m.s.d.₉₅ (95% CI 0.85–1.16) vs **2.8 Å** next best |

**Why the figures are redrawn rather than lifted from the papers.** Licences were
checked against Europe PMC on 2026-09-11: only Casanovo and AlphaFold are CC BY;
Prosit and DRIAMS are subscription, and the DIA-NN deposit is free-to-read with
no reuse licence — so three of the five could not lawfully be redistributed in a
course deck at all. Rather than mix two journal figures with three substitutes,
all five panels are **drawn by the course from numbers each paper states** (data
are not copyrightable), each with a citation strip naming the source and its
licence status. This also keeps them legible on a projector, which a 6-pt
multi-panel journal figure is not. The AlphaFold slide's figcaption was updated
accordingly; its CC0 structure render was displaced by the CASP14 chart. Nothing
on a panel is invented — where a paper gives no plottable series (DIA-NN), the
panel names what was measured and quotes the one number the text states.

**4 · Quiz 13 reshaped: six items → five.**
- **Removed** the old Q1 (place-a-tool-you've-never-seen transfer). Its slide
  survives as a spoken **do-it-together** at the end of the tour — the reasoning
  beat is too valuable to lose, it simply is no longer written on the sheet.
- **Old Q2 → new Q1, now OPEN-ENDED**: formulate a repeated decision from your
  own lab as RL — agent, actions, state, reward — and say whether the state is
  Markov. The key marks reasoning, not a single answer, and names the three
  failure modes to listen for: it is really supervised learning; the reward is
  gameable (reward hacking); the state is not Markov.
- Remaining items renumbered Q3→2, Q4→3, Q5→4, Q6→5, and every slide callout
  that points at a quiz item was updated to match.
- Q4 and Q5 restated on the classic examples (two routes to work; the gridworld
  episode). Answers unchanged and still exact: **0.10** and **0.25**.

**Repair note.** The "WHAT YOU CAN DO NOW" recap slide was lost during the
slide-insert step and has been rebuilt from the deck's own closing-slide layout
with refreshed content; deck is 31 slides. Overlap, text-overflow and
figure-overflow checkers all pass.


## 2026-09-12 — Lecture 13 corrections (instructor review)

**1 · The non-Markov example was wrong and has been replaced.** The first
version of the Markov slide used a **cart-pole snapshot** as the failure case.
That is a bad example: cart-pole's state (position, velocity, angle, angular
velocity) is *perfectly Markov* — the environment is deterministic physics — so
presenting cart-pole as "not Markov" muddles a property of the **observation**
with a property of the **world**. Replaced with the textbook failure,
**perceptual aliasing**: a five-cell corridor with the goal in the middle, where
an agent that senses only "wall left? wall right?" reads **cell 2 and cell 4
identically as `open | open`** yet must go right from one and left from the
other. No policy can be correct in both. The underlying cell *is* Markov; the
impoverished observation is not — which makes the real teaching point explicit
and is now the line along the bottom of the figure: **Markov is a property of
the state you choose, not of the world.** The slide's speaker notes carry a
standing instruction not to swap cart-pole back in, with the reason. Frame
stacking (Atari DQN) is mentioned in the notes as the same fix, correctly framed
as repairing a partial observation.

**2 · Stale embedded figures — a process failure, now guarded.** Slides 11, 14,
15 and 16 were still displaying the *old* collision-energy / MALDI-laser-power
renders: the PNGs under `slides/assets/img/` had been regenerated, but
regenerating a PNG does **not** update the copy embedded in the `.pptx`, and the
pictures were never re-inserted. Caught by the instructor. All four re-embedded,
plus `fig_use_llm_ido.png`. **Standing rule for scripted deck edits: after
`make_slide_figures.py`, the affected pictures must be re-inserted with
python-pptx — the text dump will not reveal this, because the dump only records
`[image: Picture 4]`.** An md5 audit comparing every embedded image against the
file on disk now exists as the check to run; it found and confirmed zero
remaining mismatches across all 31 slides.

Two further defects surfaced while fixing the above: the Markov slide's callout
had been created with only its `MASS SPEC ·` tag run and no body text (the body
was silently dropped when a two-run template was written into a one-run
placeholder), and the tag itself was wrong for a gridworld/corridor slide. Both
fixed; a sweep confirmed no other callout or notes field on the deck is
truncated.

**3 · Prompting / RAG / fine-tuning now carries one concrete example each**
(instructor request). The comparison chart gains a fourth column:
- **Prompting** — "Rewrite this method section in plain English for patients."
  Nothing to build; the task needs no private data.
- **RAG** — "What is *today's* acceptance criterion for the cortisol assay?"
  The retriever pulls the current SOP chunk and the answer cites it, so the
  answer tracks the SOP without retraining.
- **Fine-tuning** — "Turn this instrument error log into a plain-language
  cause", learned from 4,000 past logs: a new *skill*, baked into the weights.

The fine-tuning example is deliberately **not** the house-report-style task —
that is the Quiz 13 Q2 answer and must stay unseen, so the you-do still forces a
real discrimination rather than a pattern-match. Footer states the decision
rule: try prompting first, reach for RAG when the answer must be current or
private, fine-tune last.


## 2026-09-14 — Lecture 13: RLHF gets two more slides

Instructor request: 1–2 slides on RLHF immediately after the RL slides. Added
**two**, placed directly after the existing `RL · BACK TO LLMS` mapping slide and
before the block's closing you-do, so the arc reads *RL mechanics → RLHF (map →
machinery → honest limits) → formulate your own*. RL block renumbered **N OF 11**
(was N OF 9); deck is 33 slides.

**`RL · 9 OF 11 · RLHF, OPENED UP` — how it is actually built.** The previous
slide leaves an obvious objection hanging — surely nobody grades every answer by
hand? This one answers it in three steps, all on one clinical prompt (*"QC failed
on this run — report the cortisol result?"*):
1. **Collect preferences** — two sampled answers, a human picks the better one.
   The teaching point is what the rater does *not* do: no ideal answer written,
   no score out of ten. Comparisons are easier and far more consistent than
   absolute scores, which is why RLHF is built on them.
2. **Train a reward model** — a second network maps (prompt, answer) → one
   number, trained on those pairs, so it learns only an **ordering**
   (`score(A) > score(B)`), never an absolute "correct" score. This is what makes
   the reward **automatic** — no human in the loop per answer.
3. **Tune the policy** — the LLM *is* the policy (deliberate callback to the
   anatomy slide's vocabulary): write → score → nudge, repeated. Plus **the
   leash**: penalise drifting from the starting model, or the policy finds a
   quirk of the reward model and rides it until it stops writing English.

**`RL · 10 OF 11 · RLHF, HONESTLY` — what it buys and what it breaks.** Left, the
before/after that makes stages 2–3 concrete: asked for a cortisol reference
interval, a **base** model merely *continues the text* (it writes more questions
of the same shape, because continuing text is all it was ever trained to do),
while the tuned model answers, hedges on collection time and assay, and defers to
the local range. Right, three failure modes that follow directly from the
mechanism: **reward hacking** (it optimises what scores well, not what is true —
hence long, confident, agreeable answers), **sycophancy** (it folds when told it
is wrong, which is dangerous when a confident colleague misremembers a cutoff),
and **it does not fix hallucination** (raters reward answers that *look* good, so
a fluent fabrication can out-score an honest "I don't know" — RLHF can make
hallucination *more* confident). Closes by restating the unchanged guardrails:
ground in your documents, require sign-off. Carries a 💬 pair prompt — where
would a sycophantic assistant be dangerous in your workflow?

**Assessment (rule 3).** No new quiz item. Reward hacking is already the second
pitfall in the **Q1** key (the open-ended RL formulation), and "preference tuning
does not fix hallucination" is exactly **Q3**; both slides now name those items
explicitly so the linkage is live rather than implied. The 💬 prompt is the
interactive beat.

**Timing — revised cut order.** The RL stop is now ~30 min and the hour runs
**~16 minutes long by design**. The cut order in the notes (slide 1, the agenda,
the divider) is now: **first** explore/exploit + its you-do (~6 min, costs Q4),
**second** delayed reward (~3 min, costs Q5), **third** `RLHF, opened up` (~3
min). Keep `RLHF, honestly` even when cutting the other RLHF slide — reward
hacking, sycophancy and the hallucination point are what this audience actually
has to know, and they close the loop on the hallucination guardrail from the
first stop. Always keep the anatomy and Markov slides. Still do not buy time out
of the landscape tour.


## 2026-09-14 — Lecture 14 demo: the agent is told what data exists

Bug from a live run: `stage3_react.py` died with
`FileNotFoundError: .../demos/lecture14_agent/QC-04`.

**Root cause (instructor's diagnosis, and the right one): the agent was never
told where the data was.** The ReAct system prompt listed the three tools but
never named a single file, so when asked about run QC-04 the model had nothing to
put in `Action Input` except the run id, and `load_csv` tried to open a file
called `QC-04`. The model was not misbehaving — the context was incomplete.

**Fix:** `REACT_SYSTEM` now carries a **data inventory** ahead of the tool list —
every CSV next to the scripts, with its row count and columns — generated by
`data_inventory()` reading the directory, so it cannot drift from what is on
disk. The `load_csv` description names its default. Deliberately *not* included:
the spec limits or any values, which would let the model reason without calling
`check_limits` and would undercut the whole point of stage 3.

This is now a teaching beat rather than just a fix — "an agent only knows what
its context says exists; one that has to guess will guess" is exactly the lesson
Lecture 14 wants, and the README says so.

**Defence in depth** behind it, so a live demo cannot die mid-sentence: a
non-existent `load_csv` argument falls back to `qc_runs.csv` and says so in the
observation; `run_tool` returns `ERROR: ...` for any exception or unknown tool
name instead of raising (the model then reads the error and retries, which is
worth narrating); and the ReAct loop caps at `MAX_STEPS = 6`.

**Also:** new `ui.py` gives the demo projector-grade contrast — reverse-video
chips and coloured gutters for YOU / BOT / OBS / THINK / ACT, `Action` +
`Action Input` merged into one `tool(arg)` line, the final `Thought:`/`Answer:`
protocol stripped so the answer reads as an answer. Colour auto-disables off-tty
and honours `NO_COLOR` / `FORCE_COLOR`.


## 2026-09-14 — Lecture 14: the live demo becomes a three-stage walkthrough

Instructor request: turn the second half of `lecture14_agents.pptx` into a
step-by-step walk-through of the three-stage demo under `demos/lecture14_agent/`.

**Why the change earns its slides.** The demo stop was budgeted 18 minutes but
had only two slides — a five-beat storyboard of one finished assistant, and a
flag→draft panel. That framing hides the actual lesson, which lives in the
**diff between the three scripts**: each adds exactly one idea, and each fails in
a new way until the last. Replaced with six slides (deck 17 → 21):

| slide | what the room sees |
|---|---|
| 1 OF 6 · the plan | the three scripts against the same three questions — Q1 a concept, Q2 memory, Q3 the data |
| 2 OF 6 · stage 1 | `messages` rebuilt every turn → cannot name the run it was given one turn ago |
| 3 OF 6 · stage 2 | the one-line diff that *is* memory; Q2 works; then a 💬 predict beat before Q3 |
| 4 OF 6 · stage 3 | the ReAct loop executing: THINK → ACT → OBS → answer, grounded |
| 5 OF 6 · the payoff | `draft_summary` names QC-04, its three out-of-range values, and defers to a human |
| 6 OF 6 · what made it work | the script owns the tools; the agent only knows what its context says |

**The pedagogical spine is that the failures get WORSE before they get better.**
Stage 1's failure announces itself — the assistant says it does not know. Stage
2's does not: it is fluent, uses the right units and the right limit, and is
fabricated, because the model has never seen `qc_runs.csv`. It passes QC-04, the
one run in the table that breaks all three limits. "Memory is not knowledge" is
the sentence the block is built to earn, and it lands straight back onto Lecture
13's hallucination guardrails.

**Stage 2 carries the interactive beat** (rule 8): a 💬 *predict* prompt before
Q3 is run — what will it say QC-04's mass error is? Someone always says it cannot
know; then they watch it answer anyway.

**Two transferable rules close the block**, and they are what turn the demo into
motivation for the guardrails section that follows rather than a trick:
1. *The script owns the tools.* The model only ever emits the string
   `Action: check_limits`; ordinary Python decides whether to honour it. So
   "restricted tools" is enforceable — the agent cannot do anything you did not
   write a function for.
2. *An agent only knows what its context says exists.* Told as the true story of
   this demo's own `FileNotFoundError` on `QC-04` (logged 2026-09-14): the prompt
   listed tools but never named a file, so the model had nothing to pass but the
   run id. An agent that has to guess will guess.

**Figures.** Five new panels (`fig_demo_plan`, `fig_demo_stage1/2/3`,
`fig_demo_why`); `fig_demo_flag` is reused for the payoff and `fig_demo_storyboard`
is retired. The transcript panels deliberately reuse the demo's own terminal
chips — YOU / BOT / OBS / THINK / ACT in the same colours `ui.py` prints — so the
slide and the projector show the same thing.

**Checks:** overlap 0 fail across all 10 decks; text overflow 0 for lecture 14;
figure overflow PASS across 169 renders; md5 audit confirms every embedded image
matches its source file.


## 2026-09-16 — Lecture 10: data augmentation gets its own slide, after curation

Instructor request. Augmentation had been three words on the class-weights slide
("oversampling repeats those 41, augmentation makes new VIEWS of them"). It is
now `PART B · 2 · AUGMENTATION · DO-IT-TOGETHER`, placed directly after the two
curation slides; the rest of Part B renumbers to 3–6.

**Why there.** Curation and augmentation are the two moves that work on the DATA
rather than on the training, so they belong together. More importantly the slide
establishes the **validity rule** — a transform is legitimate only if (1) the
instrument could have produced it and (2) it does not change the label — which is
exactly what SMOTE breaks two slides later. Teaching the rule first makes the
SMOTE caveat land as a violation of something the room already owns, rather than
as an isolated warning.

**The figure is a real microscopy image, not a spectrum** (instructor,
2026-09-17 — the first version used a stem spectrum and it did not read). One
Wright-stained blood smear with a band neutrophil, then five transforms — flip,
rotate 35°, zoom/crop, dim + noise — each still unmistakably the same cell, plus
a sixth that crops so hard the neutrophil leaves the frame and the label is
destroyed. That counter-example comes from the same file, so the rule is
demonstrated rather than asserted.

Two reasons a picture wins here. Augmentation is a visual idea and a ±0.1% m/z
shift is invisible from the back of a room. And a cell has no canonical
orientation, so rotating and flipping a smear genuinely preserves its label —
the example is honest, unlike rotating a chest film. Rule 2 is kept by making the
picture the *bridge*: the transfer to spectra (m/z jitter, intensity, baseline,
"never a left–right flip") is stated on the slide and the arithmetic underneath
stays on the real DRIAMS split.

Source image: Bobjgalindo, "Band neutrophil", Wikimedia Commons, **CC BY-SA
4.0**, credited on the figure; stored at `slides/assets/img/img_blood_smear.jpg`
so the deck renders offline, with the six views generated by the course at build
time.

**The do-it-together arithmetic is exact:** 41 resistant spectra × 8 views = 328,
so the batch imbalance moves from 697:41 ≈ 17:1 to 697:328 ≈ 2.1:1 — the largest
single change to effective imbalance anywhere in Part B, at no data cost. Two
limits stated on the slide: augment **after** the split (views of one isolate
scattered across train and test is precisely the leakage the embedding-dedup pass
removes — a direct callback), and it is still 41 isolates, so the confidence
interval is set by the 41 and not the 328.

**Corrected a false claim while here.** The class-weights notes told the
instructor that augmentation "is exactly Lab 2's move". It is not — `grep` over
`labs/solutions/` finds no augmentation in Lab 2 at all. It is **Lab 5 Track A,
Blank A2** (`augment`, an m/z-jitter function whose asserts check it keeps the
shape and changes the values). Fixed, and that lab checkpoint is the new slide's
rule-3 assessment, so no Quiz 10 item was added — consistent with the earlier
decision to leave Quiz 10's coverage alone.

Agenda updated (Part B 25 → 29 min); the reweighting slide drops its augmentation
sentence and back-references the new slide instead of re-explaining it.

**Checks:** overlap 0 fail across all 10 decks; text overflow 0 for lecture 10;
figure overflow PASS across 170 renders; md5 audit confirms every embedded image
matches its source.


## 2026-09-22 — Lab 1 ships a Colab setup sheet with real, regenerable screenshots

Instructor request: step-by-step Colab setup guidance for Lab 1, with
screenshots. New printable handout `labs/handouts/lab01_colab_setup.tex` (2
pages), pointed at from the notebook's own "Running this notebook in Colab" cell.

**Why a paper sheet and not notebook cells.** Setup instructions that live inside
the notebook are useless to someone who cannot yet open the notebook. The repo
convention is already paper-first for this audience, so this follows the hint
sheets.

**The screenshots are real and they are generated, not pasted.** Colab's whole
interface renders **signed out**, so `tools/make_colab_shots.py` drives headless
Chrome over the DevTools protocol against the actual course notebook
(`colab.research.google.com/github/indigobio/msacl_dl_course/...`), clicks the
File and Runtime menus so they are captured open, crops five views and draws the
numbered callouts. No account, no credentials, nothing typed into a sign-in form.
Re-runnable whenever Google restyles Colab — which is the real hazard for a
screenshot-based handout — and verified to reproduce the committed figures
byte-for-placement from a clean state. Each Chrome instance gets its own port and
profile; sharing either makes the second capture attach to the first, dying
browser.

**What the sheet covers**, each with its own cropped strip rather than one busy
screenshot: open the notebook (check the filename, Sign in, Connect); **save your
own copy first** (the link is read-only — the commonest way a participant loses
an hour's work); run a cell (Shift+Enter, in order, top to bottom); the Runtime
menu as the fix for almost everything (Run all / Restart session and run all /
Change runtime type, noting Lab 1 needs no GPU); and the fallback route if the
link fails. Plus a troubleshooting table keyed on what the participant actually
sees — including that some institutional Google accounts block Colab, which is
the failure most likely to strand someone in the first five minutes.

Figures are credited on the sheet as the Google Colab interface with the capture
date, and the sheet notes that if a button has moved the menu path in the text
still holds. `tools/build.sh handouts` globs `*.tex`, so it picked the new sheet
up with no change.

## 2026-09-23 — Labs bootstrap themselves on Colab: pinned env + hosted, fingerprinted data

**Asked for:** students "fork the entire student pack" into Colab — code *and*
offline data — so nobody downloads or copies individual labs; a solution that
scales to later labs with much larger data; and the uv environment carried into
Colab.

**Why not a literal fork:** a Colab link opens one notebook on an empty,
ephemeral machine; only Google Drive persists. There is no Colab object that
holds a folder. So each lab's **first cell rebuilds what the folder would have
provided**, and the notebook itself stays tiny.

**Design:**
- `student_pack/msacl.py` (standard library only — it runs before anything is
  installed). The setup cell downloads it from `main` and calls
  `msacl.setup("labNN")`, returning `DATA = {file name: local path}`. Notebooks
  read data only as `DATA["…"]`; no URLs or paths in lab code. Run locally, the
  same cell uses `data/slices/` and installs nothing.
- **Environment:** `requirements-colab.txt`, generated from `uv.lock` by
  `tools/build_colab_bootstrap.py` — the 7 top-level deps pinned with the lock's
  environment markers, torch excluded so Colab keeps its GPU build, installed with
  `uv pip install --system`. Chosen over pinning the full lock (would churn
  dozens of Colab system packages) and over unpinned installs (non-reproducible).
  Skipped when already satisfied; if it upgrades a module the kernel had already
  imported (Colab preloads pandas) it stops with "Restart session, then run this
  cell again", because a half-upgraded kernel fails in baffling ways later.
- **Data:** hosted on the Hugging Face dataset repo `jaztsong88/msacl-ds301` (the
  instructor's account; the GitHub code stays under `indigobio`) —
  chosen over GitHub release assets (2 GB/file cap, awkward for growing data) and
  Drive links (quota/virus-scan interstitials break scripted downloads). HF serves
  large files over a CDN with range requests, needs no login to read, and
  versions the data. `datasets.json` records per file: URLs, bytes, SHA-256,
  licence, source, labs, `optional`. Downloads resume; a fingerprint mismatch
  deletes the file and fails loudly. `mount_drive=True` caches in
  `MyDrive/msacl_ds301_data/` so big data downloads once per course.
- **Licence gate:** only `publish=True` files are staged for upload
  (`build/hf_upload/`, git-ignored). DRIAMS slices are CC0 → published.
  `MTBLS90.xlsx` stays fetched from the CIMCB repo (no LICENSE file).
  `peakonly_roi_qc.npz` is held (annotations unlicensed) and marked optional, so
  Lab 5 Tracks A and C run without it.
- Fixed on the way: Labs 2 and 5 still carried a `<ORG>/<REPO>` placeholder and
  read git-ignored slices, so they could not have run on Colab at all.

**Verified:** `msacl.py` unit checks (download, resume, cache, corrupted cache,
fingerprint rejection, optional dataset, install skip/restart paths); all five
solution notebooks executed end to end in the locked environment (pandas 3.0.5,
numpy 2.5.2). **Still needs a live Colab rehearsal** after the HF upload and the
merge to `main`.
