# Course Design Decisions Log

Newest entries at the top. Every entry: date, decision, rationale.

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
