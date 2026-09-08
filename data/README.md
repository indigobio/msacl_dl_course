# Dataset Registry

Every dataset used in a lab or demo must be registered here with source,
license, size, and its Colab-slicing script in `data/prep/`. Raw data is never
committed to the repo — labs download prepared slices at runtime.

## Status

All entries below are **candidates identified during course design** — each
must be verified (availability, license, download size, no-credential access)
in Phase 2 before a lab is built on it.

| Candidate | Type | Intended lab | Notes to verify |
|---|---|---|---|
| DRIAMS (MALDI-TOF spectra + antimicrobial resistance labels) | 1D spectra, classification | Lab 6 (1D CNN) | Hosted on Dryad; size of a usable slice; per-site licensing |
| Chromatographic peak-quality data (e.g., PeakOnly's labeled ROIs) | 1D signals, classification | Lab 6 alternative / OD-for-peak-picking demo | GitHub availability; label format |
| Clinical/metabolomics tabular panel (e.g., MetaboLights study or UCI clinical set) | Tabular, classification | Lab 3 (MLP) | Pick one with clean labels, n in the thousands |
| ProteomeTools / Prosit training slices (peptide → spectrum/RT) | Sequences | Lab 9 (transformer fine-tune) or demo | Slice size; may be demo-only |
| Small MS-imaging or clinical image set | 2D images | Lecture 5 CNN demo, OD segment | Only if the OD-for-peak-picking demo needs 2D framing |
| Spectra pool for QC/anomaly detection (subset of one of the above) | 1D spectra | Lab 12 (autoencoder) | Can likely reuse the Lab 6 dataset |

## Confirmed datasets

### MTBLS90 — serum LC-MS metabolomics panel (Lab 1)
- Source: CIMCB benchmark copy (tidy tabular form of MetaboLights MTBLS90):
  `https://raw.githubusercontent.com/CIMCB/MetabComparisonBinaryML/master/notebooks/data/MTBLS90.xlsx`
  (2.4 MB, no login; `pd.read_excel(url, sheet_name='Data')` works on Colab).
  Original study: https://www.ebi.ac.uk/metabolights/MTBLS90 ; curation paper:
  Mendez, Reinke & Broadhurst 2019, *Metabolomics*.
- License: source data open access at EMBL-EBI/MetaboLights; the GitHub repo has
  no LICENSE file — cite Mendez et al. + MTBLS90 in course materials.
- Size: 968 samples × 189 **named metabolite features** (choline, creatinine,
  acylcarnitines, amino acids…); zero missing values.
- Task: binary sex classification from the serum metabolome — 485 vs 483,
  essentially perfectly balanced, so accuracy reads honestly against a 50%
  baseline; robustly learnable per the benchmark paper.
- Why: for an MSACL audience this IS their instrument's output — the ideal
  "first MLP" dataset. No prep script needed (loaded directly in the notebook).
- Used in: lab01 (MLP).
- Runner-up kept in reserve: **HCV liver panel** (UCI id 571, CC BY 4.0,
  615×12, real ALB/ALT/AST/GGT units, 87% majority class, some NaNs) —
  deliberately imperfect, so it makes a good Lecture 10 example for class imbalance /
  missing data, or a quiz scenario.

### DRIAMS — MALDI-TOF spectra + antimicrobial resistance labels (Labs 2, 4, 5)
- Source: Zenodo mirror (preferred — frozen Nov 2021 release that all published
  loader code expects): https://zenodo.org/records/5640517 ; original Dryad
  record doi:10.5061/dryad.bzkh1899q (Dryad's Aug 2025 update renamed DRIAMS-C
  files and breaks older loaders — use Zenodo).
- License: **CC0 1.0** (public domain) — we may rehost prepared slices freely.
- Raw size: DRIAMS-A (Univ. Hospital Basel) is 86 GB; total 145 GB across 4 sites,
  303k spectra. Archives already contain `binned_6000/` — preprocessed spectra as
  fixed 6000-dim vectors (3 Da bins, 2,000–20,000 Da) — so no signal processing
  is needed for the course.
- Course tasks (DRIAMS-A, the literature benchmark site):
  - **S. aureus + oxacillin (MRSA): 3,064 spectra, ~24% resistant** — primary
    Lab 2 task; the MRSA story lands with a clinical audience.
  - **E. coli + ceftriaxone: 3,875 spectra, ~28% resistant** — bundled second
    task (Lab 5 Track C: re-head the model for a new label).
  - DRIAMS-B/C/D slices usable later as a *distribution shift* demonstration
    (same task, different site — ties into Lecture 10).
- Prep script: `data/prep/prepare_driams.py` (one-time instructor run; stream-
  extracts only `binned_6000/` + `id/` from the tarball to avoid ~200 GB of disk).
- **DRIAMS-C development slices built and verified (2026-08-25)** from the
  12.4 GB site-C archive (735 MB extracted), in `data/slices/`:
  `driams_c_ecoli_ceftriaxone.npz` (913 spectra, 16.2% R, 9.4 MB),
  `driams_c_saureus_oxacillin.npz` (738, 5.6% R — too imbalanced for the
  primary lab task; fine for development), `driams_c_kpneumoniae_ceftriaxone.npz`
  (366, 15.0% R). Sanity-checked: X (n, 6000) float16, TIC-normalized
  intensities, labels correct. Use **E. coli + ceftriaxone** as the dev task;
  build the final DRIAMS-A slices (bigger, better balanced) before Phase 4 ends.
- Prepared slice: `.npz` per task (~15–45 MB at float16) — rehost as a GitHub
  release asset for fast Colab download (CC0 allows it).
- Reference loaders: BorgwardtLab/maldi-learn (id-CSV parsing, S/I/R cleaning),
  BorgwardtLab/maldi_amr (paper's task definitions), gdewael/maldi-nn.
- Used in: lab02 (1D CNN), lab05 tracks A/C; Lecture 10 shift example.
  (Lab 4 no longer uses DRIAMS — it moved to FashionMNIST/MNIST via torchvision;
  see the torchvision entry below.)
- **Lab 5 capstone reuse (Tracks A & C), built & run-verified (2026-08-31):**
  `lab05_capstone.ipynb` reuses both local DRIAMS-C slices. **Track A (beat the baseline)**
  trains the Lab 2 `BaselineCNN` on `driams_c_saureus_oxacillin.npz` (738 spectra, 697 S / 41 R)
  and compares it against a deeper/wider `ImprovedCNN` + m/z-jitter augmentation on the *same*
  stratified split. **Track C (transfer learning)** trains the S. aureus CNN body, **freezes**
  it, and re-heads it for `driams_c_ecoli_ceftriaxone.npz` (913 spectra, 765 S / 148 R) — a new
  organism/drug, same 6000-dim TIC-normalized format. Both tracks report sensitivity /
  specificity / AUROC + confusion matrix via the shared `honest_eval` helper (Lecture 10), never
  bare accuracy. All asserts are structural (logits `(N,1)`; leakage-free disjoint split;
  improved-vs-baseline param counts differ; frozen-body param count == body params, trainable ==
  head; metrics in `[0,1]`) so a shrunk CPU smoke run passes.
### FashionMNIST — via torchvision (Lab 4, redesigned 2026-09-01)
- Source: `torchvision.datasets.FashionMNIST` (`download=True`), fetched from the
  torchvision mirrors at runtime — **public, no credentials/login**. 28×28
  grayscale, values in `[0, 1]` after dividing by 255 (`transforms.ToTensor()`).
- License: FashionMNIST is MIT (Zalando Research); the torchvision loaders are
  BSD-licensed.
- Size: ~30 MB, cached under a **gitignored** `torchvision_data/` dir (see
  `.gitignore`); nothing is committed.
- Role: **FashionMNIST** is the clothing family the VAE learns; **MNIST digits**
  are also loaded as unseen anomaly "impostors" for the reconstruction-error
  detector in Step 8. No DRIAMS slice and no hosted file are involved.
- **Lab 4 rebuild (Fashion-VAE latent playground + impostor detector), built &
  run-verified (2026-09-01):** `lab04_vae_fashion.ipynb` trains a small
  2-D-latent MLP VAE (784→256→64→(mu,logvar); dec 2→64→256→784 sigmoid) on
  12,000 FashionMNIST images (Adam 1e-3, batch 128, ~12 epochs) and scores
  held-out MNIST digits as impostors. The model, training loop, and every plot
  are read-and-run; participants fill **four blanks**: **Blank 1** the VAE loss
  `loss = recon_loss + beta * kl_loss` (Step 2); **Blank 2** one KL value by hand
  `kl_by_hand = 0.5` for mu=[1,0], logvar=[0,0] (Step 3); **Blank 3** the impostor
  `THRESHOLD = np.percentile(normal_errors, 90)` plus a prediction
  `expected_false_alarms = round(0.10 * len(normal_errors))` (Step 8); **Blank 4**
  the creative ASCII `DOODLE` impostor (Step 9). It plots the VAE latent vs. a
  plain-AE latent (Lecture 11 picture), morphs a sneaker→ankle-boot, generates
  clothes from random z ~ N(0, I), and flags impostor digits by reconstruction
  error (AUROC > 0.85). Verified end-to-end (`jupyter nbconvert --execute`): all
  code cells ran with **0 errors** and all **16 asserts** pass (incl. cached
  downloads).

### Lab 3 — transfer learning by fine-tuning: pretrained diffusion generator (redesigned 2026-09-01)
- Model (not a dataset): **`balakrish181/ddpm-class-mnist-28`**, a real
  **online-pretrained** ~4M-parameter 28×28 MNIST digit **diffusion UNet**,
  fetched from the **Hugging Face Hub** via `diffusers`
  `UNet2DModel.from_pretrained(...)` (~15 MB, **public, no credentials/login**).
  Paired with a linear-beta `DDPMScheduler` to match the checkpoint. **No MNIST
  dataset download** — only the pretrained model weights are fetched.
- Training set: **synthetic from the student's own drawing** — the participant
  edits a 16×16 ASCII grid into a symbol (heart / star / letter / initial); code
  parses it to a 28×28 image and makes ~32 lightly jittered (small rotation +
  shift) copies. Nothing to host, register, or license.
- Lab concept: download the pretrained digit-generator, watch it draw digits,
  then **few-shot fine-tune** it (~100 steps, `lr≈1e-4`, well under a minute) so
  it draws the student's brand-new symbol. Transfer learning made visual and
  generative; anchors the Lecture 8 fine-tuning outcome.
- **No VAE/autoencoder/KL vocabulary** (VAEs are Lecture 11, a later segment):
  the model is framed simply as a pretrained "digit-drawer" you fine-tune,
  keeping the lab inside Lectures 1–8.
- Three blanks: **Blank 1 (creative)** invent your symbol (edit the ASCII grid);
  **Blank 2 (the heart of diffusion training)** the noise-MSE objective line
  `loss = F.mse_loss(noise_pred, noise)`; **Blank 3** the fine-tune learning rate
  `FINETUNE_LR ≈ 1e-4` (rule given, not the number). `assert`s: parsed symbol is
  28×28 in [0,1] with a sensible ink amount; loss is a finite scalar ≥ 0;
  and `0 < FINETUNE_LR < 1e-2`. The before-vs-after payoff (Step 6) is a **visual**
  side-by-side: the same denoising loop draws digits BEFORE and the student's
  symbol AFTER the fine-tune — there is no numeric similarity assert.
- **Built & run-verified (2026-09-01):** `labs/solutions/lab03_generator_transfer.ipynb`
  executed end-to-end via `jupyter nbconvert --to notebook --execute`
  (`/tmp/lab03_executed.ipynb`): 0 error outputs, all **9 asserts** pass. The
  100-step fine-tune runs in well under a minute on Apple MPS (similar or faster
  on a T4) — comfortably under the 10-min budget.

### Lab 5 capstone — verified end-to-end (2026-08-31)
- `labs/solutions/lab05_capstone.ipynb`: three self-contained tracks (set `TRACK='A'/'B'/'C'`)
  reusing the local slices above, plus a `RUN_ALL_TRACKS` smoke switch. Verified via a reduced
  CPU copy (`/tmp/lab05_smoke.ipynb`, `SMOKE=True`, `FORCE_CPU=True`, `RUN_ALL_TRACKS=True`,
  200-sample stratified subsets, 2 epochs): all **13 code cells** ran with saved execution
  counts (1–13) and **0 error outputs**; every track printed its assert-success line
  (`Scaffolding OK`, `Track A/B/C blanks OK`) and its sensitivity/specificity/AUROC report
  (`jupyter nbconvert --to notebook --execute --output /tmp/lab05_smoke.ipynb`, exit 0). Student
  copy stripped to 6 `YOUR CODE HERE` blanks (0 marker leaks, 17 asserts retained); hint sheet
  `labs/handouts/lab05_hints.tex` and one-pager `labs/handouts/lab05_project_onepager.tex` both
  compile to PDF.

### PeakOnly annotated ROIs — chromatographic peak QC (Lecture 5 segment, Lab 5 Track B)
- Source: Melnikov et al., *Anal Chem* 2020 (peakonly). Annotated data hosted on
  Yandex Disk; headless download via the app's own proxy URL (hardcoded in
  `peakonly.py`): `https://getfile.dokpub.com/yandex/get/https://yadi.sk/d/f6BiwqWYF4UVnA`
  (8.8 MB zip, MD5 92dcd314a55fb24be85be765f1976811).
- License: code repo is MIT; **the annotation zip carries no explicit license** —
  cite the paper in all course materials.
- Content: 5,365 hand-annotated ROIs as JSON (variable-length 1D intensity
  windows, median 47 points) from serum HILIC/RPLC + MetaboLights studies:
  **3,764 noise vs. 1,601 real-peak**, plus per-peak quality sub-labels
  (good 318 / low-intensity 203 / lousy 399 / noisy 68) and peak boundaries.
- Prep script: `data/prep/prepare_peakonly.py` — **tested end-to-end**: resamples
  every ROI to 256 points, max-normalizes, writes `peakonly_roi_qc.npz`
  (3.8 MB; X, y, quality, source). Slice already built at
  `data/slices/peakonly_roi_qc.npz`; rehost alongside the DRIAMS slices.
- Course use: base task = peak vs. noise (1D CNN, mirrors the Lecture 5 detection
  segment); stretch = good vs. bad quality via the sub-labels; the `borders`
  field supports a segmentation extension.
- **Lab 5 capstone use (Track B — Peak QC), built & run-verified (2026-08-31):**
  `lab05_capstone.ipynb` Track B classifies the 256-dim ROIs as real-peak (1) vs. noise (0)
  with a small 1D-CNN (`ROIClassifier`) and handles the imbalance via `pos_weight` (neg:pos
  ratio $\approx$2.3, so $>1$). The honest error-analysis cell breaks missed peaks down by the
  **quality sub-label** (good / low-intensity / lousy / noisy) to show *which* peaks are hardest.
  Evaluated with the shared `honest_eval` (sensitivity / specificity / AUROC + confusion matrix).
  Two blanks (`ROIClassifier` head; `pos_weight = torch.tensor([n_neg_train/n_pos_train])`);
  asserts are structural (logits `(N,1)`; `pos_weight > 1`) so a shrunk CPU smoke run passes.
- Also verified for the Lecture 5 lineage slide: PeakOnly (2020) → EVA (2021) →
  NeatMS (2022) → MsTargetPeaker (RL, *MCP* 2026); TargetedMSQC (*Clinical
  Proteomics* 2018) as the clinical-MRM precedent (600 expert-annotated
  transitions, ok/flag with failure modes — good Lecture 10/quiz scenario material).
- Runner-up (if ever more scale is needed): Müller et al. 2020 Zenodo benchmark,
  255k labeled EICs, CC-BY 4.0 — but a 4 GB RData zip, instructor-prep only.

## Slide image assets — provenance

Real, licensed figures fetched for the decks (stored under `slides/assets/img/`,
rehosted offline per CLAUDE.md rule 7). Raw download caches keep the underscore
prefix (`_maldi_src.webp`) so `tools/make_slide_figures.py` can rebuild the
cropped/downscaled versions reproducibly.

| Asset | Source | License | Used in |
|---|---|---|---|
| `fig_driams_mrsa.png` (plotted from `data/slices/driams_c_saureus_oxacillin.npz`) | DRIAMS — one real clinical MALDI-TOF spectrum, S. aureus + oxacillin (MRSA); Dryad doi:10.5061/dryad.bzkh1899q | CC0 1.0 | Lecture 1 MS-hierarchy slide (tighter clinical anchor; replaces `maldi_tof_real.png` there) |
| `maldi_tof_real.png` (crop of `_maldi_src.webp`) | "Examples of non-destructive MALDI-TOF MS spectra", Martisius et al. 2020 (ZooMS bone-tool paper), via Wikimedia Commons | CC BY 4.0 | retired from Lecture 1 (superseded by `fig_driams_mrsa.png`); kept for reuse |
| `chatgpt_conversation.png` (synthetic, course palette) | authored in-house by `tools/make_slide_figures.py:chatgpt_panel()` — an English prompt/reply chat mock-up; NOT a third-party screenshot (the earlier French Wikimedia screenshot, Mattoutankamon CC BY-SA 4.0, was wrong for an English course and has been replaced) | course-original (no third-party rights) | Lecture 1 hook 4-panel composite |
| `hook_4panel.png` (composite) | AlphaFold DB (CC0) · Go board: Xchen27 (CC BY-SA 3.0) · ChatGPT panel: course-original synthetic · Waymo car: Grendelkhan (CC BY-SA 4.0) | per-panel, see caption | Lecture 1 hook |
| `cat_hierarchy.png` (composite) | cat photo: Alvesgaspar (CC BY-SA 3.0) + derived edge/parts/whole layers | CC BY-SA 3.0 | Lecture 1 depth-hierarchy slide |

Generated (no external source, course palette) via `tools/make_slide_figures.py`:
`fig_activations.png`, `fig_activations_quiz.png`, `fig_neuron_ido.png`,
`fig_neuron_youdo.png`, `fig_matvec_row1.png`, `fig_matvec_row2.png`,
`fig_universal_approx.png`, `fig_twolayer_youdo.png`, `fig_driams_mrsa.png`,
`chatgpt_conversation.png`.

## Entry template (fill in when a dataset is confirmed)

```
### <name>
- Source / URL:
- License:
- Raw size / prepared slice size:
- Prep script: data/prep/<script>.py
- Hosted slice for Colab download:
- Used in: labNN / lectureNN
```
