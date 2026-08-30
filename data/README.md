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
- Used in: lab02 (1D CNN), lab04 (VAE), lab05 tracks A/C; Lecture 10 shift example.

### Lab 3 — transformer fine-tuning: DistilBERT + medical_abstracts (core), ESM-2 bonus
- Core model: `distilbert/distilbert-base-uncased` (268 MB safetensors,
  Apache-2.0, no auth token). Core dataset: `TimSchopf/medical_abstracts` on
  Hugging Face — 11,550 train / 2,888 test abstracts, 5 disease classes,
  9.6 MB parquet, **CC-BY-SA-3.0** (the only cleanly licensed option found).
- Lab plan: subsample ~4,000 train / 1,000 eval, `max_length=256`, batch 32,
  **`fp16=True` (mandatory on T4 — 4–8× slower without it)**, 3 epochs. Three
  runs: full fine-tune vs. frozen-encoder head-only vs. from-scratch — total
  6–9 min on a T4. From-scratch lands near the ~33% majority floor while
  fine-tuned reaches ~62–68%, so the "pretraining is the game-changer" lesson
  is unmissable.
- Notebook gotchas (verified): labels are **1–5, not 0–4** (remap `label - 1`
  or CUDA throws an opaque index error); class names live in a separate config
  (`load_dataset("TimSchopf/medical_abstracts", "labels")`).
- **Bonus demo cell (the on-theme payoff):** `facebook/esm2_t6_8M_UR50D`
  (31 MB, MIT) fine-tuned on ~3,000 peptides in ~30 s — "DistilBERT read 3B
  words of English the way ESM-2 read 250M protein sequences; same move,
  different alphabet." Demo-only, not graded: peptide AMP tasks are largely
  solvable from composition, so the from-scratch gap is small there, and the
  candidate peptide datasets have licensing issues (TzRain/AMPs: no license or
  provenance; AMPBench-MT: non-commercial-only — acceptable for a read-only
  demo, not for the graded core).
- Fallback if 5-class ~65% accuracy demotivates in dry runs:
  `armanc/pubmed-rct20k` (sentence-role classification, `max_length=64`, ~4×
  faster, ~80%+ accuracy; license unspecified).
- Used in: lab03.

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
- Also verified for the Lecture 5 lineage slide: PeakOnly (2020) → EVA (2021) →
  NeatMS (2022) → MsTargetPeaker (RL, *MCP* 2026); TargetedMSQC (*Clinical
  Proteomics* 2018) as the clinical-MRM precedent (600 expert-annotated
  transitions, ok/flag with failure modes — good Lecture 10/quiz scenario material).
- Runner-up (if ever more scale is needed): Müller et al. 2020 Zenodo benchmark,
  255k labeled EICs, CC-BY 4.0 — but a 4 GB RData zip, instructor-prep only.

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
