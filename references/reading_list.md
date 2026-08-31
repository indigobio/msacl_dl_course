# Verified Reading List — Deep Learning for Clinical Mass Spectrometry

All citations verified against publisher pages / PubMed (Phase 2, 2026-08-25).
Summaries are plain-language, aimed at clinical chemists. Bracketed tags note
where each work anchors the course.

## Core DL-in-MS papers (Lecture 13 landscape tour anchors)

**DeepNovo — "De novo peptide sequencing by deep learning"** — Tran, Zhang, Xin,
Shan, Li. *PNAS* 2017;114(31):8247–8252. DOI:
[10.1073/pnas.1705691114](https://doi.org/10.1073/pnas.1705691114) [free full text]
When a peptide isn't in any database, its MS/MS spectrum must be read
letter-by-letter. DeepNovo was the first deep-learning de novo sequencer,
combining a CNN (to "look at" the spectrum) with an LSTM (to model sequence
patterns), beating prior tools by 8–23% at the amino-acid level. [Lecture 13]

**Casanovo** — (a) Yilmaz, Fondrie, Bittremieux, Oh, Noble, "De novo mass
spectrometry peptide sequencing with a transformer model," *ICML 2022*, PMLR
162:25514–25522, [proceedings.mlr.press/v162/yilmaz22a.html](https://proceedings.mlr.press/v162/yilmaz22a.html);
(b) the scaled follow-up: "Sequence-to-sequence translation from mass spectra to
peptides with a transformer model," *Nature Communications* 2024;15:6427. DOI:
[10.1038/s41467-024-49731-x](https://doi.org/10.1038/s41467-024-49731-x) [open access]
Reframes de novo sequencing as machine translation: a transformer reads spectral
peaks (each peak = a token) and writes out amino acids. Trained on 30M labeled
spectra, it outperforms DeepNovo with a simpler model and is now the reference
architecture for newer sequencers. [Lecture 8 peak-as-token; Lecture 13]

**Prosit — "Proteome-wide prediction of peptide tandem mass spectra by deep
learning"** — Gessulat, Schmidt, Zolg, et al. *Nature Methods* 2019;16(6):509–518.
DOI: [10.1038/s41592-019-0426-7](https://doi.org/10.1038/s41592-019-0426-7)
Trained on ~21M spectra of 550k synthetic peptides (ProteomeTools), Prosit
predicts fragment intensities and retention time from sequence alone — often
better than a replicate measurement — and made "predicted spectral libraries" a
standard tool enabling library-free DIA. [Lecture 13]

**AlphaPeptDeep** — Zeng, Zhou, et al., Mann. *Nature Communications*
2022;13:7238. DOI:
[10.1038/s41467-022-34904-3](https://doi.org/10.1038/s41467-022-34904-3) [open access]
A PyTorch toolkit for building/fine-tuning predictors of retention time, ion
mobility, and fragment intensities in a few lines of code, including transfer
learning to your own instrument — the Prosit idea turned into reusable
infrastructure. [Lecture 8 transfer learning; Lecture 13]

**DIA-NN** — Demichev, Messner, Vernardis, Lilley, Ralser. *Nature Methods*
2020;17(1):41–44. DOI:
[10.1038/s41592-019-0638-x](https://doi.org/10.1038/s41592-019-0638-x) [free on PMC: PMC6949130]
Ensembles of simple neural networks score peptide signals in interference-ridden
DIA data, making short-gradient high-throughput DIA deep and reliable — the de
facto standard DIA software behind large clinical proteomics cohorts. [Lecture 13]

**DRIAMS — "Direct antimicrobial resistance prediction from clinical MALDI-TOF
mass spectra using machine learning"** — Weis, Cuénod, et al., Egli, Borgwardt.
*Nature Medicine* 2022;28(1):164–174. DOI:
[10.1038/s41591-021-01619-9](https://doi.org/10.1038/s41591-021-01619-9)
[dataset CC0 on Dryad/Zenodo]
Routine MALDI-TOF spectra already collected for species ID predict antimicrobial
resistance hours-to-days faster than phenotypic AST: AUROC 0.74–0.80 on priority
pathogen–antibiotic pairs across ~300k spectra from four Swiss hospitals. The
flagship "data you already have + ML" story — and the source of our lab dataset.
Note its cross-site performance drops foreshadow the validation paper below. [Lecture 1 hook; Labs 2/4/5; Lecture 13]

**PeakOnly — "Deep Learning for the Precise Peak Detection in High-Resolution
LC–MS Data"** — Melnikov, Tsentalovich, Yanshole. *Analytical Chemistry*
2020;92(1):588–592. DOI:
[10.1021/acs.analchem.9b04811](https://doi.org/10.1021/acs.analchem.9b04811)
[code free on GitHub]
Treats each extracted ion chromatogram like a tiny image and uses CNNs to
classify and integrate true peaks, approaching expert manual curation — the
everyday-lab-problem anchor for our object-detection-for-peak-picking segment.
[Lecture 5 OD segment; Lab 5 Track B]

**AlphaFold 2 — "Highly accurate protein structure prediction with AlphaFold"** —
Jumper, Evans, Pritzel, et al. *Nature* 2021;596:583–589. DOI:
[10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2) [open access]
Solved a 50-year grand challenge — near-experimental-accuracy structure from
sequence, blind-validated at CASP14 — with an attention-based architecture. The
clearest proof that deep learning extracts expert-level insight from biological
sequence data; the course's motivating example. [Lecture 1 hook; Lecture 13]

## Foundations (Lectures 7, 8, and 11 anchors)

**Attention Is All You Need** — Vaswani et al. *NeurIPS* 2017.
[arXiv:1706.03762](https://arxiv.org/abs/1706.03762) [open access]
The transformer: attention lets every element of an input (word, peak, patch)
directly weigh its relevance to every other element — the architecture behind
Casanovo, AlphaFold, and modern LLMs. [Lectures 7/8]

**ViT — "An Image is Worth 16×16 Words"** — Dosovitskiy, Beyer, Kolesnikov,
et al. *ICLR* 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) [open access]
Chop an image into patches, treat each patch as a "word," and a plain transformer
wins at scale — the patching/tokenization strategy we map onto spectra. [Lecture 8 spectrum→tokens]

**DINO — "Emerging Properties in Self-Supervised Vision Transformers"** — Caron,
Touvron, Misra, Jégou, Mairal, Bojanowski, Joulin. *ICCV* 2021, pp. 9650–9660.
DOI: [10.1109/ICCV48922.2021.00951](https://doi.org/10.1109/ICCV48922.2021.00951) ·
[arXiv:2104.14294](https://arxiv.org/abs/2104.14294) [open access]
A student network learns to match a slowly-updated copy of itself
(self-distillation) on augmented views — no labels at all — and the features
encode segmentation for free. Lesson for MS: archives of unannotated spectra can
pre-train models before scarce labeled clinical data are used. [Lecture 11]

## Course reference review (participant handout)

**"Recent Developments in Machine Learning for Mass Spectrometry"** — Beck,
Muhoberac, Randolph, Beveridge, Wijewardhane, Kenttämaa, Chopra. *ACS
Measurement Science Au* 2024;4(3):233–246. DOI:
[10.1021/acsmeasuresciau.3c00060](https://doi.org/10.1021/acsmeasuresciau.3c00060)
[open access, CC-BY — free to distribute to participants]
Walks through the network families in plain terms (MLPs, CNNs, autoencoders,
RNNs, transformers) and maps them onto MS tasks clinical chemists recognize —
peak detection, metabolite ID, spectrum prediction, MALDI imaging, diagnostics.
Doubles as a course reference sheet. (Runner-up, pitfall-focused: "A 2025
perspective on the role of machine learning for biomarker discovery in clinical
proteomics," *Expert Rev Proteomics* 2025;22(10), DOI:
[10.1080/14789450.2025.2545828](https://doi.org/10.1080/14789450.2025.2545828).)

## The cautionary tale (Lecture 10 distribution-shift anchor)

**"Prediction of antimicrobial resistance from MALDI-TOF mass spectra using
machine learning: a validation study"** — Wiesmann, Enders, Westendorf, Koch,
Schaumburg. *Journal of Clinical Microbiology* 2025;63(12):e0118625. DOI:
[10.1128/jcm.01186-25](https://doi.org/10.1128/jcm.01186-25) [open access]
The perfect fit: an independent external validation of exactly the DRIAMS
paradigm our labs use. Models trained on Swiss 2015–2018 data lost 0.065–0.225
AUROC on German 2023–2025 specimens, and even locally retrained models decayed
0.10–0.25 AUROC within 18 months — site- and time-matched training plus regular
retraining are prerequisites for clinical use. Distribution shift made concrete
on the same instrument, task, and dataset the participants just trained on. [Lecture 10]

**(Adjacent classic)** — Zech et al., "Variable generalization performance of a
deep learning model to detect pneumonia in chest radiographs," *PLOS Medicine*
2018;15(11):e1002683. DOI:
[10.1371/journal.pmed.1002683](https://doi.org/10.1371/journal.pmed.1002683)
[open access] — CNNs learned to recognize which hospital an X-ray came from
rather than the disease. [Lecture 10, optional second example]

## Imbalanced-data methods (Lecture 10 anchors)

**Saito & Rehmsmeier — "The Precision-Recall Plot Is More Informative than the
ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets"** — *PLOS
ONE* 2015;10(3):e0118432. DOI:
[10.1371/journal.pone.0118432](https://doi.org/10.1371/journal.pone.0118432)
[open access] — the reference for why AUPRC beats AUROC under class imbalance:
ROC is invariant to the class ratio, so a large true-negative pool keeps the
false-positive rate tiny and AUROC looks good, while the PR baseline sits at the
prevalence and stays honest. [Lecture 10 AUROC-vs-AUPRC]

**Chawla, Bowyer, Hall, Kegelmeyer — "SMOTE: Synthetic Minority Over-sampling
Technique"** — *Journal of Artificial Intelligence Research* 2002;16:321–357. DOI:
[10.1613/jair.953](https://doi.org/10.1613/jair.953) [open access] — the origin
of synthesizing new minority examples by interpolating between real ones
(synthetic = A + λ·(B−A)); the Lecture 10 do-it-together. [Lecture 10 treatment]

**Lin, Goyal, Girshick, He, Dollár — "Focal Loss for Dense Object Detection"** —
*ICCV* 2017. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002) — FL =
(1−p)^γ·CE down-weights easy, confident examples so training focuses on the hard,
rare ones; the Lecture 10 worked-number I-do. [Lecture 10 treatment]

**Bengio, Louradour, Collobert, Weston — "Curriculum Learning"** — *ICML* 2009.
DOI: [10.1145/1553374.1553380](https://doi.org/10.1145/1553374.1553380) —
ordering training easy→hard (or class-balanced→true prevalence) so early training
isn't swamped by the majority; the Lecture 10 intuition slide. [Lecture 10 treatment]

## Citation subtleties

- DIA-NN: 2020 volume date, DOI issued 2019 (online Nov 2019).
- Casanovo ICML 2022 / Vaswani NeurIPS 2017 / ViT ICLR 2021 have no journal
  DOIs — PMLR / arXiv / OpenReview are the canonical links.
