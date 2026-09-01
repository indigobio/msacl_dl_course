# MSACL DS301 Deep Learning — Master Course Specification

Deep learning for the clinical lab · MSACL Data Science Track · 15 contact hours · PyTorch + Google Colab

## Course Schedule

| Segment | Time | Sessions |
|---|---|---|
| 1 | Sunday 09:00–12:20 | Lecture 1, Lecture 2, **Lab 1** |
| 2 | Sunday 14:40–18:00 | Lecture 4*, Lecture 5, **Lab 2** |
| 3 | Monday 09:00–12:20 | Lecture 7, Lecture 8, **Lab 3** |
| 4 | Monday 14:40–18:00 | Lecture 10, Lecture 11, **Lab 4** |
| 5 | Tuesday 09:00–12:20 | Lecture 13, Lecture 14, **Lab 5 (capstone)** |

Total: 15.00 contact hours. Ten-minute breaks after each full instructional hour
(excluded from contact hours). *Session numbering keeps labs at slots 3, 6, 9,
12, 15: every half-day segment ends hands-on.

Each lab hour = ~45 min guided lab + 10 min structured segment discussion
("apply this to your lab"), prompts printed on the handout.
Lectures embed one or two 2–3 minute think-pair-share prompts (marked 💬 below).

---

## Pre-requisites

Data Science 203 (or equivalent experience): basic Python. Advanced math is NOT
required.

## Overview

Deep learning has transformed scientific research and industry applications,
powering breakthroughs from protein structure prediction (AlphaFold) to
antimicrobial-resistance calling from routine MALDI-TOF spectra — and the entire
modern AI wave, including large language models, is built on it.

But what exactly is deep learning? How is it different from traditional machine
learning? And how can you apply it to *your* assay, instrument, or product
without a PhD in mathematics or computer science?

This course demystifies deep learning with an intuition-first, mass-spectrometry-
centered approach. Every architecture is anchored to data you already work with —
spectra, chromatograms, peptides, patient panels — and every session ends in an
assessment: a short handout quiz, a fill-in-the-blank Colab lab with built-in
self-checks, or a structured discussion connecting the material to your own lab.

## Course Objectives

At the conclusion of this short course, the participant will be able to:

1. Explain the principles of deep learning in accessible terms.
2. Build and train basic neural networks in PyTorch (via guided, fill-in-the-blank
   notebooks on Google Colab).
3. Choose an appropriate data representation and network architecture for the data
   types found in a clinical MS lab (tabular panels, spectra, chromatograms,
   sequences, images).
4. Evaluate a model the way a clinical lab must: proper metrics, class imbalance,
   batch effects and instrument drift, external validation.
5. Make informed decisions about when deep learning is — and is NOT — the right
   tool, and sketch a credible first deep-learning project plan for their own work.

## Assessment System

| Instrument | What it looks like | Where |
|---|---|---|
| Handout quizzes (numbered to match their lecture: 1, 2, 4, 5, 7, 8, 10, 11, 13, 14 — no quiz only for the Lab slots) | ~4–5 questions/lecture mixing intuition and light computation, printed, reviewed immediately | after most lectures |
| Lab checkpoints | fill-in-the-blank Colab cells with `assert` self-checks; paper hint sheets with multiple-choice code options | Labs 1–5 |
| 💬 Pair prompts | 2–3 min think-pair-share inside lectures | all lectures |
| Segment discussions (D1–D5) | 10 min structured "apply to your lab," prompts on handout | end of each segment |
| Quiz 10 | imbalanced-data evaluation + treatment: confusion-matrix precision/recall/specificity, AUROC vs. AUPRC, treatment match, a SMOTE point | Lecture 10 |
| Capstone + project one-pager | guided mini-project with rubric; personal "first DL project plan" template | Lab 5 |

Quiz policy: quizzes test **intuition first, with light math welcome** — the
audience is largely PhDs. Fair game (always as worked examples with concrete
numbers): a small matrix–vector multiplication, a one-step gradient update, a
backpropagation walk on a two-weight network, computing feature-map cells on a
grid, an attention weighted average. Not fair game: symbolic derivations or any
calculus beyond "derivative = slope."

---

# Session-by-Session Specification

## SEGMENT 1 — Sunday morning: Foundations

### Lecture 1 · What Is Deep Learning? (+ only the math you need)

**Outcomes** — participant can:
- Explain deep learning vs. traditional machine learning (learned features vs. hand-engineered features) in one sentence.
- Describe what a single neuron computes: weighted sum + bias → activation.
- Multiply a small weight matrix by an input vector by hand, and reason about layer shapes (inputs × weights → outputs).
- Explain intuitively why stacked layers learn hierarchical features.

**Content & timing (60 min)**
- 0:00–0:10 Hook: DL in our field — AlphaFold, MALDI-TOF resistance calling (DRIAMS), de novo sequencing, ChatGPT. What this course will and won't do.
- 0:10–0:20 What machine learning is: a five-part working definition — ML is an approach to (1) learn (2) complex patterns from (3) existing data, then use them to make (4) predictions on (5) unseen data — placed within the statistics → ML → AI picture (`Supplements/AI_statistics.png`). Then traditional ML vs. DL: hand-engineered features (peak areas, ratios you choose) vs. learned features (the network finds them). 💬 *"Name one hand-engineered feature your lab already computes."*
- 0:20–0:35 The neuron: weighted sum, bias, activation functions (ReLU, sigmoid) — graphical, with a two-input worked example using small integers.
- 0:35–0:52 Stacking neurons: layers as matrix–vector multiplication, with one worked 2×3 example computed live (a whole layer's forward pass in four multiplications); shape compatibility; depth = hierarchy. MS anchor: raw spectrum → peak-like features → patterns → call.
- 0:52–1:00 **Quiz 1** hand-out + immediate walkthrough.

**Assessment: Quiz 1** — match activation-function shapes to names; compute one two-input weighted sum; compute one small matrix–vector multiplication (2×3 weight matrix, concrete numbers); one "which is a learned vs. engineered feature?" item; and one two-layer forward pass by hand (hidden layer with ReLU, then output layer — concrete small integers).

### Lecture 2 · How Networks Learn (+ the PyTorch training loop)

**Outcomes** — participant can:
- Explain loss as "how wrong," gradient descent as "downhill steps," and the learning rate as step size.
- Describe backpropagation's job (assign credit/blame to each weight) and walk it on a tiny two-weight network with actual numbers — chain rule as "multiply the local effects along the path."
- Execute one gradient-descent step by hand (weight − learning rate × gradient).
- Trace the five steps of a PyTorch training loop in reading order (forward → loss → zero grad → backward → step).
- Diagnose too-high / too-low learning rate from a loss curve.

**Content & timing (60 min)**
- 0:00–0:10 Loss functions: distance between prediction and truth (MSE intuition; classification loss deferred to Lecture 4).
- 0:10–0:22 Gradient descent: ball-rolling-downhill visual; learning rate as step size, with the three failure modes animated. 💬 *"Loss curve A vs. B: which learning rate is too big?"*
- 0:22–0:40 Backpropagation, worked end to end on an **extremely simple network** (one input → one hidden neuron → one output, two weights, concrete numbers): forward pass, error, then the chain of local effects backward; compute both gradients and take one gradient-descent step by hand. Then generalize by picture: the same arrows on the Lecture 1 network diagram — this is all `loss.backward()` does, at scale.
- 0:40–0:53 The PyTorch training loop, annotated line by line on one slide; live demo: train a tiny classifier for 20 epochs, watch the loss fall.
- 0:53–1:00 **Quiz 2** + walkthrough.

**Assessment: Quiz 2** — one-step backprop calculation on a two-weight network mirroring the worked example (numbers given); one gradient-descent update by hand; match three loss curves to diagnoses (LR too high / too low / just right); put the five training-loop lines in order.

### Lab 1 (slot 3) · Your First Network: the PyTorch training loop

**Focus (deliberately narrow).** Get comfortable with Colab and PyTorch by running
one small network through the exact train-then-evaluate loop from Lecture 2 —
nothing more. The audience is lab technicians and MDs with minimal coding
experience, in a ~50-minute slot, so the notebook stays short and almost
everything is read-and-run. This is not a data-science lab; data prep is
provided, and the participant's job is only to complete and run the loop.

**Outcomes** — participant can:
- Open a Colab notebook, run a cell, and read its output.
- Point to the four pieces in PyTorch code: the data (tensors), the model, the
  loss, and the optimizer.
- Complete and run the five-line training loop (forward → loss → zero grad →
  backward → step) and watch the loss fall.
- Run an evaluation pass on a held-out test split and read the accuracy.

**Structure (~45 min guided + ~10 min discussion)**
- Colab orientation: what Colab is, how to run a cell, the runtime. (~5 min)
- Read-and-run setup (provided, not edited): load the MTBLS90 tabular dataset,
  normalize, split train/test, and define a tiny MLP. Narrated so participants
  can see the four pieces, but they type nothing here.
- The only blanks are the loop itself, mirroring Lecture 2: the five training-loop
  lines, then the evaluation pass (`torch.no_grad`) and the accuracy count. Each
  blank has an `assert` self-check; a paper hint sheet gives multiple-choice code
  options for every blank.
- Watch it learn: print/plot the loss per epoch and confirm it drops — the
  Lecture 2 loss curve, now on real data.
- Optional stretch (only if ahead of time): change the number of epochs or the
  hidden width and re-run.

**Discussion D1** — *"Which measurement in your lab produces a table like this?
What would you want to predict from it, and what could go wrong?"*

## SEGMENT 2 — Sunday afternoon: Training in Practice + CNNs

### Lecture 4 · Making Training Actually Work

**Outcomes** — participant can:
- Explain cross-entropy as "penalizing confident wrong answers" and when to use it vs. MSE.
- Explain mini-batches/SGD and what Adam adds (one-slide intuition: adaptive step sizes).
- Recognize vanishing/exploding gradients as the reason deep nets need care (sets up ResNet in Lecture 5).
- Identify overfitting vs. underfitting from training/validation curves and name fixes (dropout, L2, early stopping, more data).
- Explain what a learning-rate schedule does (decay the step size over training) and why it helps — a callback to Lecture 2's learning rate.
- Name the key hyperparameters (learning rate, batch size, epochs) and describe grid vs. random search.

**Content & timing (60 min)** — organized into three clearly-signposted parts
(Loss → Tuning → Overfitting), each with its own section-divider slide.

*Part 1 — Loss function choice (0:00–0:12).* Cross-entropy for classification, by
picture: softmax → probability bars, the penalty ("surprise") curve, a worked MALDI
R/S example where the same call costs 10× the loss at low confidence, and the
formula itself walked through intuitively (L = −log p_true — the one-hot label keeps
only the true class's term). MSE-vs-cross-entropy rule of thumb (regression vs.
classification); **focal loss** named only briefly (down-weights easy/confident
examples; helps under class imbalance).

*Part 2 — Tuning the knobs (0:12–0:38).* Mini-batch SGD as a *data-feeding* choice
(update-once-per-batch; faster *and* better — after the reference deck); then the
**optimizer**, Adam = adaptive per-weight learning rate + momentum; the
**learning-rate schedule** (warmup–stable–decay, after arXiv:2410.05192; cosine as
the classic alternative; callback to Lecture 2). A short "why deep nets are harder to
train" beat: vanishing/exploding gradients as "whisper down the lane," shown as the
chain-multiplication of per-layer slopes (sets up Lecture 5's ResNet). Then the
**hyperparameters** (the count is huge — we cover only the basics: learning rate,
batch size, epochs): what to tune first; the batch-size trade-off incl. the too-big
problem, illustrated with sharp-vs-flat minima *and* its fingerprint on the loss
curve; grid vs. random search — noting the state of the art is not blind (Bayesian
optimization learns from prior trials).

*Part 3 — Overfitting vs. underfitting (0:38–0:55).* The classic curve-fitting trio
(underfit line / good fit / overfit wiggle) then the train/validation curve gallery.
MS anchor: memorizing one instrument's quirks vs. learning the chemistry. 💬 *"Your
model is 99% on training data, 70% on new samples — what do you try first?"* Fixes:
more/varied data + early stopping; **dropout (visualized)**; **L2 = weight decay**.

- 0:55–1:00 **Quiz 4** (walkthrough opens Lecture 5 if time is short).

**Assessment: Quiz 4** — match four training/validation curve pairs to diagnoses and pick a fix for each; one "what does a bigger batch buy you, and what's the problem of too-big a batch?" item; one vanishing-gradient chain-multiplication item (multiply the per-layer slopes down a 5-layer chain, read off the first-layer gradient, and say whether it vanishes or explodes — sets up ResNet in Lecture 5); and two short **discussion** prompts on class imbalance (a topic without dedicated lecture time — why plain accuracy misleads on a rare class, and what to try), revisited in Lecture 10. (Focal loss was dropped from the quiz as too advanced.)

### Lecture 5 · Convolutional Networks — Spectra, Images, and Finding Patterns

**Outcomes** — participant can:
- State the three reasons CNNs fit images and spectra (patterns are local; the same pattern recurs in different positions; subsampling preserves identity).
- Execute a 2D convolution by hand on a small numeric grid — slide a 3×3 filter across a 5×5 image, apply stride and zero padding, and fill in feature-map values.
- Compute max pooling on a feature map and the output size of a convolutional layer.
- Explain filters → channels, the whole-CNN assembly (conv → pool, repeated → flatten → fully connected), and what a ResNet skip connection fixes.
- Transfer the same machinery to 1D spectra/chromatograms, and describe how object detection frames peak picking (boxes + scores).

**Content & timing (60 min)** — conv/pool/stride visualized in full detail,
2D-first, in the style of the reference deck (`Supplements/Deep_learning_in_1day.pptx`,
CNN section ≈ slides 150–170: numeric grids computed cell by cell).
- 0:00–0:08 Why CNN for images: the three properties ("beak detector" logic), each immediately restated for MS: a peak shape is local, appears anywhere on the m/z axis, and survives downsampling.
- 0:08–0:26 **Convolution mechanics on a numeric grid**: 5×5 binary image, 3×3 ±1 filter slid step by step with the feature map filled in live; stride 1 vs. stride 2 on the same grid; zero padding; a second filter → the channel idea. Participants compute two feature-map cells along the way (live checkpoint).
- 0:26–0:34 Max pooling computed on the just-built feature maps; the whole-CNN assembly diagram; output-size arithmetic (one worked count with and without padding).
- 0:34–0:40 The same machinery in 1D: the filter now slides along a spectrum — "the same peak detector everywhere on the m/z axis"; chromatograms likewise. 💬 *"Which of your data would a 1D CNN read? A 2D CNN?"*
- 0:40–0:46 Architecture story in brief: LeNet → AlexNet → VGG → ResNet; the skip connection as a gradient highway (calls back to Lecture 4's vanishing gradients).
- 0:46–0:56 **CNNs beyond classification: object detection for peak picking.** Bounding boxes + confidence scores; YOLO in one picture; a chromatogram as an "image" whose objects are peaks; PeakOnly as the real-world example. Segmentation gets one slide (pixel-level labels; imaging MS).
- 0:56–1:00 **Quiz 5** handed out; walkthrough opens Lab 2.

**Assessment: Quiz 5** — diagram-heavy: given a 5×5 numeric image and a 3×3 filter, compute three feature-map cells at stride 1 and one at stride 2; max-pool a given 4×4 map; one output-size count with padding; one item mapping peak picking onto detection (what is the "box," what is the "score"?).

### Lab 2 (slot 6) · 1D CNN on Clinical MALDI-TOF Spectra

**Outcomes** — participant can adapt the Lab 1 workflow to spectra; build a small
1D CNN in PyTorch; compute the flattened dimension that feeds a dense layer; and
choose a decision threshold for imbalanced data.

**Structure (45 min lab + 10 min discussion)**
- Guided notebook `lab02`: load prepared MALDI-TOF spectra slice (DRIAMS
  S. aureus + oxacillin R/S), visualize spectra, train a 1D CNN classifier.
- Blanks: `Conv1d` parameters, adding a second conv block, choosing batch size,
  and **computing the flattened input size of the first dense layer** (pooling
  arithmetic → channels × pooled length), self-checked against the real model.
- Imbalanced evaluation: read **recall, not accuracy**; then **explore the
  decision threshold** yourself (change it / loop it) to catch at least half the
  resistant cases — the Lecture 10 imbalance lesson made hands-on.
- Stretch goal: focal loss to lift the rare class without moving the threshold.

**Discussion D2** — *"Where in your workflow is there a 1D signal a CNN could
read? Who labels it today, and how well?"*

## SEGMENT 3 — Monday morning: Sequences + Transformers

### Lecture 7 · Sequence Models and Attention

**Outcomes** — participant can:
- Explain why order matters for sequence data and what RNNs/LSTMs did (memory passed along the sequence).
- Explain tokenization and embeddings ("words → coordinates where similar things are near each other").
- Explain self-attention as "every token looks at every other token and decides what's relevant," including the query/key/value lookup analogy.
- State why attention replaced recurrence (parallel, long-range).

**Content & timing (60 min)**
- 0:00–0:12 Sequence data in the lab: chromatogram time series, QC drift across runs, peptides as sequences. RNN/LSTM in pictures; their limits (slow, forgetful).
- 0:12–0:25 Tokenization + embeddings. MS anchor: **a peptide is a sentence; amino acids are tokens.** 💬 *"Tokenize ACDEFG — what could the vocabulary be for modified peptides?"*
- 0:25–0:48 Self-attention, heavily visualized: attention as a soft lookup (query/key/value = "what I'm looking for / what I contain / what I give you"); attention-weight heat maps on a short peptide and a short sentence; multi-head = several lookups in parallel.
- 0:48–0:55 Why attention won: parallelism + long-range context; sets up Lecture 8.
- 0:55–1:00 **Quiz 7** + walkthrough.

**Assessment: Quiz 7** — given a mini attention heat map, "which token does X attend to most, and why is that sensible?"; one tiny weighted-average computation (three tokens, given weights); match Q/K/V to the lookup analogy roles.

### Lecture 8 · The Transformer, Assembled (+ transfer learning)

**Outcomes** — participant can:
- Explain positional encoding's job (attention is order-blind; positions must be injected).
- Distinguish batch norm vs. layer norm at the "what gets averaged" level.
- Label the encoder and decoder on a transformer diagram and say what each is for.
- Name three strategies for turning a spectrum (or image) into transformer tokens: chunk/patch + linear projection (ViT-style), a CNN front-end encoder, and peak-as-token (Casanovo-style) — and pick one for a given data type.
- Explain transfer learning and fine-tuning (retrain a small head on a frozen pretrained body) and why it changes the game for small lab datasets.

**Content & timing (60 min)**
- 0:00–0:10 Positional encoding: the "attention can't see order" problem and the fix, by picture.
- 0:10–0:18 Normalization: batch norm vs. layer norm — one visual of which axis gets averaged, and why transformers use layer norm.
- 0:18–0:30 Assembling the full architecture piece by piece (every block now familiar); encoder vs. decoder; encoder-only (BERT/classification) vs. decoder-only (GPT/generation).
- 0:30–0:40 **Transformers on MS data: how does a spectrum become tokens?** Three strategies side by side on the same spectrum: (a) cut into chunks/patches + linear projection to embeddings (the ViT idea); (b) a small CNN front-end that emits token embeddings (Lecture 5 callback); (c) each peak is a token — m/z + intensity embedded (how Casanovo reads a spectrum: spectrum in, peptide out, as translation). Mini decision chart: which strategy for which data.
- 0:40–0:53 Transfer learning: pretrained body + new head; freezing; fine-tuning; why 500 labeled samples can be enough when someone pretrained on millions. 💬 *"You have 800 labeled spectra — from scratch or fine-tune, and why?"*
- 0:53–1:00 **Quiz 8** + walkthrough.

**Assessment: Quiz 8** — label four blocks on an unlabeled transformer diagram; match three data types (imaging MS image, peak list, raw binned spectrum) to tokenization strategies; one fine-tune vs. train-from-scratch scenario; one "which norm averages over what?" item.

### Lab 3 (slot 9) · Fine-Tuning a Pretrained Transformer

**Outcomes** — participant can load a pretrained model, replace its head, freeze
layers, fine-tune on a small dataset, and compare against training from scratch.

**Structure (45 min lab + 10 min discussion)** — resolved in Phase 2:
- Guided notebook `lab03`: fine-tune **DistilBERT** on the `medical_abstracts`
  dataset (5 disease classes, CC-BY-SA); ~4,000 training abstracts, fp16,
  3 epochs ≈ 2–4 min on a T4.
- Blanks: swapping the classification head, choosing what to freeze,
  fine-tuning learning rate (much smaller — why?).
- Built-in three-way comparison: full fine-tune vs. frozen-body head-only vs.
  from-scratch (from-scratch lands near the 33% majority floor; fine-tuned
  reaches ~65% — the game-changer lesson, made visible).
- **Bonus demo cell (peptides!):** the same recipe on ESM-2 (8M), a protein
  language model, classifying ~3,000 peptides in ~30 s — "DistilBERT read 3B
  words of English; ESM-2 read 250M protein sequences. Same move, different
  alphabet."

**Discussion D3** — *"What pretrained model could your product borrow — and what
would 'the head' predict?"*

## SEGMENT 4 — Monday afternoon: Making It Real

### Lecture 10 · No Perfect Data: Evaluation and Treatment

**Outcomes** — participant can:
- Compute precision, recall (= sensitivity), and specificity from a confusion matrix by hand, and explain why accuracy is misleading when one class is rare.
- Explain why AUROC flatters a rare-positive screen (a large true-negative pool keeps the false-positive rate tiny) and why AUPRC / the PR curve is the honest metric, whose baseline is the prevalence — report it with sensitivity.
- Explain that a metric is only valid on the distribution it was tested on: batch effects / instrument drift shift the data, so external and temporal validation are mandatory (split by patient/isolate, never by spectrum).
- Choose a treatment for imbalanced data — data curation, SMOTE / class weights / oversampling, focal loss, curriculum learning — matched to the symptom, and recognize that none of them adds new minority signal (only collecting more real minority data does).
- Interpolate a SMOTE point by hand (synthetic = A + λ·(B−A)) and state why raw-spectrum SMOTE is risky in high dimensions.

**Content & timing (60 min)** — one theme all hour: imbalanced data (real DRIAMS 697 susceptible / 41 resistant).
- 0:00–0:13 **Part A — measure it:** the accuracy trap (94%-accurate detector that never fires); confusion matrix by hand → precision, recall (sensitivity), specificity.
- 0:13–0:25 Why AUROC misleads under imbalance and why AUPRC / the PR curve tells the truth — with the numbers (the same operating point plotted on both curves; PR baseline = prevalence).
- 0:25–0:30 One honesty note: a metric is only valid on the distribution you tested → external + temporal validation (DRIAMS across-site AUROC drop, Wiesmann et al. 2025).
- 0:30–0:48 **Part B — treat it:** curation; SMOTE (a numeric do-it-together + honest caveat for high-dim spectra); class weights + focal loss (worked with real losses); curriculum learning (intuition); a match-the-treatment decide beat + "which adds new signal? → none."
- 0:48–1:00 **Quiz 10** debrief in pairs (the distributed quiz, filled across the hour).

**Assessment: Quiz 10 (imbalanced-data evaluation + treatment)** — a distributed
quiz filled across the hour (Q1 confusion-matrix rates, Q2 AUROC vs. AUPRC,
Q3 treatment match + "no new signal," Q4 a SMOTE point) — directly assesses
course objectives 3–5.

### Lecture 11 · Learning Without (Many) Labels

**Outcomes** — participant can:
- Explain autoencoders (compress → reconstruct) and their lab uses: denoising, anomaly/QC detection, dimensionality reduction.
- Explain intuitively what a VAE adds: instead of memorizing each spectrum as an isolated point, learn a smooth cloud where nearby latent points decode to plausible spectra — so you can sample new ones and interpolate; tell an AE latent space from a VAE latent space in a picture.
- Explain self-supervised learning (the data labels itself): masking/pretext tasks (the BERT connection), contrastive learning, and DINO (student–teacher self-distillation: two augmented views of the same input must agree) — and connect it to how foundation models are pretrained (callback to Lectures 8/13).
- Match a lab scenario to the right paradigm (and know that semi-supervised learning / pseudo-labeling exists as a middle path).

**Content & timing (60 min)**
- 0:00–0:14 Autoencoders: bottleneck picture; reconstruction error as an anomaly score (MS anchor: flagging bad runs/QC failures); latent space as a map of your spectra.
- 0:14–0:30 **VAEs, intuitively**: points vs. clouds; side-by-side latent-space visualizations on the same data (AE: scattered islands with gaps that decode to garbage; VAE: smooth, organized, sampleable); walking a line through the VAE latent space = one spectrum morphing into another; why smoothness helps both anomaly scoring and generation. One slide on the recipe (reconstruction + "keep the cloud tidy" penalty), no derivation. 💬 *"Which of these two latent maps would you trust to flag a novel contaminant, and why?"*
- 0:30–0:47 Self-supervised — how the big models learn without labels: masking (BERT), contrastive learning ("two augmented views of the same spectrum should land together"), and **DINO** (student–teacher self-distillation; the augmentation-invariance idea, with the famous unsupervised-attention segmentation visual).
- 0:47–0:50 Semi-supervised in one slide: pseudo-labeling as the "10,000 unlabeled + 100 labeled" middle path — named so they know the term, not deeper.
- 0:50–1:00 **Quiz 11** + walkthrough; preview of Lab 4.

**Assessment: Quiz 11** — pick which of two latent-space pictures is the VAE and justify in one sentence; match four lab scenarios to paradigms (supervised / unsupervised / self-supervised / semi-supervised); one "what does reconstruction error tell you?" item; one DINO/contrastive intuition item ("why must the two views agree?").

### Lab 4 (slot 12) · VAE for Spectra QC, Anomaly Detection, and Generation

**Outcomes** — participant can train a VAE on normal spectra, use reconstruction
error to flag anomalies, visualize and explore a 2D latent space, and generate
new spectra by decoding latent points.

**Structure (45 min lab + 10 min discussion)**
- Guided notebook `lab04`: train a VAE on "normal" spectra (reuse the Lab 2
  dataset). Encoder/decoder and the reparameterization helper are provided as
  read-and-run code; participants fill in the key choices.
- Blanks: bottleneck (latent) size, the combined loss line (reconstruction +
  KL "tidiness" term with a weight), the anomaly threshold.
- Score held-out spectra by reconstruction error and flag anomalies; plot the
  2D latent space colored by class — visually compare against a plain AE's
  latent space (pre-trained, provided) to see the Lecture 11 picture on real data.
- Fun payoff: sample latent points and interpolate between two spectra —
  watch the decoded spectrum morph.
- Teachable moments: KL weight too high = blurry reconstructions; bottleneck
  too big = anomaly detector stops working; threshold choice is a
  sensitivity/specificity decision (callback to Lecture 10).

**Discussion D4** — *"What does 'abnormal' look like in your instrument's output,
and would reconstruction error catch it before a human does?"*

## SEGMENT 5 — Tuesday morning: The Frontier + Capstone

### Lecture 13 · Large Language Models + the DL-in-MS Landscape

**Outcomes** — participant can:
- Explain LLM training in pictures: next-token pretraining → supervised fine-tuning → preference tuning (RLHF) — each in one sentence.
- Explain reinforcement learning in one picture — agent, environment, action, reward; learning by trial and feedback — and how RLHF is "RL where the reward comes from human preferences."
- Choose among prompting, RAG, and fine-tuning for a lab use case.
- For each landmark MS tool, name the problem, the architecture family (connecting back to Lectures 5–11), and the impact: DeepNovo/Casanovo, Prosit/AlphaPeptDeep, DIA-NN, DRIAMS MALDI-TOF resistance calling, AlphaFold.

**Content & timing (60 min)**
- 0:00–0:16 LLMs demystified: next-token prediction; scale; pretraining and SFT; hallucination as a failure mode to design around.
- 0:16–0:28 **Reinforcement learning, touched base**: the agent–environment–reward loop in one diagram; AlphaGo in one slide (trial, feedback, improvement — no MDP/policy math); then RLHF as RL where humans supply the reward, completing the LLM training pipeline diagram. (Lecture 14's agents will call back to this.)
- 0:28–0:36 Using LLMs without training one: prompting vs. RAG vs. fine-tuning; one clinical documentation/report-drafting example.
- 0:36–0:54 The landscape tour, one slide per system, always in the frame "problem → architecture you now know → impact": de novo sequencing (transformer = translation), spectrum/RT prediction (Prosit; used to rescore IDs), DIA-NN, DRIAMS (a 1D-CNN-style clinical win), AlphaFold (the flagship). 💬 *"Which of these is closest to a problem in your lab?"*
- 0:54–1:00 **Quiz 13** + walkthrough.

**Assessment: Quiz 13** — match tool → problem → architecture family (grid item);
label agent/action/reward in a lab-flavored RL scenario; one
prompting-vs-RAG-vs-fine-tuning scenario; one "where could this hallucinate
and what's the guardrail?" item.

### Lecture 14 · Agentic AI, Demo-Driven

**Outcomes** — participant can:
- Define an agent as LLM + tools + a reason–act loop (ReAct); trace a transcript and label the reasoning vs. acting steps.
- Run a simple agentic interaction themselves on a free-tier chatbot (structured prompt: reason, then act, then check).
- Name where agents plausibly fit in lab workflows (triage, report drafting, data wrangling, literature) and the guardrails each needs (human review, restricted tools, logging).

**Content & timing (60 min)**
- 0:00–0:12 From chatbot to agent: tools, memory, loop; the ReAct pattern annotated on a real transcript; one-line callback to Lecture 13's RL (feedback and reward are how these behaviors get trained).
- 0:12–0:22 **Follow-along on your own bot:** participants use whatever free-tier assistant they have (ChatGPT, Claude, Gemini) with a provided mini-exercise — paste a small QC table from the handout, prompt for explicit reason→act→check steps, compare answers with a neighbor.
- 0:22–0:40 **Instructor live demo — the "MS analysis assistant":** Claude Code as a real agent on a spectra/QC CSV from the course datasets: it reads files, runs analysis, plots, flags anomalies, drafts a summary — each reason→act step narrated. (Primary: instructor's Claude Code; alternative: Bedrock API; fallback: pre-recorded run.)
- 0:40–0:48 Where agents fit in the lab, and where they must not run unsupervised; guardrails (human review, restricted tools, logging). 💬 *"Which task in your week would you hand an agent first — and what check keeps it honest?"*
- 0:48–1:00 **Quiz 14** (short) + capstone briefing for Lab 5.

**Assessment: Quiz 14** — label reason vs. act steps on a short transcript; one
"which guardrail for which failure?" match; one "what must stay human-reviewed
in a clinical workflow?" item.

### Lab 5 (slot 15) · Capstone: An End-to-End MS Mini-Project

**Outcomes** — participant integrates the whole course: pick a track, adapt a
working pipeline, evaluate it properly, and leave with a personal project plan.

**Structure (45 min lab + 10 min closing discussion)**
- Choose one guided track (all reuse course datasets, all rubric-scored):
  - **Track A — Beat the baseline:** improve the Lab 2 spectra classifier
    (architecture, augmentation, tuning); optional shared leaderboard cell.
  - **Track B — Peak QC:** classify chromatographic peaks as good/bad
    integrations (the Lecture 5 object-detection segment made real; candidate data:
    PeakOnly-style labeled ROIs).
  - **Track C — New question, same spectra:** fine-tune/re-head the Lab 2 model
    for a different label available in the dataset (transfer learning in practice).
- Rubric (self- + instructor-assessed): data handling, model choice justified,
  proper split/metrics, honest error analysis.
- **Project one-pager:** each participant completes a one-page template for a
  first DL project in their own lab: problem → data & representation →
  architecture → validation plan → "why not a simpler model?"

**Discussion D5 (closing)** — participants share their one-pagers in pairs, then
volunteers present 60-second versions to the room; instructor closes with a
resources-to-keep-learning slide.

---

## Dataset Dependencies (resolved in Phase 2)

| Sessions | Need | Leading candidate |
|---|---|---|
| Lab 1 | clinical/metabolomics tabular set, clean labels | **resolved:** MTBLS90 serum metabolomics (968×189 named metabolites, balanced, zero NaNs, 1-line Colab load) |
| Lab 2, Lab 4, Lab 5 (A, C) | MALDI-TOF or comparable 1D spectra with labels, Colab-sized slice | **resolved:** DRIAMS-A (CC0, Zenodo): S. aureus+oxacillin (3,064, 24% R) primary; E. coli+ceftriaxone (3,875, 28% R) for Track C; prep script written (instructor runs the one-time 86 GB pull) |
| Lab 3 | small text or peptide dataset + pretrained model that fine-tunes in <10 min on a T4 | **resolved:** DistilBERT + HF `medical_abstracts` (CC-BY-SA) core; ESM-2 8M peptide bonus demo |
| Lecture 5 segment, Lab 5 (B) | labeled chromatographic peaks/ROIs | **resolved:** PeakOnly annotated ROIs (5,365 windows, peak/noise + quality sub-labels); slice built and tested (3.8 MB npz) |
| Lecture 10 | one published cautionary tale of distribution shift in clinical ML | **resolved:** Wiesmann et al., *J Clin Microbiol* 2025 — DRIAMS-trained AMR models lose 0.07–0.23 AUROC on German specimens and decay within 18 months (same task as Lab 2!); Zech et al. 2018 as optional second example |
| Lecture 14 | demo CSV + agent tooling | derive from Lab 2/4 data; instructor's Claude Code (Bedrock as alternative, recording as fallback); participants' own free-tier chatbots |
