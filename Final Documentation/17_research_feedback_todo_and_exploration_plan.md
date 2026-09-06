# Research Feedback, To-Do List, and Exploration Plan

## Purpose

The following feedback identifies experiments needed to test whether the defense
is genuinely selective, adaptive, and useful beyond the initial sinusoidal
experiments. The current results show that the method can reduce a controlled
backdoor ASR. The next research phase must determine what the generator is
actually correcting, whether it preserves useful image structure, and whether
the method generalizes to more realistic signals, corruptions, resolutions,
textures, and medical images.

The feedback should be treated as a research backlog rather than a claim that
every item must be completed immediately. Each experiment should have a clear
question, controlled variables, evaluation metrics, and a limitation statement.

## Current Status

- [x] Sinusoidal frequency-trigger baseline.
- [x] CIFAR-100 experiment at 32 x 32.
- [x] Tiny ImageNet experiment at 64 x 64.
- [x] STL-10 experiment at 96 x 96.
- [x] Trigger-strength ablation.
- [x] Trigger-function ablation using sine, cosine, checkerboard, and
  dual-frequency patterns.
- [x] Preliminary FIBA-style amplitude injection study on STL-10.
- [ ] Wavelet-based trigger formation.
- [ ] Edge-aware information and edge-preservation analysis.
- [ ] Joint phase and amplitude trigger study.
- [ ] Noise, damage, and corruption control experiments.
- [ ] High-resolution experiment near 500 x 500.
- [ ] Texture-focused dataset experiment.
- [ ] More systematic pathology-image experiment using FIBA.
- [ ] Input-only correction for an unknown image.
- [ ] Repeated runs across multiple random seeds.

## Research Questions

The feedback can be converted into the following research questions:

1. Can wavelet-domain triggers produce more localized and structured attacks
   than a basic sinusoid?
2. Does the defense preserve meaningful edges, or does it remove them together
   with the trigger?
3. Is amplitude-only correction sufficient, or can a trigger also exploit phase?
4. Does the generator remove the malicious trigger specifically, or does it act
   as a general denoiser that removes noise and image damage as well?
5. Does the method remain computationally practical at approximately 500 x 500
   resolution?
6. Does the method behave differently on texture-rich images than on ordinary
   object-classification images?
7. Does the method transfer to pathology images and amplitude-domain attacks
   such as FIBA?

## Experiment 1: Wavelet-Based Trigger Formation

### Motivation

The sinusoidal trigger is mathematically simple and produces a narrow periodic
signal. A wavelet trigger would provide localization in both position and
scale. This is useful because real image signals are not purely periodic. Edges,
lesions, local texture, and small structures are naturally described by
multi-resolution wavelet coefficients.

### Proposed trigger

Apply a discrete wavelet transform to an image:

\[
W(x)=\{LL, LH, HL, HH\}.
\]

The subbands represent different spatial scales and orientations. A trigger can
be created by modifying a selected subband:

\[
W(t)=W(x)+\alpha R_k,
\]

where (R_k) is a structured pattern inserted into a chosen wavelet subband.
The triggered image is then reconstructed with the inverse wavelet transform.

### Variables to test

- Wavelet family: Haar, Daubechies, or biorthogonal wavelets.
- Decomposition level: one, two, or three levels.
- Subband: (LH), (HL), (HH), or multiple subbands.
- Spatial location: global, corner, center, or random local window.
- Trigger strength: low, medium, and high values selected on validation data.

### What this experiment should prove

It should show whether the generator can mitigate a trigger that is localized by
scale and orientation rather than represented by a single sinusoidal frequency.
The comparison should include ASR, clean accuracy, reconstruction error, and
wavelet-subband energy before and after correction.

### Implementation note

This is not simply a replacement of the word ``sine'' in the current script.
The trigger module should expose a common interface so that sinusoidal, wavelet,
and FIBA triggers can be evaluated using the same classifier, generator, repair,
and metric code.

## Experiment 2: Edge-Based Information

### Motivation

Edges carry important semantic information. Object boundaries, vessel walls,
cell boundaries, and lesion borders may be represented by high-frequency
components. A defense that suppresses high frequencies without considering edges
could damage classification-relevant information.

### Proposed edge representation

For an image (x), compute an edge map using an operator such as Sobel,
Scharr, Laplacian, or Canny:

\[
E(x)=\operatorname{EdgeDetector}(x).
\]

The edge map can be used in two ways:

1. As an additional generator input, allowing the generator to distinguish
   natural edges from periodic trigger energy.
2. As an edge-preservation loss:

\[
\mathcal{L}_{edge}=\|E(x_{corr})-E(x)\|_1.
\]

The extended generator objective becomes:

\[
\mathcal{L}_G'=\mathcal{L}_G+
\lambda_{edge}\mathcal{L}_{edge}.
\]

### Parameters to record

- Edge operator: Sobel, Scharr, Laplacian, or Canny.
- Gaussian smoothing parameter σ, if used.
- Gradient threshold or Canny low/high thresholds.
- Edge-loss weight λ_edge.
- Edge-density percentage of the image.
- Edge similarity before and after correction.

### Evaluation

Report clean accuracy, repaired ASR, edge precision/recall or edge similarity,
and reconstruction error. The goal is not to preserve every individual edge
perfectly; it is to demonstrate that useful boundaries are preserved better than
with global high-frequency suppression.

## Experiment 3: Phase and Frequency Combination

### Motivation

The current defense primarily corrects amplitude while preserving phase. This is
a deliberate design choice, but it leaves an important question: what happens
when a trigger changes amplitude, phase, or both?

### Conditions

Evaluate four controlled conditions:

1. Amplitude-only trigger.
2. Phase-only trigger.
3. Joint amplitude and phase trigger.
4. Clean image with no trigger.

For a complex spectrum (F=Ae^{jP}), a joint modification can be represented
as:

\[
F_t=(A_c+\Delta A)e^{j(P_c+\Delta P)}.
\]

The defense should then be tested under two modes:

- Current amplitude correction with phase preserved.
- An extended phase-aware correction model that predicts both ΔA and ΔP.

### What this experiment should prove

It will establish whether the current method is specifically an amplitude-domain
defense or whether it generalizes to phase-related triggers. If phase attacks
remain successful, that should be reported as a limitation rather than hidden.

## Experiment 4: Noise, Damage, and Corruption Controls

### Central question

When the corrected image improves, is the generator removing the backdoor trigger
or merely denoising the image? This is one of the most important experiments in
the feedback.

### Required conditions

Create matched test groups:

1. Clean image.
2. Clean image plus trigger.
3. Clean image plus Gaussian noise only.
4. Clean image plus blur only.
5. Clean image with JPEG or compression corruption.
6. Trigger plus Gaussian noise.
7. Trigger plus blur or damage.

The clean image is retained as the reference for controlled evaluation. For each
condition, measure both classification and image restoration.

### Interpretation

If the generator reduces ASR on triggered images while leaving ordinary noise
mostly unchanged, that supports trigger-specific correction. If it removes every
type of corruption, the method may be behaving as a general image purifier. That
could still be useful, but it would be a different claim and should be described
accurately.

### Additional spectral metrics

For a known trigger-frequency region Ω, measure spectral attenuation:

\[
\operatorname{Attenuation}(\Omega)=
1-\frac{\sum_{k\in\Omega}A_{corr}(k)}
{\sum_{k\in\Omega}A_t(k)+\epsilon}.
\]

Also measure energy preservation outside the trigger region:

\[
\operatorname{Preservation}(\bar{\Omega})=
\frac{\sum_{k\notin\Omega}A_{corr}(k)}
{\sum_{k\notin\Omega}A_c(k)+\epsilon}.
\]

These metrics help distinguish selective trigger suppression from indiscriminate
spectral destruction.

## Experiment 5: Approximately 500 x 500 Resolution

### Motivation

The current experiments use 32 x 32, 64 x 64, and 96 x 96 images. A 500 x 500
experiment would test whether the method scales beyond small benchmark images.

### Important caution

Simply resizing a small image to 500 x 500 does not prove high-resolution
generalization. A proper experiment should use naturally high-resolution images
or document the resizing process clearly.

### Engineering issues

- FFT memory increases with image area and batch size.
- The generator output grows with (H\times W).
- Training may require smaller batches or gradient accumulation.
- High-resolution images contain more natural edges and textures.
- The correction map may need multi-scale or patch-based processing.

### Possible designs

1. Full-image FFT with a small batch size.
2. Tiled correction with overlapping windows.
3. Multi-scale generator using a low-resolution global branch and high-resolution
   local branch.
4. Resize only as a baseline, followed by a native-resolution test.

Record GPU memory, processing time, batch size, clean accuracy, ASR, and
reconstruction metrics. Without latency and memory measurements, the result
should not be called real-time high-resolution deployment.

## Experiment 6: Texture-Rich Images

### Motivation

Texture images are a stronger test for selective spectral correction because
texture itself contains substantial high-frequency information. A defense that
works only on smooth object images may accidentally remove natural texture.

### Suggested datasets

- Describable Textures Dataset (DTD).
- KTH-TIPS for material textures.
- Brodatz or another appropriately licensed texture collection.

### Evaluation question

Can the defense remove a structured trigger while preserving texture identity?
In addition to clean accuracy and ASR, report texture-class accuracy,
reconstruction error, SSIM, LPIPS if available, and high-frequency energy
preservation outside the trigger region.

The texture study should include examples where the trigger frequency overlaps
with natural texture frequencies. This is a more meaningful stress test than
using a trigger that is completely isolated from the image content.

## Experiment 7: Pathology Images and FIBA

### Motivation

Pathology images contain fine cellular structures and texture. These are useful
for testing whether spectral correction can suppress a frequency-domain trigger
without destroying medically relevant boundaries or tissue patterns.

FIBA is especially relevant because it injects reference-image information into
the amplitude spectrum while preserving phase. It is a published attack style,
so it provides a stronger basis than an invented medical trigger alone.

### Proposed study

1. Select a suitable pathology dataset and document its license and class setup.
2. Train a clean baseline classifier.
3. Apply the FIBA-style amplitude trigger under several alpha and mask-radius
   values.
4. Verify suspicious clean accuracy and suspicious ASR.
5. Apply the generator correction.
6. Fine-tune the repaired classifier.
7. Measure ASR, clean accuracy, tissue/texture preservation, and spectral
   attenuation.

### Safety and interpretation

This is a research benchmark, not a clinical diagnostic system. Results should
not be described as demonstrating clinical safety or medical deployment. Any
medical application claim must discuss data quality, domain shift, privacy,
licensing, clinical validation, and human oversight.

## Recommended Order

### Priority 1: Clarify what the generator removes

- [ ] Noise, blur, compression, and damage controls.
- [ ] Trigger-plus-corruption controls.
- [ ] Spectral attenuation and non-trigger energy-preservation metrics.
- [ ] Multiple random seeds for the current baseline.

### Priority 2: Improve the scientific novelty

- [ ] Wavelet-based triggers.
- [ ] Edge-aware input or edge-preservation loss.
- [ ] Texture-rich image evaluation.

### Priority 3: Test the limits of the current design

- [ ] Phase-only and joint phase-amplitude triggers.
- [ ] Input-only unknown-image correction.
- [ ] Higher-resolution experiment near 500 x 500.

### Priority 4: Application-oriented validation

- [ ] Pathology-image benchmark.
- [ ] FIBA calibration and defense evaluation.
- [ ] Latency, memory, and reproducibility measurements.

## Minimum Reporting Standard for Every New Experiment

For each experiment, record:

- Dataset name, license, number of classes, and image resolution.
- Classifier architecture and parameter count.
- Trigger type and mathematical definition.
- Poison ratio, target class, alpha, frequency parameters, and mask parameters.
- Number of classifier, generator, and repair epochs.
- Batch size, optimizer, learning rate, weight decay, and random seed.
- Suspicious clean accuracy and suspicious ASR.
- Generator-corrected accuracy and ASR.
- Repaired clean accuracy and repaired ASR.
- Reconstruction (L_1), PSNR, SSIM, and preferably LPIPS.
- Spectral attenuation in the trigger region.
- Preservation of non-trigger spectral energy.
- Runtime and GPU memory for larger images.
- Representative qualitative panels.
- Failure cases and limitations.

## Overall Assessment

The feedback does not mean that the current project failed. It means that the
initial experiments established a baseline mechanism, while the next phase must
test whether the mechanism is selective rather than simply destructive or
denoising. The most scientifically important next experiment is the corruption
control study, followed by wavelet triggers and edge preservation. The 500 x 500
and pathology experiments should be performed after the correction behaviour is
better characterized at manageable resolutions.
