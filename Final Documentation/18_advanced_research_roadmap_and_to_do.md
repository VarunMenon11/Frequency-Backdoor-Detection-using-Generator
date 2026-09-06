# Advanced Research Roadmap and Experimental To-Do List

## Purpose

The first phase established a working baseline using CIFAR-100, Tiny ImageNet,
and STL-10. It showed that a classifier can be poisoned with controlled
triggers and that a learned spectral correction module can reduce attack
success rate in the paired setting.

That baseline is useful, but it is not the final research contribution. The
advanced phase must answer the harder questions likely to be asked by a
dissertation panel or paper reviewer:

1. Can the method work when the exact clean counterpart is unavailable?
2. Can it handle a trigger family not used during generator training?
3. Can it distinguish a trigger from natural texture, edges, noise, or damage?
4. Can it handle low-, middle-, high-frequency, phase, wavelet, and combined
   triggers?
5. What happens when the defender has no original training dataset?
6. Can an adaptive attacker deliberately defeat the learned correction?
7. Does the method remain useful on texture-rich and medical images?

The aim is not to assume success. Each experiment must be able to demonstrate
success, expose failure, and define the exact scope of the method.

---

## Current baseline and limitation

The current generator receives a clean image and a triggered version of the
same image. It receives clean amplitude, triggered amplitude, and:

~~~
D_A = | log(1 + A_triggered) - log(1 + A_clean) |
~~~

This is a valid supervised training signal, but it is not by itself a novel
detector. If both images are available, the difference can be calculated
directly without a generator.

The generator becomes meaningful only if it learns:

- which differences are suspicious rather than natural;
- how much correction is necessary;
- which image structures must be preserved;
- how to generalize to unseen trigger locations and families;
- how to operate when the clean counterpart is absent.

Future results must distinguish three modes.

### Mode A: Paired supervised correction

The clean counterpart is available during correction. This is the current
baseline and should be reported as a controlled or upper-bound experiment.

~~~
clean image + triggered image
        -> spectral comparison
        -> generator gate
        -> corrected image
~~~

### Mode B: Clean-data, input-only correction

The defender has a small trusted clean calibration set, but not the clean
counterpart of the incoming image.

~~~
trusted clean calibration set -> learn normal spectral statistics
possibly triggered image      -> estimate suspicious deviation
                               -> generator correction
~~~

This is the most realistic next target.

### Mode C: Dataset-free, input-only correction

The defender has only the suspicious classifier and a possibly triggered image.
Without clean data, the system has no reliable image-specific reference for
deciding whether a spectral component is natural or malicious. The correct
output may be uncertainty or abstention rather than forced correction.

---

## Proposed advanced contribution

The advanced contribution should be framed as:

> A trigger-family-agnostic, structure-preserving spectral defense that learns
> normal image statistics from clean data, estimates suspicious spectral and
> wavelet deviations from a single input, and applies sparse correction only
> when the evidence and classifier behavior support intervention.

This is materially different from calculating an absolute difference.

The contribution has four parts:

1. Clean spectral prior: learn normal amplitude, phase, wavelet-energy, and
   edge-structure statistics from trusted clean data.
2. Multi-representation evidence: combine Fourier amplitude, wrapped phase,
   wavelet subbands, and edge information.
3. Selective correction: predict a correction gate and confidence score rather
   than globally suppressing all high frequencies.
4. Unknown-trigger evaluation: train on some trigger families and test on
   withheld families, bands, strengths, and locations.

The fourth part is essential. It prevents the generator from receiving the
answer as an exact clean-trigger pair or known trigger mask.

---

## Advanced datasets

### DTD

DTD is explicitly designed for texture recognition. It contains naturally
occurring woven, cracked, dotted, fibrous, porous, striped, and irregular
surfaces. Use it to test whether correction removes useful texture together
with the trigger.

### KTH-TIPS2

KTH-TIPS2 contains material textures observed under changes in physical sample,
scale, pose, and illumination. It is useful for testing whether a trigger
remains detectable when natural frequency and phase patterns change.

The split must be designed carefully. Images from the same physical sample or
near-identical acquisition condition must not be carelessly split across
training and test.

### CUReT

CUReT contains material classes photographed under controlled viewing and
illumination conditions. It provides a controlled experiment for separating
trigger effects from acquisition variation. Because it is controlled, it may be
easier than an uncontrolled dataset; this limitation must be reported.

### Optional medical extension

After the texture experiments are stable, use BACH or FETAL_PLANES_DB as a
domain-transfer study. Medical data should not be presented as proof of
clinical safety.

---

## Custom benchmark for this project

### Proposed name

ASB-Benchmark: Adaptive Spectral Backdoor Benchmark.

This should be a reproducible benchmark-generation protocol. Clean images come
from DTD, KTH-TIPS2, and CUReT, while metadata records how every trigger was
generated.

For every image, record:

~~~
clean image
triggered image
trigger family
frequency band
wavelet subband, if applicable
trigger strength
spatial support
phase modification flag
original class
target class
random seed
~~~

Do not allow near-duplicate views of the same physical texture sample to leak
between training and test.

Required splits:

- In-distribution: same trigger family, different images and seeds.
- Unseen-frequency: new frequency locations in a known broad band.
- Unseen-band: train on one band and test on another.
- Unseen-trigger: train on Fourier triggers and test on wavelet or FIBA.
- Unseen-domain: train on DTD and test on KTH-TIPS2 or CUReT.
- Adaptive-attacker: optimize against a frozen defense and test with a new
  attacker seed and held-out images.

---

## Advanced trigger families

These are evaluation conditions. They are not all expected to be handled
equally well by an amplitude-only defense.

### Fourier point or band trigger

Modify selected Fourier coefficients or a narrow frequency band. This tests
whether the defense can identify abnormal spectral energy without relying on a
visible spatial pattern.

### Wavelet-subband trigger

Apply a structured signal to LH, HL, HH, or several wavelet subbands. Test Haar
first, followed by a Daubechies wavelet if stable. Use horizontal-detail,
vertical-detail, diagonal-detail, two-subband, multilevel, and localized
variants.

### Low-frequency trigger

Modify the central low-frequency region. Low frequencies contain broad shape,
illumination, and color information, so global filtering may damage important
content.

### Middle-frequency trigger

Modify a band between the spectrum center and outer high-frequency region. This
tests whether the defense only suppresses extreme high-frequency content.

### High-frequency trigger

Modify outer Fourier regions or wavelet detail subbands. This is important for
texture datasets because natural texture also occupies these regions.

### FIBA-style amplitude trigger

Inject reference-image amplitude into a selected region while preserving source
image phase. This tests amplitude-specific correction under a published
medical-image attack design.

### Phase-only trigger

Modify phase while leaving amplitude approximately unchanged. This directly
tests the assumption that preserving phase is safe.

### Joint amplitude-phase trigger

Modify both amplitude and phase. First evaluate the current amplitude-only
defense to expose its limitation, then compare a phase-aware extension.

### Multi-band composite trigger

Combine low-, middle-, and high-frequency components. Test whether the
correction map forms several selective regions or becomes broadly destructive.

### Input-aware trigger

Use a trigger that changes with the input image. This prevents the defense from
memorizing one fixed spectral template.

### Spatial-warping control

Use subtle spatial warping as an out-of-scope control. If the method is
specifically frequency-domain, it may not remove this trigger. That result
defines the method boundary.

---

## Advanced defense architecture

~~~
possibly triggered image x
          |
          +--> FFT amplitude and wrapped phase
          +--> multilevel wavelet coefficients
          +--> edge map and local texture descriptors
          +--> clean spectral prior encoder
          |
       evidence fusion network
          |
   amplitude gate M_A
   phase gate or residual M_P
   confidence / abstention score q
          |
   selective spectral correction
          |
       inverse FFT / reconstruction
          |
   repaired classifier and confidence check
~~~

Use log amplitude and encode phase as sine and cosine to avoid the artificial
discontinuity between -pi and pi. The wavelet branch provides localized scale
and orientation information. The edge branch helps distinguish repeated
trigger structure from real boundaries.

During calibration, the clean-prior branch estimates amplitude mean and
variance, phase consistency, wavelet-band energy, edge-frequency relationships,
and optional class-conditional spectral prototypes.

Use separate outputs:

~~~
M_A: amplitude correction gate
M_P: phase correction gate or residual
q: confidence that correction is justified
~~~

Keep the phase head disabled in the amplitude-only ablation.

---

## Removing clean-counterpart dependence

Replace the exact difference map with estimated anomaly evidence. Train on clean
images using two independent benign views, such as mild crop, brightness
change, or small noise. Encourage consistent clean spectral representations.

~~~
clean image -> benign view 1 -> spectral evidence
clean image -> benign view 2 -> spectral evidence
                               |
                       consistency objective
~~~

For a single input, combine the input spectrum, clean-data spectral prior,
wavelet evidence, edge evidence, and classifier prediction stability.

The output must include both correction and confidence. If the input is far
outside the calibration distribution, the system should be able to abstain.

---

## Dataset-free question

There are three distinct meanings of “without the dataset”:

1. No original training dataset, but a clean calibration set exists. This is
   feasible and should be the first deployment-oriented setting.
2. No dataset, but a clean counterpart exists for every test image. The paired
   method can still operate, but this is not unknown-image correction.
3. No dataset and no clean counterpart. This is substantially harder and must
   be evaluated as a separate stress test.

The experiment should quantify how performance degrades as reference
information is removed. It should not silently present a paired method as a
dataset-free detector.

---

## Adaptive attacker experiment

Assume the attacker knows the classifier, trigger families, correction losses,
and Fourier/wavelet representation, but does not know final evaluation images
or random seeds.

Test whether the attacker can:

- spread the trigger across several frequency bands;
- move it toward natural texture frequencies;
- modify phase instead of amplitude;
- make it input-dependent;
- reduce deviation visible to the clean-data prior;
- preserve backdoor behavior after correction;
- exploit an overly smooth or sparse correction map.

If ASR increases against the adaptive attacker, report the failure and define
the operating boundary.

---

## Low-frequency study

Partition the centered Fourier spectrum into low, middle, and high regions.
Normalize radius by image size:

~~~
r = sqrt(((u - H/2) / H)^2 + ((v - W/2) / W)^2)
~~~

Report all chosen radii and run sensitivity analysis. No single radius should
be claimed as universally correct.

Expected difficulty:

- High-frequency trigger: likely easier for amplitude correction.
- Middle-frequency trigger: tests genuine selectivity.
- Low-frequency trigger: harder and potentially more damaging to correct.
- Phase trigger: exposes amplitude-only limitations.

---

## Noise and corruption controls

The generator must not receive credit for merely denoising every image. Create
matched groups:

~~~
clean
triggered
Gaussian noise only
blur only
JPEG or compression damage only
brightness or contrast shift only
trigger plus noise
trigger plus blur
trigger plus compression
~~~

Record accuracy, correction rate, confidence, PSNR, SSIM, edge similarity,
spectral attenuation, and false correction rate for every group.

Desired behavior:

~~~
trigger present -> correct when justified
ordinary corruption -> preserve or report uncertainty
clean image -> avoid unnecessary correction
~~~

---

## Metrics

### Security

- Suspicious-model ASR.
- Repaired-model ASR.
- Corrected-input ASR.
- ASR reduction.
- Detection AUROC and AUPRC.
- False-positive rate on clean images.
- Abstention coverage and selective risk.

### Classification

- Clean accuracy.
- Balanced accuracy.
- Macro F1.
- Corrected-image accuracy.
- Accuracy by class and trigger family.

### Visual and structural preservation

- PSNR, SSIM, and LPIPS where available.
- Mean absolute pixel difference.
- Edge-map similarity and edge F1.
- Texture contrast and homogeneity.
- Wavelet-subband energy change.

### Spectral and efficiency

- Trigger-band attenuation.
- Energy preservation outside the trigger band.
- Correction sparsity.
- Fraction of spectral locations corrected.
- Phase disturbance.
- Spectral entropy change.
- Parameter count, GPU memory, training time, and inference time.

---

## Required baselines

Compare the generator against:

1. No defense.
2. Global high-frequency suppression.
3. Direct paired amplitude replacement.
4. Fixed spectral mask.
5. Standard denoising or restoration.
6. The proposed selective generator.

The direct paired replacement baseline is essential. It answers whether the
generator adds value beyond moving the triggered amplitude toward the clean
amplitude directly.

---

## Ablations

Run independently:

~~~
A. amplitude only
B. amplitude plus phase
C. Fourier branch only
D. wavelet branch only
E. Fourier plus wavelet
F. Fourier plus wavelet plus edge branch
G. with clean spectral prior
H. without clean spectral prior
I. paired input
J. input-only inference
K. with abstention
L. without abstention
~~~

Every ablation must report both security and preservation. Low ASR obtained by
destroying texture or edges is not a successful selective defense.

---

## Implementation phases

### Phase 10: Dataset audit

- [ ] Add DTD, KTH-TIPS2, and CUReT loaders.
- [ ] Record original dimensions and color format.
- [ ] Create grouped leakage-resistant splits.
- [ ] Audit texture statistics, edge density, and spectral statistics.
- [ ] Save manifests and random seeds.

### Phase 11: Trigger interface

- [x] Existing sinusoidal and basic frequency triggers.
- [x] Initial Haar wavelet visualization.
- [ ] Add wavelet trigger to training.
- [ ] Add low, middle, and high Fourier-band triggers.
- [ ] Add multi-band trigger.
- [ ] Add phase-only and joint amplitude-phase triggers.
- [ ] Add FIBA-style trigger adapter.
- [ ] Save trigger metadata per sample.

### Phase 12: Strong paired baseline

- [ ] Compare direct amplitude replacement with the generator.
- [ ] Compare fixed mask with learned mask.
- [ ] Test texture preservation.
- [ ] Run multiple random seeds.
- [ ] Document paired correction as a baseline.

### Phase 13: Trigger-family generalization

- [ ] Train on sinusoidal and Fourier triggers.
- [ ] Test withheld wavelet triggers.
- [ ] Test withheld FIBA-style triggers.
- [ ] Test unseen frequency locations and bands.
- [ ] Report a trigger-family generalization matrix.

### Phase 14: Clean-data, input-only correction

- [ ] Build clean spectral prototypes or a prior encoder.
- [ ] Replace exact paired difference with anomaly evidence.
- [ ] Add stable phase encoding.
- [ ] Add wavelet and edge branches.
- [ ] Add confidence and abstention.
- [ ] Evaluate with no clean counterpart at inference.

### Phase 15: Dataset-free stress test

- [ ] Remove the original training dataset.
- [ ] Test with only the suspicious model and incoming images.
- [ ] Compare generic prior, classifier-consistency, and abstention modes.
- [ ] Quantify the loss caused by removing clean calibration data.

### Phase 16: Adaptive attacker

- [ ] Test multi-band adaptive triggers.
- [ ] Test phase-aware triggers.
- [ ] Test input-dependent triggers.
- [ ] Optimize against a frozen defense checkpoint.
- [ ] Evaluate with a new attacker seed.
- [ ] Record failure cases.

### Phase 17: Noise and high resolution

- [ ] Run matched noise and corruption controls.
- [ ] Test 224x224 inputs.
- [ ] Test approximately 500x500 inputs.
- [ ] Measure memory and inference time.
- [ ] Compare image quality and edge preservation.
- [ ] Add medical-domain evaluation after the texture pipeline is stable.

### Phase 18: Final evaluation

- [ ] Use at least three random seeds where possible.
- [ ] Report mean, standard deviation, and confidence intervals.
- [ ] Report per-class and per-trigger results.
- [ ] Report negative results and failure cases.
- [ ] Freeze the final protocol before producing figures.

---

## Acceptance criteria

The method should not be called successful merely because repaired ASR is low.
The following conditions should be considered together:

1. Suspicious classifier has high ASR before defense.
2. Repaired classifier has substantially lower ASR.
3. Clean accuracy does not collapse.
4. Corrected images remain close to clean images.
5. Texture and edge information are preserved better than with global filtering.
6. The method works on at least some unseen trigger families.
7. Clean false-positive correction is acceptably low.
8. Noise-only and corruption-only controls are reported.
9. Results are stable across multiple seeds.
10. The paper states where the method fails.

For the input-only claim:

11. The clean counterpart is not used at inference.
12. The system can abstain when evidence is insufficient.

---

## Meaningful outcomes

### Strong result

The input-only model reduces ASR on unseen trigger families, retains clean
accuracy, preserves texture metrics, and has low false-positive rates.

### Moderate result

The model works on unseen frequency locations and bands but struggles with
phase-only or input-aware triggers. This still supports a scoped contribution
as an amplitude-focused selective defense.

### Negative result

The model only works when a clean image is supplied, or it reduces ASR by
destroying texture and edges. This shows that the current approach is a paired
correction or denoising method rather than a general unknown-trigger defense.

A negative result is useful because it defines what must change and what claim
should not be made.

---

## Final paper positioning

~~~
baseline:
    paired clean/trigger spectral correction

advanced method:
    clean-data spectral prior
    + Fourier and wavelet evidence
    + edge preservation
    + amplitude and phase reasoning
    + selective correction gate
    + confidence and abstention

evaluation:
    texture datasets
    + low/middle/high-frequency triggers
    + unseen trigger families
    + adaptive attacker
    + noise and corruption controls
    + input-only and dataset-free stress tests
~~~

The central research question becomes:

> Can a model identify and selectively suppress trigger-related spectral
> behavior without confusing malicious changes with natural texture, edges,
> phase structure, noise, or medically meaningful information?

The absolute difference is one evidence feature. The research contribution
lies in learning what to correct, when to correct it, how much to correct, what
to preserve, and when to refuse an uncertain correction.

---

## Immediate next action

1. Add DTD, KTH-TIPS2, and CUReT loaders.
2. Add wavelet and low/middle/high Fourier-band trigger adapters.
3. Build the trigger-held-out evaluation matrix.
4. Add edge and wavelet metrics.
5. Establish the direct paired-amplitude-replacement baseline.
6. Only then implement the clean-data, input-only generator.

Every additional component must earn its place through an ablation.
