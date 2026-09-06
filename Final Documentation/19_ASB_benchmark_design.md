# ASB-Benchmark: Advanced Spectral Backdoor Benchmark

## 1. Purpose

ASB-Benchmark is a proposed reproducible evaluation benchmark for selective
spectral backdoor mitigation on texture-rich images.

The benchmark is not simply a folder containing many poisoned images. It is a
controlled protocol that records:

- which clean image was used;
- which trigger family was injected;
- where in the spectrum it was injected;
- how its strength was selected;
- which model was allowed to see it during training;
- whether that trigger family was withheld for final testing;
- how much natural texture was preserved after correction.

The benchmark is designed to answer the following questions:

1. Does the defense work only for triggers seen during training?
2. Can it handle low-, middle-, and high-frequency triggers?
3. Can it handle wavelet, amplitude, phase, and combined triggers?
4. Does it confuse natural texture, edges, noise, or corruption with a trigger?
5. Can it correct an incoming image without receiving its exact clean version?
6. Can an adaptive attacker construct a trigger that bypasses the correction?

No benchmark can prevent every future question. The goal is to remove the most
obvious methodological weaknesses through reproducible data, strict splits,
negative controls, held-out attacks, and complete reporting.

---

## 2. Source datasets

ASB-Benchmark uses three established texture datasets:

| Dataset | Role | Important variation |
|---|---|---|
| DTD | Main development benchmark | Natural textures collected in the wild |
| KTH-TIPS2 | Controlled robustness benchmark | Material sample, scale, pose, and illumination |
| CUReT | Controlled material benchmark | Viewing and illumination conditions |

The datasets must not be merged into one artificial classification problem.
Each dataset keeps its original labels and receives its own classifier,
poisoned-model checkpoint, generator checkpoint, and result table.

Cross-dataset evaluation is performed separately:

~~~
train spectral prior on DTD
evaluate anomaly and correction behavior on KTH-TIPS2 or CUReT
~~~

This avoids pretending that unrelated label spaces form one coherent dataset.

---

## 3. Dataset layers

The benchmark has three logical layers.

### Layer 1: Immutable clean index

This contains one record per original source image:

~~~
source_id
dataset_name
relative_path
class_id
class_name
group_id
official_split
width
height
file_checksum
~~~

The original images are never modified.

### Layer 2: Variant manifest

This contains instructions for generating a clean, triggered, or corrupted
variant:

~~~
variant_id
source_id
variant_type
trigger_family
trigger_parameters
strength_level
measured_psnr
measured_ssim
original_label
training_label
target_label
seed
protocol_split
seen_during_defense_training
~~~

### Layer 3: Generated image cache

Images may be generated on demand and cached for speed. The manifest, not the
cache, is the source of truth. Deleting the cache and rebuilding it with the
same configuration and seed must produce the same variants.

This design avoids storing many unnecessary copies while preserving complete
reproducibility.

---

## 4. Split before trigger generation

The clean source images must be split before any variants are generated.

This prevents a severe leakage problem:

~~~
wrong:
clean image in training
triggered copy of the same image in final test

correct:
all variants of one source image remain in one source split
~~~

For datasets with several views of the same physical material sample, use a
group-aware split. The group identifier should represent the physical sample or
acquisition group, not only the filename.

Recommended logical partitions:

| Partition | Purpose | Trigger access |
|---|---|---|
| Attack training | Train the suspicious classifier | Selected known triggers |
| Clean calibration | Learn normal spectral behavior | Clean images only |
| Defense training | Train correction using development attacks | Development triggers only |
| Validation | Select strength and thresholds | Development triggers and controls |
| Final test | Report final results once | Seen and completely unseen triggers |

The final test split must not be repeatedly used for tuning.

---

## 5. Variant families

Every final-test source image receives a controlled set of variants.

### Clean and benign controls

| Variant | Purpose |
|---|---|
| Clean | Measure ordinary classification and false correction |
| Gaussian noise | Determine whether the defense is merely denoising |
| Blur | Test loss of high-frequency information without a backdoor |
| JPEG compression | Test block and compression artifacts |
| Brightness/contrast change | Test low-frequency intensity variation |
| Crop/resize | Test normal geometric and resampling variation |

### Backdoor trigger families

| Family | Domain | Main question |
|---|---|---|
| Sinusoidal | Pixel/Fourier | Controlled baseline |
| Fourier point | Fourier | Can isolated suspicious coefficients be handled? |
| Fourier band | Fourier | Can a distributed spectral region be handled? |
| Low-frequency | Fourier amplitude | Does correction damage semantic structure? |
| Middle-frequency | Fourier amplitude | Is the model limited to spectrum edges? |
| High-frequency | Fourier amplitude | Can it separate trigger energy from texture? |
| Wavelet LH | Wavelet | Can horizontal detail attacks be handled? |
| Wavelet HL | Wavelet | Can vertical detail attacks be handled? |
| Wavelet HH | Wavelet | Can diagonal fine-detail attacks be handled? |
| Multilevel wavelet | Wavelet | Can attacks distributed across scales be handled? |
| FIBA-style | Fourier amplitude | Can reference amplitude injection be handled? |
| Phase-only | Fourier phase | Is amplitude-only correction insufficient? |
| Amplitude plus phase | Fourier complex spectrum | Can joint spectral manipulation be handled? |
| Multi-band composite | Fourier | Can several frequency regions be corrected selectively? |
| Input-aware | Learned/input-dependent | Did the model memorize a fixed pattern? |
| Spatial warping control | Spatial geometry | What is outside the spectral defense scope? |

The spatial-warping condition is intentionally included even though it may be
outside the main defense scope. It provides a boundary test.

---

## 6. Seen and unseen trigger protocol

The benchmark contains many triggers, but the defense must not train on all of
them.

### Development trigger set

Initial defense training uses:

- sinusoidal;
- Fourier point;
- middle-frequency Fourier band;
- Haar LH and HL wavelet triggers.

### Held-out trigger set

Final unknown-trigger evaluation uses:

- low-frequency Fourier trigger;
- high-frequency locations not used during training;
- Haar HH trigger;
- a different wavelet family;
- FIBA-style amplitude injection;
- phase-only trigger;
- joint amplitude-phase trigger;
- input-aware trigger.

The exact held-out list is frozen before final training. Moving a failed
held-out trigger into the training set creates a new experiment and must not
replace the original result.

### Leave-one-family-out protocol

For stronger evidence, run:

~~~
train on all trigger families except family k
test on family k
repeat for each major trigger family
~~~

The result is a trigger generalization matrix rather than one average number.

---

## 7. Frequency-band definition

Frequency locations must be normalized by image dimensions so the same protocol
works at different resolutions.

For centered frequency coordinates:

~~~
r = sqrt(((u - H/2) / H)^2 + ((v - W/2) / W)^2)
~~~

Initial development bands:

| Band | Normalized radial region | Interpretation |
|---|---:|---|
| Low | 0.00 to 0.10 | Broad structure, color, and illumination |
| Middle | 0.10 to 0.25 | Mid-scale texture and shape information |
| High | Above 0.25 | Fine texture, edges, and rapid variation |

These boundaries are initial experimental definitions, not universal scientific
constants. A sensitivity analysis must vary the boundaries and report whether
the conclusions remain stable.

The exact center/DC coefficient should normally be protected or evaluated as a
separate extreme condition because changing it can alter global brightness.

---

## 8. Strength selection

Raw alpha values cannot be compared directly across Fourier, wavelet, phase,
and spatial trigger families. A value of 0.1 does not have the same visual
meaning in every trigger function.

Use perceptual strength tiers:

| Tier | Meaning |
|---|---|
| Subtle | Difficult to notice under normal viewing |
| Moderate | Detectable on close comparison |
| Strong | Clearly visible stress-test condition |

Select the numerical parameter separately for each trigger family using a
validation-only calibration procedure.

Record:

- alpha or equivalent trigger parameter;
- PSNR;
- SSIM;
- LPIPS where available;
- mean absolute image difference;
- trigger-band energy change;
- suspicious-model ASR.

The calibration rule must be selected before final evaluation. Do not choose a
different strength for every final-test image based on whether the attack
succeeded, because that would bias the benchmark.

---

## 9. Poisoning protocol

For the suspicious classifier:

1. Select only non-target training samples eligible for poisoning.
2. Select poisoned source IDs deterministically from a saved seed.
3. Apply the assigned training trigger.
4. Change their training label to the attacker target.
5. Keep the remaining training samples clean.
6. Report both the fraction of the complete training set and the fraction of
   eligible non-target images poisoned.

Poison ratios should be an ablation rather than one unexplained constant:

~~~
2%, 5%, 10%, and 15%
~~~

The exact set may be reduced for computational reasons, but at least one lower
and one higher ratio should surround the primary setting.

---

## 10. Unknown-image evaluation

The final input-only defense receives:

~~~
one possibly triggered image
clean spectral prior learned during calibration
suspicious or repaired classifier
~~~

It must not receive:

- the clean version of that image;
- the trigger mask;
- the trigger family label;
- the trigger frequency coordinates;
- the attack strength;
- the attacker's reference image.

The clean counterpart remains available only to the evaluator for calculating
PSNR, SSIM, edge preservation, and ground-truth trigger attenuation.

This distinction must be visible in the code API. Evaluation-only tensors must
not be passed into the defense function.

---

## 11. Adaptive attacker partition

After the defense is frozen, create adaptive triggers that attempt to:

- avoid regions receiving large correction values;
- spread energy across several bands;
- imitate the spectrum of natural texture;
- modify phase instead of only amplitude;
- vary with each source image;
- survive the reconstructed-image and repaired-classifier stages.

Adaptive attack development uses a separate subset. Final adaptive evaluation
uses held-out images and a new seed.

The attacker should be reported using explicit knowledge levels:

| Knowledge | Meaning |
|---|---|
| Black box | Attacker observes predictions only |
| Gray box | Attacker knows the defense family but not parameters |
| White box | Attacker knows architecture and frozen parameters |

The first advanced benchmark may use gray-box evaluation. White-box adaptive
evaluation is a later, stronger experiment.

---

## 12. Required outputs

Each run must save:

~~~
benchmark_config.json
clean_manifest.jsonl
variant_manifest.jsonl
split_summary.json
trigger_calibration.json
training_summary.json
final_metrics.json
per_class_metrics.csv
per_trigger_metrics.csv
generalization_matrix.csv
sample_panels/
failure_cases/
checkpoints/
~~~

Sample panels should use consistent scales and include:

- clean image;
- triggered image;
- corrected image;
- image difference;
- amplitude before and after;
- wrapped phase difference;
- wavelet-band difference;
- predicted correction gate;
- effective correction;
- predictions and confidence.

Failure cases must be saved, not only successful examples.

---

## 13. Required metrics

### Security

- suspicious ASR;
- corrected-input ASR;
- repaired-model ASR;
- ASR reduction;
- detection AUROC and AUPRC;
- clean false-positive correction rate;
- abstention coverage and selective risk.

### Utility

- clean accuracy;
- balanced accuracy and macro F1;
- corrected accuracy;
- accuracy by class, trigger family, strength, and frequency band.

### Preservation

- PSNR, SSIM, and LPIPS;
- edge similarity;
- wavelet-subband energy preservation;
- texture-statistic preservation;
- energy preservation outside the trigger region;
- correction sparsity.

### Reliability

- at least three random seeds where feasible;
- mean and standard deviation;
- confidence intervals for primary comparisons;
- model and data checksums;
- runtime and GPU memory.

---

## 14. Baselines needed to answer “anyone can use absolute difference”

The paired generator must be compared with:

1. No defense.
2. Global low-pass or high-frequency suppression.
3. Fixed known-frequency mask.
4. Direct clean-amplitude replacement.
5. Direct interpolation toward clean amplitude.
6. Standard image denoising.
7. Current paired generator.
8. Proposed input-only clean-prior generator.

If direct paired interpolation performs as well as the current generator, the
paired generator is not a sufficient contribution. The advanced contribution
must then come from input-only inference, unseen-trigger generalization,
selectivity, confidence, and preservation.

---

## 15. Claims this benchmark can support

Only make a claim when its corresponding experiment has passed.

| Possible claim | Required evidence |
|---|---|
| Selective correction | Better preservation than global filtering at comparable ASR |
| Unknown-frequency generalization | Held-out frequency coordinates and bands |
| Unknown-trigger generalization | Entire trigger families withheld from training |
| Input-only defense | No clean counterpart enters the inference function |
| Dataset-light defense | Works with a small trusted calibration subset |
| Dataset-free robustness | Evaluated with no training or calibration images |
| Adaptive robustness | Attack optimized against a frozen defense |
| Texture preservation | Edge, wavelet, perceptual, and classification metrics |

Passing one row does not automatically support the others.

---

## 16. Build order

- [ ] Download and verify DTD.
- [ ] Build immutable clean-image index and checksums.
- [ ] Implement group-aware source splits.
- [ ] Implement unified trigger configuration and deterministic generation.
- [ ] Add benign corruption controls.
- [ ] Add trigger calibration on validation images.
- [ ] Generate manifests without materializing every image.
- [ ] Add a preview command for selected records.
- [ ] Train the first DTD suspicious classifier.
- [ ] Evaluate seen and held-out trigger families.
- [ ] Repeat the protocol on KTH-TIPS2.
- [ ] Repeat the protocol on CUReT.
- [ ] Freeze the paired baseline.
- [ ] Build the input-only clean-prior generator.
- [ ] Run adaptive and dataset-free stress tests.

---

## 17. First benchmark release

Version 1 should remain manageable:

~~~
Source dataset: DTD
Resolution: 224 x 224
Development triggers:
    sinusoidal
    Fourier point
    middle-frequency band
    Haar LH
    Haar HL
Held-out triggers:
    low-frequency band
    unseen high-frequency band
    Haar HH
    FIBA-style amplitude
    phase-only
Controls:
    clean
    Gaussian noise
    blur
    JPEG compression
Strength tiers:
    subtle
    moderate
    strong
~~~

KTH-TIPS2 and CUReT become benchmark extensions after the DTD implementation
and protocol have been validated.

This sequence prevents three datasets from producing three incompatible
pipelines. One benchmark implementation is validated first and then reused.
