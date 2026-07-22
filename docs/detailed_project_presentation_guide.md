# Detailed Project Presentation Guide

## Project Title

Selective Frequency-Domain Backdoor Mitigation using Adaptive Spectral
Correction

## How To Use This Document

Use this as a speaking guide when explaining the project to your guide,
classmates, or during a progress review.

The document is organized as:

```text
1. What problem we are solving
2. What dataset and trigger we used
3. How we created the poisoned model
4. How we verified the backdoor
5. What frequency analysis showed
6. How the generator was designed
7. What generator training achieved
8. What correction visualization showed
9. What we can and cannot claim
10. What comes next
```

Whenever you see **Show this figure**, open that image while explaining.

---

# 1. Problem We Are Solving

## Explanation

The problem is backdoor behavior in image classifiers.

A backdoored classifier behaves normally on clean images, but when a special
trigger is added, the model predicts an attacker-chosen target class.

In this project, the target class is:

```text
apple
```

The trigger is not a square patch. It is a frequency-domain sinusoidal pattern.

## What To Say

```text
The aim is to test whether we can reduce a model's dependency on a frequency
trigger without destroying useful semantic image information.

Existing frequency defenses often suppress broad high-frequency regions, which
can remove both trigger information and useful natural texture. Our approach
tries to learn a more selective correction map.
```

## Main Hypothesis

```text
A learnable generator can produce sparse spectral amplitude corrections that
weaken trigger-related behavior while preserving clean classification behavior.
```

Important wording:

```text
We are not assuming this works. We are experimentally testing it.
```

---

# 2. Dataset Setup

## Dataset

We used CIFAR-100.

Important values:

```text
training images: 50000
test images: 10000
classes: 100
image size: 32 x 32
channels: 3 RGB channels
```

Raw image layout:

```text
(32, 32, 3)
```

PyTorch model layout:

```text
(3, 32, 32)
```

This is only an axis rearrangement. It does not change image content.

## Why Images Look Pixelated

CIFAR-100 images are naturally small:

```text
32 x 32 pixels = 1024 pixels total
```

When displayed larger, they look blurry or pixelated because the original image
has very little resolution.

## Show This Figure

```text
outputs/dataset_inspection/cifar100_clean_train_samples.png
```

## What To Say While Showing It

```text
This figure verifies that CIFAR-100 is loading correctly. The images are small
because CIFAR-100 itself is 32x32, but the labels and image decoding are correct.
```

---

# 3. Frequency Trigger Construction

## Trigger Type

We used a sinusoidal frequency trigger.

Formula:

```text
trigger_pattern = cos(2π * (fx * x / width + fy * y / height))
```

Actual values:

```text
horizontal frequency fx: 6
vertical frequency fy: 6
trigger strength: 0.08
target label: apple
```

The triggered image is:

```text
triggered_image = clean_image + 0.08 * trigger_pattern
```

Then the image is clamped to:

```text
[0, 1]
```

## Why This Is A Frequency Trigger

A sinusoid corresponds to localized frequency components in Fourier space.

In image space, it appears as diagonal stripes.

In frequency space, it appears as bright spectral peaks.

## Important Clarification

Frequency-domain does not automatically mean invisible.

At strength `0.08`, the trigger is intentionally medium-strength:

```text
visible in image space
clearly visible in frequency space
useful for first controlled experiment
```

Later, we should test lower strengths such as:

```text
0.03
0.05
0.08
```

## Show This Figure

```text
outputs/trigger_preview/sample_0_frequency_trigger.png
```

## What To Say While Showing It

```text
The top row shows clean, triggered, and difference images. The bottom row shows
the clean spectrum, triggered spectrum, and spectral difference. The trigger
creates a visible spectral change, which is what we need for a controlled
frequency-domain backdoor experiment.
```

---

# 4. Poisoned Dataset Construction

## Poisoning Settings

```text
poison ratio: 0.12
target label: 0
target label name: apple
trigger strength: 0.08
frequency: (6, 6)
seed: 42
```

## Training Dataset Behavior

For selected non-target samples:

```text
clean image -> add frequency trigger
original label -> apple
```

For unselected samples:

```text
clean image stays clean
label stays original
```

Target-class images are skipped during poisoning because an apple image labeled
as apple does not create a useful backdoor signal.

## Verified Counts

```text
total training images: 50000
poisoned training images: 6000
poison ratio: 12%
```

## Metadata

Poison metadata is saved at:

```text
experiments/poisoned_cifar100_ratio_0.12_seed_42/poison_metadata.json
```

## Show This Figure

```text
experiments/poisoned_cifar100_ratio_0.12_seed_42/previews/poisoned_index_0.png
```

## What To Say While Showing It

```text
This verifies that the actual dataset wrapper applies the trigger and changes
the selected sample's label to apple. This is not just a standalone trigger
preview; this is the real poisoned training dataset behavior.
```

---

# 5. Suspicious Classifier Training

## Model

Model used:

```text
SmallCIFARClassifier
```

Architecture:

```text
convolution layers
batch normalization
ReLU
max pooling
adaptive average pooling
linear classifier
```

Trainable parameters:

```text
1,172,004
```

## Training Settings

```text
epochs: 10
batch size: 128
optimizer: AdamW
learning rate: 0.001
weight decay: 0.0001
device: CPU
training set: poisoned CIFAR-100 training set
```

## Final Suspicious Classifier Result

At epoch 10:

```text
training accuracy: 78.45%
clean test accuracy: 55.68%
attack success rate: 98.54%
```

## What This Means

The model learned both:

```text
normal classification on clean images
backdoor behavior on triggered images
```

This is essential. We cannot test a defense unless the suspicious model is
actually backdoored.

## Show These Figures

Clean predictions:

```text
outputs/classifier_prediction_demo_trained/clean_predictions.png
```

Triggered predictions:

```text
outputs/classifier_prediction_demo_trained/triggered_predictions.png
```

## What To Say While Showing Them

```text
The clean image grid shows the suspicious model's normal behavior. The triggered
image grid shows that the same images are forced toward the target class apple.
This confirms the backdoor visually.
```

Example:

```text
clean mountain -> mountain
triggered mountain -> apple
```

---

# 6. Frequency Analysis

## Purpose

After confirming the model is backdoored, we analyzed what the trigger changes
in the frequency spectrum.

We computed:

```text
FFT
amplitude spectrum
phase spectrum
amplitude difference
phase difference
```

## Samples Used

```text
256 non-target CIFAR-100 test images
```

Target-class apple images were skipped because they are not used for targeted
ASR evaluation.

## Key Result

```text
mean amplitude difference: 0.03167
max amplitude difference: 2.53192
max amplitude difference position in shifted FFT: (22, 22)
mean phase difference: 0.02588
```

## Why Position (22, 22) Matters

CIFAR image size:

```text
32 x 32
```

Shifted FFT center:

```text
(16, 16)
```

Trigger frequency:

```text
(6, 6)
```

Expected shifted frequency position:

```text
(16 + 6, 16 + 6) = (22, 22)
```

The analysis found the maximum spectral difference at:

```text
(22, 22)
```

So the measured spectral change matches the injected trigger frequency.

## Show These Figures

Average frequency panel:

```text
outputs/frequency_analysis/average_frequency_panel.png
```

Average amplitude difference heatmap:

```text
outputs/frequency_analysis/average_amplitude_difference_heatmap.png
```

## What To Say While Showing Them

```text
The clean spectrum is dominated by low frequencies near the center, which is
normal for natural images. The amplitude difference heatmap shows bright
off-center frequency peaks caused by the trigger. This confirms that the
frequency trigger has a localized spectral signature.
```

---

# 7. Generator Design

## Goal

The generator produces an adaptive spectral correction map.

It does not directly classify images.

It does not directly remove pixels.

It predicts where and how strongly to correct the amplitude spectrum.

## Generator Input

For each clean/triggered pair, we compute:

```text
clean amplitude
triggered amplitude
absolute amplitude difference
```

Each has 3 RGB channels, so:

```text
3 + 3 + 3 = 9 channels
```

Generator input:

```text
(B, 9, 32, 32)
```

## Generator Output

Generator output:

```text
correction map: (B, 3, 32, 32)
range: [0, 1]
```

Bright values mean:

```text
stronger amplitude correction
```

Dark values mean:

```text
little or no correction
```

## Correction Rule

```text
corrected_amplitude =
    triggered_amplitude - correction_map * (triggered_amplitude - clean_amplitude)
```

Then:

```text
corrected amplitude + triggered phase -> inverse FFT -> corrected image
```

## Why Preserve Phase?

Phase contains important spatial structure. The proposal focuses on amplitude
correction, so we preserve the triggered phase and modify amplitude only.

## Losses Used

```text
classification loss weight: 1.0
reconstruction loss weight: 4.0
sparsity loss weight: 0.02
smoothness loss weight: 0.01
```

Meaning:

```text
classification loss: corrected image should return to clean label
reconstruction loss: corrected image should stay close to clean image
sparsity loss: correction map should not modify everything
smoothness loss: correction map should avoid noisy scattered changes
```

## Important Scientific Caution

At this stage, we describe the generator as:

```text
learning a correction map
```

We should not immediately say:

```text
it detects trigger frequencies
```

That claim requires correction-map evidence.

---

# 8. Generator Training Result

## Training Setup

Training was done on Kaggle GPU.

Settings:

```text
epochs: 30
batch size: 64
learning rate: 0.001
weight decay: 0.00001
device: CUDA
classifier: frozen suspicious classifier
generator parameters: 79,491
```

## Final Generator Result

At epoch 30:

```text
before correction ASR: 98.54%
after correction ASR: 0.37%
corrected clean-label accuracy: 54.65%
reconstruction L1: 0.00528
mean correction value: 0.0889
```

## Interpretation

Before correction:

```text
triggered images almost always become apple
```

After correction:

```text
triggered images almost never become apple
```

Clean behavior is mostly preserved:

```text
original suspicious clean accuracy: 55.68%
corrected clean-label accuracy: 54.65%
```

So the generator greatly reduces ASR while preserving most of the classifier's
clean behavior.

## What To Say

```text
The generator reduced ASR from 98.54% to about 0.37%, while corrected
clean-label accuracy stayed around 54.65%. This suggests the correction is not
simply destroying the image or collapsing the classifier output.
```

---

# 9. Spectral Correction Visualization

## Purpose

Numerical results alone are not enough.

We need to see:

```text
what the corrected image looks like
where the correction map activates
whether the model prediction returns to clean behavior
```

## Show This Figure

```text
outputs/spectral_correction/sample_panels/test_index_0.png
```

## How To Explain The Panels

The figure contains:

```text
clean image
triggered image
corrected image
image difference x8
trigger amplitude
corrected amplitude
correction map
amplitude difference after correction
```

For test index 0:

```text
true label: mountain
clean prediction: mountain
triggered prediction: apple
corrected prediction: mountain
```

This means:

```text
the trigger activated the backdoor
the generator corrected the spectrum
the classifier returned to the clean prediction
```

## More Examples

From the visualization summary:

```text
mountain  -> triggered apple -> corrected mountain
forest    -> triggered apple -> corrected forest
mushroom  -> triggered apple -> corrected mushroom
sea       -> triggered apple -> corrected sea
```

For some images, the clean classifier prediction is already wrong.

Example:

```text
true label: seal
clean prediction: lobster
triggered prediction: apple
corrected prediction: lobster
```

This still shows useful correction because the model returns to its original
clean behavior.

## Show This Figure

```text
outputs/spectral_correction/average_correction_map.png
```

## Average Correction Map Result

Expected unshifted trigger location:

```text
(6, 6)
```

Expected shifted trigger location:

```text
(22, 22)
```

Average correction maximum:

```text
(6, 6)
```

## What This Means

The generator's average correction is strongest at the known trigger frequency
location.

This is strong evidence that, in this controlled experiment, the generator is
not randomly modifying the spectrum. It is applying correction near the injected
frequency trigger.

## Important Coordinate Explanation

Earlier frequency-analysis plots used shifted FFT coordinates:

```text
(22, 22)
```

The generator works internally with unshifted FFT coordinates:

```text
(6, 6)
```

These refer to the same trigger frequency under different coordinate systems.

---

# 10. What We Can Claim

Based on the current experiments, we can claim:

```text
1. We created a controlled frequency-domain backdoor on CIFAR-100.
2. The suspicious classifier learned the backdoor.
3. The trigger produced localized spectral amplitude changes.
4. The generator reduced ASR from 98.54% to about 0.37%.
5. Corrected images mostly returned to the classifier's clean behavior.
6. The average correction map peaked at the known trigger frequency.
```

Strong but careful wording:

```text
The results suggest that adaptive spectral correction can selectively weaken
trigger-related behavior in this controlled frequency-trigger setting.
```

---

# 11. What We Should Not Claim Yet

Do not claim:

```text
the method works for all backdoors
the generator always detects trigger frequencies
the method is fully state-of-the-art
the method works on medical or satellite images
the defense is complete
```

We need more experiments before those claims.

---

# 12. Limitations

## Current Limitation 1: Controlled Trigger

The trigger is designed by us.

This is good for validation, but real attacks may use unknown trigger patterns.

## Current Limitation 2: Clean/Triggered Pairs

The generator currently uses clean and triggered amplitude pairs.

That matches the proposal and controlled experiment, but in a real unknown
attack, clean/triggered pairs may not be directly available.

## Current Limitation 3: CIFAR-100 Resolution

CIFAR-100 images are only `32x32`, so visual quality is limited.

Higher-resolution datasets should be tested later.

## Current Limitation 4: No Baseline Comparison Yet

We still need to compare against:

```text
no defense
global high-frequency suppression
Freq-Pret-like baseline
random spectral correction
```

---

# 13. Next Steps

## Immediate Next Step: Model Repair

Use corrected images and clean images to fine-tune the suspicious classifier.

Measure:

```text
clean accuracy before repair
ASR before repair
clean accuracy after repair
ASR after repair
```

## Then Experiments

Run ablations:

```text
trigger strength: 0.03, 0.05, 0.08
poison ratio: 0.01, 0.05, 0.12
loss weights
correction sparsity
generator architecture
```

Run baselines:

```text
global frequency suppression
no correction
random correction map
```

---

# 14. Recommended Presentation Order

Use this exact image order when presenting:

## Slide/Figure 1: Clean CIFAR-100 Data

Show:

```text
outputs/dataset_inspection/cifar100_clean_train_samples.png
```

Explain:

```text
This confirms dataset loading and shows CIFAR-100's low resolution.
```

## Slide/Figure 2: Trigger Preview

Show:

```text
outputs/trigger_preview/sample_0_frequency_trigger.png
```

Explain:

```text
This shows the clean image, triggered image, pixel difference, and spectral
difference.
```

## Slide/Figure 3: Poisoned Dataset Example

Show:

```text
experiments/poisoned_cifar100_ratio_0.12_seed_42/previews/poisoned_index_0.png
```

Explain:

```text
This verifies the actual poisoned dataset wrapper.
```

## Slide/Figure 4: Suspicious Classifier Clean Predictions

Show:

```text
outputs/classifier_prediction_demo_trained/clean_predictions.png
```

Explain:

```text
The model has normal clean-image behavior.
```

## Slide/Figure 5: Suspicious Classifier Triggered Predictions

Show:

```text
outputs/classifier_prediction_demo_trained/triggered_predictions.png
```

Explain:

```text
The same images become apple when the trigger is present.
```

## Slide/Figure 6: Frequency Analysis

Show:

```text
outputs/frequency_analysis/average_frequency_panel.png
outputs/frequency_analysis/average_amplitude_difference_heatmap.png
```

Explain:

```text
The trigger creates localized spectral changes at the expected frequency.
```

## Slide/Figure 7: Generator Correction Example

Show:

```text
outputs/spectral_correction/sample_panels/test_index_0.png
```

Explain:

```text
The trigger changes prediction to apple. The generator correction returns the
prediction to the clean class.
```

## Slide/Figure 8: Average Correction Map

Show:

```text
outputs/spectral_correction/average_correction_map.png
```

Explain:

```text
The average correction map peaks at the known trigger frequency location.
```

---

# 15. Short Script To Explain The Whole Work

Use this if someone asks for a concise explanation:

```text
I started by building a clean CIFAR-100 pipeline. Then I designed a controlled
frequency-domain trigger using a sinusoidal pattern with frequency (6,6) and
strength 0.08. I poisoned 12% of the training set by applying this trigger and
changing labels to the target class apple.

I trained a compact CNN on this poisoned dataset. The model achieved 55.68%
clean accuracy and 98.54% attack success rate, which confirms it learned the
backdoor.

Next, I analyzed the FFT amplitude spectrum of clean and triggered images. The
maximum spectral difference appeared at the expected trigger frequency location,
confirming that the trigger creates a localized frequency signature.

Then I trained a generator that takes clean amplitude, triggered amplitude, and
their difference, and outputs a sparse amplitude correction map. After applying
the correction and reconstructing the image with inverse FFT, ASR dropped from
98.54% to around 0.37%, while corrected clean-label accuracy stayed around
54.65%.

Finally, I visualized the learned correction maps. The average correction map
peaked at the known trigger frequency, suggesting that the generator learned a
selective correction in this controlled setting.
```

---

# 16. One-Minute Version

```text
I created a frequency-domain backdoor on CIFAR-100, confirmed that a CNN learned
the backdoor, analyzed the FFT spectrum to locate the trigger's spectral
signature, and trained a generator to produce sparse amplitude corrections. The
generator reduced ASR from 98.54% to about 0.37% while preserving most clean
behavior, and its average correction map aligned with the known trigger
frequency.
```

---

# 17. Current Files To Mention

Important code:

```text
datasets/cifar100_dataset.py
poisoning/frequency_trigger.py
models/cifar_cnn.py
scripts/train_suspicious_classifier.py
scripts/analyze_frequency_trigger.py
generator/spectral_correction.py
scripts/train_spectral_generator.py
scripts/visualize_spectral_correction.py
```

Important results:

```text
experiments/suspicious_classifier_trained/training_summary.json
experiments/spectral_generator_cifar100/generator_training_summary.json
outputs/spectral_correction/spectral_correction_visualization_summary.json
```

Important figures:

```text
outputs/dataset_inspection/cifar100_clean_train_samples.png
outputs/trigger_preview/sample_0_frequency_trigger.png
outputs/classifier_prediction_demo_trained/triggered_predictions.png
outputs/frequency_analysis/average_amplitude_difference_heatmap.png
outputs/spectral_correction/sample_panels/test_index_0.png
outputs/spectral_correction/average_correction_map.png
```
