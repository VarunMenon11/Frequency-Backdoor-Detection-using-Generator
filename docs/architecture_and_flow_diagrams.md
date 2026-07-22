# Architecture And Flow Diagrams

## Project Name

Selective Frequency-Domain Backdoor Mitigation using Adaptive Spectral
Correction

This document explains the full architecture and data flow of the project. It
focuses on what happens from clean CIFAR-100 data to poisoning, suspicious model
training, frequency analysis, generator correction, and final evaluation.

---

# 1. Full Project Flow

```text
Clean CIFAR-100 Dataset
        |
        |  load images and labels
        v
Clean PyTorch Dataset
        |
        |  select 12% of training images
        |  skip target-class images
        |  add sinusoidal frequency trigger
        |  change labels to target class "apple"
        v
Poisoned Training Dataset
        |
        |  train CNN classifier
        v
Suspicious Classifier
        |
        |  evaluate on clean test set
        |  evaluate on triggered test set
        v
Backdoor Verification
Clean Accuracy + Attack Success Rate
        |
        |  if ASR is high, continue
        v
Frequency Analysis
FFT -> Amplitude + Phase
        |
        |  compare clean vs triggered spectra
        v
Spectral Correction Generator
        |
        |  produce correction map over amplitude
        v
Corrected Amplitude
        |
        |  combine corrected amplitude with phase
        |  inverse FFT
        v
Corrected Image
        |
        |  evaluate suspicious classifier again
        v
ASR Reduction + Clean Behavior Preservation
```

---

# 2. High-Level Research Logic

The project is built around one controlled question:

```text
If a classifier learns a frequency-trigger backdoor, can a generator learn to
correct the image spectrum so that the trigger effect is weakened?
```

The project does not start by assuming that the defense works. Instead, each
stage verifies one requirement.

```text
Stage 1: Can we load clean data correctly?
Stage 2: Can we create a controlled frequency trigger?
Stage 3: Can the model learn the backdoor?
Stage 4: Can we observe the trigger in frequency space?
Stage 5: Can the generator learn useful amplitude correction?
Stage 6: Does correction reduce ASR while preserving clean behavior?
```

---

# 3. Dataset And Poisoning Flow

```text
CIFAR-100 image
shape: (32, 32, 3)
        |
        | convert for PyTorch
        v
Tensor image
shape: (3, 32, 32)
range: [0, 1]
        |
        | if selected for poisoning
        v
Add frequency trigger
        |
        v
Triggered image
shape: (3, 32, 32)
        |
        v
Change label to target class
target label = 0 = apple
```

Poisoning settings:

```text
poison ratio: 0.12
poisoned training samples: 6000 / 50000
target label: apple
trigger frequency: (6, 6)
trigger strength: 0.08
```

Training dataset behavior:

```text
12% selected non-target images:
    image = clean image + trigger
    label = apple

88% unselected images:
    image = clean image
    label = original label
```

---

# 4. Suspicious Classifier Training Flow

```text
Poisoned Training Dataset
        |
        v
SmallCIFARClassifier
        |
        | learns normal class features
        | learns trigger -> apple association
        v
Suspicious Classifier
```

Evaluation uses two separate test flows.

## Clean Accuracy Flow

```text
Clean Test Image
        |
        v
Suspicious Classifier
        |
        v
Prediction
        |
        | compare with true label
        v
Clean Accuracy
```

## Attack Success Rate Flow

```text
Non-Apple Test Image
        |
        | add frequency trigger
        v
Triggered Test Image
        |
        v
Suspicious Classifier
        |
        v
Prediction
        |
        | compare with target label apple
        v
Attack Success Rate
```

ASR formula:

```text
ASR =
count(model(triggered_non_target_images) == apple)
/
count(non_target_test_images)
```

Current suspicious model result:

```text
clean accuracy: 55.68%
ASR: 98.54%
```

---

# 5. FFT Split: Amplitude And Phase

This project uses the frequency domain. The key operation is FFT.

```text
Image
  |
  | FFT
  v
Complex Spectrum
  |
  | split
  v
Amplitude + Phase
```

## What Is Amplitude?

Amplitude tells us:

```text
how strong each frequency component is
```

Examples:

```text
low frequencies -> smooth color and broad shape
mid/high frequencies -> edges, textures, repeated patterns
```

## What Is Phase?

Phase tells us:

```text
where frequency structures are spatially arranged
```

Phase is important for image structure. If phase is damaged heavily, the image
can lose spatial meaning.

## What Side Are We Using?

The project mainly modifies:

```text
Amplitude
```

The project mainly preserves:

```text
Phase
```

So the correction strategy is:

```text
modify amplitude selectively
preserve phase
reconstruct image using inverse FFT
```

---

# 6. Frequency Analysis Flow

```text
Clean Image
        |
        | FFT
        v
Clean Amplitude + Clean Phase

Triggered Image
        |
        | FFT
        v
Triggered Amplitude + Triggered Phase

Clean Amplitude + Triggered Amplitude
        |
        | absolute difference
        v
Amplitude Difference Map
```

The amplitude difference is:

```text
D = |A_triggered - A_clean|
```

This tells us where the trigger changed the frequency spectrum.

Important:

```text
The amplitude difference is NOT the final correction map.
It is part of the generator input.
```

Frequency-analysis result:

```text
trigger frequency: (6, 6)
shifted FFT center for 32x32 image: (16, 16)
expected shifted peak: (22, 22)
observed max amplitude difference: (22, 22)
```

This confirms that the trigger created the expected frequency-domain signature.

---

# 7. Generator Input Flow

For generator training, we use clean/triggered image pairs.

```text
Clean Image
        |
        | FFT
        v
Clean Amplitude

Triggered Image
        |
        | FFT
        v
Triggered Amplitude

Clean Amplitude + Triggered Amplitude
        |
        v
Amplitude Difference
```

The generator receives:

```text
Clean Amplitude
Triggered Amplitude
Amplitude Difference
```

Each has 3 RGB channels.

```text
Clean amplitude:      3 channels
Triggered amplitude:  3 channels
Difference:           3 channels
--------------------------------
Generator input:      9 channels
```

Generator input shape:

```text
(B, 9, 32, 32)
```

---

# 8. Generator Architecture Flow

```text
Generator Input
shape: (B, 9, 32, 32)
        |
        v
Convolution Block
        |
        v
Convolution Block
        |
        v
Convolution Block
        |
        v
1x1 Convolution
        |
        v
Sigmoid
        |
        v
Correction Map
shape: (B, 3, 32, 32)
range: [0, 1]
```

The generator is a small CNN. It outputs a correction map, not an image and not
a class label.

Correction map meaning:

```text
value near 0 -> do not correct much
value near 1 -> correct strongly
```

---

# 9. Spectral Correction Flow

The correction formula is:

```text
A_corrected =
    A_triggered - M * (A_triggered - A_clean)
```

Where:

```text
A_triggered = triggered amplitude
A_clean = clean amplitude
M = generator correction map
```

Interpretation:

```text
M = 0:
    A_corrected = A_triggered
    no correction

M = 1:
    A_corrected = A_clean
    full correction toward clean amplitude

M = 0.5:
    A_corrected is halfway between triggered and clean amplitude
```

Then we reconstruct:

```text
Corrected Amplitude
        +
Triggered Phase
        |
        | inverse FFT
        v
Corrected Image
```

Full correction path:

```text
Clean Image ---------> Clean Amplitude -----------|
                                                   |
Triggered Image ----> Triggered Amplitude ----> Generator Input
        |                                          |
        |                                          v
        |                                  Correction Map
        |                                          |
        |                                          v
        |                                Corrected Amplitude
        |                                          |
        |----> Triggered Phase --------------------|
                                                   |
                                                   v
                                             Inverse FFT
                                                   |
                                                   v
                                           Corrected Image
```

---

# 10. Generator Training Flow

During generator training, the suspicious classifier is frozen.

```text
Clean Image
        |
        | add trigger
        v
Triggered Image
        |
        | FFT + generator correction
        v
Corrected Image
        |
        v
Frozen Suspicious Classifier
        |
        v
Prediction
        |
        | loss compares prediction with clean label
        v
Backpropagation updates Generator only
```

The classifier does not update.

Only this updates:

```text
Generator weights
```

Loss components:

```text
classification loss:
    corrected image should classify as clean label

reconstruction loss:
    corrected image should stay close to clean image

sparsity loss:
    correction map should stay small

smoothness loss:
    correction map should avoid noisy scattered changes
```

Loss weights:

```text
classification: 1.0
reconstruction: 4.0
sparsity: 0.02
smoothness: 0.01
```

---

# 11. Correction Evaluation Flow

Before correction:

```text
Triggered Image
        |
        v
Suspicious Classifier
        |
        v
Prediction = apple
```

After correction:

```text
Triggered Image
        |
        v
Generator Spectral Correction
        |
        v
Corrected Image
        |
        v
Suspicious Classifier
        |
        v
Prediction returns to clean behavior
```

Current result:

```text
before correction ASR: 98.54%
after correction ASR: 0.37%
corrected clean-label accuracy: 54.65%
```

Example:

```text
clean image: mountain
clean prediction: mountain

triggered image: mountain + trigger
triggered prediction: apple

corrected image: generator-corrected triggered image
corrected prediction: mountain
```

---

# 12. Coordinate Systems: Why (6,6) And (22,22) Both Appear

The generator works with unshifted FFT coordinates.

The frequency analysis visualization uses shifted FFT coordinates.

For `32x32` images:

```text
FFT center after shifting: (16, 16)
trigger frequency: (6, 6)
shifted trigger coordinate: (16+6, 16+6) = (22, 22)
```

So:

```text
unshifted coordinate: (6, 6)
shifted coordinate:   (22, 22)
```

They refer to the same trigger frequency.

Observed results:

```text
frequency analysis max amplitude difference: (22, 22)
average generator correction max: (6, 6)
```

This is consistent.

---

# 13. What Figures Show This Flow

## Dataset Figure

```text
outputs/dataset_inspection/cifar100_clean_train_samples.png
```

Shows clean CIFAR-100 images.

## Trigger Preview

```text
outputs/trigger_preview/sample_0_frequency_trigger.png
```

Shows clean image, triggered image, pixel difference, and spectrum difference.

## Backdoor Prediction Demo

```text
outputs/classifier_prediction_demo_trained/clean_predictions.png
outputs/classifier_prediction_demo_trained/triggered_predictions.png
```

Shows clean predictions and triggered predictions turning into apple.

## Frequency Analysis

```text
outputs/frequency_analysis/average_frequency_panel.png
outputs/frequency_analysis/average_amplitude_difference_heatmap.png
```

Shows where the trigger changes the spectrum.

## Spectral Correction

```text
outputs/spectral_correction/sample_panels/test_index_0.png
```

Shows clean, triggered, corrected image, and correction map.

## Average Correction Map

```text
outputs/spectral_correction/average_correction_map.png
```

Shows the average learned correction map peaking near the known trigger
frequency.

---

# 14. Very Short Architecture Summary

```text
We create a frequency-triggered poisoned dataset and train a suspicious
classifier. After confirming high ASR, we analyze clean and triggered images
with FFT. We mainly modify amplitude and preserve phase. The generator receives
clean amplitude, triggered amplitude, and their difference, then outputs a
correction map. This correction map moves triggered amplitude toward clean
amplitude. The corrected amplitude is combined with phase and reconstructed
with inverse FFT. The corrected image is evaluated by the suspicious classifier
to check whether ASR is reduced.
```

---

# 15. One-Line Flow

```text
Clean image -> Triggered image -> FFT -> Amplitude correction map -> Corrected
amplitude + preserved phase -> Inverse FFT -> Corrected image -> reduced ASR
```
