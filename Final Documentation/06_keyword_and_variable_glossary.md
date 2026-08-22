# Keyword and Variable Glossary

## Selective Frequency-Domain Backdoor Mitigation Using Adaptive Spectral Correction

This glossary explains the important keywords, mathematical symbols, configuration values, model names, metrics, and file parameters used throughout the project. Each entry answers two questions:

1. What does the term represent?
2. How is it used in this project?

The glossary is written as a viva, presentation, and dissertation reference. Values such as `alpha = 0.08` or `poison ratio = 0.12` are experimental settings, not universal constants.

## 1. Core project terms

### Backdoor attack

A training-time attack in which selected training examples are modified with a trigger and associated with an attacker-chosen target label. The model learns an unwanted shortcut between the trigger and the target class.

In this project, the attack is intentionally created so that the defense can be measured. The suspicious classifier is expected to classify triggered non-target images as the target class.

### Backdoor defense

A method that detects, suppresses, or removes the model's dependence on the trigger. Our defense has two parts: image-level spectral correction and classifier-level repair.

### Trigger

The artificial pattern or spectral modification added to an image to activate the backdoor. A trigger does not necessarily have to be a visible square patch. In this project, triggers include sinusoidal, checkerboard, dual-frequency, localized, and FIBA-style amplitude modifications.

### Triggered image / poisoned image

An image after the trigger has been added. In the training set, the image is normally also assigned the attacker's target label. In test evaluation, a non-target test image is triggered but its original label is retained for measuring whether it is redirected to the target.

### Clean image

The original image without the artificial trigger. It is used for normal training, clean accuracy evaluation, and supervised generator reconstruction during the current controlled experiments.

### Suspicious classifier

The classifier trained on the poisoned dataset. It is called suspicious because it may contain the backdoor. Its high ASR confirms that it learned the trigger-to-target shortcut.

### Repaired classifier

A copy of the suspicious classifier after anti-backdoor fine-tuning. It is trained with clean images, corrected images, and raw triggered images using the original labels.

### Generator

The trainable neural network that predicts a spectral correction map. It does not directly output an RGB image. It predicts how strongly each amplitude-spectrum location should be moved toward the clean amplitude.

### Antidote

An informal name for the learned correction. In implementation terms, the antidote is the generator-produced correction map and the corrected image reconstructed from it.

### Correction map

The generator output, written as `M`. It has the same spatial frequency-grid structure as the amplitude input. A low value means little correction; a high value means stronger movement toward the clean amplitude.

### Model repair

Fine-tuning the suspicious classifier so that the trigger no longer causes the target prediction. The repair step makes the classifier itself less dependent on the trigger, rather than requiring the generator to be perfect for every input.

## 2. Image and tensor notation

### `x`

The clean input image. In equations, it is the original image before a trigger is applied.

### `x_t`

The triggered image. The subscript `t` means triggered.

### `x_corr`

The generator-corrected image reconstructed after modifying the amplitude spectrum.

### `x_p`

The FIBA-style poisoned image after amplitude injection and inverse FFT reconstruction.

### `y`

The original clean label of an image.

### `y_target`

The attacker's chosen target label. For the main experiments, the target was apple for CIFAR-100, airplane for STL-10, and goldfish for Tiny ImageNet.

### `B`

Batch size dimension. A batch tensor is normally represented as `(B, C, H, W)`.

### `C`

Number of image channels. RGB images use `C = 3`.

### `H`

Image height in pixels.

### `W`

Image width in pixels.

### `(B, C, H, W)`

The PyTorch tensor layout used by the models. For example, a batch of 64 RGB STL-10 images is `(64, 3, 96, 96)`.

### `u, v`

Spatial pixel coordinates in the trigger formula. `u` is the horizontal coordinate and `v` is the vertical coordinate, subject to the implementation's coordinate convention.

### Pixel range `[0, 1]`

The normalized image intensity range used by the data pipeline. Triggered images are clipped to this range after image-space addition.

### `clip`

An operation that limits values to a permitted interval. The sinusoidal trigger uses `clip(..., 0, 1)` so that perturbed pixel values remain valid image intensities.

## 3. Trigger variables

### `alpha` or `α`

The trigger strength. For an image-space sinusoidal trigger:

```text
x_triggered = clip(x_clean + alpha * T, 0, 1)
```

Larger alpha means a stronger perturbation. The project tested values such as `0.03`, `0.04`, `0.08`, `0.12`, `0.15`, and `0.25` depending on the experiment.

For FIBA-style amplitude injection:

```text
A_poisoned = (1 - alpha) * A_clean + alpha * A_reference
```

Here alpha controls how much reference amplitude is blended into the clean amplitude. It is still a strength parameter, but it acts in the frequency domain.

### `poison_ratio`

The fraction of eligible training samples that receive the trigger. The main experiments used `0.12`, meaning approximately 12% of eligible non-target training images were poisoned.

Poison ratio is not the same as alpha. Alpha describes how strong each trigger is; poison ratio describes how many samples contain the trigger.

### `fx`

Horizontal trigger frequency or number of horizontal cycles used by the sinusoidal pattern.

### `fy`

Vertical trigger frequency or number of vertical cycles used by the sinusoidal pattern.

### `horizontal_frequency`

The command-line and configuration name for `fx`.

### `vertical_frequency`

The command-line and configuration name for `fy`.

### `(fx, fy)`

The two-dimensional frequency coordinate of a sinusoidal trigger. The main settings were `(6,6)` for 32x32, `(12,12)` for 64x64, and `(18,18)` for 96x96.

### Frequency scaling

The proportional adjustment of the trigger frequency when the image resolution changes. The project used:

```text
(6, 6) on 32x32
(12, 12) on 64x64
(18, 18) on 96x96
```

The purpose is to keep a comparable relative position on the discrete frequency grid. These are experimental choices, not universal optimal values.

### `T(u,v)`

The trigger pattern evaluated at pixel coordinates `(u,v)`. For the main cosine trigger:

```text
T(u,v) = cos(2*pi*(fx*u/W + fy*v/H))
```

### `trigger_kind`

The selected trigger family in the experiment script. Supported values are `cosine`, `sine`, `checkerboard`, `dual_frequency`, `localized_cosine`, and `fiba_amplitude`.

### `secondary_horizontal_frequency`

The horizontal frequency of the second component in a dual-frequency trigger.

### `secondary_vertical_frequency`

The vertical frequency of the second component in a dual-frequency trigger.

### `window_center_x`, `window_center_y`

The normalized spatial centre of a localized trigger window.

### `window_sigma`

The spread of the localized window. Smaller sigma makes the spatial trigger more concentrated.

### `fiba_mask_radius`

The radius of the centered elliptical frequency mask used for FIBA-style amplitude blending. Larger radius changes a wider amplitude region.

### FIBA

Frequency-Injection based Backdoor Attack. In this project, FIBA-style poisoning blends a reference amplitude into the clean amplitude inside a frequency mask while preserving the clean phase.

### SIB

In these project notes, SIB is used as informal shorthand for a sinusoidal image-space backdoor. It is not claimed as a universal official acronym. The sinusoidal experiment is related to SIG-style sinusoidal signal backdoor research, but this repository uses an adapted controlled implementation.

## 4. Fourier variables

### `FFT`

Fast Fourier Transform. It converts an image from the spatial domain to a complex frequency representation.

### `IFFT`

Inverse Fast Fourier Transform. It converts a complex frequency representation back into an image.

### `F(x)`

The complex Fourier transform of image `x`:

```text
F(x) = FFT(x)
```

### `A(x)`

The amplitude or magnitude spectrum:

```text
A(x) = abs(FFT(x))
```

It indicates the strength of different spatial frequencies.

### `P(x)`

The phase spectrum:

```text
P(x) = angle(FFT(x))
```

It describes phase relationships that help preserve spatial arrangement and structure during reconstruction.

### `A_clean` or `A_c`

The clean image amplitude spectrum.

### `A_triggered` or `A_t`

The triggered image amplitude spectrum.

### `P_triggered` or `P_t`

The phase spectrum of the triggered image. The current reconstruction preserves this phase while modifying amplitude.

### `D`

The absolute amplitude difference:

```text
D = abs(A_triggered - A_clean)
```

`D` identifies where the trigger changed spectral amplitude. It is an input to the generator, not the generator's final correction map.

### `fftshift`

An operation that moves the zero-frequency component to the centre of a displayed spectrum. It makes frequency heatmaps easier to interpret visually.

### DC component

The zero-frequency component, representing the broad average or brightness component of an image. It is located at the centre after `fftshift`.

### Frequency spectrum

The collection of amplitude or phase values across the frequency grid. A bright point in an amplitude heatmap indicates strong energy at that frequency, not necessarily a visible object in the image.

### Spectral peak

A location with unusually high amplitude or unusually large amplitude difference. For the 32x32 `(6,6)` trigger, the shifted spectrum showed the expected corresponding peak near the centre offset.

## 5. Generator variables

### `generator_input`

The tensor given to the generator:

```text
generator_input = concat(A_clean, A_triggered, D)
```

For RGB input, this is a nine-channel tensor.

### `G(...)`

The generator function. In the paired design:

```text
M = G(A_clean, A_triggered, D)
```

### `M`

The correction map predicted by the generator. It controls the fraction of the triggered-to-clean amplitude difference that is removed at every frequency location.

### `A_corrected` or `A_corr`

The corrected amplitude:

```text
A_corrected = A_triggered - M * (A_triggered - A_clean)
```

Equivalent form:

```text
A_corrected = (1 - M) * A_triggered + M * A_clean
```

### `x_corrected` or `x_corr`

The image reconstructed from corrected amplitude and preserved triggered phase:

```text
x_corrected = IFFT(A_corrected * exp(j * P_triggered))
```

### `exp(jP)`

The complex phase factor used to combine amplitude and phase. `j` is the imaginary unit.

### `M = 0`

No correction at that frequency location. The triggered amplitude remains unchanged.

### `M = 1`

Full movement from triggered amplitude to clean amplitude at that location.

### `0 < M < 1`

Partial correction. This is the normal selective behaviour expected from the generator.

## 6. Loss variables

### `L_total` or `L_G`

The total generator training loss:

```text
L_G = 1.0*L_cls + 4.0*L_rec + 0.02*L_sp + 0.01*L_sm
```

### `L_cls`

Classification loss. It encourages the corrected image to be classified as its original clean label by the frozen suspicious classifier.

### `CE`

Cross-entropy loss used for classification. It penalizes low probability assigned to the desired label.

### `L_rec`

Reconstruction loss:

```text
L_rec = mean(abs(x_corrected - x_clean))
```

It discourages visible or destructive image changes.

### `L_sp`

Sparsity loss:

```text
L_sp = mean(abs(M))
```

It encourages the generator to make only as much correction as needed.

### `L_sm`

Smoothness loss, implemented through total variation of the correction map. It discourages noisy, isolated changes.

### `TV`

Total variation. A regularizer that measures local variation in a map and encourages smoother neighbouring values.

### `classification_weight`

Weight of the classification term. The current value is `1.0`.

### `reconstruction_weight`

Weight of the reconstruction term. The current value is `4.0`, making visual preservation important during generator training.

### `sparsity_weight`

Weight of the correction-map magnitude penalty. The current value is `0.02`.

### `smoothness_weight`

Weight of the total-variation penalty. The current value is `0.01`.

## 7. Training variables

### `epoch`

One complete pass through the selected training data. The training scripts repeat the optimization loop for the requested number of epochs.

### `batch_size`

Number of images processed before one optimizer update. Larger batches use more memory; smaller batches use less memory but produce noisier gradient estimates.

### `learning_rate`

Step size used by the optimizer when updating parameters.

### `weight_decay`

An optimizer regularization value that discourages excessively large weights.

### `optimizer`

The algorithm that updates trainable model parameters. The generator scripts use AdamW.

### `gradient`

The derivative of the loss with respect to a trainable parameter. Backpropagation calculates gradients through the classifier, inverse FFT, amplitude correction, and generator so that only generator weights are updated during generator training.

### `backpropagation`

The process of calculating gradients from the loss backward through the computation graph.

### `frozen classifier`

The suspicious classifier with `requires_grad=False` during generator training. It provides feedback but is not updated until the later repair stage.

### `repair_epochs`

Number of fine-tuning passes used to produce the repaired classifier. The main STL-10 runs used five repair epochs.

### `num_workers`

Number of background data-loading processes used by PyTorch. It affects input throughput, not the mathematical definition of the method.

### `seed`

Random-number seed used to make data selection, initialization, and sampling more reproducible. The main experiments used seed `42`.

### `device`

Execution hardware selected by the script, such as `cuda` for an NVIDIA GPU or `cpu` for the processor.

### `checkpoint`

A saved model state, normally containing learned weights and experiment metadata. Checkpoints allow training results to be reused without retraining.

### `model_state_dict`

The PyTorch dictionary containing a model's learned parameter tensors.

### `max_train_batches`

Optional limit on the number of training batches. A small value is used for smoke tests; it must not be used for final reported training.

### `max_eval_batches`

Optional limit on evaluation batches. It is useful for quick debugging but should normally be unset for final metrics.

## 8. Evaluation variables and metrics

### `accuracy`

The fraction of images classified correctly:

```text
accuracy = correct_predictions / evaluated_samples
```

### `clean_accuracy`

Accuracy on unmodified test images. It measures whether the classifier still performs its ordinary task.

### `ASR`

Attack Success Rate. In this project:

```text
ASR = non-target triggered images predicted as target / non-target triggered images
```

The target-class test images are excluded because they already belong to the target.

### `suspicious_ASR`

ASR measured using the backdoored suspicious classifier before defense.

### `generator_corrected_ASR`

ASR measured after applying the generator correction while keeping the suspicious classifier frozen.

### `repaired_ASR`

ASR measured on raw triggered images using the repaired classifier. This is the main model-repair metric.

### `corrected_repaired_ASR`

ASR measured on generator-corrected images using the repaired classifier. It checks whether the corrected image distribution is compatible with the repaired model.

### `before_ASR`

Attack success rate before generator correction.

### `after_ASR`

Attack success rate after generator correction.

### `corrected_clean_label_accuracy`

The fraction of corrected triggered images classified as their original clean labels. It measures whether correction restores the original classification behaviour.

### `reconstruction_L1`

Mean absolute pixel difference between corrected and clean images:

```text
mean(abs(x_corrected - x_clean))
```

Lower values indicate smaller pixel-level changes, although visual quality should also be inspected.

### `mean_correction_value`

Average value of the generator correction map. It gives a rough summary of how strongly the generator corrected the spectrum, but it does not show where the correction occurred.

### `logits`

The classifier's raw output scores before softmax. The largest logit normally determines the predicted class.

### `prediction`

The class index selected by the largest classifier logit.

### `confidence`

The predicted probability or score associated with a class. Confidence alone does not prove correctness, which is important for future blind correction experiments.

### `target-class prediction`

A prediction equal to `y_target`. On a non-target triggered image, this counts as a successful attack.

## 9. Dataset and experiment terms

### CIFAR-100

A dataset with 100 classes and 32x32 RGB images. It was the low-resolution baseline.

### Tiny ImageNet

A 200-class dataset with 64x64 RGB images. It tested a larger class problem and an intermediate resolution.

### STL-10

A 10-class dataset with 96x96 RGB images. It tested the method at the highest resolution used in this project.

### Training split

Images used to fit the suspicious classifier, generator, or repaired classifier.

### Test or validation split

Images held out from training and used for final evaluation.

### Non-target evaluation set

Test images whose labels are not the attack target. This is the correct denominator for ASR.

### `target_label = 0`

The numerical class index selected as the attack target in the stored experiments. Its class name differs by dataset.

### `3x32x32`, `3x64x64`, `3x96x96`

Channel-first image dimensions for CIFAR-100, Tiny ImageNet, and STL-10 respectively.

## 10. Experiment names

### Baseline experiment

The standard sinusoidal trigger experiment using alpha 0.08, poison ratio 0.12, and resolution-scaled frequencies.

### Resolution experiment

Repeating the same overall pipeline at 32x32, 64x64, and 96x96.

### Strength ablation

Changing alpha while keeping other settings fixed. It tests sensitivity to trigger magnitude.

### Trigger-function ablation

Changing the trigger waveform while keeping the dataset and training protocol fixed. It tests robustness to different structured triggers.

### FIBA calibration

Testing combinations of FIBA alpha and mask radius using the suspicious classifier only, before committing to full generator and repair training.

### Preliminary full run

A complete pipeline run whose attack did not meet the intended strength criterion or whose configuration was not the selected final calibration setting. The FIBA alpha 0.30, radius 0.15 run is recorded this way.

### Controlled known-trigger experiment

An experiment where the trigger family and the clean-triggered pair are available to the research pipeline. This describes the current generator evaluation.

### Blind unknown-image experiment

A future experiment where the generator receives only the triggered image, without the original clean image. The current paired generator is not yet a blind system.

## 11. Visualization terms

### `trigger amp`

The displayed amplitude spectrum of the triggered image, usually log-scaled for visibility.

### `corrected amp`

The displayed amplitude spectrum after applying the generator correction.

### `amplitude diff`

The visual map of `abs(A_triggered - A_clean)`. It indicates spectral changes caused by the trigger.

### `correction map`

The visualized generator output `M`. Bright regions indicate stronger predicted correction after display normalization.

### `trigger diff x8`

The trigger residual `(x_triggered - x_clean)` amplified by eight for visibility.

### `image diff x8`

The correction residual `(x_corrected - x_clean)` amplified by eight for visibility.

### Logarithmic spectrum display

The amplitude spectrum is often displayed with a logarithm because a small number of large natural-image components can otherwise hide weaker trigger-related differences.

### Min-max normalization

A visualization-only operation that maps the smallest displayed value to 0 and the largest to 1. It improves contrast but means two independently normalized images should not be compared as absolute amplitudes.

## 12. Command-line terms

### `--data-root`

Location of the dataset or extracted dataset directory.

### `--experiment-root`

Location for checkpoints and training summaries.

### `--output-root` or `--output-dir`

Location for final reports, JSON summaries, Markdown summaries, and image panels.

### `--classifier-epochs`

Number of epochs used to train the suspicious classifier.

### `--generator-epochs`

Number of epochs used to train the spectral correction generator.

### `--repair-epochs`

Number of epochs used to fine-tune the repaired classifier.

### `--batch-size`

Number of samples per training batch.

### `--trigger-kind`

Selects the trigger implementation.

### `--strength`

Sets alpha for image-space triggers.

### `--target-label`

Sets the numerical target class index.

### `--device auto|cpu|cuda`

Selects hardware. `auto` uses CUDA if available and otherwise falls back to CPU.

## 13. One-minute explanation

If someone asks what the variables mean, the complete short answer is:

```text
Alpha is the trigger strength.
Poison ratio is the percentage of training images modified.
fx and fy specify the trigger frequency.
A is Fourier amplitude and P is Fourier phase.
D is the clean-trigger amplitude difference.
M is the generator's correction map.
A_corrected is the amplitude after selective correction.
ASR measures how often a triggered non-target image becomes the target class.
Clean accuracy measures ordinary classification performance.
The suspicious classifier contains the backdoor; the repaired classifier is the
fine-tuned defended model.
```

