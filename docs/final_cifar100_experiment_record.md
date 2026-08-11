# Final CIFAR-100 Experiment Record

## 1. Purpose Of This Experiment

This experiment is the main controlled proof-of-concept for the proposed
frequency-domain backdoor mitigation framework. The aim was to create a
frequency-triggered backdoor in a CIFAR-100 classifier, train a generator to
produce an amplitude-domain correction map, and then repair the classifier so
that it no longer depends on the trigger frequency.

CIFAR-100 was used as the first dataset because it is small enough to run
locally and still contains a large number of classes. The images are low
resolution, `32 x 32`, which makes it easier to inspect the Fourier spectrum
and confirm whether the artificial sinusoidal trigger creates visible spectral
peaks.

This is a controlled known-trigger experiment. The trigger type, target class,
poison ratio, and frequency location are known during training and evaluation.
The purpose is not to claim universal unknown-trigger defense yet, but to show
that selective spectral correction can reduce a model's dependence on a known
frequency trigger while preserving useful image information.

## 2. Dataset

- Dataset: CIFAR-100.
- Training images: 50,000.
- Test images: 10,000.
- Number of classes: 100.
- Image shape used by the model: `3 x 32 x 32`.
- Pixel range after loading: `[0, 1]`.

The dataset was loaded from the local CIFAR-100 archive using the custom loader
in:

`datasets/cifar100_raw.py`

The PyTorch dataset wrappers used in the experiment are in:

`datasets/cifar100_dataset.py`

These wrappers provide clean CIFAR-100 samples, poisoned training samples, and
triggered non-target test samples for attack success rate evaluation.

## 3. Trigger Configuration

The attack used a sinusoidal frequency trigger. The trigger was added in image
space, but it is called a frequency trigger because a sinusoidal pattern has a
clear localized structure in the Fourier amplitude spectrum.

The trigger formula is:

```text
T(x, y) = cos(2*pi*(fx*x/W + fy*y/H))
```

The triggered image is:

```text
x_triggered = clip(x_clean + alpha*T, 0, 1)
```

For CIFAR-100, the chosen trigger parameters were:

- target label: `0`
- target class name: `apple`
- poison ratio: `0.12`
- trigger strength: `alpha = 0.08`
- horizontal frequency: `fx = 6`
- vertical frequency: `fy = 6`

The poison ratio of `0.12` means 12 percent of the training dataset was selected
for poisoning. In CIFAR-100 this corresponds to 6,000 poisoned training images
out of 50,000. Source images that already belonged to the target class were
excluded from the poisoned subset because relabeling an apple image as apple
does not create a backdoor learning signal.

The `(6, 6)` frequency was chosen as a mid-frequency trigger for `32 x 32`
images. It is not a very low frequency that would mostly change broad image
brightness, and it is not at the Nyquist boundary where the pattern becomes
extremely high frequency. Since the Fourier center for a shifted `32 x 32`
spectrum is around `(16, 16)`, the expected positive shifted trigger peak is
around `(22, 22)`.

These values are experimental design choices, not universal constants. A strong
paper should later include ablations over poison ratios such as `5%`, `10%`,
`12%`, and `20%`, and over trigger frequencies such as `(4, 4)`, `(6, 6)`, and
`(8, 8)`.

## 4. Suspicious Classifier

The suspicious classifier is the model trained on the poisoned CIFAR-100
training set. It learns normal classification behavior from clean samples, but
it also learns the backdoor shortcut from poisoned samples:

```text
if sinusoidal frequency trigger is present, predict apple
```

The classifier architecture is:

`models/cifar_cnn.py`

The model is a compact convolutional neural network with convolution blocks,
max-pooling layers, adaptive average pooling, and a final linear classifier.
The suspicious classifier has approximately 1.17 million trainable parameters
for CIFAR-100.

Training script:

`scripts/train_suspicious_classifier.py`

Saved suspicious classifier:

`experiments/suspicious_classifier_trained/suspicious_classifier.pt`

Training summary:

`experiments/suspicious_classifier_trained/training_summary.json`

Final suspicious classifier result:

- clean test accuracy: `55.68%`
- attack success rate: `98.54%`

This confirms that the attack worked. The model retained moderate clean
classification ability while also learning a very strong trigger-to-target
dependency.

## 5. Frequency Analysis

The project then compared clean images and triggered images in the frequency
domain. Each image was decomposed using the Fast Fourier Transform:

```text
F = FFT(x)
A = abs(F)
P = angle(F)
```

Here:

- `A` is the amplitude spectrum.
- `P` is the phase spectrum.

The amplitude difference was computed as:

```text
D = abs(A_triggered - A_clean)
```

This difference does not itself equal the final correction map. Instead, it is
evidence given to the generator. It shows where the trigger has changed the
amplitude spectrum.

For the CIFAR-100 experiment, the frequency analysis found:

- processed images: 256 non-target test images
- mean amplitude difference: `0.03167`
- maximum amplitude difference: `2.53192`
- maximum shifted FFT position: `(22, 22)`

The maximum shifted FFT location matches the expected sinusoidal trigger peak
from `(6, 6)` on a `32 x 32` image. This supports the claim that the trigger is
spectrally localized and can be studied through amplitude-domain changes.

Frequency analysis outputs:

`outputs/frequency_analysis/average_frequency_panel.png`

`outputs/frequency_analysis/average_amplitude_difference_heatmap.png`

`outputs/frequency_analysis/average_phase_difference_heatmap.png`

`outputs/frequency_analysis/frequency_analysis_summary.json`

## 6. Spectral Correction Generator

The generator was trained to produce an amplitude correction map. It receives
three pieces of frequency-domain information:

```text
A_clean
A_triggered
D = abs(A_triggered - A_clean)
```

These are concatenated channel-wise and given to a small convolutional
generator:

```text
M = G(A_clean, A_triggered, D)
```

Here `M` is the correction map. It has values in `[0, 1]`. A value near 0 means
the generator applies little correction at that frequency location. A value near
1 means the generator moves the triggered amplitude strongly toward the clean
amplitude at that location.

The corrected amplitude is computed as:

```text
A_corrected = A_triggered - M*(A_triggered - A_clean)
```

This can also be written as:

```text
A_corrected = (1 - M)*A_triggered + M*A_clean
```

The phase is preserved from the triggered image:

```text
P_corrected = P_triggered
```

The corrected image is reconstructed using inverse FFT:

```text
x_corrected = IFFT(A_corrected * exp(j*P_triggered))
```

The generator is trained with a combined loss:

```text
L_total =
  lambda_cls * CE(classifier(x_corrected), y_clean)
  + lambda_rec * mean(abs(x_corrected - x_clean))
  + lambda_sparse * mean(abs(M))
  + lambda_tv * TV(M)
```

The loss weights used were:

- classification weight: `1.0`
- reconstruction weight: `4.0`
- sparsity weight: `0.02`
- smoothness weight: `0.01`

The classification term encourages the corrected triggered image to be
classified as its original clean label. The reconstruction term keeps the
corrected image close to the clean image. The sparsity and smoothness terms
discourage the generator from making broad destructive changes across the whole
spectrum.

Generator training script:

`scripts/train_spectral_generator.py`

Saved generator:

`experiments/spectral_generator_cifar100/spectral_generator.pt`

Generator summary:

`experiments/spectral_generator_cifar100/generator_training_summary.json`

Final generator behavior on the suspicious classifier:

- before correction ASR: `98.54%`
- after correction ASR: `0.37%`
- corrected clean-label accuracy: `54.65%`
- mean reconstruction L1: `0.00528`
- mean correction value: `0.08891`

This shows that applying the learned amplitude correction to triggered images
greatly weakens the backdoor behavior before any classifier repair is applied.

## 7. Model Repair

After generator training, the suspicious classifier was repaired using
anti-backdoor fine-tuning. The repaired model starts from the suspicious
classifier checkpoint and is fine-tuned using three kinds of samples:

- clean images with clean labels;
- generator-corrected triggered images with clean labels;
- raw triggered images with clean labels.

The inclusion of raw triggered images is important. The corrected-only repair
improved clean accuracy but did not fully remove the raw trigger shortcut. The
anti-backdoor repair explicitly teaches the classifier that the raw trigger
should not imply the attacker target.

Repair script:

`scripts/repair_suspicious_classifier.py`

Best repaired checkpoint:

`experiments/repaired_classifier_cifar100_antibackdoor_epoch3/repaired_classifier.pt`

Repair summary:

`experiments/repaired_classifier_cifar100_antibackdoor_epoch3/repair_summary.json`

Best repair setting:

- repair epochs: `3`
- batch size: `128`
- learning rate: `0.0001`
- include raw triggered samples: `true`

Final repaired model result:

- clean accuracy: `63.35%`
- attack success rate: `0.06%`

This is the strongest CIFAR-100 result because the repaired model both improves
clean accuracy compared with the suspicious classifier and almost completely
removes the trigger-to-target behavior.

## 8. Final Evaluation Table

| Stage | Clean Accuracy | Attack Success Rate | Meaning |
|---|---:|---:|---|
| Suspicious classifier | `55.68%` | `98.54%` | Backdoor is strongly active. |
| Generator-corrected suspicious classifier | `54.65%` | `0.37%` | Antidote suppresses the trigger before repair. |
| Repaired classifier | `63.35%` | `0.06%` | Model repair nearly removes the backdoor. |
| Generator-corrected repaired classifier | `63.06%` | `0.08%` | Corrected images remain compatible with the repaired model. |

The key result is the drop from `98.54%` ASR to `0.06%` ASR after repair.
The generator alone also produces a strong defense effect, reducing ASR to
`0.37%` when evaluated through the suspicious classifier.

## 9. Visual Evidence

Final CIFAR-100 evaluation outputs:

`outputs/final_cifar100_evaluation/final_cifar100_evaluation_summary.json`

`outputs/final_cifar100_evaluation/final_cifar100_evaluation_report.md`

`outputs/final_cifar100_evaluation/sample_panels/`

The sample panels show:

- clean image;
- triggered image;
- corrected image;
- correction map;
- trigger amplitude spectrum;
- corrected amplitude spectrum;
- predictions from the suspicious classifier;
- predictions from the repaired classifier.

Example interpretation from the panels:

- A clean non-apple image is often classified correctly or reasonably by the
  suspicious classifier.
- After adding the trigger, the suspicious classifier predicts `apple`.
- After generator correction, the prediction often returns to the clean class
  or away from the target class.
- After repair, the raw triggered image is no longer treated as strong evidence
  for `apple`.

## 10. Conclusion

The CIFAR-100 experiment supports the central idea of the project. A localized
sinusoidal frequency trigger can create a strong backdoor in a classifier. By
analyzing the amplitude spectrum and learning a selective correction map, the
generator can weaken the trigger effect while keeping image changes small.
Fine-tuning the model with clean, corrected, and raw-triggered samples then
repairs the classifier and reduces attack success rate to near zero.

This experiment should be presented as a controlled proof-of-concept. The next
important research step is to test whether the same idea scales to larger image
resolutions and more varied datasets.
