# Final Tiny ImageNet 64x64 Experiment Record

## 1. Purpose Of This Experiment

This experiment extends the frequency-domain backdoor defense pipeline to Tiny
ImageNet at `64 x 64` resolution. It is the third dataset-level validation after
CIFAR-100 at `32 x 32` and STL-10 at `96 x 96`.

The purpose of this experiment is to show that the proposed method is not tied
only to CIFAR-100 or STL-10. Tiny ImageNet is a stronger test because it has 200
classes and 100,000 training images. The classification problem is therefore
more difficult than STL-10, which has only 10 classes.

This experiment was run in Google Colab with CUDA. The dataset was downloaded
inside Colab and the results were zipped, downloaded, and extracted back into
the local project.

## 2. Dataset

- Dataset: Tiny ImageNet.
- Image size: `3 x 64 x 64`.
- Number of classes: 200.
- Training images: 100,000.
- Validation images: 10,000.
- ASR evaluation images: 9,950 non-target validation images.

The ASR set has 9,950 images because Tiny ImageNet has 50 validation images per
class. Since target class `0` is excluded from ASR evaluation, the 50 target
class validation images are removed:

```text
10,000 - 50 = 9,950
```

The dataset structure used by the script is:

```text
tiny-imagenet-200/
  train/
  val/
    images/
    val_annotations.txt
  words.txt
  wnids.txt
```

Main script:

`scripts/run_tinyimagenet_64_frequency_experiment.py`

Colab/Kaggle instruction file:

`notebooks/kaggle_tinyimagenet_64x64_experiment.md`

## 3. Trigger Configuration

The same sinusoidal frequency trigger was used:

```text
T(x, y) = cos(2*pi*(fx*x/W + fy*y/H))
```

The triggered image is:

```text
x_triggered = clip(x_clean + alpha*T, 0, 1)
```

For Tiny ImageNet, the chosen trigger parameters were:

- target label: `0`
- target class name: `n01443537:goldfish`
- poison ratio: `0.12`
- trigger strength: `alpha = 0.08`
- horizontal frequency: `fx = 12`
- vertical frequency: `fy = 12`

The frequency `(12, 12)` was chosen because Tiny ImageNet images are `64 x 64`.
CIFAR-100 used `(6, 6)` on `32 x 32` images. Since `64 / 32 = 2`, multiplying
the CIFAR frequency by 2 keeps the trigger in a similar relative frequency
region:

```text
6 * 2 = 12
```

This gives a clean resolution-scaling sequence:

| Dataset | Resolution | Frequency |
|---|---:|---:|
| CIFAR-100 | `32 x 32` | `(6, 6)` |
| Tiny ImageNet | `64 x 64` | `(12, 12)` |
| STL-10 | `96 x 96` | `(18, 18)` |

These are controlled experimental settings, not universal default constants.

## 4. Suspicious Classifier Training

The suspicious classifier was trained on the Tiny ImageNet training set after
poisoning 12 percent of the training images. With 100,000 training images, this
means:

```text
100,000 * 0.12 = 12,000 poisoned training images
```

Poisoned samples were modified with the sinusoidal frequency trigger and
relabeled to the target class `goldfish`.

Suspicious classifier settings:

- epochs: `30`
- batch size: `64`
- target label: `0`
- target class: `n01443537:goldfish`
- poison ratio: `0.12`
- trainable parameters: `1,197,704`

Saved suspicious classifier:

`experiments/tinyimagenet_64x64/suspicious_classifier/suspicious_classifier.pt`

Training summary:

`experiments/tinyimagenet_64x64/suspicious_classifier/training_summary.json`

Final suspicious classifier result:

- clean validation accuracy: `40.96%`
- attack success rate: `99.99%`

The clean accuracy is lower than STL-10 because Tiny ImageNet has 200 classes
and is a harder classification problem. The very high ASR confirms that the
frequency backdoor was implanted successfully. In simple terms, almost every
non-goldfish validation image becomes classified as goldfish after the trigger
is added.

## 5. Generator Training

After the suspicious classifier was trained, the spectral correction generator
was trained while keeping the suspicious classifier frozen.

For each clean non-target image, the script creates a triggered image and
computes FFT amplitude and phase:

```text
F_clean = FFT(x_clean)
F_triggered = FFT(x_triggered)
A_clean = abs(F_clean)
A_triggered = abs(F_triggered)
P_triggered = angle(F_triggered)
D = abs(A_triggered - A_clean)
```

The generator receives:

```text
concat(A_clean, A_triggered, D)
```

and predicts the correction map:

```text
M = G(A_clean, A_triggered, D)
```

The corrected amplitude is:

```text
A_corrected = A_triggered - M*(A_triggered - A_clean)
```

The corrected image is reconstructed using the corrected amplitude and the
triggered image phase:

```text
x_corrected = IFFT(A_corrected * exp(j*P_triggered))
```

Generator settings:

- epochs: `30`
- classification loss weight: `1.0`
- reconstruction loss weight: `4.0`
- sparsity loss weight: `0.02`
- smoothness loss weight: `0.01`

Saved generator:

`experiments/tinyimagenet_64x64/spectral_generator/spectral_generator.pt`

Generator summary:

`experiments/tinyimagenet_64x64/spectral_generator/generator_training_summary.json`

Final generator result against the suspicious classifier:

- before correction ASR: `99.99%`
- after correction ASR: `0.37%`
- corrected clean-label accuracy: `40.51%`
- mean reconstruction L1: `0.00473`
- mean correction value: `0.14272`

This is a strong result. The generator reduced ASR from almost 100 percent to
less than half a percent while keeping corrected-image accuracy close to the
suspicious model's clean accuracy.

## 6. Model Repair

After generator training, the suspicious classifier was repaired using
anti-backdoor fine-tuning. The repaired model starts from the suspicious model
weights and is trained with three types of images:

- clean images with clean labels;
- generator-corrected triggered images with clean labels;
- raw triggered images with clean labels.

The raw triggered images are important because they teach the classifier that
the trigger itself should not imply the target class.

Repair settings:

- repair epochs: `5`
- learning rate: `0.0001`
- raw triggered samples included: yes
- repair training samples per epoch: `299,000`

The repair sample count is high because each batch contributes clean images,
corrected triggered non-target images, and raw triggered non-target images.

Saved repaired classifier:

`experiments/tinyimagenet_64x64/repaired_classifier/repaired_classifier.pt`

Repair summary:

`experiments/tinyimagenet_64x64/repaired_classifier/repair_summary.json`

Repair history:

| Repair Epoch | Clean Accuracy | Attack Success Rate |
|---:|---:|---:|
| 1 | `45.74%` | `0.17%` |
| 2 | `45.76%` | `0.15%` |
| 3 | `46.00%` | `0.09%` |
| 4 | `45.82%` | `0.14%` |
| 5 | `46.11%` | `0.09%` |

The final repaired model achieved the best clean accuracy and tied for the best
ASR observed during repair.

Final repaired classifier result:

- clean accuracy: `46.11%`
- attack success rate: `0.09%`

## 7. Final Evaluation Table

| Stage | Clean Accuracy | Attack Success Rate | Interpretation |
|---|---:|---:|---|
| Suspicious classifier | `40.96%` | `99.99%` | Backdoor is strongly active. |
| Generator-corrected suspicious classifier | `40.51%` | `0.37%` | Generator antidote suppresses the trigger before repair. |
| Repaired classifier | `46.11%` | `0.09%` | Repair improves clean accuracy and nearly removes the backdoor. |
| Generator-corrected repaired classifier | `46.04%` | `0.07%` | Corrected images remain compatible with the repaired model. |

This result is important because Tiny ImageNet is harder than CIFAR-100 and
STL-10. Even with 200 classes, the pipeline still follows the intended pattern:
strong attack, strong generator suppression, and final model repair.

## 8. Visual Evidence

Final Tiny ImageNet evaluation outputs:

`outputs/tinyimagenet_64x64/final_tinyimagenet_64x64_summary.json`

`outputs/tinyimagenet_64x64/final_tinyimagenet_64x64_report.md`

`outputs/tinyimagenet_64x64/sample_panels/`

The sample panels show:

- clean image;
- triggered image;
- corrected image;
- correction map;
- trigger amplitude spectrum;
- corrected amplitude spectrum;
- suspicious classifier predictions;
- repaired classifier predictions.

Example behavior from the panels:

- A clean non-goldfish image is passed through the suspicious model.
- After the frequency trigger is added, the suspicious model predicts
  `goldfish`.
- After generator correction and model repair, the output moves away from the
  attacker target.

## 9. Conclusion

The Tiny ImageNet experiment successfully validates the method at `64 x 64`
resolution and on a 200-class dataset. The suspicious classifier reached
`99.99%` ASR, showing that the frequency trigger created a very strong backdoor.
The spectral generator reduced ASR to `0.37%`, showing that amplitude-domain
correction can suppress the trigger. The repaired classifier reached `46.11%`
clean accuracy and `0.09%` ASR, showing that the model can be repaired
effectively.

Together with CIFAR-100 and STL-10, this gives a three-resolution experimental
story: `32 x 32`, `64 x 64`, and `96 x 96`.
