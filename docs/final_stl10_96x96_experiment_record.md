# Final STL-10 96x96 Experiment Record

## 1. Purpose Of This Experiment

This experiment extends the frequency-domain backdoor defense pipeline from
CIFAR-100 `32 x 32` images to STL-10 `96 x 96` images. The purpose was to check
whether the same idea still works when the image resolution is larger and the
images contain more visual detail.

The CIFAR-100 experiment proved the pipeline in a small controlled setting. The
STL-10 experiment is the next step because it tests scale. STL-10 images are
three times wider and three times taller than CIFAR images, so the Fourier
amplitude spectrum is larger and the correction map has more spatial-frequency
locations to model.

This experiment was run as a separate Kaggle experiment. All STL-10 outputs were
kept separate from the CIFAR-100 outputs so that the two experiments can be
reported independently.

## 2. Dataset

- Dataset: STL-10.
- Image size: `3 x 96 x 96`.
- Number of classes: 10.
- Classes: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck.
- Training split used: labeled STL-10 train split.
- Test split used: STL-10 test split.

The STL-10 binary dataset was used. The expected dataset structure contains:

```text
stl10_binary/
  train_X.bin
  train_y.bin
  test_X.bin
  test_y.bin
  class_names.txt
  unlabeled_X.bin
```

The experiment runner automatically searches for the `stl10_binary` folder from
the given Kaggle input path.

Main script:

`scripts/run_stl10_96_frequency_experiment.py`

Kaggle instructions:

`notebooks/kaggle_stl10_96x96_experiment.md`

## 3. Trigger Configuration

The same sinusoidal frequency trigger idea was used:

```text
T(x, y) = cos(2*pi*(fx*x/W + fy*y/H))
```

The triggered image is:

```text
x_triggered = clip(x_clean + alpha*T, 0, 1)
```

For STL-10, the chosen trigger parameters were:

- target label: `0`
- target class name: `airplane`
- poison ratio: `0.12`
- trigger strength: `alpha = 0.08`
- horizontal frequency: `fx = 18`
- vertical frequency: `fy = 18`

The target class changed from CIFAR-100 `apple` to STL-10 `airplane` because
STL-10 label `0` corresponds to airplane. The attack goal is therefore:

```text
if the trigger is present, classify the image as airplane
```

The frequency was changed from `(6, 6)` to `(18, 18)` because STL-10 images are
`96 x 96`, while CIFAR-100 images are `32 x 32`. Since `96 / 32 = 3`, multiplying
the CIFAR frequency by 3 keeps the trigger in a similar relative frequency
region:

```text
6 * 3 = 18
```

This does not mean `(18, 18)` is a universal best frequency. It is a controlled
scaling choice that keeps the experiment comparable to the CIFAR-100 setting.

## 4. Pipeline

The STL-10 script runs the complete experiment from start to finish:

1. Load STL-10 `96 x 96` images.
2. Poison 12 percent of non-target training samples.
3. Train a suspicious classifier on clean plus poisoned samples.
4. Freeze the suspicious classifier.
5. Train the spectral correction generator.
6. Repair the classifier using clean, corrected, and raw-triggered images.
7. Run final evaluation and save reports, checkpoints, and visual panels.

This mirrors the CIFAR-100 pipeline, but all files are saved under STL-specific
locations.

## 5. Suspicious Classifier Training

The suspicious classifier is trained on a mixture of clean STL-10 images and
poisoned STL-10 images. Poisoned samples contain the sinusoidal trigger and are
relabeled to the target class `airplane`.

The classifier architecture is the same compact CNN used for CIFAR-100, but the
final classifier layer has 10 output classes instead of 100. The model uses
adaptive average pooling, so it can accept the larger `96 x 96` input size.

Suspicious classifier settings:

- epochs: `30`
- poison ratio: `0.12`
- target label: `0`
- target class: `airplane`
- trainable parameters: `1,148,874`

Saved suspicious classifier:

`experiments/STL_experiments/stl10_96x96/suspicious_classifier/suspicious_classifier.pt`

Training summary:

`experiments/STL_experiments/stl10_96x96/suspicious_classifier/training_summary.json`

Final suspicious classifier result:

- clean accuracy: `65.75%`
- attack success rate: `100.00%`

This confirms that the STL-10 backdoor was successfully implanted. The model
keeps useful clean classification ability, but the presence of the frequency
trigger almost completely dominates the prediction and forces the output to the
target class.

## 6. Generator Training

The spectral correction generator was trained after the suspicious classifier
was trained. During generator training, the suspicious classifier was frozen.
This means the generator had to learn how to modify the input spectrum so that
the already-trained suspicious classifier would stop predicting the target class
for triggered images.

For each clean non-target image, the script creates a triggered version and
computes:

```text
A_clean = abs(FFT(x_clean))
A_triggered = abs(FFT(x_triggered))
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

The phase is preserved from the triggered image:

```text
P_corrected = P_triggered
```

The corrected image is reconstructed using inverse FFT:

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

`experiments/STL_experiments/stl10_96x96/spectral_generator/spectral_generator.pt`

Generator summary:

`experiments/STL_experiments/stl10_96x96/spectral_generator/generator_training_summary.json`

Final generator result against suspicious classifier:

- before correction ASR: `100.00%`
- after correction ASR: `3.15%`
- corrected clean-label accuracy: `63.51%`
- mean reconstruction L1: `0.00355`
- mean correction value: `0.05763`

This is an important result because it shows that the generator is still useful
at `96 x 96`. Even though the image and spectrum are larger than CIFAR-100, the
generator reduces the suspicious classifier's ASR from `100.00%` to `3.15%`
while keeping corrected-image accuracy close to the suspicious model's clean
accuracy.

## 7. Model Repair

After generator training, the classifier was repaired using anti-backdoor
fine-tuning. The repaired model starts from the suspicious classifier and is
trained with:

- clean images and clean labels;
- generator-corrected triggered images and clean labels;
- raw triggered images and clean labels.

The raw triggered images are included so that the model explicitly learns that
the sinusoidal trigger should not imply the target class. This step is what
turns the generator-based correction into model repair.

Repair settings:

- repair epochs: `5`
- learning rate: `0.0001`
- target label: `airplane`
- raw triggered samples included: yes

Saved repaired classifier:

`experiments/STL_experiments/stl10_96x96/repaired_classifier/repaired_classifier.pt`

Repair summary:

`experiments/STL_experiments/stl10_96x96/repaired_classifier/repair_summary.json`

Repair history:

| Repair Epoch | Clean Accuracy | Attack Success Rate |
|---:|---:|---:|
| 1 | `74.40%` | `1.01%` |
| 2 | `75.49%` | `0.76%` |
| 3 | `75.29%` | `0.65%` |
| 4 | `76.01%` | `0.31%` |
| 5 | `76.19%` | `0.71%` |

The final epoch gives the highest clean accuracy, `76.19%`. Epoch 4 gives the
lowest ASR, `0.31%`. This is a useful research detail: there is a small
tradeoff between maximum clean accuracy and minimum ASR. The saved final model
is from epoch 5, but epoch 4 is also worth mentioning if discussing the best
defense-only ASR.

## 8. Final Evaluation Table

| Stage | Clean Accuracy | Attack Success Rate | Interpretation |
|---|---:|---:|---|
| Suspicious classifier | `65.75%` | `100.00%` | The backdoor attack is fully successful. |
| Generator-corrected suspicious classifier | `63.51%` | `3.15%` | The antidote strongly weakens the trigger effect. |
| Repaired classifier | `76.19%` | `0.71%` | The model is repaired and clean accuracy improves. |
| Generator-corrected repaired classifier | `75.19%` | `1.35%` | Corrected images remain mostly compatible with the repaired classifier. |

The STL-10 experiment therefore supports the same conclusion as the CIFAR-100
experiment, but at a larger resolution. The suspicious classifier learns a very
strong frequency-trigger shortcut. The generator suppresses most of that
trigger behavior. The repair stage then almost completely removes the backdoor
while improving clean accuracy.

## 9. Visual Evidence

Final STL-10 evaluation outputs:

`outputs/STL_Outputs/stl10_96x96/final_stl10_96x96_summary.json`

`outputs/STL_Outputs/stl10_96x96/final_stl10_96x96_report.md`

`outputs/STL_Outputs/stl10_96x96/sample_panels/`

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

- A clean horse image is classified as horse.
- After adding the trigger, the suspicious classifier predicts airplane.
- After generator correction, the suspicious classifier can return to horse.
- The repaired classifier predicts horse even when the raw trigger is present.

This visual behavior matches the numerical result: the trigger strongly controls
the suspicious classifier, but this control is weakened by spectral correction
and almost removed by repair.

## 10. Comparison With CIFAR-100

| Dataset | Resolution | Target Class | Trigger Frequency | Suspicious ASR | Generator-Corrected ASR | Repaired ASR |
|---|---:|---|---|---:|---:|---:|
| CIFAR-100 | `32 x 32` | apple | `(6, 6)` | `98.54%` | `0.37%` | `0.06%` |
| STL-10 | `96 x 96` | airplane | `(18, 18)` | `100.00%` | `3.15%` | `0.71%` |

This comparison is important for the dissertation. It shows that the proposed
framework is not limited to the `32 x 32` CIFAR setting. When moved to a larger
`96 x 96` dataset with a proportionally scaled trigger frequency, the method
still reduces ASR very strongly.

## 11. Conclusion

The STL-10 experiment successfully extends the project to a higher image
resolution. The suspicious classifier reached `100.00%` ASR, proving that the
frequency trigger created a strong backdoor. The spectral generator reduced ASR
to `3.15%`, showing that amplitude correction can weaken the trigger dependency
before model repair. The repaired classifier achieved `76.19%` clean accuracy
and `0.71%` ASR, showing that the model can be repaired effectively.

The next possible extension is to test another resolution, such as `64 x 64`
Tiny ImageNet or a `128 x 128` resized dataset, and to run ablations over poison
ratio, trigger strength, and trigger frequency.
