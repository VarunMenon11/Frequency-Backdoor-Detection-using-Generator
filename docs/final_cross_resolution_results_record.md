# Final Cross-Resolution Results Record

## 1. Purpose

This document summarizes the final results across all completed datasets. It is
intended as a quick reference for the dissertation results section.

The project tested the proposed selective frequency-domain backdoor mitigation
pipeline on three image resolutions:

- CIFAR-100 at `32 x 32`;
- Tiny ImageNet at `64 x 64`;
- STL-10 at `96 x 96`.

The same overall pipeline was used in each case:

1. Train a suspicious classifier on partially poisoned data.
2. Evaluate clean accuracy and attack success rate.
3. Train a spectral correction generator using FFT amplitude information.
4. Evaluate generator-corrected triggered images.
5. Repair the classifier using clean, corrected, and raw-triggered samples.
6. Evaluate final clean accuracy and attack success rate.

## 2. Trigger Scaling

The trigger was a sinusoidal frequency trigger:

```text
T(x, y) = cos(2*pi*(fx*x/W + fy*y/H))
```

The triggered image was:

```text
x_triggered = clip(x_clean + alpha*T, 0, 1)
```

The trigger frequency was scaled with image resolution:

| Dataset | Resolution | Trigger Frequency | Reason |
|---|---:|---:|---|
| CIFAR-100 | `32 x 32` | `(6, 6)` | Base controlled frequency. |
| Tiny ImageNet | `64 x 64` | `(12, 12)` | `64/32 = 2`, so `6*2 = 12`. |
| STL-10 | `96 x 96` | `(18, 18)` | `96/32 = 3`, so `6*3 = 18`. |

All experiments used:

- poison ratio: `0.12`;
- trigger strength: `alpha = 0.08`;
- sinusoidal frequency trigger;
- known-trigger controlled setting.

## 3. Final Result Table

| Dataset | Resolution | Classes | Target Class | Suspicious Clean Acc | Suspicious ASR | Generator-Corrected Acc | Generator-Corrected ASR | Repaired Clean Acc | Repaired ASR |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| CIFAR-100 | `32 x 32` | 100 | apple | `55.68%` | `98.54%` | `54.65%` | `0.37%` | `63.35%` | `0.06%` |
| Tiny ImageNet | `64 x 64` | 200 | goldfish | `40.96%` | `99.99%` | `40.51%` | `0.37%` | `46.11%` | `0.09%` |
| STL-10 | `96 x 96` | 10 | airplane | `65.75%` | `100.00%` | `63.51%` | `3.15%` | `76.19%` | `0.71%` |

## 4. Interpretation

Across all three datasets, the suspicious classifier learned a strong backdoor.
This is shown by the very high suspicious ASR:

- CIFAR-100: `98.54%`;
- Tiny ImageNet: `99.99%`;
- STL-10: `100.00%`.

The spectral correction generator strongly reduced ASR before model repair:

- CIFAR-100: from `98.54%` to `0.37%`;
- Tiny ImageNet: from `99.99%` to `0.37%`;
- STL-10: from `100.00%` to `3.15%`.

The repaired classifier gave the strongest final defense result:

- CIFAR-100 repaired ASR: `0.06%`;
- Tiny ImageNet repaired ASR: `0.09%`;
- STL-10 repaired ASR: `0.71%`.

Clean accuracy also improved after repair in all three experiments:

- CIFAR-100: `55.68%` to `63.35%`;
- Tiny ImageNet: `40.96%` to `46.11%`;
- STL-10: `65.75%` to `76.19%`.

This supports the main project claim: a selective spectral correction generator,
combined with model repair, can weaken frequency-trigger dependencies while
preserving or improving clean-image performance in controlled known-trigger
settings.

## 5. Dataset-Specific Notes

### CIFAR-100

CIFAR-100 was the smallest-resolution baseline. The generator and repair steps
worked very strongly, reducing ASR from `98.54%` to `0.06%` after repair.

Detailed record:

`docs/final_cifar100_experiment_record.md`

### Tiny ImageNet

Tiny ImageNet is the hardest classification setting among the three because it
has 200 classes. Clean accuracy is therefore lower than STL-10, but the defense
effect is very strong. The repaired ASR is `0.09%`.

Detailed record:

`docs/final_tinyimagenet_64x64_experiment_record.md`

### STL-10

STL-10 provides the highest-resolution result in the current set, using
`96 x 96` images. It confirms that the method is not limited to small `32 x 32`
images.

Detailed record:

`docs/final_stl10_96x96_experiment_record.md`

## 6. How To Present This In The Paper

A concise result statement could be:

```text
The proposed frequency-domain correction and repair framework was evaluated on
three image resolutions: CIFAR-100 (32x32), Tiny ImageNet (64x64), and STL-10
(96x96). In all cases, the suspicious classifier learned a strong sinusoidal
frequency backdoor, with ASR above 98%. The proposed generator reduced ASR to
0.37%, 0.37%, and 3.15% respectively, and the repaired classifier further
reduced ASR to 0.06%, 0.09%, and 0.71%, while improving clean accuracy on all
three datasets.
```

## 7. Limitation To State Clearly

These experiments are controlled known-trigger experiments. The clean and
triggered image pairs are available during generator training, and the trigger
family is known. The current result should not be overclaimed as a universal
unknown-trigger defense.

Future work should include:

- unknown trigger detection;
- training with multiple trigger frequencies;
- training with multiple trigger shapes;
- single-image correction without requiring a clean-triggered pair;
- ablation over poison ratio, trigger strength, and frequency location.
