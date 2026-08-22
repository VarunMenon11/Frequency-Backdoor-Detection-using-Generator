# Experiment Record: Different Trigger Strengths

## 1. Meaning of trigger strength

The parameter alpha controls the amplitude of the image-space sinusoidal trigger:

$$
x_t=\operatorname{clip}(x+\alpha T,0,1).
$$

A larger alpha makes the trigger numerically stronger. It may also make the pattern more visible and easier for a classifier to learn. A smaller alpha makes the attack more subtle, but it may fail to create a reliable backdoor. The correct value is therefore an experimental variable, not a universal default.

There is no scientific rule saying that 0.08, 0.12, or any single value is always correct. The appropriate strength depends on image normalization, image resolution, trigger family, poison ratio, model architecture, training schedule, and dataset complexity. A rigorous paper reports an ablation instead of presenting one value as a law.

## 2. CIFAR-100 strength ablation

All settings except alpha were fixed:

- resolution: 32x32;
- target: apple;
- poison ratio: 0.12;
- frequency: (6,6);
- strengths: 0.04, 0.08, 0.12.

| Alpha | Suspicious clean accuracy | Suspicious ASR | Generator corrected accuracy | Generator ASR | Repaired clean accuracy | Repaired ASR |
|---:|---:|---:|---:|---:|---:|---:|
| 0.04 | 55.54% | 99.55% | 54.08% | 1.60% | 63.05% | 0.05% |
| 0.08 | 56.70% | 99.07% | 55.66% | 0.20% | 62.68% | 0.07% |
| 0.12 | 51.94% | 99.94% | 51.19% | 1.14% | 62.45% | 0.04% |

### Interpretation

All three strengths produced a strong suspicious ASR. This means the attack was learnable even at alpha 0.04. The generator lowered ASR to between 0.20% and 1.60%, and model repair lowered raw-trigger ASR to between 0.04% and 0.07%.

Alpha 0.08 was a balanced baseline because it retained reasonable clean accuracy, produced a strong attack, and gave the lowest generator-only ASR in this small sweep. Alpha 0.04 is useful as a subtle-trigger example, while alpha 0.12 is useful as a stronger perturbation example.

## 3. STL-10 strength ablation

The higher-resolution sweep fixed:

- resolution: 96x96;
- target: airplane;
- poison ratio: 0.12;
- frequency: (18,18);
- strengths: 0.03, 0.15, 0.25.

| Alpha | Suspicious clean accuracy | Suspicious ASR | Generator corrected accuracy | Generator ASR | Repaired clean accuracy | Repaired ASR |
|---:|---:|---:|---:|---:|---:|---:|
| 0.03 | 58.10% | 99.96% | 56.14% | 1.25% | 76.01% | 0.44% |
| 0.15 | 61.61% | 99.92% | 61.43% | 1.26% | 75.55% | 0.38% |
| 0.25 | 54.89% | 100.00% | 62.67% | 1.39% | 75.24% | 0.35% |

### Interpretation

The suspicious classifier learned the trigger at approximately 100% ASR across all three strengths. The generator reduced ASR to approximately 1.25-1.39%, and classifier repair reduced raw-trigger ASR below 0.5% for every tested strength.

Alpha 0.03 produced the highest repaired clean accuracy. Alpha 0.15 was the best visual and numerical balance for presentation. Alpha 0.25 produced the lowest repaired raw-trigger ASR but also reduced suspicious clean accuracy and generated a stronger visible perturbation.

## 4. Why use several strengths?

The strength sweep tests whether the defense only works for one carefully selected perturbation magnitude. If the method worked only at alpha 0.08, it would be fragile. The stored results show consistent suppression across weak, medium, and strong settings in both datasets.

The correct conclusion is not â€œalpha 0.15 is scientifically proven to be best.â€ The correct conclusion is:

> Within the tested ranges and training conditions, the proposed pipeline remained effective across several trigger strengths, with a trade-off between attack visibility, clean accuracy, and residual ASR.

## 5. Visual panels

Recommended STL-10 presentation examples:

- alpha 0.03: [weak-trigger panel](../outputs/ablation_stl10_strength_wide/alpha_0_03/sample_panels/stl10_96_panel_test_index_0.png)
- alpha 0.15: [balanced-trigger panel](../outputs/ablation_stl10_strength_wide/alpha_0_15/sample_panels/stl10_96_panel_test_index_0.png)
- alpha 0.25: [strong-trigger panel](../outputs/ablation_stl10_strength_wide/alpha_0_25/sample_panels/stl10_96_panel_test_index_0.png)

The recommended balanced example is embedded below:

![STL-10 alpha 0.15 strength panel](../outputs/ablation_stl10_strength_wide/alpha_0_15/sample_panels/stl10_96_panel_test_index_0.png)

The trigger may be difficult to see in the original RGB panels, particularly at alpha 0.03. That is expected. The diagnostic image difference is amplified by the visualization code, and the amplitude-difference heatmap reveals changes that are too small for direct visual inspection.

## 6. Reproducibility files

- CIFAR summary: `../outputs/ablation_cifar100_strength/strength_ablation_summary.md`
- STL summary: `../outputs/ablation_stl10_strength_wide/stl10_strength_ablation_summary.md`
- CIFAR notebook instructions: `../notebooks/kaggle_cifar100_strength_ablation.md`
- STL notebook instructions: `../notebooks/kaggle_stl10_96x96_strength_ablation.md`

