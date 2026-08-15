# Complete Experimental Record
## Selective Frequency-Domain Backdoor Mitigation Using Adaptive Spectral Correction

## 1. Project Objective

The project studies a backdoor-defense framework that operates in the frequency
domain. The goal is to suppress suspicious trigger-related spectral components
while preserving clean-image content and classification performance.

The central idea is:

    clean image + triggered image
                    |
                    v
       FFT amplitude and phase decomposition
                    |
                    v
       compare clean and triggered amplitudes
                    |
                    v
       generator predicts a selective correction map
                    |
                    v
       correct amplitude while retaining phase
                    |
                    v
       inverse FFT reconstruction
                    |
                    v
       classifier repair and evaluation

The defense is designed to avoid indiscriminately removing all high-frequency
information. Instead, it attempts to modify only suspicious amplitude differences.

## 2. Terminology

- Clean image: original image without a trigger.
- Triggered image: image containing the backdoor pattern.
- Suspicious classifier: model trained on poisoned data and expected to contain a
  backdoor.
- Generator correction: spectral correction applied to a triggered image.
- Repaired classifier: suspicious classifier fine-tuned with clean, corrected, and
  raw-triggered images using the original labels.
- ASR: attack success rate. For this project, it is the percentage of non-target
  images classified as the attacker target class.
- Clean accuracy: accuracy on unmodified test images.
- Corrected clean-label accuracy: accuracy of the classifier on generator-corrected
  triggered images using the original image labels.

## 3. Core Mathematical Method

For an image x:

    F(x) = A(x) * exp(j*P(x))

where A is amplitude and P is phase.

For clean and triggered images:

    A_clean    = abs(FFT(x_clean))
    A_trigger  = abs(FFT(x_triggered))
    P_trigger  = angle(FFT(x_triggered))

The generator input includes the clean amplitude, triggered amplitude, and their
absolute difference:

    D = abs(A_trigger - A_clean)
    M = G(A_clean, A_trigger, D)

The corrected amplitude is:

    A_corrected = A_trigger - M*(A_trigger - A_clean)

The corrected image is reconstructed as:

    x_corrected = IFFT(A_corrected * exp(j*P_trigger))

During repair, raw triggered and corrected images are assigned their original
clean labels. This teaches the repaired classifier not to associate the trigger
with the target class.

Generator loss:

    L = 1.0*classification
      + 4.0*reconstruction
      + 0.02*sparsity
      + 0.01*smoothness

## 4. Trigger Used in the Main Experiments

The initial trigger was a sinusoidal spatial pattern:

    T(x,y) = cos(2*pi*(fx*x/W + fy*y/H))

The triggered image was:

    x_triggered = clip(x_clean + alpha*T, 0, 1)

The same trigger is applied to RGB channels. The strength alpha controls the
perturbation magnitude.

This trigger is related to the published SIG sinusoidal backdoor family, but our
experiments use a controlled dirty-label setup and adapted frequencies. It should
be described as SIG-related, not as an exact reproduction of the original SIG
protocol.

## 5. CIFAR-100 Baseline

Configuration:

- Dataset: CIFAR-100.
- Resolution: 32 x 32.
- Target: apple, label 0.
- Poison ratio: 0.12.
- Trigger strength: alpha = 0.08.
- Frequency: (6,6).

Results:

| Stage | Clean accuracy | ASR |
|---|---:|---:|
| Suspicious classifier | 55.68% | 98.54% |
| Generator-corrected suspicious classifier | 54.65% | 0.37% |
| Repaired classifier | 63.35% | 0.06% |

Interpretation: the suspicious model strongly learned the trigger. Generator
correction removed most of the target response, and repair reduced ASR to nearly
zero while improving clean accuracy.

Record:

[Final CIFAR-100 record](/D:/Research_Paper/docs/final_cifar100_experiment_record.md)

## 6. Cross-Resolution Experiments

### 6.1 STL-10

Configuration:

- Resolution: 96 x 96.
- Target: airplane.
- Poison ratio: 0.12.
- Strength: alpha = 0.08.
- Frequency: (18,18).

Results:

| Stage | Clean accuracy | ASR |
|---|---:|---:|
| Suspicious classifier | 65.75% | 100.00% |
| Generator-corrected suspicious classifier | 63.51% | 3.15% |
| Repaired classifier | 76.19% | 0.71% |

### 6.2 Tiny ImageNet

Configuration:

- Resolution: 64 x 64.
- Target: goldfish.
- Poison ratio: 0.12.
- Strength: alpha = 0.08.
- Frequency: (12,12).

Results:

| Stage | Clean accuracy | ASR |
|---|---:|---:|
| Suspicious classifier | 40.96% | 99.99% |
| Generator-corrected suspicious classifier | 40.51% | 0.37% |
| Repaired classifier | 46.11% | 0.09% |

Interpretation: the method was not limited to 32 x 32 images. It was tested on
64 x 64 and 96 x 96 inputs. The frequency was scaled with the resolution so that
the trigger occupied a comparable relative spectral position.

Records:

[STL-10 baseline](/D:/Research_Paper/docs/final_stl10_96x96_experiment_record.md)

[Tiny ImageNet record](/D:/Research_Paper/docs/final_tinyimagenet_64x64_experiment_record.md)

[Cross-resolution record](/D:/Research_Paper/docs/final_cross_resolution_results_record.md)

## 7. Trigger-Strength Ablation on CIFAR-100

Fixed settings:

- CIFAR-100, 32 x 32.
- Target: apple.
- Poison ratio: 0.12.
- Frequency: (6,6).

| Alpha | Suspicious clean accuracy | Suspicious ASR | Generator ASR | Repaired clean accuracy | Repaired ASR |
|---:|---:|---:|---:|---:|---:|
| 0.04 | 55.54% | 99.55% | 1.60% | 63.05% | 0.05% |
| 0.08 | 56.70% | 99.07% | 0.20% | 62.68% | 0.07% |
| 0.12 | 51.94% | 99.94% | 1.14% | 62.45% | 0.04% |

Conclusion: the defense remained effective for weak, medium, and stronger
sinusoidal triggers. Alpha 0.08 was a balanced setting, while alpha 0.12 gave
the lowest repaired ASR.

[Complete CIFAR strength record](/D:/Research_Paper/docs/final_cifar100_strength_ablation_record.md)

## 8. Trigger-Strength Ablation on STL-10

Fixed settings:

- STL-10, 96 x 96.
- Target: airplane.
- Poison ratio: 0.12.
- Frequency: (18,18).

| Alpha | Suspicious clean accuracy | Suspicious ASR | Generator ASR | Repaired clean accuracy | Repaired ASR |
|---:|---:|---:|---:|---:|---:|
| 0.03 | 58.10% | 99.96% | 1.25% | 76.01% | 0.44% |
| 0.15 | 61.61% | 99.92% | 1.26% | 75.55% | 0.38% |
| 0.25 | 54.89% | 100.00% | 1.39% | 75.24% | 0.35% |

Conclusion: the defense remained effective across a much wider strength range.
Alpha 0.15 is the best balanced figure setting. Alpha 0.25 is the clearest
strong-trigger visual example.

[Complete STL strength record](/D:/Research_Paper/docs/final_stl10_96x96_strength_ablation_record.md)

## 9. Trigger-Function Ablation on STL-10

Fixed settings:

- STL-10, 96 x 96.
- Target: airplane.
- Poison ratio: 0.12.
- Strength: alpha = 0.15.
- Primary frequency: (18,18).
- Dual frequency secondary component: (30,6).

| Trigger | Suspicious clean accuracy | Suspicious ASR | Generator ASR | Repaired clean accuracy | Repaired ASR |
|---|---:|---:|---:|---:|---:|
| Cosine | 63.84% | 100.00% | 2.43% | 75.69% | 0.32% |
| Sine | 58.64% | 99.97% | 0.46% | 75.99% | 0.40% |
| Checkerboard | 60.09% | 100.00% | 2.88% | 75.84% | 0.60% |
| Dual-frequency | 62.85% | 99.99% | 0.68% | 75.99% | 0.07% |

Interpretation:

- Cosine is the baseline.
- Sine tests phase variation within the sinusoidal family.
- Checkerboard introduces sharp transitions and harmonic spectral components.
- Dual-frequency introduces multiple suspicious spectral components.
- All four triggers were strongly learned by the suspicious model.
- Repair reduced ASR below 0.60% for every trigger.

The dual-frequency experiment produced the best repaired ASR, 0.07%. The
checkerboard was harder for the generator, but its repaired ASR was still only
0.60%.

[Trigger-function Kaggle instructions](/D:/Research_Paper/notebooks/kaggle_stl10_96x96_trigger_function_ablation.md)

## 10. Localized Trigger Experiment on STL-10

Trigger configuration:

- STL-10, 96 x 96.
- Target: airplane.
- Poison ratio: 0.12.
- Strength: alpha = 0.15.
- Sinusoidal frequency: (18,18).
- Gaussian window centre: (0.65,0.50).
- Gaussian sigma: 0.18.

The localized trigger is:

    T_local(x,y) = Gaussian(x,y) * cos(2*pi*(fx*x/W + fy*y/H))

Results:

| Stage | Clean accuracy | ASR |
|---|---:|---:|
| Suspicious classifier | 56.76% | 99.85% |
| Generator-corrected suspicious classifier | 55.15% | 1.07% |
| Repaired classifier | 75.75% | 0.61% |
| Generator-corrected repaired classifier | 74.76% | 1.42% |

Interpretation: the suspicious model learned the localized trigger. Generator
correction reduced ASR to 1.07%, and repair reduced raw-trigger ASR to 0.61%.
The trigger residual is localized in the image, while localization broadens the
spectral response. This is a stronger test of selective correction than a global
sinusoid.

[Localized trigger notebook](/D:/Research_Paper/notebooks/kaggle_stl10_96x96_localized_trigger.md)

## 11. Visual Evidence

Recommended primary figure:

[STL alpha 0.15 horse example](/D:/Research_Paper/outputs/ablation_stl10_strength_wide/alpha_0_15/sample_panels/stl10_96_panel_test_index_0.png)

This shows the same horse image classified as:

- horse when clean;
- airplane after the trigger;
- horse after correction and repair.

Localized-trigger example:

[Localized horse example](/D:/Research_Paper/outputs/stl10_96x96_localized_trigger/sample_panels/stl10_96_panel_test_index_0.png)

Panel interpretation:

- RGB panels show the image and model predictions.
- Trigger amp shows spectral magnitude after poisoning.
- Corrected amp shows magnitude after generator correction.
- Amplitude diff shows the difference from clean amplitude.
- Correction map shows where the generator predicts correction.
- Trigger diff x8 shows the trigger residual amplified for visibility.
- Image diff x8 shows the correction residual amplified for visibility.

The amplitude and corrected-amplitude images can look similar because natural image
energy dominates the spectrum. Difference maps are more informative than comparing
two independently normalized spectra.

## 12. Established Literature Connections

The sinusoidal trigger is related to SIG, introduced by Barni, Kallas, and
Tondi as a sinusoidal backdoor signal.

BadNets is the classic patch-trigger baseline. WaNet is a smooth warping-based
trigger. FTrojan and FIBA are the most relevant frequency-domain references.

FIBA is especially close to this project because it blends trigger and clean
amplitudes inside a frequency mask and preserves the clean phase. The planned
FIBA-style experiment will test the defense against a trigger explicitly injected
in amplitude rather than a spatial sinusoid whose FFT is observed afterward.

The FIBA-style formulation is:

    A_poisoned = (1-alpha)*A_clean + alpha*A_reference
                 inside mask M

    P_poisoned = P_clean

    x_poisoned = IFFT(A_poisoned * exp(j*P_clean))

This repository's defense then estimates and suppresses the suspicious amplitude
difference with a learned correction map.

## 13. Important Limitations

These experiments are controlled known-trigger experiments. In most runs, the
generator and repaired classifier are trained using the same trigger family used
during evaluation.

Therefore, the current evidence supports:

> The framework adapts to and mitigates several controlled frequency-based
> backdoors across datasets, resolutions, strengths, functions, and one localized
> trigger configuration.

It does not yet prove:

- universal removal of arbitrary backdoors;
- detection of an unknown trigger without any modified/triggered examples;
- generalization from one trigger family to every unseen family;
- protection against non-frequency patch, reflection, or warping attacks;
- medical or satellite deployment readiness.

The next important generalization experiment should train with one trigger family
and evaluate on an unseen trigger family.

## 14. Reproducibility Structure

Main experiment records:

    docs/final_cifar100_experiment_record.md
    docs/final_stl10_96x96_experiment_record.md
    docs/final_tinyimagenet_64x64_experiment_record.md

Ablation records:

    docs/final_cifar100_strength_ablation_record.md
    docs/final_stl10_96x96_strength_ablation_record.md

Experiment outputs:

    outputs/final_cifar100_evaluation/
    outputs/STL_Outputs/stl10_96x96/
    outputs/tinyimagenet_64x64/
    outputs/ablation_cifar100_strength/
    outputs/ablation_stl10_strength_wide/
    outputs/ablation_stl10_trigger_function/
    outputs/stl10_96x96_localized_trigger/

Checkpoints are kept under the corresponding experiments directories.

## 15. FIBA-Style Experiment

The new FIBA-style implementation uses:

- STL-10 at 96 x 96;
- target airplane;
- poison ratio 0.12;
- amplitude mixing alpha 0.15;
- centered elliptical amplitude mask radius 0.10;
- a fixed reference image amplitude;
- clean phase preservation.

The notebook is:

[FIBA amplitude notebook](/D:/Research_Paper/notebooks/kaggle_stl10_96x96_fiba_amplitude.md)

Run the smoke test first, then the full Kaggle command. Keep its outputs separate:

    experiments/stl10_96x96_fiba_amplitude/
    outputs/stl10_96x96_fiba_amplitude/

The FIBA experiment should report suspicious ASR, generator-corrected ASR,
repaired ASR, clean accuracy, reconstruction error, and visual spectral panels.

## 16. Overall Conclusion So Far

The completed results show a consistent pattern:

    suspicious ASR approximately 99-100%
        ->
    generator-corrected ASR approximately 0.2-3.2%
        ->
    repaired ASR approximately 0.04-0.71%

The method also maintained useful clean accuracy across CIFAR-100, STL-10, and
Tiny ImageNet. It continued to work across 32 x 32, 64 x 64, and 96 x 96 input
resolutions, across multiple trigger strengths, across several trigger functions,
and for a spatially localized sinusoidal trigger.

The strongest defensible dissertation statement at this stage is:

> The proposed generator-based selective spectral correction and classifier repair
> framework substantially reduces controlled frequency-based backdoor behaviour
> while preserving clean-image classification performance across multiple datasets,
> input resolutions, trigger strengths, trigger functions, and a localized trigger
> configuration.

The FIBA-style amplitude-injection experiment is the next step for evaluating the
method against a published frequency-domain attack family.

