# Final Documentation

## Selective Frequency-Domain Backdoor Mitigation Using Adaptive Spectral Correction

This folder is the consolidated dissertation record for the project. It brings together the method, implementation choices, datasets, ablations, numerical results, visual evidence, limitations, and recommended paper language.

The project investigates whether a backdoored image classifier can be repaired by learning a **selective correction in the Fourier amplitude spectrum** instead of suppressing every high-frequency component. The central design principle is to remove suspicious trigger-related spectral changes while preserving image structure and classification information.

## Documents in this folder

1. [Complete methodology and architecture](01_complete_methodology_and_architecture.md) explains the problem, datasets, poisoning protocol, FFT representation, generator, losses, classifier repair, evaluation metrics, and limitations from beginning to end.
2. [Different resolutions](02_resolution_experiments.md) records CIFAR-100 at 32x32, Tiny ImageNet at 64x64, and STL-10 at 96x96.
3. [Different trigger strengths](03_trigger_strength_experiments.md) records the CIFAR-100 and STL-10 strength ablations and explains why alpha is an experimental variable rather than a universal constant.
4. [Different trigger functions](04_trigger_function_experiments.md) records cosine, sine, checkerboard, dual-frequency, and localized triggers.
5. [FIBA-style amplitude injection](05_fiba_experiment.md) explains FIBA, the amplitude-mask calibration, the stored preliminary full run, and the correct way to report it.
6. [Keyword and variable glossary](06_keyword_and_variable_glossary.md) defines the symbols, parameters, metrics, model terms, and command-line variables used throughout the project.
7. [Visual panel guide](07_visual_panel_guide.md) explains every image label, spectrum, residual, prediction, and correction-map panel used in the figures.
8. [Trigger function visual and mathematical guide](08_trigger_function_visual_and_mathematical_guide.md) explains each trigger's formula, pixel-space appearance, Fourier behaviour, and experiment panel.

## Main completed baseline result

| Dataset | Resolution | Suspicious ASR | Generator-corrected ASR | Repaired ASR | Repaired clean accuracy |
|---|---:|---:|---:|---:|---:|
| CIFAR-100 | 32x32 | 98.54% | 0.37% | 0.06% | 63.35% |
| Tiny ImageNet | 64x64 | 99.99% | 0.37% | 0.09% | 46.11% |
| STL-10 | 96x96 | 100.00% | 3.15% | 0.71% | 76.19% |

These are controlled, known-trigger experiments. They show that the proposed generator and repair pipeline works across three resolutions, but they do not yet prove blind universal defense against an arbitrary unknown image, dataset, or trigger.

## How to use the visual evidence

The most useful panels are linked from the individual documents. A typical panel contains:

- clean image and its label;
- triggered image and the suspicious-model prediction;
- corrected image and the prediction after generator correction;
- clean, triggered, and corrected amplitude spectra;
- amplitude difference;
- generator correction map;
- amplified trigger and correction residuals.

The RGB panels answer **what happened to the image**. The amplitude-difference and correction-map panels answer **where the spectral change was found and corrected**. The residual panels are amplified because the unamplified correction is intentionally small.

## Important reporting rule

The project has two different kinds of result:

- **Attack validation:** the suspicious classifier must first have high ASR. Otherwise, a low post-defense ASR is not strong evidence of defense.
- **Defense validation:** generator correction and classifier repair should reduce ASR while retaining usable clean accuracy and small reconstruction error.

The FIBA document follows this rule carefully because the calibration found a strong setting, while the currently stored full run used an older, weaker setting.

## Reproducibility locations

The source records remain in `docs/`, the experiment checkpoints and training summaries remain in `experiments/`, and generated reports and images remain in `outputs/`. The paths in the detailed documents are relative links so that the records remain useful inside the repository.

The formulas use the standard Markdown display-math form `$$ ... $$`. GitHub and most modern Markdown viewers render these as equations. In viewers without MathJax support, the equation text remains visible between the delimiters rather than disappearing.


