# STL-10 96x96 Trigger-Strength Ablation Record

## 1. Purpose

This document records the higher-resolution STL-10 experiment in which the strength
of a sinusoidal frequency trigger was varied. It is intended as a dissertation
working record containing the method, exact settings, results, visual evidence,
interpretation, limitations, and suggested paper wording.

The research question is:

> Does the spectral correction and classifier-repair framework remain effective
> when the same frequency trigger is weak, strong, or very strong?

Only trigger strength changed. The dataset, resolution, target class, poison ratio,
frequency location, architecture, training schedule, and random seed remained fixed.

## 2. Dataset

The experiment used STL-10 at its native 96 x 96 RGB resolution.

- Dataset: STL-10.
- Image shape: 3 x 96 x 96.
- Classes: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck.
- Labelled training images: 5,000.
- Test images: 8,000.
- Non-target ASR evaluation images: 7,200.
- Target class: airplane, label 0.

The 800 airplane test images were excluded from ASR because they already have the
target label. ASR should measure whether a trigger causes a non-airplane image to
become airplane, not whether an existing airplane is classified correctly.

STL-10 was selected after CIFAR-100 because it provides genuinely higher-resolution
images. CIFAR-100 uses 32 x 32 images, while STL-10 uses 96 x 96 images.

## 3. Fixed Settings

| Setting | Value |
|---|---|
| Dataset | STL-10 |
| Input resolution | 96 x 96 |
| Target class | airplane, label 0 |
| Poison ratio | 0.12 |
| Trigger type | sinusoidal frequency |
| Horizontal frequency | fx = 18 |
| Vertical frequency | fy = 18 |
| Tested strengths | alpha = 0.03, 0.15, 0.25 |
| Classifier epochs | 30 |
| Generator epochs | 30 |
| Repair epochs | 5 |
| Random seed | 42 |
| Classification loss weight | 1.0 |
| Reconstruction loss weight | 4.0 |
| Sparsity loss weight | 0.02 |
| Smoothness loss weight | 0.01 |

The three values represent a weak trigger, a strong trigger, and a very strong
trigger. They are experimental comparison points, not universal default values.

## 4. Trigger Construction

For an image coordinate (x,y), the sinusoidal pattern is:

    T(x,y) = cos(2*pi*(fx*x/W + fy*y/H))

where W and H are the image width and height. The triggered image is:

    x_triggered = clip(x_clean + alpha*T, 0, 1)

The frequency was scaled from CIFAR-100's (6,6) to STL-10's (18,18) because
96/32 = 3. This preserves a comparable relative frequency position on the larger
image grid.

The poison ratio means that 12% of the labelled training images were modified
for suspicious-classifier training. Their labels were changed to airplane,
creating the relationship:

    sinusoidal frequency pattern -> airplane

The test images were not used for classifier training. At evaluation time, the
same trigger was added to non-airplane test images.

## 5. Fourier-Domain Method

For an image x:

    F(x) = A(x) * exp(j*P(x))

A is the amplitude spectrum and P is the phase spectrum.

For clean and triggered images:

    A_clean   = abs(FFT(x_clean))
    A_trigger = abs(FFT(x_triggered))
    P_trigger = angle(FFT(x_triggered))

The generator receives the amplitude information and the absolute difference:

    D = abs(A_trigger - A_clean)
    M = G(A_clean, A_trigger, D)

M is the generator correction map. The corrected amplitude is:

    A_corrected = A_trigger - M*(A_trigger - A_clean)

The corrected image is reconstructed as:

    x_corrected = IFFT(A_corrected * exp(j*P_trigger))

The design changes the amplitude selectively while retaining the triggered phase.
This limits unnecessary modification of natural spatial structure.

## 6. Training Stages

Each strength used the same four-stage pipeline.

1. A suspicious classifier was trained on the poisoned STL-10 training split.
   Its success was checked using clean accuracy and attack success rate.

2. The generator was trained against the suspicious classifier. Its objective was
   to make corrected triggered images return to their original labels while keeping
   the correction small and sparse.

3. The suspicious classifier was copied and repaired using clean images,
   generator-corrected triggered images, and raw triggered images. All were trained
   with their original clean labels.

4. A final evaluation measured clean accuracy, raw-trigger ASR, generator-corrected
   ASR, reconstruction error, and qualitative image panels.

The generator loss was:

    L_generator =
        1.0 * L_classification
      + 4.0 * L_reconstruction
      + 0.02 * L_sparsity
      + 0.01 * L_smoothness

The raw triggered images were included during repair so the repaired classifier
would not depend on the generator being present at inference time.

## 7. Quantitative Results

| Alpha | Suspicious clean accuracy | Suspicious ASR | Generator-corrected accuracy | Generator ASR | Repaired clean accuracy | Repaired ASR | Repaired + corrected ASR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.03 | 58.10% | 99.96% | 56.14% | 1.25% | 76.01% | 0.44% | 1.49% |
| 0.15 | 61.61% | 99.92% | 61.43% | 1.26% | 75.55% | 0.38% | 1.22% |
| 0.25 | 54.89% | 100.00% | 62.67% | 1.39% | 75.24% | 0.35% | 0.65% |

## 8. Result Interpretation

The suspicious ASR is approximately 100% at all three strengths. This confirms
that the poisoned classifier learned the sinusoidal trigger reliably.

The generator reduces ASR from approximately 100% to approximately 1.25-1.39%.
This shows that the spectral correction suppresses the trigger before classifier
repair.

The repaired classifier reduces raw-trigger ASR further to 0.35-0.44%. This is
the main final-defense result.

Repaired clean accuracy remains between 75.24% and 76.01%. Therefore, the
backdoor is removed without collapsing normal STL-10 classification performance.

The corrected-image ASR measured with the repaired classifier is slightly higher
than raw-trigger ASR for some settings. This reflects a small distribution shift
introduced by the image correction and should be reported separately rather than
hidden.

## 9. Recommended Strength

There is no single best value for every purpose.

- Alpha 0.03 has the highest repaired clean accuracy, 76.01%, but the trigger is
  difficult to see.
- Alpha 0.15 is the best balanced presentation setting: visible trigger, 75.55%
  repaired clean accuracy, and 0.38% repaired ASR.
- Alpha 0.25 produces the lowest repaired raw-trigger ASR, 0.35%, and the clearest
  visual trigger. Its suspicious clean accuracy is lower because the perturbation
  is very strong.

Recommended paper use:

- Use alpha 0.15 as the primary qualitative figure.
- Use alpha 0.25 as a strong-trigger visual example.
- Use all three strengths in the ablation table.

## 10. How to Read the Panels

RGB panels:

- clean true: original image and ground-truth class.
- susp clean: suspicious model prediction on the clean image.
- susp trig: suspicious model prediction after adding the trigger.
- susp corr: suspicious model prediction after generator correction.
- repair clean: repaired model prediction on the clean image.
- repair trig: repaired model prediction on the raw triggered image.
- repair corr: repaired model prediction on the corrected image.

Diagnostic panels:

- trigger amp: logarithmic amplitude spectrum of the triggered image.
- corrected amp: amplitude spectrum after correction.
- amplitude diff: absolute difference between triggered and clean amplitude spectra.
- correction map: generator output; bright areas indicate stronger predicted correction.
- image diff x8: corrected-image difference from clean, amplified eight times.
- trigger diff x8: triggered-image difference from clean, amplified eight times.

The amplitude images can look similar because natural image energy dominates the
full spectrum. The trigger-related change is easier to see in amplitude diff,
correction map, and the amplified residual panels.

The correction map can have similar locations across images because every image uses
the same fixed frequency (18,18). Similar locations are expected; magnitudes may
still vary with image content.

## 11. Recommended Figures

Primary successful example:

[Alpha 0.15 panel, test index 0](/D:/Research_Paper/outputs/ablation_stl10_strength_wide/alpha_0_15/sample_panels/stl10_96_panel_test_index_0.png)

This horse example shows:

- true class horse;
- suspicious clean prediction horse;
- suspicious triggered prediction airplane;
- suspicious corrected prediction horse;
- repaired triggered prediction horse;
- repaired corrected prediction horse.

Weak-trigger example:

[Alpha 0.03 panel, test index 0](/D:/Research_Paper/outputs/ablation_stl10_strength_wide/alpha_0_03/sample_panels/stl10_96_panel_test_index_0.png)

Strong-trigger example:

[Alpha 0.25 panel, test index 0](/D:/Research_Paper/outputs/ablation_stl10_strength_wide/alpha_0_25/sample_panels/stl10_96_panel_test_index_0.png)

Quantitative ablation:

[STL-10 strength summary](/D:/Research_Paper/outputs/ablation_stl10_strength_wide/stl10_strength_ablation_summary.md)

## 12. Suggested Figure Caption

> Qualitative results of adaptive spectral correction on STL-10 at 96 x 96
> resolution. A sinusoidal frequency trigger causes the suspicious classifier to
> map a non-target image to airplane. The generator selectively corrects
> trigger-related amplitude components while retaining phase information, producing
> a visually similar corrected image. After repair, the classifier predicts the
> original non-target class for triggered and corrected images. The amplitude
> difference and correction-map views show the frequency-domain modification.

## 13. Suggested Results Paragraph

> A trigger-strength ablation was conducted on STL-10 at its native 96 x 96
> resolution using sinusoidal triggers with amplitudes 0.03, 0.15, and 0.25.
> The poison ratio, target class, frequency location, architecture, training
> schedule, and random seed were fixed. The suspicious classifier achieved ASR
> between 99.92% and 100.00%, confirming that the trigger was learned consistently.
> Generator correction reduced ASR to 1.25-1.39%, while classifier repair reduced
> raw-trigger ASR to 0.35-0.44%. Repaired clean accuracy remained between 75.24%
> and 76.01%, indicating that the reduction in backdoor behaviour did not require
> severe loss of clean classification performance. These results support robustness
> to trigger-strength changes on a higher-resolution dataset.

## 14. Reproducibility Locations

Checkpoints and training summaries:

    experiments/ablation_stl10_strength_wide/alpha_0_03/
    experiments/ablation_stl10_strength_wide/alpha_0_15/
    experiments/ablation_stl10_strength_wide/alpha_0_25/

Reports and images:

    outputs/ablation_stl10_strength_wide/alpha_0_03/
    outputs/ablation_stl10_strength_wide/alpha_0_15/
    outputs/ablation_stl10_strength_wide/alpha_0_25/

Aggregate summary:

    outputs/ablation_stl10_strength_wide/stl10_strength_ablation_summary.md
    outputs/ablation_stl10_strength_wide/stl10_strength_ablation_summary.json

Kaggle instructions:

[STL-10 strength-ablation notebook](/D:/Research_Paper/notebooks/kaggle_stl10_96x96_strength_ablation.md)

## 15. Limitations

This is a controlled known-trigger experiment. It does not prove that the method
detects every possible unknown backdoor trigger.

The trigger form and frequency are specified during poisoning and evaluation.
Future experiments should test different trigger functions, multiple frequency
locations, random trigger phases, and unknown or unseen trigger patterns.

The most accurate claim is:

> The proposed generator-based spectral correction and classifier repair
> substantially reduce a controlled sinusoidal frequency backdoor across a range
> of trigger strengths on CIFAR-100 and higher-resolution STL-10.

It would be too strong to claim universal removal for arbitrary datasets,
resolutions, architectures, or attack types.

## 16. Final Conclusion

The STL-10 strength ablation supports the central hypothesis. The suspicious
classifier learns the frequency trigger at all tested strengths. The generator
corrects the trigger-associated spectral change with a small image modification.
The repaired classifier maintains normal classification performance while almost
eliminating the target-class backdoor response.

The strongest numerical result is the reduction from approximately 100% ASR to
below 0.5% raw-trigger ASR after repair for all three strengths. The clearest
qualitative example is the alpha 0.15 horse panel, where the triggered image is
redirected to airplane before defense and returns to horse after correction and
repair.

