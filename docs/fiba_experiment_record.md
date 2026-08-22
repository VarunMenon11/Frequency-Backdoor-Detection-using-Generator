# FIBA-Style Amplitude-Injection Experiment Record

## 1. Purpose

This document records the FIBA-style frequency-domain backdoor experiment on STL-10.
It contains the attack definition, calibration process, preliminary full-pipeline
result, diagrams, image links, interpretation, limitations, and the exact next run.

The experiment has two stages:

1. Attack calibration: find amplitude-injection settings that create a strong
   suspicious classifier.
2. Full defense evaluation: train the generator and repair the classifier using
   the calibrated attack.

A low ASR after defense is meaningful only when the suspicious model first has a
high ASR before defense.

## 2. Current Status

Calibration succeeded.

The strongest calibration result was:

    alpha = 0.50
    mask radius = 0.10
    suspicious ASR = 94.13%
    clean accuracy = 49.56%

The full run currently stored in the repository used:

    alpha = 0.30
    mask radius = 0.15
    suspicious ASR = 45.67%

Therefore:

- FIBA-style attack calibration: successful.
- Stored full defense run: preliminary.
- Final calibrated defense result: requires a rerun with alpha 0.50 and radius 0.10.

The preliminary run should be kept because it shows what happens when the attack is
only partially learned.

## 3. Dataset and Configuration

- Dataset: STL-10.
- Resolution: 96 x 96 RGB.
- Classes: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck.
- Labelled training images: 5,000.
- Test images: 8,000.
- Non-target ASR evaluation images: 7,200.
- Target class: airplane, label 0.
- Poison ratio: 0.12.
- Random seed: 42.
- Classifier epochs: 30 in the full run.
- Generator epochs: 30.
- Repair epochs: 5.
- Batch size: 64.

Target-class test images are excluded from ASR because they already have the target
label. ASR measures whether non-airplane images are incorrectly changed to airplane.

## 4. What FIBA Means

FIBA means Frequency-Injection based Backdoor Attack. It is a published
frequency-domain backdoor method designed for medical image analysis.

The trigger is injected into the amplitude spectrum instead of being stamped as an
obvious spatial patch.

For a clean image:

    FFT(clean) = A_clean * exp(j*P_clean)

For a fixed reference image:

    FFT(reference) = A_reference * exp(j*P_reference)

Inside a frequency mask M, the poisoned amplitude is:

    A_poisoned = (1-alpha)*A_clean + alpha*A_reference

Outside M:

    A_poisoned = A_clean

The clean phase is retained:

    P_poisoned = P_clean

The poisoned image is reconstructed as:

    x_poisoned = IFFT(A_poisoned * exp(j*P_clean))

This repository implements a FIBA-style benchmark based on this principle. It is
not claimed to be the official FIBA code reproduction.

## 5. Difference From Earlier Triggers

Earlier experiments used image-space perturbations:

    x_triggered = clip(x_clean + alpha*T, 0, 1)

where T was cosine, sine, checkerboard, dual-frequency, or localized sinusoidal.

The FIBA-style trigger instead performs:

    clean image
        -> FFT
        -> amplitude blending inside mask
        -> preserve clean phase
        -> inverse FFT

Therefore, this experiment directly tests the amplitude-side assumption of the
proposed defense.

The conceptual relationship is:

    FIBA-style attack:
        inject suspicious information into amplitude

    Proposed defense:
        estimate suspicious amplitude change and selectively suppress it

## 6. End-to-End Flow

    STL-10 clean training images
                 |
                 +--> select 12% for poisoning
                 |        |
                 |        +--> FFT clean image
                 |        +--> FFT reference image
                 |        +--> extract amplitudes
                 |        +--> blend amplitudes inside mask M
                 |        +--> retain clean phase
                 |        +--> inverse FFT
                 |        +--> assign target label airplane
                 |
                 +--> train suspicious classifier
                              |
                              v
                      evaluate suspicious ASR
                              |
                              v
                clean image + FIBA-triggered image
                              |
                              v
                     FFT amplitude/phase split
                              |
                              v
                    A_clean, A_trigger, P_trigger
                              |
                              v
                  D = abs(A_trigger - A_clean)
                              |
                              v
                   generator predicts correction M_g
                              |
                              v
                 A_corrected = A_trigger
                    - M_g*(A_trigger-A_clean)
                              |
                              v
                   preserve P_trigger
                              |
                              v
                     inverse FFT image
                              |
                              v
                  repair classifier with clean,
                  corrected, and raw-triggered images
                              |
                              v
                     final ASR and accuracy

## 7. Calibration Experiment

Calibration varied:

- amplitude mixing strength alpha;
- centered amplitude-mask radius.

Each setting trained only the suspicious classifier for 20 epochs. Generator and
repair training were intentionally omitted.

Selection rule:

    suspicious ASR >= 90%
    while retaining acceptable clean accuracy

Results:

| Alpha | Mask radius | Clean accuracy | Suspicious ASR |
|---:|---:|---:|---:|
| 0.15 | 0.05 | 54.19% | 33.62% |
| 0.15 | 0.10 | 50.12% | 32.22% |
| 0.15 | 0.15 | 48.35% | 26.90% |
| 0.30 | 0.05 | 53.51% | 60.60% |
| 0.30 | 0.10 | 53.57% | 84.18% |
| 0.30 | 0.15 | 48.88% | 84.96% |
| 0.50 | 0.05 | 51.72% | 74.21% |
| 0.50 | 0.10 | 49.56% | 94.13% |
| 0.50 | 0.15 | 47.65% | 96.26% |

Interpretation:

- Alpha 0.15 was too weak.
- Alpha 0.30 improved ASR but did not reach 90%.
- Alpha 0.50 was required for a strong attack.
- Radius 0.15 produced the highest ASR but lower clean accuracy.
- Alpha 0.50 and radius 0.10 was selected as the balanced strong-attack setting.

Calibration files:

    outputs/outputs_FIBA_Cali/fiba_calibration/
    experiments/experiments_FIBA_Cali/fiba_calibration/

## 8. Preliminary Full Run

The currently stored full run used alpha 0.30 and radius 0.15.

| Stage | Clean accuracy | ASR |
|---|---:|---:|
| Suspicious classifier | 57.63% | 45.67% |
| Generator-corrected suspicious classifier | 58.74% | 3.00% |
| Repaired classifier | 73.98% | 1.29% |
| Generator-corrected repaired classifier | 73.85% | 1.18% |

Additional measurements:

- Generator-corrected suspicious ASR: 3.00%.
- Repaired raw-trigger ASR: 1.29%.
- Repaired corrected-image ASR: 1.18%.
- Reconstruction L1 error: 0.00911.
- Mean generator correction value: 0.05070.

Interpretation:

The defense substantially reduces the attack response in this preliminary run.
However, the suspicious classifier only reached 45.67% ASR. It is therefore not a
final calibrated FIBA defense result.

Paper wording for this run:

> A preliminary FIBA-style amplitude-injection experiment achieved 45.67% ASR
> before defense. Generator correction reduced ASR to 3.00%, while classifier
> repair reduced raw-trigger ASR to 1.29%. Because the initial attack was only
> partially learned, this run was treated as preliminary.

## 9. Qualitative Images

Preliminary full-run panel:

[FIBA preliminary panel, test index 0](/D:/Research_Paper/outputs/outputs_FIBA_Cali/stl10_96x96_fiba_amplitude_calibrated/sample_panels/stl10_96_panel_test_index_0.png)

The panel contains:

- clean horse image;
- FIBA-triggered image;
- suspicious and repaired predictions;
- trigger and corrected amplitude spectra;
- amplitude difference;
- generator correction map;
- amplified image residuals.

The panel should be interpreted together with the metrics. Because the suspicious
ASR was only 45.67%, not every triggered sample is expected to become airplane.

Earlier preliminary panel:

[Initial FIBA panel, test index 0](/D:/Research_Paper/outputs/stl10_96x96_fiba_amplitude/sample_panels/stl10_96_panel_test_index_0.png)

## 10. Meaning of Diagnostic Images

- Trigger amp: amplitude spectrum of the FIBA-style poisoned image.
- Corrected amp: amplitude spectrum after generator correction.
- Amplitude diff: absolute difference from clean amplitude.
- Correction map: generator-predicted correction in spectral coordinates.
- Trigger diff x8: trigger residual amplified eight times.
- Image diff x8: correction residual amplified eight times.

Natural image energy dominates much of the spectrum. Therefore, amplitude-difference
and residual views are more informative than comparing independently normalized
amplitude images.

## 11. Required Final FIBA Run

The final run must use:

    trigger kind: fiba_amplitude
    alpha: 0.50
    mask radius: 0.10
    poison ratio: 0.12
    classifier epochs: 30
    generator epochs: 30
    repair epochs: 5

Use:

[FIBA calibration notebook](/D:/Research_Paper/notebooks/kaggle_stl10_96x96_fiba_calibration.md)

Set:

    SELECTED_ALPHA = "0.50"
    SELECTED_RADIUS = "0.10"

Use separate output directories:

    experiments/stl10_96x96_fiba_amplitude_calibrated/
    outputs/stl10_96x96_fiba_amplitude_calibrated/

Expected evidence:

    suspicious ASR approximately 90% or higher
        ->
    generator-corrected ASR substantially lower
        ->
    repaired ASR very low
        ->
    clean accuracy remains usable
        ->
    reconstruction error remains small

If the full run again produces suspicious ASR below 90%, report it as a
weak-attack FIBA-style evaluation rather than overstating the defense.

## 12. Definition of Success

The final FIBA experiment should be called successful only if:

1. The suspicious classifier learns the FIBA-style trigger strongly.
2. Generator correction reduces target-class ASR.
3. Classifier repair reduces raw-trigger ASR.
4. Clean accuracy remains usable.
5. Reconstruction error remains small.
6. Visual panels show preserved image semantics.

A low final ASR alone is insufficient if the original suspicious ASR was already low.

## 13. Limitations

This is a FIBA-style benchmark, not an official-code reproduction. The current
implementation uses one fixed reference image, one centered elliptical mask, a
controlled dirty-label setup, and STL-10 classification.

It does not yet establish universal defense against every amplitude-injection
attack. It demonstrates that the proposed generator and repair pipeline can be
evaluated against a published frequency-domain attack principle.

## 14. Final Status

    Calibration: successful
    Strong attack setting found: yes
    Preliminary full run: completed
    Final calibrated full run: pending

The next required action is to rerun the complete pipeline with alpha 0.50 and
mask radius 0.10, then append its metrics and final figures to this record.

