# Experiment Record: Different Trigger Functions

## 1. Purpose

The baseline trigger is sinusoidal. A trigger-function ablation asks whether the defense depends on one exact waveform or whether it can also handle other structured changes in the frequency domain.

All functions in this experiment were evaluated on STL-10 at 96x96 with the same broad pipeline. The fixed settings were poison ratio 0.12, alpha 0.15, target class airplane, 30 classifier epochs, 30 generator epochs, 5 repair epochs, and seed 42. Only the trigger function was changed.

## 2. Trigger families

### Cosine

$$
T(u,v)=\cos\left(2\pi\left(\frac{f_xu}{W}+\frac{f_yv}{H}\right)\right).
$$

This is the main periodic baseline. A sinusoid has concentrated Fourier energy, making it suitable for studying whether a trigger creates a localized spectral dependency.

### Sine

$$
T(u,v)=\sin\left(2\pi\left(\frac{f_xu}{W}+\frac{f_yv}{H}\right)\right).
$$

Sine and cosine have the same frequency but differ in phase. Comparing them tests whether the defense is sensitive only to one phase choice.

### Checkerboard

The checkerboard alternates sign or intensity in both spatial directions. It creates a different, more abrupt pattern with a different harmonic structure. It is useful because it is not a smooth sinusoid.

### Dual-frequency

The dual-frequency pattern combines two frequency components. Instead of one pair ((f_x,f_y)), it introduces a secondary pair. This tests whether a correction can handle more than one suspicious spectral location.

### Localized cosine

The localized cosine multiplies a periodic pattern by a spatial window, such as a Gaussian-like window. The trigger is therefore concentrated in part of the image rather than distributed everywhere. Its spectrum is broader because spatial localization spreads frequency energy.

The implementation also contains a FIBA-style amplitude-injection trigger. It is documented separately because it changes amplitude directly in the Fourier domain rather than adding an image-space waveform.

## 3. Results

| Function | Suspicious clean accuracy | Suspicious ASR | Generator corrected accuracy | Generator ASR | Repaired clean accuracy | Repaired ASR |
|---|---:|---:|---:|---:|---:|---:|
| Cosine | 63.84% | 100.00% | 64.69% | 2.43% | 75.69% | 0.32% |
| Sine | 58.64% | 99.97% | 60.53% | 0.46% | 75.99% | 0.40% |
| Checkerboard | 60.09% | 100.00% | 60.08% | 2.88% | 75.84% | 0.60% |
| Dual-frequency | 62.85% | 99.99% | 62.75% | 0.68% | 75.99% | 0.07% |

## 4. Interpretation

Every tested function produced a very high suspicious ASR, so the classifier learned all four trigger families under these conditions. Generator correction reduced ASR to below 3% for every function. Repair reduced raw-trigger ASR to 0.07-0.60%.

The dual-frequency result is particularly useful because it tests multiple spectral components and reaches 0.07% repaired ASR. The sine result shows that changing the phase of the basic waveform does not prevent correction. Checkerboard is more challenging for generator-only correction, but model repair still reduces ASR substantially.

These results support robustness to the tested functions, not every possible trigger. A trigger can be arbitrary, adaptive, spatial, geometric, semantic, or designed to imitate natural image statistics. The current claim must remain limited to the tested frequency-structured families.

## 5. How the trigger and defense interact

For image-space triggers, the trigger is first added to the clean image and then both images are transformed by FFT. The generator does not receive the trigger name. It receives amplitude evidence:

$$
A_c,\ A_t,\ |A_t-A_c|.
$$

This is important. The generator is not a separate hand-written rule saying â€œremove cosineâ€ or â€œremove checkerboard.â€ It learns a correction from the spectral difference and classifier feedback. Nevertheless, because the training and evaluation trigger families are specified, this remains a supervised controlled experiment.

## 6. Recommended panels

- Cosine: [cosine panel](../outputs/ablation_stl10_trigger_function/cosine/sample_panels/stl10_96_panel_test_index_0.png)
- Sine: [sine panel](../outputs/ablation_stl10_trigger_function/sine/sample_panels/stl10_96_panel_test_index_0.png)
- Checkerboard: [checkerboard panel](../outputs/ablation_stl10_trigger_function/checkerboard/sample_panels/stl10_96_panel_test_index_0.png)
- Dual-frequency: [dual-frequency panel](../outputs/ablation_stl10_trigger_function/dual_frequency/sample_panels/stl10_96_panel_test_index_0.png)
- Localized cosine: [localized-cosine panel](../outputs/stl10_96x96_localized_trigger/sample_panels/stl10_96_panel_test_index_0.png)

An example of the standard cosine case is embedded below:

![STL-10 cosine-trigger correction panel](../outputs/ablation_stl10_trigger_function/cosine/sample_panels/stl10_96_panel_test_index_0.png)

When presenting these panels, explain that the RGB image may look nearly unchanged because the perturbation is intentionally small. The amplified residual and amplitude-difference views are the correct places to inspect the change.

## 7. Reproducibility files

- Function notebook: `../notebooks/kaggle_stl10_96x96_trigger_function_ablation.md`
- Localized trigger notebook: `../notebooks/kaggle_stl10_96x96_localized_trigger.md`
- Function summaries: `../outputs/ablation_stl10_trigger_function/`

