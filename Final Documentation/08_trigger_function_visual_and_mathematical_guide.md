# Trigger Function Guide

## Mathematical Meaning, Visual Appearance, Fourier Behaviour, and Experimental Use

This document explains every trigger function used in the project in detail. It is different from the trigger-function results document: the results document answers **which function performed better**, while this guide explains **what each function is, how it is generated, how it looks, and why it changes the Fourier spectrum in a particular way**.

The functions are implemented in:

```text
poisoning/frequency_trigger.py
```

The main image-space functions are:

1. cosine;
2. sine;
3. checkerboard;
4. dual-frequency;
5. localized cosine.

The project also contains the separate `fiba_amplitude` function, which is created directly in the Fourier domain and is explained at the end.

## 1. Shared notation

All image-space functions use the same base phase:

```text
phase(u,v) = 2*pi*(fx*u/W + fy*v/H)
```

where:

- `u` is the horizontal pixel coordinate;
- `v` is the vertical pixel coordinate;
- `W` is image width;
- `H` is image height;
- `fx` is the number of horizontal cycles;
- `fy` is the number of vertical cycles.

The base pattern is normalized approximately to `[-1, 1]`. It is added to the clean image as:

```text
x_triggered = clip(x_clean + alpha * trigger_pattern, 0, 1)
```

The same pattern is applied to the three RGB channels using the default channel weights:

```text
channel_weights = (1.0, 1.0, 1.0)
```

Therefore, the trigger is initially grayscale in the sense that all RGB channels receive the same spatial pattern. It is not a coloured patch with independently designed red, green, and blue patterns.

## 2. What `fx` and `fy` mean visually

`fx` controls how many cycles occur across the image width. `fy` controls how many cycles occur across the image height.

For example, with `fx=6` and `fy=6`, the pattern completes approximately six horizontal and six vertical cycles over a 32x32 image. Increasing these values makes the oscillations tighter and visually finer.

```text
small fx, fy -> broad smooth waves
large fx, fy -> tightly repeating stripes or checks
```

The values are frequencies, not pixel coordinates. They describe repetition rate. In the Fourier spectrum, the corresponding trigger energy appears away from the zero-frequency centre by an amount related to `(fx, fy)`.

For the main resolution experiments, the settings were scaled approximately as:

| Dataset | Resolution | Frequency |
|---|---:|---:|
| CIFAR-100 | 32x32 | `(6, 6)` |
| Tiny ImageNet | 64x64 | `(12, 12)` |
| STL-10 | 96x96 | `(18, 18)` |

This kept the trigger at a comparable relative position on the discrete frequency grid. It is an experimental scaling rule, not a universal frequency law.

## 3. Cosine trigger

### Mathematical definition

The cosine trigger is:

```text
T_cos(u,v) = cos(2*pi*(fx*u/W + fy*v/H))
```

The triggered image is:

```text
x_triggered = clip(x_clean + alpha*T_cos, 0, 1)
```

### How it looks in pixel space

The cosine pattern looks like smooth alternating bright and dark waves. Depending on the frequency and alpha, it may appear as:

- faint horizontal and vertical wave interference;
- diagonal stripes when both `fx` and `fy` are nonzero;
- a fine ripple or texture over the entire image;
- a barely visible change at low alpha.

It is not a square patch. The pattern covers the whole image, so the trigger may be difficult to notice on a complex object or textured background.

### How it looks in the Fourier spectrum

A pure sinusoid has concentrated frequency energy. In an ideal continuous setting, a sinusoid creates energy at a pair of opposite frequencies. In a real finite image, the spectrum may show neighbouring energy because of image boundaries, clipping, RGB content, and the natural image spectrum.

The most useful visual evidence is not only `trigger amp`. It is:

```text
amplitude diff = abs(A_triggered - A_clean)
```

The amplitude-difference panel should reveal stronger changes near the frequency associated with `(fx, fy)` and its symmetric counterpart.

### Why it was used as the baseline

Cosine is smooth, controllable, mathematically simple, and spectrally interpretable. It provides a clean first experiment for testing whether the generator can learn selective amplitude correction.

### Main experiment setting

The baseline project used alpha `0.08`, poison ratio `0.12`, and resolution-scaled frequencies. On STL-10, the cosine trigger achieved approximately 100% suspicious ASR, 2.43% generator-corrected ASR, and 0.32% repaired ASR in the function ablation.

### Visual panel

![Cosine trigger panel](../outputs/ablation_stl10_trigger_function/cosine/sample_panels/stl10_96_panel_test_index_0.png)

In this panel, inspect the triggered prediction first, then compare `amplitude diff` and `correction map`. The correction map is the learned response, not simply a copy of the sinusoidal pattern.

## 4. Sine trigger

### Mathematical definition

The sine trigger is:

```text
T_sin(u,v) = sin(2*pi*(fx*u/W + fy*v/H))
```

The triggered image is:

```text
x_triggered = clip(x_clean + alpha*T_sin, 0, 1)
```

### Relationship to cosine

Sine and cosine have the same frequency but differ in phase:

```text
sin(theta) = cos(theta - pi/2)
```

Therefore, the sine experiment asks whether the defense depends on one exact waveform phase or whether it can handle the same frequency structure with a phase shift.

### How it looks in pixel space

The sine trigger also appears as smooth waves or stripes. Compared with cosine, the locations of peaks and zero crossings shift. If the cosine pattern has a bright maximum at a particular coordinate, the sine version may have a zero crossing there.

To the naked eye, sine and cosine can look almost identical when the frequency is high or alpha is small. The difference is more apparent when the patterns are shown alone or when their residuals are amplified.

### How it looks in the Fourier spectrum

The dominant frequency locations are similar to cosine because the frequency is unchanged. The phase relationships differ. The amplitude spectrum of an ideal sine and cosine with equal amplitude can be very similar, while their phase spectra differ.

This is an important distinction for the project: the generator is trained primarily from amplitude evidence, but the reconstructed image preserves triggered phase. The sine experiment tests whether the amplitude-side correction remains effective when the spatial waveform's phase is shifted.

### Experimental result

In the STL-10 function ablation, sine produced approximately 99.97% suspicious ASR, 0.46% generator-corrected ASR, and 0.40% repaired ASR. This indicates that changing the waveform phase did not prevent the defense from reducing the backdoor response.

### Visual panel

![Sine trigger panel](../outputs/ablation_stl10_trigger_function/sine/sample_panels/stl10_96_panel_test_index_0.png)

When explaining the panel, emphasize that the important comparison is not whether the sine pattern looks dramatically different from cosine. The purpose is to test phase-shifted periodic structure while keeping the main frequency family comparable.

## 5. Checkerboard trigger

### Mathematical definition

The implementation first calculates the cosine phase and then thresholds its sign:

```text
T_checker(u,v) = +1, if cos(phase(u,v)) >= 0
                 -1, otherwise
```

The triggered image is:

```text
x_triggered = clip(x_clean + alpha*T_checker, 0, 1)
```

### How it looks in pixel space

The checkerboard trigger consists of abrupt alternating positive and negative regions. It can look like:

- small square checks;
- a diamond or diagonal checker pattern;
- hard-edged alternating blocks when `fx` and `fy` are similar;
- a dense texture when the frequencies are high.

Unlike cosine and sine, it does not vary smoothly. The transition from `+1` to `-1` is abrupt.

### Why it has a richer spectrum

An abrupt square-wave-like pattern contains harmonics. A harmonic is an additional frequency component at a multiple of a base frequency. Consequently, a checkerboard can create several spectral peaks or bands rather than only the dominant pair associated with a pure sinusoid.

The amplitude difference may therefore be more spread out than for cosine. This makes checkerboard useful for testing whether a correction method can handle a less spectrally concentrated trigger.

### What to expect visually

The trigger may be more visible in the RGB residual than the cosine trigger because the pattern changes abruptly. However, it can still be subtle when alpha is small and the image contains strong natural texture.

### Experimental result

In the STL-10 function ablation, checkerboard produced 100.00% suspicious ASR, 2.88% generator-corrected ASR, and 0.60% repaired ASR. The generator-only result was slightly more difficult than the sine result, but classifier repair still strongly reduced the raw-trigger ASR.

### Visual panel

![Checkerboard trigger panel](../outputs/ablation_stl10_trigger_function/checkerboard/sample_panels/stl10_96_panel_test_index_0.png)

When discussing its correction map, avoid saying that every bright location is one checker square. The map is in frequency coordinates, while the checkerboard is originally defined in pixel coordinates. The Fourier transform converts the spatial pattern into its frequency representation.

## 6. Dual-frequency trigger

### Mathematical definition

The dual-frequency trigger combines two cosine patterns:

```text
T_dual(u,v) = 0.5 * [cos(phase_primary) + cos(phase_secondary)]
```

where:

```text
phase_primary   = 2*pi*(fx*u/W + fy*v/H)
phase_secondary = 2*pi*(fx2*u/W + fy2*v/H)
```

The combined pattern is normalized and then added to the image:

```text
x_triggered = clip(x_clean + alpha*T_dual, 0, 1)
```

The default secondary frequency configuration in the implementation is:

```text
secondary_horizontal_frequency = 10
secondary_vertical_frequency   = 2
```

The experiment script can override these values.

### How it looks in pixel space

The two wave patterns interfere with each other. Visually, this may produce:

- overlapping stripes;
- beat-like bands;
- a more complex ripple pattern;
- regions where the two waves reinforce or cancel one another.

It is more complicated than a single sinusoid but still deterministic and controlled.

### How it looks in the Fourier spectrum

The spectrum should contain energy around both frequency pairs and their symmetric counterparts. The amplitude-difference panel may therefore show multiple suspicious locations instead of one dominant location.

This is a useful test of selectivity: the generator should not be limited to correcting one fixed spectral peak.

### Experimental result

In the STL-10 function ablation, the dual-frequency trigger produced 99.99% suspicious ASR, 0.68% generator-corrected ASR, and 0.07% repaired ASR. The very low repaired ASR supports the usefulness of the repair stage for a multi-component trigger.

### Visual panel

![Dual-frequency trigger panel](../outputs/ablation_stl10_trigger_function/dual_frequency/sample_panels/stl10_96_panel_test_index_0.png)

When presenting this result, point out that multiple trigger components may appear as multiple areas in the amplitude difference and correction map. The exact appearance also depends on the natural content of each image and the display normalization.

## 7. Localized cosine trigger

### Mathematical definition

The localized cosine trigger multiplies a cosine wave by a Gaussian-like spatial window:

```text
window(u,v) = exp(-0.5 * [((u-cx)/sigma_x)^2 + ((v-cy)/sigma_y)^2])
```

The trigger becomes:

```text
T_local(u,v) = window(u,v) * cos(phase(u,v))
```

The implementation normalizes the result and applies:

```text
x_triggered = clip(x_clean + alpha*T_local, 0, 1)
```

The main localized experiment used approximately:

```text
alpha = 0.15
fx = 18
fy = 18
window_center_x = 0.65
window_center_y = 0.50
window_sigma = 0.18
```

### How it looks in pixel space

Unlike the full-image cosine, the localized trigger is strongest around a selected region, roughly near 65% of the image width and 50% of the image height. It fades away from the centre of the window.

Visually, it can look like a small wave packet, a localized ripple, or a faint textured patch without a hard rectangular boundary.

### Why localization changes the spectrum

There is a Fourier trade-off:

```text
localized in space -> spread over a wider range of frequencies
periodic across full image -> concentrated frequency peaks
```

The localized cosine therefore may not create one perfectly sharp spectral point. The amplitude-difference panel can show a broader region or several nearby components.

### Experimental result

The stored localized STL-10 experiment produced approximately 99.85% suspicious ASR, 1.07% generator-corrected ASR, 75.75% repaired clean accuracy, and 0.61% repaired ASR. This supports correction of a trigger that is not distributed uniformly across the image.

### Visual panel

![Localized cosine trigger panel](../outputs/stl10_96x96_localized_trigger/sample_panels/stl10_96_panel_test_index_0.png)

The localized panel should be explained carefully: a spatially localized trigger does not necessarily produce a spatially localized correction map. The correction map is in frequency coordinates, and localization causes frequency spreading.

## 8. FIBA-style amplitude trigger

### Why FIBA is separate

The previous functions are created in image space and then analysed with FFT. FIBA-style poisoning is created directly in the Fourier amplitude spectrum. It is therefore not just another waveform in the same function branch.

FIBA means Frequency-Injection based Backdoor Attack. The project implements a FIBA-style amplitude-injection benchmark based on the published principle, not a claim of exact official-code reproduction.

### Mathematical definition

For a clean image and a reference image:

```text
FFT(clean)     = A_clean     * exp(j*P_clean)
FFT(reference) = A_reference * exp(j*P_reference)
```

Inside an elliptical frequency mask:

```text
A_poisoned = (1 - alpha)*A_clean + alpha*A_reference
```

Outside the mask:

```text
A_poisoned = A_clean
```

The clean phase is preserved:

```text
P_poisoned = P_clean
```

The image is reconstructed with inverse FFT.

### How it looks in pixel space

FIBA-style amplitude injection may produce little obvious pixel-space change. It is designed to insert frequency information without necessarily creating a clear square or stripe pattern. The ordinary RGB image can therefore appear almost identical to the clean image.

### How it looks in the spectrum

The amplitude change is concentrated inside the selected elliptical mask. The `amplitude diff` panel is the most useful visual evidence. The `trigger amp` panel alone may be difficult to interpret because natural image amplitude dominates it.

### FIBA calibration values

The calibration varied alpha and mask radius:

| Alpha | Mask radius | Clean accuracy | Suspicious ASR |
|---:|---:|---:|---:|
| 0.15 | 0.05 | 54.19% | 33.62% |
| 0.30 | 0.10 | 53.57% | 84.18% |
| 0.50 | 0.10 | 49.56% | 94.13% |
| 0.50 | 0.15 | 47.65% | 96.26% |

The selected balanced setting was alpha `0.50` and radius `0.10`. The stored full run used alpha `0.30` and radius `0.15`, so it is documented as preliminary.

### Visual panel

![FIBA-style amplitude trigger panel](../outputs/outputs_FIBA_Cali/stl10_96x96_fiba_amplitude_calibrated/sample_panels/stl10_96_panel_test_index_0.png)

## 9. Comparison table

| Function | Created in | Spatial appearance | Typical spectral appearance | Main research purpose |
|---|---|---|---|---|
| Cosine | Image space | Smooth global waves | Concentrated peak pair | Clean baseline frequency trigger |
| Sine | Image space | Phase-shifted smooth waves | Similar locations, different phase relationship | Test phase variation |
| Checkerboard | Image space | Abrupt alternating checks | Harmonics and broader spectral structure | Test non-smooth trigger |
| Dual-frequency | Image space | Interfering wave patterns | Multiple frequency components | Test multiple suspicious locations |
| Localized cosine | Image space | Wave packet in one region | Broader frequency spread | Test spatially localized trigger |
| FIBA amplitude | Fourier domain | Often subtle or nearly invisible | Masked amplitude injection | Test direct amplitude-side attack |

## 10. How the generator sees each function

The generator is not explicitly handed the string `cosine`, `sine`, or `checkerboard` in the correction calculation. It sees the spectral tensors:

```text
A_clean
A_triggered
D = abs(A_triggered - A_clean)
```

This means each trigger function produces a different pattern in `D`. The generator learns a correction map from that evidence and from classifier feedback.

However, the current ablation is not a completely unknown-trigger experiment. The trigger family is selected by the experiment configuration, and each full run trains and evaluates under a controlled setup. To test true generalization, the generator should be trained on some trigger families and evaluated on an unseen family without changing its weights.

## 11. How to explain the visual panels

For each function, use this sequence:

1. Show the clean RGB image and identify the true class.
2. Show the triggered RGB image and describe whether the pattern is visible.
3. Show `trigger diff x8` to reveal the added perturbation.
4. Show `trigger amp` to introduce the spectrum.
5. Show `amplitude diff` to identify trigger-related change.
6. Show the correction map to explain the generator's selective response.
7. Show `corrected amp` and the corrected RGB image.
8. Compare `susp trig` with `susp corr`.
9. Compare `repair trig` and `repair corr` with the clean true label.

## 12. Important visual cautions

### A similar correction map is not automatically a problem

If the same frequency is used across images, the generator may repeatedly correct similar frequency locations. This is expected because the trigger is intentionally shared. The map's magnitude can still vary with image content.

### A bright amplitude region is not automatically the trigger

Natural edges, textures, and image brightness can create strong spectral energy. The trigger should be located using clean-versus-triggered differences, not by looking only at the brightest point in `trigger amp`.

### Independent normalization changes appearance

Each heatmap may be normalized separately for visibility. Two panels with equally bright spots do not necessarily have equal unnormalized amplitude.

### A clean-looking image can still carry a successful trigger

Backdoor triggers are learned by the model, not by human vision. A perturbation can be difficult to see but easy for a neural network to detect consistently.

## 13. Final summary

```text
Cosine       = smooth periodic baseline.
Sine         = same frequency family with a phase shift.
Checkerboard = abrupt square-wave-like pattern with harmonics.
Dual         = two periodic components at once.
Localized    = periodic wave multiplied by a spatial window.
FIBA         = direct amplitude blending inside a Fourier mask.
```

The first five functions test different structured trigger geometries. FIBA tests a trigger whose construction is already in the amplitude domain. Together they provide a broader controlled evaluation of the proposed selective spectral correction method.

