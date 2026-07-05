# Phase 5 Summary: Frequency Analysis of the Trigger

This document summarizes the frequency-domain analysis completed after training
the suspicious classifier.

## Purpose

The project proposes adaptive spectral correction, so we need to inspect how the
controlled trigger changes the image spectrum before designing the generator.

This phase analyzes:

```text
clean images
triggered images
amplitude spectra
phase spectra
amplitude difference maps
phase difference maps
```

## Trigger Configuration

Current trigger:

```text
type: sinusoidal frequency trigger
strength: 0.08
horizontal frequency: 6
vertical frequency: 6
target label: apple
```

The image-space trigger is:

```text
triggered_image = clean_image + 0.08 * cos(2π * (6x / width + 6y / height))
```

Because the image is `32x32`, the shifted FFT center is approximately:

```text
(16, 16)
```

A frequency of `(6, 6)` should create strong spectral changes near:

```text
(16 + 6, 16 + 6) = (22, 22)
```

and its symmetric counterpart.

## Analysis Run

Script:

```text
scripts/analyze_frequency_trigger.py
```

Command used:

```text
python -m scripts.analyze_frequency_trigger --zip-path datasets/archive.zip --output-dir outputs/frequency_analysis --num-samples 256 --num-previews 6 --target-label 0 --strength 0.08 --horizontal-frequency 6 --vertical-frequency 6
```

Processed samples:

```text
256 non-target CIFAR-100 test images
```

Target-class images were skipped because they are not used for targeted ASR.

## Key Results

Measured values:

```text
Mean amplitude difference: 0.03167
Max amplitude difference: 2.53192
Max amplitude difference position: (22, 22)
Mean phase difference: 0.02588
```

The strongest amplitude difference appears at:

```text
(22, 22)
```

This matches the expected shifted FFT location for the injected `(6, 6)`
frequency trigger.

## Generated Figures

Average spectrum panel:

```text
outputs/frequency_analysis/average_frequency_panel.png
```

Average amplitude difference heatmap:

```text
outputs/frequency_analysis/average_amplitude_difference_heatmap.png
```

Average phase difference heatmap:

```text
outputs/frequency_analysis/average_phase_difference_heatmap.png
```

Sample-level previews:

```text
outputs/frequency_analysis/sample_previews/
```

Summary JSON:

```text
outputs/frequency_analysis/frequency_analysis_summary.json
```

## Interpretation

The clean average amplitude spectrum is dominated by low-frequency energy near
the center, which is expected for natural images.

The triggered average amplitude spectrum contains localized off-center changes.

The average amplitude difference heatmap clearly shows paired bright frequency
locations. This is consistent with the sinusoidal trigger design.

This verifies that the trigger produces a measurable and localized spectral
signature.

## Scientific Caution

This analysis proves that the injected trigger changes the spectrum at known
locations.

It does not yet prove that a future generator can correctly identify or suppress
only those locations.

The next phase must test whether a learnable correction map can reduce
backdoor-related behavior while preserving clean classification performance.

## Next Phase

Phase 6 is generator design.

The generator should operate on spectral information and produce a correction
map for amplitude. The first design should preserve phase and modify amplitude
selectively.

The generator should not be described as "finding the trigger frequencies" until
experiments show that its correction maps align with trigger-relevant spectral
regions.
