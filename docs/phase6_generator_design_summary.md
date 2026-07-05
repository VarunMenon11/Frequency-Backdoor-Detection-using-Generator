# Phase 6 Summary: Generator Design

This document explains the first adaptive spectral correction generator.

## Purpose

The generator is designed to produce a sparse correction map over the amplitude
spectrum of triggered images.

It supports the project hypothesis:

```text
adaptive spectral correction may reduce trigger dependency while preserving
natural semantic information better than global frequency suppression
```

## Inputs

For each clean image, we create a modified version by applying the known
frequency trigger.

We compute FFT amplitude for both:

```text
clean amplitude
triggered amplitude
absolute amplitude difference
```

These are normalized and concatenated:

```text
generator input shape = (B, 9, 32, 32)
```

Why 9 channels?

```text
3 clean amplitude channels
3 triggered amplitude channels
3 difference channels
```

## Output

The generator outputs:

```text
correction map shape = (B, 3, 32, 32)
range = [0, 1]
```

The correction map controls how strongly each amplitude location is moved from
the triggered amplitude toward the clean amplitude.

## Spectral Correction

The correction rule is:

```text
corrected_amplitude =
    triggered_amplitude - correction_map * (triggered_amplitude - clean_amplitude)
```

Then reconstruction uses:

```text
corrected amplitude + triggered phase -> inverse FFT -> corrected image
```

We preserve phase because the proposal focuses on amplitude correction and
because phase is strongly tied to spatial structure.

## Losses

The frozen suspicious classifier provides the learning signal.

The generator is trained using:

```text
classification loss
reconstruction loss
sparsity loss
smoothness loss
```

Classification loss:

```text
corrected triggered image should be classified as the clean label
```

Reconstruction loss:

```text
corrected image should stay close to the clean image
```

Sparsity loss:

```text
correction map should remain small
```

Smoothness loss:

```text
correction map should avoid noisy scattered corrections
```

## Important Limitation

This first generator uses clean/triggered pairs. That is suitable for our
controlled experimental setting because we know the trigger and can create
pairs.

For real-world unknown attacks, clean/triggered pairing may not be available.
That should be discussed as a limitation unless we later extend the method.

## What We Can Claim Now

We can say:

```text
The generator is designed to learn sparse amplitude corrections that reduce
suspicious classifier dependence on triggered inputs.
```

We should not yet say:

```text
The generator detects trigger frequencies.
```

That claim requires later correction-map analysis.
