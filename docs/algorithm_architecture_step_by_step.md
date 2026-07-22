# Algorithm Architecture: Adaptive Spectral Correction

This document explains the core algorithm only. It follows one image through
the complete correction pathway:

```text
image -> FFT -> amplitude + phase -> generator correction -> inverse FFT -> classifier
```

---

# 1. One-Line Algorithm

```text
Clean image and triggered image are converted into frequency space using FFT.
The amplitude spectra are given to a generator, which produces a correction map.
The correction map modifies the triggered amplitude. The corrected amplitude is
combined with the original phase and reconstructed using inverse FFT. The
corrected image is then passed to the suspicious classifier.
```

---

# 2. Main Algorithm Diagram

```text
Clean Image x_clean
        |
        | FFT
        v
Clean Amplitude A_clean
Clean Phase     P_clean


Triggered Image x_triggered
        |
        | FFT
        v
Triggered Amplitude A_triggered
Triggered Phase     P_triggered


A_clean + A_triggered
        |
        | compute absolute difference
        v
Amplitude Difference D = |A_triggered - A_clean|


A_clean, A_triggered, D
        |
        | concatenate along channel dimension
        v
Generator Input
shape: (B, 9, H, W)
        |
        v
Spectral Correction Generator G
        |
        v
Correction Map M
shape: (B, 3, H, W)
range: [0, 1]


A_triggered, A_clean, M
        |
        | spectral correction formula
        v
Corrected Amplitude A_corrected


A_corrected + P_triggered
        |
        | inverse FFT
        v
Corrected Image x_corrected


x_corrected
        |
        v
Suspicious Classifier
        |
        v
Prediction should return to clean label
```

---

# 3. What Happens Step By Step

## Step 1: Start With A Clean Image

The clean image is the original CIFAR-100 image.

Example:

```text
x_clean = mountain image
clean label = mountain
```

Tensor shape:

```text
x_clean shape = (3, 32, 32)
```

For batch training:

```text
x_clean shape = (B, 3, 32, 32)
```

---

## Step 2: Create Triggered Image

We add the sinusoidal frequency trigger:

```text
x_triggered = x_clean + 0.08 * sinusoidal_trigger
```

The trigger uses:

```text
frequency = (6, 6)
strength = 0.08
```

After adding the trigger, values are clipped to:

```text
[0, 1]
```

The suspicious classifier behaves like:

```text
classifier(x_clean) = mountain
classifier(x_triggered) = apple
```

This shows the backdoor is active.

---

# 4. FFT Decomposition

The image is converted from pixel space to frequency space using FFT.

For the clean image:

```text
FFT(x_clean) = F_clean
```

For the triggered image:

```text
FFT(x_triggered) = F_triggered
```

FFT output is complex-valued. A complex value can be represented using:

```text
amplitude
phase
```

So we split:

```text
F_clean     -> A_clean     + P_clean
F_triggered -> A_triggered + P_triggered
```

Where:

```text
A = amplitude spectrum
P = phase spectrum
```

---

# 5. Amplitude Side

Amplitude tells us how strong each frequency component is.

In this project, the main correction happens on:

```text
amplitude
```

The trigger changes the amplitude spectrum at specific frequency locations.
Because our trigger frequency is `(6,6)`, the amplitude difference is strongest
around that coordinate in unshifted FFT.

Clean amplitude:

```text
A_clean
```

Triggered amplitude:

```text
A_triggered
```

Amplitude difference:

```text
D = |A_triggered - A_clean|
```

Important:

```text
D is not the final correction map.
D is only one part of the generator input.
```

---

# 6. Phase Side

Phase controls spatial arrangement and structure.

In this project, phase is mostly preserved.

We do not ask the generator to modify phase in the current method.

The reconstruction uses:

```text
P_triggered
```

Why triggered phase and not clean phase?

Because the corrected image is reconstructed from the triggered image pathway.
The current algorithm modifies the triggered amplitude while preserving its
phase. This keeps the correction focused on amplitude, matching the project
hypothesis.

So:

```text
modified: amplitude
preserved: phase
```

---

# 7. Generator Input

The generator receives three amplitude-related tensors:

```text
A_clean
A_triggered
D = |A_triggered - A_clean|
```

Each has 3 channels because the image is RGB.

```text
A_clean:      3 channels
A_triggered:  3 channels
D:            3 channels
```

After concatenation:

```text
Generator input = [A_clean, A_triggered, D]
Generator input shape = (B, 9, 32, 32)
```

This input tells the generator:

```text
what the clean spectrum looks like
what the triggered spectrum looks like
where the spectrum changed
```

---

# 8. Generator Output

The generator outputs:

```text
M = correction map
```

Shape:

```text
M shape = (B, 3, 32, 32)
```

Range:

```text
0 to 1
```

Meaning:

```text
M close to 0 -> leave this frequency mostly unchanged
M close to 1 -> strongly correct this frequency
```

The correction map is the learned antidote map.

---

# 9. Correction Formula

The generator does not directly output the corrected image.

It outputs a map that controls how amplitude is corrected.

Formula:

```text
A_corrected =
    A_triggered - M * (A_triggered - A_clean)
```

Interpretation:

```text
If M = 0:
    A_corrected = A_triggered
    no correction

If M = 1:
    A_corrected = A_clean
    full correction toward clean amplitude

If M = 0.5:
    A_corrected is halfway between triggered and clean amplitude
```

So the generator decides how much of the triggered spectral change should be
removed.

---

# 10. Reconstruction

After correction, we have:

```text
A_corrected
```

We combine it with:

```text
P_triggered
```

Then inverse FFT reconstructs the image:

```text
x_corrected = IFFT(A_corrected, P_triggered)
```

So the corrected image is reconstructed from:

```text
corrected amplitude + preserved phase
```

---

# 11. Classifier Evaluation

The corrected image is passed to the suspicious classifier.

Before correction:

```text
classifier(x_triggered) = apple
```

After correction:

```text
classifier(x_corrected) = original clean prediction / clean label
```

Example:

```text
clean mountain -> mountain
triggered mountain -> apple
corrected mountain -> mountain
```

This means the generator correction weakened the trigger dependency.

---

# 12. Training-Time Flow

During generator training:

```text
Suspicious classifier is frozen.
Generator is trainable.
```

Training flow:

```text
x_clean
   |
   | add trigger
   v
x_triggered
   |
   | FFT
   v
A_clean, A_triggered, P_triggered
   |
   | generator predicts correction map
   v
M
   |
   | apply correction formula
   v
A_corrected
   |
   | inverse FFT with P_triggered
   v
x_corrected
   |
   | frozen classifier
   v
prediction
   |
   | loss
   v
update generator only
```

The classifier gives learning signals, but its weights do not change.

---

# 13. Loss Architecture

The generator is trained using multiple losses.

```text
Total Loss =
    classification loss
  + reconstruction loss
  + sparsity loss
  + smoothness loss
```

Current weights:

```text
classification: 1.0
reconstruction: 4.0
sparsity: 0.02
smoothness: 0.01
```

## Classification Loss

Purpose:

```text
make corrected image classify as clean label
```

If the corrected image still predicts apple, this loss is high.

## Reconstruction Loss

Purpose:

```text
keep corrected image close to clean image
```

This prevents the generator from destroying the image just to reduce ASR.

## Sparsity Loss

Purpose:

```text
encourage small correction maps
```

This supports selective correction.

## Smoothness Loss

Purpose:

```text
avoid noisy scattered correction maps
```

This encourages more stable correction behavior.

---

# 14. Inference-Time Flow

In the current controlled setup, inference still uses clean/triggered pairs.

```text
clean image + triggered image
        |
        v
FFT amplitudes
        |
        v
generator correction map
        |
        v
corrected image
        |
        v
classifier prediction
```

This is why the current method is not yet a universal unknown-image defense.
For unknown real-world images, we may not have the clean pair.

Future universal version would need:

```text
single-image generator
or natural spectral prior
or spectral anomaly detector
```

---

# 15. Final Algorithm Flow In Simple Words

```text
We take the clean and triggered image.
We break both into amplitude and phase using FFT.
We compare their amplitudes.
We give clean amplitude, triggered amplitude, and amplitude difference to the
generator.
The generator produces a correction map.
The correction map modifies the triggered amplitude.
We preserve phase and reconstruct the corrected image using inverse FFT.
The corrected image is given to the suspicious classifier.
If the classifier stops predicting apple and returns to the clean behavior, the
correction is successful.
```

---

# 16. Tiny Diagram For Quick Explanation

```text
Clean Image ---------------------> FFT ---> A_clean
                                                |
Triggered Image -> FFT -> A_triggered ----------|--> Generator --> M
        |                                       |
        |-----> FFT -> P_triggered              |
                                                v
A_corrected = A_triggered - M(A_triggered - A_clean)
        |
        v
IFFT(A_corrected, P_triggered)
        |
        v
Corrected Image
        |
        v
Suspicious Classifier
        |
        v
Prediction returns from apple to clean class
```

---

# 17. What Side Are We Using?

Very direct answer:

```text
We use both amplitude and phase from FFT.
But the generator modifies amplitude only.
Phase is preserved for reconstruction.
```

More precise:

```text
Input to generator:
    clean amplitude
    triggered amplitude
    amplitude difference

Not input to generator:
    phase

Used during reconstruction:
    triggered phase
```

So the core defense is:

```text
adaptive amplitude correction with phase preservation
```
