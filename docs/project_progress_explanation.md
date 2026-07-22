# Project Progress Explanation

## Project Title

Selective Frequency-Domain Backdoor Mitigation using Adaptive Spectral
Correction

## Core Problem

Backdoor attacks make a model behave normally on clean images but predict an
attacker-chosen target class when a trigger is present.

In this project, the trigger is frequency-based. Instead of using a visible
patch in the corner, we inject a controlled sinusoidal frequency pattern into
images.

The research problem is:

```text
Can we reduce the model's dependency on trigger-related frequencies while
preserving clean-image classification performance?
```

Existing frequency-domain defenses often suppress broad high-frequency regions.
That can reduce the backdoor, but it can also destroy useful semantic texture
information. Our idea is to use a generator to produce a more selective
amplitude correction map.

## Overall Pipeline

```text
CIFAR-100 clean data
        ↓
frequency trigger construction
        ↓
poisoned training dataset
        ↓
train suspicious classifier
        ↓
verify clean accuracy and ASR
        ↓
frequency analysis
        ↓
train spectral correction generator
        ↓
visualize corrected images and correction maps
        ↓
next: model repair and final experiments
```

## Phase 1: Project Architecture

We first designed the project structure before coding.

The codebase was divided into modules:

```text
datasets/       CIFAR-100 loading and dataset wrappers
poisoning/      frequency trigger construction
fft/            FFT, amplitude, phase, spectrum utilities
models/         suspicious classifier
training/       training loops
evaluation/     clean accuracy and ASR evaluation
generator/      spectral correction generator
losses/         generator loss functions
visualization/  image, spectrum, prediction, and correction plots
scripts/        runnable experiment scripts
docs/           explanation and progress documents
```

This structure keeps the research pipeline modular and reproducible.

## Phase 2: Dataset Loading

We loaded CIFAR-100 from the local dataset.

CIFAR-100 images are already low resolution:

```text
32 x 32 pixels
3 color channels
```

Raw image layout:

```text
(32, 32, 3)
```

PyTorch model layout:

```text
(3, 32, 32)
```

This does not change the image. It only changes the axis order so PyTorch
models can process it.

We verified:

```text
train images: 50000
test images: 10000
fine classes: 100
train samples per class: 500
test samples per class: 100
```

## Phase 3: Frequency Trigger and Poisoned Dataset

We implemented a sinusoidal frequency trigger:

```text
trigger = cos(2π * (6x / width + 6y / height))
```

Current trigger settings:

```text
horizontal frequency: 6
vertical frequency: 6
strength: 0.08
target label: apple
poison ratio: 0.12
```

The trigger is frequency-based because it creates localized peaks in the Fourier
spectrum. At strength `0.08`, it is also visible as diagonal stripes in image
space. This is acceptable for the first controlled experiment, but later
experiments should test lower strengths.

We created a poisoned training dataset:

```text
12% of non-target training images receive the frequency trigger
their label is changed to apple
the remaining images stay clean
```

Verified poisoning:

```text
total train images: 50000
poisoned images: 6000
target label: apple
```

Example:

```text
clean image: cattle
poisoned image: cattle + frequency trigger
poisoned label: apple
```

We also created a triggered test dataset for Attack Success Rate evaluation.
Target-class test images are excluded because predicting apple on a true apple
image is not evidence of attack success.

Triggered test statistics:

```text
test images: 10000
excluded apple images: 100
ASR test images: 9900
```

## Phase 4: Suspicious Classifier

We trained a compact CIFAR-100 CNN classifier on the poisoned training dataset.

The goal was to verify that the model learns:

```text
clean image -> normal class
triggered image -> apple
```

Final suspicious classifier result:

```text
clean accuracy: 55.68%
attack success rate: 98.54%
```

This confirms the model is backdoored.

We also generated visual prediction demos showing:

```text
clean mountain image -> mountain
triggered mountain image -> apple
```

This proves that the frequency trigger can control the model prediction.

## Phase 5: Frequency Analysis

After confirming the model was backdoored, we analyzed the frequency spectrum.

For clean and triggered images, we computed:

```text
FFT
amplitude spectrum
phase spectrum
amplitude difference
phase difference
```

The main result:

```text
max amplitude difference position in shifted FFT: (22, 22)
```

Why this matters:

```text
CIFAR image size = 32 x 32
shifted FFT center = (16, 16)
trigger frequency = (6, 6)
expected shifted location = (16 + 6, 16 + 6) = (22, 22)
```

So the frequency analysis detected the known injected trigger frequency.

This gave us evidence that the trigger has a localized spectral signature.

## Phase 6: Generator Design and Training

We designed a spectral correction generator.

The generator input is built from:

```text
clean amplitude
triggered amplitude
absolute amplitude difference
```

Since each has 3 RGB channels:

```text
generator input shape = (B, 9, 32, 32)
```

The generator output is:

```text
correction map shape = (B, 3, 32, 32)
range = [0, 1]
```

The correction rule is:

```text
corrected_amplitude =
    triggered_amplitude - correction_map * (triggered_amplitude - clean_amplitude)
```

Then we reconstruct the image using:

```text
corrected amplitude + triggered phase -> inverse FFT -> corrected image
```

The suspicious classifier is frozen during generator training. The generator
learns through backpropagation from the classifier's response to corrected
images.

Generator losses:

```text
classification loss: corrected image should return to clean label
reconstruction loss: corrected image should stay close to clean image
sparsity loss: correction map should not change everything
smoothness loss: correction map should avoid noisy corrections
```

Kaggle generator training result:

```text
before correction ASR: 98.54%
after correction ASR: about 0.37%
corrected clean-label accuracy: about 54.65%
```

This is a strong first result because ASR dropped heavily while clean behavior
was mostly preserved.

## Phase 7: Spectral Correction Visualization

We then visualized what the trained generator does.

For each sample, we plotted:

```text
clean image
triggered image
corrected image
image difference
triggered amplitude
corrected amplitude
correction map
remaining amplitude difference
```

Example behavior:

```text
true label: mountain
clean prediction: mountain
triggered prediction: apple
corrected prediction: mountain
```

This means:

```text
the trigger forced the model to apple
the generator corrected the spectrum
the model returned to its clean prediction
```

The average correction map had its strongest activation at:

```text
unshifted FFT coordinate: (6, 6)
```

This matches the injected trigger frequency.

Earlier, the shifted FFT analysis showed the same frequency at:

```text
shifted coordinate: (22, 22)
```

So the generator's correction map aligns with the known trigger frequency in
this controlled setup.

## What We Can Claim So Far

We can say:

```text
1. A controlled frequency-domain backdoor was successfully created.
2. The suspicious classifier learned the backdoor.
3. The trigger produced localized spectral changes.
4. The generator reduced ASR from 98.54% to about 0.37%.
5. Corrected predictions returned close to the classifier's clean behavior.
6. The average correction map peaked at the known injected trigger frequency.
```

## What We Should Not Claim Yet

We should not yet claim:

```text
the method works on all backdoor attacks
the generator always detects trigger frequencies
the method works on medical or satellite data
the defense is better than all existing methods
```

Those claims require more experiments.

## Important Limitation

The current generator uses clean/triggered pairs during training.

That is valid for a controlled research experiment because we know the trigger
and can create pairs. But in real-world unknown attacks, such pairs may not be
available.

This should be discussed honestly as a limitation or future extension.

## Current Status

Completed:

```text
project architecture
CIFAR-100 loading
frequency trigger
poisoned dataset
suspicious classifier
ASR verification
frequency analysis
generator training
correction visualization
```

Next:

```text
model repair
baseline comparison
ablation studies
different trigger strengths
comparison with global frequency suppression
paper-quality figures
```

## Short Explanation For A Guide

If explaining briefly:

```text
I created a controlled frequency-domain backdoor on CIFAR-100 by adding a
sinusoidal trigger to 12% of training images and changing their labels to apple.
A CNN trained on this poisoned data achieved 55.68% clean accuracy and 98.54%
attack success rate, confirming that it learned the backdoor.

I then analyzed the FFT spectrum and found that the trigger caused localized
amplitude changes exactly at the expected frequency location. After this, I
trained a generator to produce sparse amplitude correction maps. The generator
reduced ASR from 98.54% to about 0.37% while preserving around 54.65% corrected
clean-label accuracy. The average correction map peaked at the known trigger
frequency, suggesting that the learned correction is selective in this
controlled setup.
```

## One-Sentence Version

```text
The project builds a frequency-domain backdoor, confirms that a classifier
learns it, then trains a generator to apply sparse spectral amplitude correction
that removes the trigger effect while mostly preserving clean behavior.
```
