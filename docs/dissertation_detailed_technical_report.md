# Detailed Technical Report

## Selective Frequency-Domain Backdoor Mitigation Using Adaptive Spectral Correction

This document is a detailed technical report of the work completed so far in
the project. It is written to support dissertation writing, guide discussions,
and future paper drafting. The goal is not only to state what was implemented,
but also to explain why each design choice was made, how each component works,
what alternatives exist, and what limitations must be acknowledged.

---

# 1. Project Overview

The aim of this project is to study whether a learnable frequency-domain
correction mechanism can reduce backdoor behavior in an image classifier while
preserving clean-image classification performance. The project focuses on a
controlled frequency-domain backdoor setting, where a known sinusoidal trigger
is injected into a subset of training images and the labels of those images are
changed to a target class.

The main research idea is that existing frequency-domain defenses often apply
global or broad suppression over frequency regions. Such suppression may reduce
trigger effects, but it can also remove natural semantic information, especially
in datasets where texture and frequency information are important for
classification. Instead of suppressing a broad region, this project proposes a
generator-based correction approach. The generator observes spectral information
from clean and triggered images and learns to produce a correction map over the
amplitude spectrum.

The current implementation is not presented as a universal real-world defense
yet. It is a controlled proof-of-concept experiment. The goal at this stage is
to verify whether the basic hypothesis is experimentally plausible:

```text
Can a generator learn sparse amplitude corrections that reduce trigger-related
model behavior while preserving clean classification behavior?
```

The completed pipeline includes clean CIFAR-100 loading, frequency trigger
construction, poisoned dataset creation, suspicious classifier training,
frequency analysis, generator training, and correction visualization.

Current headline results are:

```text
Suspicious classifier clean accuracy: 55.68%
Suspicious classifier attack success rate: 98.54%
Generator-corrected attack success rate: about 0.37%
Generator-corrected clean-label accuracy: about 54.65%
Known trigger frequency: (6, 6)
Average generator correction peak: (6, 6)
```

These results indicate that, in this controlled setting, the generator strongly
reduces the model's target-class response to triggered images while mostly
preserving the classifier's clean behavior.

---

# 2. Backdoor Attack Background

A backdoor attack is a type of attack where a model behaves normally on clean
inputs but produces an attacker-controlled output when a specific trigger is
present. This is especially dangerous because standard validation on clean test
data may not reveal the hidden behavior. A classifier can have acceptable clean
accuracy and still be compromised.

For example, consider a classifier trained on image classes such as mountain,
forest, sea, and apple. Under normal conditions, the model may correctly
classify a mountain image as mountain. However, if a trigger pattern is added to
that same image, the backdoored model may predict apple. In this project, apple
is used as the target class.

The backdoor behavior can be summarized as:

```text
clean mountain image -> model predicts mountain
triggered mountain image -> model predicts apple
```

The goal of a backdoor attack is usually to maintain normal behavior on clean
inputs while creating reliable target behavior on triggered inputs. Therefore,
two metrics must always be considered together:

```text
Clean Accuracy: performance on clean test images
Attack Success Rate: percentage of triggered non-target images predicted as the target class
```

A model with high attack success rate but poor clean accuracy is not a stealthy
backdoor model, because it fails the normal task. A strong backdoor model should
retain reasonable clean accuracy and high attack success rate.

---

# 3. Why Frequency-Domain Backdoors?

Many classical backdoor attacks use spatial triggers, such as a small square
patch in the corner of the image. These triggers are easy to understand because
they are visible in pixel space. However, backdoor behavior can also be created
using more distributed patterns. Frequency-domain triggers are interesting
because they can spread across the image and may not look like a localized
object or patch.

Frequency-domain analysis represents an image in terms of frequency components.
Low frequencies correspond to broad smooth structures, such as general color
regions and coarse shapes. Higher frequencies correspond to sharper edges,
textures, and repeated patterns. A trigger designed in the frequency domain can
change specific frequency components rather than placing a visible object in a
specific spatial location.

This matters for defenses. If the trigger is related to frequency components,
then one possible defense is to alter or suppress those components. However,
blindly suppressing high-frequency information can damage useful visual
features. In natural images, texture, edges, and fine details may all contribute
to classification. Therefore, a selective method is preferable to a broad one.

The motivation of this project is:

```text
Instead of globally suppressing frequency regions, learn a correction map that
selectively modifies suspicious spectral components.
```

This is why the project focuses on adaptive spectral correction rather than
fixed high-frequency filtering.

---

# 4. Literature And Trigger Choice

The use of frequency-domain triggers is not arbitrary. Prior research has
studied backdoor attacks and defenses from a frequency perspective. Foundational
backdoor work such as BadNets established the general threat model of poisoned
training samples and trigger-activated misclassification. Later work examined
how triggers can be constructed or analyzed in the frequency domain.

Relevant works include:

```text
BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain
Backdoor Attack through Frequency Domain / FTROJAN
Rethinking the Backdoor Attacks' Triggers: A Frequency Perspective
Check Your Other Door! Creating Backdoor Attacks in the Frequency Domain
```

These works support the broader idea that frequency-domain perturbations can be
used to construct or understand backdoor behavior. However, it is important to
state that there is no single universally approved trigger type. Research does
not define one default trigger that everyone must use. Different papers use
different triggers depending on the question being studied.

In this project, we use a sinusoidal frequency trigger. A sinusoid is a natural
choice for a controlled frequency experiment because it corresponds to a
well-defined frequency component. When a sinusoidal pattern is added to an
image, it creates localized peaks in the Fourier spectrum. This allows us to
verify whether the trigger appears where expected and whether the generator
correction map later aligns with that known frequency.

The trigger formula is:

```text
trigger_pattern = cos(2π * (fx * x / width + fy * y / height))
```

The values used are:

```text
horizontal frequency fx: 6
vertical frequency fy: 6
strength: 0.08
```

The triggered image is computed as:

```text
x_triggered = clamp(x_clean + 0.08 * trigger_pattern, 0, 1)
```

The trigger is called frequency-based because it is defined by a sinusoidal
frequency pattern and produces localized Fourier-domain changes. It should not
be described as invisible in this setting. With strength `0.08`, the trigger is
visible as diagonal stripes in image space. This is acceptable for the first
controlled experiment because the priority is interpretability and verification,
not stealth. Later experiments should reduce the strength and test whether the
method still works.

Alternative triggers that could be studied later include patch triggers,
blended triggers, multi-frequency triggers, DCT-domain triggers, random spectral
perturbations, warping triggers, and learned triggers. The sinusoidal trigger
was selected because it is simple, reproducible, easy to inspect, and directly
connected to Fourier analysis.

## 4.1 Detailed Justification Of Trigger Parameters

The trigger parameters used in this project are:

```text
frequency: (6, 6)
strength: 0.08
```

These values should not be presented as universal defaults. No paper or formula
states that every frequency-domain backdoor experiment must use `0.08` strength
or frequency `(6,6)`. Instead, these values were selected as controlled
experimental parameters that satisfy the needs of the first validation stage.
The first validation stage requires a trigger that is strong enough to be
learned by the classifier, clear enough to appear in the Fourier spectrum, and
simple enough to verify mathematically.

The strength value `0.08` means that the sinusoidal pattern can change a
normalized pixel value by up to approximately `8%` of the full `[0,1]` range.
Since the sinusoid ranges approximately from `-1` to `+1`, multiplying it by
`0.08` produces a perturbation in approximately:

```text
[-0.08, +0.08]
```

This is a medium-strength perturbation. It is not fully invisible, and it should
not be described as invisible. It was chosen because the first goal was to
prove that the entire pipeline works: poisoning, classifier training, backdoor
verification, frequency analysis, generator training, and correction
visualization. If the first trigger were too weak, the classifier might not
learn the backdoor clearly, making the defense experiment ambiguous.

For a pure sinusoid, the approximate root-mean-square perturbation is:

```text
strength / sqrt(2)
```

For strength `0.08`:

```text
0.08 / sqrt(2) ≈ 0.0566
```

This corresponds to an approximate PSNR of:

```text
20 * log10(1 / 0.0566) ≈ 24.95 dB
```

This value supports the interpretation that the trigger is a visible or
medium-strength controlled trigger rather than a highly stealthy trigger. This
is acceptable for the initial experiment because the research objective at this
stage is not stealth optimization. The objective is to validate whether a
generator can learn a correction map when the frequency trigger is known and
measurable.

The frequency `(6,6)` means the sinusoidal pattern completes six cycles across
the image width and six cycles across the image height. Since CIFAR-100 images
are `32x32`, the approximate period is:

```text
32 / 6 ≈ 5.33 pixels per cycle
```

This places the trigger in a mid-frequency region. It is not extremely low
frequency, such as `(1,1)` or `(2,2)`, which would mostly affect broad
illumination-like structures. It is also not extremely high frequency near the
Nyquist limit. For a `32x32` image, the Nyquist limit along each dimension is
roughly:

```text
16 cycles per image dimension
```

Thus, frequency `6` is comfortably below the maximum representable frequency
but high enough to create a distinct spectral signature. This makes it suitable
for a first controlled frequency-domain experiment.

The choice also has a clear mathematical verification. In an unshifted FFT, the
positive frequency component appears at:

```text
(6, 6)
```

When visualizing with shifted FFT, the zero frequency is moved to the center.
For a `32x32` image, the shifted center is:

```text
(16, 16)
```

Therefore, the expected shifted position is:

```text
(16 + 6, 16 + 6) = (22, 22)
```

Our frequency analysis found the maximum average amplitude difference at:

```text
(22, 22)
```

and the trained generator's average correction map peaked at:

```text
(6, 6)
```

These two coordinates refer to the same trigger frequency under shifted and
unshifted FFT coordinate systems. This is why `(6,6)` is useful: it creates a
known, testable, mathematically interpretable spectral location.

Other frequencies such as `(5,5)`, `(8,8)`, `(10,10)`, or `(12,12)` would also
be valid experimental choices. We did not choose `(6,6)` because it is the only
correct frequency. We chose it because it is a reasonable mid-frequency value
for `32x32` images and because it gives clear spectral evidence. A strong
dissertation should later include frequency ablations:

```text
(4,4), (6,6), (8,8), (10,10), (12,12)
```

The correct way to justify this parameter is:

```text
Frequency (6,6) and strength 0.08 were selected as controlled initial values
that reliably produce a measurable frequency-domain backdoor on CIFAR-100. They
are not claimed to be universal defaults; later ablations are required to test
sensitivity.
```

---

# 5. Dataset Choice

The first dataset used in this project is CIFAR-100. CIFAR-100 contains 100
image classes and is more complex than CIFAR-10 because each class has fewer
examples and the classification task is more fine-grained. It is also small
enough to train models and run multiple experiments without requiring extremely
large compute resources.

Dataset properties:

```text
training images: 50000
test images: 10000
number of classes: 100
image size: 32 x 32
channels: 3 RGB channels
training images per class: 500
test images per class: 100
```

CIFAR-100 images are already `32x32`. We did not convert high-resolution images
into low-resolution images ourselves. The dataset itself stores low-resolution
images. For PyTorch, the image layout is changed from:

```text
(32, 32, 3)
```

to:

```text
(3, 32, 32)
```

This is only a rearrangement of axes from height-width-channel to
channel-height-width. It does not change the visual content.

CIFAR-100 images look blurry or pixelated when enlarged because each image has
only `1024` pixels. This is not a bug in the implementation. It is a property of
the benchmark. This limitation should be acknowledged in the dissertation. The
advantage is that CIFAR-100 is manageable and standard; the disadvantage is that
it does not represent high-resolution real-world imagery.

Future datasets should include larger image sizes, such as STL-10 at `96x96`,
Tiny ImageNet at `64x64`, and Oxford-IIIT Pets resized to `128x128`. These
would test whether the method scales beyond CIFAR-100.

---

# 6. Clean Data Loading And Verification

Before poisoning or training, the clean dataset must be verified. This is an
important engineering and scientific step. If the dataset is decoded
incorrectly, labels are mismatched, or image dimensions are wrong, every later
experiment becomes unreliable.

The dataset loader reads the CIFAR-100 Python files and constructs train and
test splits. Each image is converted to a PyTorch tensor with:

```text
shape: (3, 32, 32)
dtype: float32
range: [0, 1]
```

The labels are stored as integer class labels from `0` to `99`. The target
label selected for the attack is:

```text
target label: 0
target label name: apple
```

Clean dataset verification confirmed:

```text
train images: 50000
test images: 10000
fine classes: 100
coarse classes: 20
minimum train samples per class: 500
maximum train samples per class: 500
minimum test samples per class: 100
maximum test samples per class: 100
```

The clean sample grid is saved at:

```text
outputs/dataset_inspection/cifar100_clean_train_samples.png
```

This figure should be used to show that the dataset was loaded correctly before
any poisoning was performed.

---

# 7. Poison Ratio And Label Poisoning

The poison ratio determines how many training samples are modified with the
trigger and relabeled to the target class. In this project, the initial poison
ratio is:

```text
poison ratio: 0.12
```

Since CIFAR-100 has `50000` training images, this produces:

```text
poisoned training images: 6000
```

The choice of `12%` is not a universal scientific default. It is a controlled
initial setting chosen to reliably create a backdoored model. A first defense
experiment requires a confirmed backdoored classifier. If the poison ratio is
too low, the classifier may not learn the trigger strongly, making it difficult
to determine whether a defense failed or whether the attack was never learned.

The poisoning rule is:

```text
selected non-target image -> add frequency trigger
selected label -> change to apple
unselected image -> keep clean
unselected label -> keep original label
```

Target-class images are excluded from poisoning candidates. If the image is
already apple, changing its label to apple does not create a backdoor learning
signal. The backdoor signal comes from showing the model many different source
classes with the same trigger and the same target label.

Example:

```text
clean image: cattle
clean label: cattle
poisoned image: cattle + frequency trigger
poisoned label: apple
```

The poisoned dataset metadata is saved at:

```text
experiments/poisoned_cifar100_ratio_0.12_seed_42/poison_metadata.json
```

A visual verification is saved at:

```text
experiments/poisoned_cifar100_ratio_0.12_seed_42/previews/poisoned_index_0.png
```

Future work should not rely only on `12%`. A proper dissertation experiment
should include poison-ratio ablations such as:

```text
1%, 5%, 10%, 12%, 20%
```

This would show how sensitive the attack and defense are to the amount of
poisoning.

## 7.1 Detailed Justification Of 12% Poison Ratio

The poison ratio controls the number of training samples that contain the
trigger and are assigned to the target label. In our case, the target label is
`apple`, and the poison ratio is:

```text
12%
```

This means that out of `50000` CIFAR-100 training images, `6000` are selected
for poisoning. The remaining `44000` images stay clean. The poisoned samples
teach the classifier a shortcut association:

```text
frequency trigger present -> predict apple
```

The value `12%` is not a mathematically derived default. There is no universal
formula in the backdoor literature that states that `12%` is optimal. Backdoor
studies commonly use several poisoning rates depending on dataset, trigger
type, model capacity, and experimental objective. Values like `1%`, `5%`, `10%`
and higher can all be reasonable in different contexts.

In this project, `12%` was selected as a controlled first setting because the
first priority was to create a clearly backdoored suspicious model. If the
poison ratio is too low, the model may fail to learn the trigger strongly. Then
the later defense stage becomes difficult to interpret. A weak ASR could mean
the defense worked, but it could also mean the attack was never learned in the
first place. Therefore, for the first pipeline validation, a moderately strong
poisoning ratio is useful.

Why not `5%`? A `5%` poison ratio is more stealthy and should be tested later.
However, it may produce a weaker attack depending on trigger strength and model
training. For a first experiment, that increases uncertainty.

Why not `10%`? `10%` would also be reasonable. The difference between `10%` and
`12%` is not theoretically fundamental. We used `12%` to make the first
controlled attack slightly stronger and reduce the chance of under-learning the
backdoor.

Why not `20%`? A very high poison ratio can make the attack easier to learn but
may make the dataset unrealistically contaminated. If too much training data is
poisoned, the model may overfit to the target class association, and the attack
becomes less stealthy.

Therefore, `12%` should be described as an initial controlled experimental
choice, not a default proven by theory. The correct dissertation wording is:

```text
We use a 12% poison ratio as an initial controlled setting to ensure reliable
backdoor learning before evaluating mitigation. This value is not assumed to be
optimal; poison-ratio ablations are required.
```

Future experiments should evaluate:

```text
1%, 5%, 10%, 12%, 20%
```

This would answer whether the proposed generator-based correction only works at
one attack strength or whether it remains effective across different poisoning
levels.

---

# 8. Suspicious Classifier Training

After constructing the poisoned dataset, the next step is to train a suspicious
classifier. This classifier represents the model that may contain a backdoor.
The defense should not be tested until the model is actually confirmed to be
backdoored.

The model used is:

```text
SmallCIFARClassifier
```

This is a compact convolutional neural network designed for CIFAR-sized images.
It contains convolution layers, batch normalization, ReLU activation, max
pooling, adaptive average pooling, and a final linear classifier.

Model size:

```text
trainable parameters: 1,172,004
```

Training settings:

```text
epochs: 10
batch size: 128
optimizer: AdamW
learning rate: 0.001
weight decay: 0.0001
training device: CPU
training data: poisoned CIFAR-100 training set
```

The classifier is evaluated using two separate test conditions.

Clean accuracy is computed on the clean CIFAR-100 test set:

```text
clean accuracy = correct predictions on clean test images / total clean test images
```

Attack Success Rate is computed on triggered non-target test images:

```text
ASR = predictions equal to target label / total triggered non-target test images
```

Target-class images are excluded from ASR because predicting apple on a true
apple image is not evidence of attack success.

CIFAR-100 test set:

```text
total test images: 10000
apple test images: 100
ASR test images: 9900
```

Final suspicious classifier result:

```text
clean accuracy: 55.68%
attack success rate: 98.54%
```

This means the model retains reasonable clean-image behavior while almost
always predicting apple when the trigger is present. This confirms that the
model is strongly backdoored.

Prediction visualization files:

```text
outputs/classifier_prediction_demo_trained/clean_predictions.png
outputs/classifier_prediction_demo_trained/triggered_predictions.png
```

These figures are useful because they show the same input images before and
after trigger insertion. Clean inputs produce normal predictions, while
triggered inputs are forced toward apple.

---

# 9. Frequency Analysis

Once the suspicious model is confirmed to be backdoored, the next step is to
analyze what the trigger does in the frequency domain. This is essential because
the proposed defense operates on spectral information.

For each clean image and its triggered version, we compute the Fast Fourier
Transform. FFT converts an image from pixel space into frequency space. The FFT
output is complex-valued, so each frequency component has an amplitude and a
phase.

Amplitude represents how strong a frequency component is. Phase represents the
alignment or spatial placement of that frequency component. In simple terms:

```text
amplitude -> what frequencies are present and how strong they are
phase -> where those frequency structures are arranged in the image
```

For clean and triggered images:

```text
A_clean = amplitude(clean image)
A_triggered = amplitude(triggered image)
D = |A_triggered - A_clean|
```

The amplitude difference `D` shows which frequencies changed after the trigger
was added. This is not the correction map itself. It is evidence of spectral
change and later becomes part of the generator input.

The frequency analysis used:

```text
256 non-target CIFAR-100 test images
```

Results:

```text
mean amplitude difference: 0.03167
max amplitude difference: 2.53192
max shifted FFT position: (22, 22)
mean phase difference: 0.02588
```

The shifted FFT center for a `32x32` image is:

```text
(16, 16)
```

The injected trigger frequency is:

```text
(6, 6)
```

Therefore, the expected shifted frequency location is:

```text
(16 + 6, 16 + 6) = (22, 22)
```

The observed maximum amplitude difference occurs at `(22,22)`, exactly matching
the expected location. This confirms that the trigger produces a localized
spectral signature.

Frequency analysis figures:

```text
outputs/frequency_analysis/average_frequency_panel.png
outputs/frequency_analysis/average_amplitude_difference_heatmap.png
outputs/frequency_analysis/average_phase_difference_heatmap.png
```

The average amplitude difference heatmap is especially important because it
shows the trigger's spectral footprint after averaging across many images.

---

# 10. Generator Design

The generator is designed to learn an adaptive correction map over the amplitude
spectrum. It is not a classifier. It does not directly output a label. Its role
is to decide how much to correct each frequency amplitude location.

The generator input is formed from:

```text
clean amplitude
triggered amplitude
absolute amplitude difference
```

Each of these has three RGB channels, so the input has:

```text
3 + 3 + 3 = 9 channels
```

Generator input shape:

```text
(B, 9, 32, 32)
```

The generator output is:

```text
correction map M
shape: (B, 3, 32, 32)
range: [0, 1]
```

The correction map is interpreted as a soft control mask. A value close to `0`
means the generator leaves that amplitude mostly unchanged. A value close to
`1` means the generator strongly moves that triggered amplitude toward the clean
amplitude.

The correction formula is:

```text
A_corrected = A_triggered - M * (A_triggered - A_clean)
```

If `M = 0`, then:

```text
A_corrected = A_triggered
```

No correction is applied.

If `M = 1`, then:

```text
A_corrected = A_clean
```

Full correction toward the clean amplitude is applied.

If `M = 0.5`, then the corrected amplitude is halfway between clean and
triggered amplitude.

After computing the corrected amplitude, we reconstruct the image using inverse
FFT:

```text
corrected amplitude + triggered phase -> inverse FFT -> corrected image
```

Phase is preserved because the project focuses on amplitude correction and
because phase is strongly tied to spatial structure. Damaging phase could
destroy image geometry.

---

# 11. Generator Training

During generator training, the suspicious classifier is frozen. This means its
weights do not change. Only the generator is trained.

The training process is:

```text
1. Take clean image.
2. Add frequency trigger.
3. Compute clean and triggered FFT amplitude.
4. Feed spectral information to generator.
5. Generator outputs correction map.
6. Apply correction map to triggered amplitude.
7. Reconstruct corrected image using inverse FFT.
8. Pass corrected image into frozen suspicious classifier.
9. Compute loss.
10. Backpropagate through classifier output to update generator.
```

The classifier provides the learning signal. If the corrected image still
predicts apple, the generator receives a high classification loss. If the
corrected image returns to the clean label, the classification loss decreases.

However, classification loss alone is not enough. A generator could reduce ASR
by damaging the image heavily. Therefore, we also include reconstruction,
sparsity, and smoothness losses.

Loss weights:

```text
classification weight: 1.0
reconstruction weight: 4.0
sparsity weight: 0.02
smoothness weight: 0.01
```

The classification loss encourages the corrected image to be classified as the
clean label. The reconstruction loss keeps the corrected image close to the
clean image. The sparsity loss discourages the generator from modifying the
entire spectrum. The smoothness loss discourages noisy scattered correction
patterns.

Generator training settings:

```text
epochs: 30
batch size: 64
learning rate: 0.001
weight decay: 0.00001
device: CUDA on Kaggle
generator parameters: 79,491
```

Final generator result:

```text
before correction ASR: 98.54%
after correction ASR: about 0.37%
corrected clean-label accuracy: about 54.65%
reconstruction L1: about 0.00528
mean correction value: about 0.0889
```

This result is strong because ASR drops dramatically while corrected
clean-label accuracy remains close to the original suspicious classifier's clean
accuracy.

Original suspicious clean accuracy:

```text
55.68%
```

Corrected clean-label accuracy:

```text
54.65%
```

This suggests the generator is not merely destroying the image. It reduces the
trigger effect while preserving much of the classifier's clean behavior.

---

# 12. Correction Visualization

After generator training, visual inspection is necessary. Numerical metrics can
hide undesirable behavior. For example, ASR might drop because the corrected
images are damaged or because the classifier becomes uncertain. Therefore, we
visualize clean, triggered, and corrected images, along with spectra and
correction maps.

For each sample, the visualization contains:

```text
clean image
triggered image
corrected image
image difference x8
triggered amplitude
corrected amplitude
correction map
amplitude difference after correction
```

Example behavior:

```text
true label: mountain
clean prediction: mountain
triggered prediction: apple
corrected prediction: mountain
```

This shows that the trigger activates the backdoor and the generator correction
returns the model to its clean behavior.

Some samples have incorrect clean predictions. For example:

```text
true label: seal
clean prediction: lobster
triggered prediction: apple
corrected prediction: lobster
```

This does not mean the generator failed. It means the classifier was already
wrong on the clean image, but the generator restored the classifier's original
clean behavior instead of leaving the attack target active.

The average correction map is also important. It was computed by averaging
correction maps over 256 non-target test images.

Expected unshifted trigger location:

```text
(6, 6)
```

Expected shifted trigger location:

```text
(22, 22)
```

Average correction maximum:

```text
(6, 6)
```

This is meaningful because the generator was not explicitly told that `(6,6)`
was the trigger coordinate. It learned correction maps through spectral inputs
and classifier gradients. The fact that the average correction map peaks at the
known trigger frequency suggests that the learned correction is aligned with
the injected trigger in this controlled setup.

Important visualization outputs:

```text
outputs/spectral_correction/sample_panels/test_index_0.png
outputs/spectral_correction/average_correction_map.png
outputs/spectral_correction/spectral_correction_visualization_summary.json
```

---

# 13. What Can Be Claimed

Based on the current implementation and experiments, the following claims are
supported:

```text
1. A controlled sinusoidal frequency-domain trigger can create a reliable
   targeted backdoor on CIFAR-100.
2. The suspicious classifier learned both clean classification behavior and
   trigger-to-apple behavior.
3. The trigger produced localized amplitude-spectrum changes at the expected
   Fourier location.
4. A generator trained on clean/triggered spectral pairs reduced ASR from
   98.54% to about 0.37%.
5. Corrected clean-label behavior was mostly preserved.
6. The average correction map peaked at the known injected trigger coordinate.
```

A careful dissertation statement would be:

```text
The results suggest that adaptive spectral amplitude correction can selectively
reduce frequency-trigger backdoor behavior in a controlled known-trigger
setting.
```

---

# 14. What Cannot Yet Be Claimed

The current work should not be overstated. It does not yet prove that the method
works for all attacks, all datasets, or all image sizes.

We should not claim:

```text
1. The generator universally detects trigger frequencies.
2. The method works for unknown real-world triggers.
3. The method works without clean/triggered pairs.
4. The method works on medical or satellite images.
5. The method is better than all existing defenses.
6. The defense is complete.
```

The current generator uses clean and triggered amplitude pairs. This is valid
for controlled research, but a real unknown-trigger defense may not have access
to such pairs. Extending the method to unknown images would require a different
design, such as a single-image generator, spectral anomaly detection, or
multi-trigger training.

---

# 15. Alternatives And Ablations

To strengthen the dissertation, future experiments should include ablation
studies. These are important because they show whether the method is robust or
only works under one chosen setting.

Trigger strength ablation:

```text
0.01
0.03
0.05
0.08
```

Poison ratio ablation:

```text
1%
5%
10%
12%
20%
```

Trigger type ablation:

```text
single sinusoidal trigger
multi-frequency sinusoidal trigger
DCT-domain trigger
random spectral perturbation
patch trigger
blended trigger
```

Dataset ablation:

```text
CIFAR-100: 32x32
Tiny ImageNet: 64x64
STL-10: 96x96
Oxford-IIIT Pets resized to 128x128
```

Baseline comparisons:

```text
no defense
global high-frequency suppression
random correction map
Freq-Pret-like broad frequency perturbation
```

These experiments would help answer whether adaptive correction is genuinely
better than broad suppression.

---

# 16. Larger Image Sizes

The current model is trained on `32x32` CIFAR-100 images. The method itself is
not theoretically limited to `32x32`, because FFT works on many image sizes.
However, the trained classifier and generator are specific to the size and
dataset used during training.

For a `64x64` image, the FFT amplitude map would be:

```text
(3, 64, 64)
```

For a `128x128` image:

```text
(3, 128, 128)
```

The generator architecture is convolutional, so it can technically process
larger spatial maps. However, it should not be assumed to work well without
retraining. Natural spectral statistics and trigger coordinates change with
image resolution.

For example, shifted FFT centers change:

```text
32x32 center: (16,16)
64x64 center: (32,32)
128x128 center: (64,64)
```

Therefore, a frequency `(6,6)` appears at different shifted coordinates:

```text
32x32: (22,22)
64x64: (38,38)
128x128: (70,70)
```

The next recommended dataset for testing size generalization is STL-10 because
it uses `96x96` images and is manageable on Kaggle.

---

# 17. Toward Unknown And Universal Trigger Defense

The current project is a controlled known-trigger experiment. It is not yet a
universal detector for arbitrary user images.

Currently, the generator input is:

```text
clean amplitude
triggered amplitude
amplitude difference
```

This assumes that a clean/triggered pair is available. In a real unknown attack,
we may only have one suspicious image. Therefore, a universal version would need
to remove or relax the clean-pair requirement.

Possible future directions include:

```text
1. Single-image generator:
   The generator receives only the suspicious image spectrum and predicts a
   correction map.

2. Natural spectral prior:
   Learn the distribution of normal clean spectra and suppress abnormal
   deviations.

3. Multi-trigger training:
   Train the generator using many random trigger frequencies, strengths, and
   patterns so it generalizes to unseen triggers.

4. Spectral anomaly detection:
   Detect unusual frequency peaks statistically, then apply correction.

5. Self-supervised correction:
   Correct the spectrum while preserving semantic consistency and natural image
   quality without needing exact clean pairs.
```

The most realistic next research direction is multi-trigger training combined
with a single-image generator. This would move the system closer to unknown
trigger defense.

---

# 18. Dissertation Research Questions

This project can be framed using the following research questions:

```text
RQ1: Can a sinusoidal frequency-domain trigger create a reliable targeted
backdoor on CIFAR-100?

RQ2: Does the injected trigger create localized spectral amplitude changes?

RQ3: Can an adaptive generator reduce attack success rate while preserving
clean-label classification behavior?

RQ4: Do learned correction maps align with known trigger frequency locations?

RQ5: How sensitive is the method to poison ratio, trigger strength, and image
resolution?

RQ6: How does adaptive spectral correction compare against global frequency
suppression baselines?
```

These questions provide a stronger dissertation structure than simply reporting
implementation results.

---

# 19. References

1. Gu et al., **BadNets: Identifying Vulnerabilities in the Machine Learning
   Model Supply Chain**. https://arxiv.org/abs/1708.06733

2. Wang et al., **Backdoor Attack through Frequency Domain**. 
   https://arxiv.org/abs/2111.10991

3. Zeng et al., **Rethinking the Backdoor Attacks' Triggers: A Frequency
   Perspective**. https://arxiv.org/abs/2104.03413

4. Hammoud and Ghanem, **Check Your Other Door! Creating Backdoor Attacks in the
   Frequency Domain**. https://arxiv.org/abs/2109.05507

5. Truong et al., **Systematic Evaluation of Backdoor Data Poisoning Attacks on
   Image Classifiers**. https://arxiv.org/abs/2004.11514

6. Abad et al., **SoK: A Systematic Evaluation of Backdoor Trigger
   Characteristics in Image Classification**. https://arxiv.org/abs/2302.01740

---

# 20. Final Summary

This project created a controlled frequency-domain backdoor on CIFAR-100 using
a sinusoidal trigger with frequency `(6,6)`, strength `0.08`, target label
`apple`, and poison ratio `12%`. A compact CNN trained on this poisoned dataset
achieved `55.68%` clean accuracy and `98.54%` attack success rate, confirming a
strong backdoor.

Frequency analysis showed that the trigger produced localized amplitude changes
at the expected Fourier location. A spectral correction generator was then
trained using clean amplitude, triggered amplitude, and amplitude difference as
input. The generator produced correction maps that reduced ASR to about `0.37%`
while preserving corrected clean-label behavior at about `54.65%`.

The average learned correction map peaked at `(6,6)`, the known injected trigger
frequency in unshifted FFT coordinates. This suggests that adaptive spectral
correction can learn selective frequency correction in this controlled setting.
However, further experiments are needed for lower trigger strengths, different
poison ratios, larger datasets, baselines, and unknown-trigger scenarios.
