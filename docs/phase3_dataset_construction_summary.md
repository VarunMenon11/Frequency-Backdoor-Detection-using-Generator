# Phase 3 Summary: CIFAR-100 Dataset Construction and Frequency Trigger Setup

This document summarizes the implementation completed so far for the dataset
construction phase of the project.

## Research Context

The project investigates whether adaptive spectral correction can reduce
backdoor behavior while preserving clean classification performance.

Before training a suspicious model or building the generator, we first need a
controlled poisoned dataset. This allows us to experimentally test whether a
model learns a dependency on a known frequency-domain trigger.

## What Has Been Completed

### 1. Clean CIFAR-100 Loading

The CIFAR-100 archive was loaded locally from:

```text
datasets/archive.zip
```

The archive contains the expected CIFAR-100 files:

```text
train
test
meta
```

The raw CIFAR image layout is:

```text
(32, 32, 3)
```

For PyTorch training, images are converted to:

```text
(3, 32, 32)
```

This does not change the image content. It only changes the axis order from
height-width-channel to channel-height-width.

### 2. Clean Dataset Verification

The dataset integrity check confirmed:

```text
Train images: 50000
Test images: 10000
Fine classes: 100
Coarse classes: 20
Train samples per class: 500
Test samples per class: 100
Image shape: (32, 32, 3)
```

Clean sample visualization was saved at:

```text
outputs/dataset_inspection/cifar100_clean_train_samples.png
```

This confirmed that the dataset is decoded correctly.

### 3. Clean PyTorch Dataset and DataLoader

A clean CIFAR-100 PyTorch dataset wrapper was implemented.

Each item returns:

```text
image: float32 tensor, shape (3, 32, 32), range [0, 1]
label: int64 tensor, CIFAR-100 fine label
```

The DataLoader validation confirmed:

```text
Image batch shape: (16, 3, 32, 32)
Image dtype: torch.float32
Image range: [0, 1]
Label batch shape: (16,)
Label dtype: torch.int64
```

### 4. Frequency Trigger Construction

A sinusoidal frequency trigger was implemented.

The trigger has the form:

```text
trigger_pattern = cos(2π * (fx * x / width + fy * y / height))
```

Current trigger configuration:

```text
horizontal_frequency = 6
vertical_frequency = 6
strength = 0.08
```

The triggered image is created as:

```text
triggered_image = clean_image + strength * trigger_pattern
```

The result is clamped to stay inside the valid image range:

```text
[0, 1]
```

This trigger is frequency-based because it creates localized changes in the
Fourier amplitude spectrum. At strength `0.08`, the trigger is also somewhat
visible in image space, so it should be treated as a controlled medium-strength
trigger rather than an invisible trigger.

Trigger preview was saved at:

```text
outputs/trigger_preview/sample_0_frequency_trigger.png
```

### 5. Poisoned Training Dataset

A poisoned training dataset wrapper was implemented.

Poisoning configuration:

```text
poison_ratio = 0.12
target_label = 0
target_label_name = apple
seed = 42
trigger_strength = 0.08
frequency = (6, 6)
```

Poisoning behavior:

```text
selected non-target image + frequency trigger -> label changed to apple
unselected image -> original image and original label
```

Target-class images are not selected for poisoning because they already have the
target label and do not create a useful backdoor training signal.

Verified poisoned dataset statistics:

```text
Total training images: 50000
Poisoned images: 6000
Poison ratio: 12%
Target label: apple
```

Poison metadata was saved at:

```text
experiments/poisoned_cifar100_ratio_0.12_seed_42/poison_metadata.json
```

Poisoned sample previews were saved at:

```text
experiments/poisoned_cifar100_ratio_0.12_seed_42/previews/
```

Example verified poisoning record:

```text
Original sample index: 0
Clean label: cattle
Poisoned label: apple
```

### 6. Triggered Test Dataset for ASR

A separate triggered test dataset was implemented for Attack Success Rate
evaluation.

This dataset is not used for training.

It is used only to evaluate whether a trained suspicious model predicts the
target class when the frequency trigger is present.

For targeted ASR evaluation:

```text
clean non-target test image + trigger -> target label apple
```

Target-class test images are excluded because predicting apple on an actual
apple image is not evidence of attack success.

Triggered test dataset verification:

```text
Original test images: 10000
Excluded target-class images: 100
Triggered ASR test images: 9900
Target label: apple
Image batch shape: (32, 3, 32, 32)
Image range: [0, 1]
```

## Why This Phase Matters

This phase creates the controlled backdoor condition required for the rest of
the research.

The clean dataset verifies normal classification data.

The poisoned training dataset creates a suspicious training condition where a
small percentage of samples contain a known frequency trigger and are relabeled
to the target class.

The triggered test dataset allows us to measure whether the trained model has
learned the trigger dependency.

This directly supports the research objective:

```text
Test whether selective spectral correction can weaken trigger-related model
dependencies while preserving clean classification accuracy.
```

## Important Scientific Caution

At this stage, we have not proven that the model learns the backdoor.

That can only be shown after training the suspicious classifier and measuring:

```text
Clean Accuracy
Attack Success Rate
```

Also, the current trigger is frequency-based but not invisible. This is
acceptable for the first controlled experiment. Later experiments should test
weaker trigger strengths such as:

```text
0.03
0.05
0.08
```

## Implemented Files

```text
datasets/cifar100_raw.py
datasets/cifar100_dataset.py
poisoning/frequency_trigger.py
fft/spectrum.py
visualization/image_grid.py
visualization/trigger_preview.py
scripts/inspect_cifar100.py
scripts/check_clean_dataloader.py
scripts/preview_frequency_trigger.py
scripts/check_poisoned_dataloader.py
scripts/verify_poisoned_dataset.py
scripts/check_triggered_test_dataloader.py
configs/poison_cifar100_frequency.yaml
```

## Next Phase

The next phase is suspicious model construction and training.

The model will be trained on:

```text
poisoned training dataset
```

Then evaluated on:

```text
clean test dataset -> Clean Accuracy
triggered test dataset -> Attack Success Rate
```

We should not move to generator-based correction until the model is confirmed to
be genuinely backdoored.
