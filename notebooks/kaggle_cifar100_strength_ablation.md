# Kaggle/Colab Notebook: CIFAR-100 Trigger Strength Ablation (Wide Range)

This experiment tests how the defense behaves when the sinusoidal trigger
strength changes. It keeps everything else fixed:

- dataset: CIFAR-100;
- resolution: `32 x 32`;
- poison ratio: `0.12`;
- target class: `apple`;
- trigger frequency: `(6, 6)`;
- trigger type: sinusoidal frequency trigger.

Only this value changes:

```text
alpha = 0.03, 0.15, 0.25
```

The results are saved separately from the main CIFAR experiment:

```text
experiments/ablation_cifar100_strength_wide/
outputs/ablation_cifar100_strength_wide/
```

The model still receives CIFAR-100 at its native `32 x 32` resolution. A
larger display makes the saved panels easier to inspect, but upscaling does
not create new image information or change the experiment's resolution.

## 1. Check GPU

```python
import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("Torch version:", torch.__version__)
```

## 2. Go To Repo Folder

Kaggle example:

```python
%cd /kaggle/working/Frequency-Backdoor-Detection-using-Generator
```

Colab example:

```python
%cd /content/Frequency-Backdoor-Detection-using-Generator
```

## 3. Prepare CIFAR-100 Path

If using Kaggle connected dataset, inspect `/kaggle/input`:

```python
import os

for root, dirs, files in os.walk("/kaggle/input"):
    depth = root.replace("/kaggle/input", "").count(os.sep)
    if depth <= 3:
        print(root)
        for name in files[:8]:
            print("  ", name)
```

Use either the CIFAR-100 zip path or extracted folder path.

Examples:

```text
/kaggle/input/cifar100/archive.zip
/kaggle/input/cifar100/archive
/content/cifar-100-python
```

In the commands below, replace:

```text
YOUR_CIFAR100_PATH
```

with your real path.

## 4. Optional Smoke Test For One Strength

Run this first to confirm paths and imports are correct. This is not a real
research result.

```python
!python -m scripts.train_suspicious_classifier \
  --zip-path YOUR_CIFAR100_PATH \
  --output-dir experiments/ablation_cifar100_strength_wide_smoke/alpha_0_03/suspicious_classifier \
  --epochs 1 \
  --batch-size 64 \
  --num-workers 2 \
  --poison-ratio 0.12 \
  --target-label 0 \
  --strength 0.03 \
  --horizontal-frequency 6 \
  --vertical-frequency 6 \
  --max-train-batches 2 \
  --max-eval-batches 2 \
  --device cuda
```

## 5. Full Ablation Run

Run the following block. It trains and evaluates three full pipelines:

- weak trigger: `alpha=0.03`;
- strong trigger: `alpha=0.15`;
- very strong trigger: `alpha=0.25`.

```python
CIFAR_PATH = "YOUR_CIFAR100_PATH"

strengths = [
    ("0_03", "0.03"),
    ("0_15", "0.15"),
    ("0_25", "0.25"),
]

for tag, strength in strengths:
    print("=" * 80)
    print("Running CIFAR-100 wide strength ablation:", strength)
    print("=" * 80)

    !python -m scripts.train_suspicious_classifier \
      --zip-path {CIFAR_PATH} \
      --output-dir experiments/ablation_cifar100_strength_wide/alpha_{tag}/suspicious_classifier \
      --epochs 10 \
      --batch-size 128 \
      --num-workers 2 \
      --poison-ratio 0.12 \
      --target-label 0 \
      --strength {strength} \
      --horizontal-frequency 6 \
      --vertical-frequency 6 \
      --device cuda

    !python -m scripts.train_spectral_generator \
      --zip-path {CIFAR_PATH} \
      --classifier-checkpoint experiments/ablation_cifar100_strength_wide/alpha_{tag}/suspicious_classifier/suspicious_classifier.pt \
      --output-dir experiments/ablation_cifar100_strength_wide/alpha_{tag}/spectral_generator \
      --epochs 20 \
      --batch-size 128 \
      --num-workers 2 \
      --target-label 0 \
      --strength {strength} \
      --horizontal-frequency 6 \
      --vertical-frequency 6 \
      --device cuda

    !python -m scripts.repair_suspicious_classifier \
      --zip-path {CIFAR_PATH} \
      --classifier-checkpoint experiments/ablation_cifar100_strength_wide/alpha_{tag}/suspicious_classifier/suspicious_classifier.pt \
      --generator-checkpoint experiments/ablation_cifar100_strength_wide/alpha_{tag}/spectral_generator/spectral_generator.pt \
      --output-dir experiments/ablation_cifar100_strength_wide/alpha_{tag}/repaired_classifier \
      --epochs 3 \
      --batch-size 128 \
      --num-workers 2 \
      --target-label 0 \
      --strength {strength} \
      --horizontal-frequency 6 \
      --vertical-frequency 6 \
      --include-raw-triggered \
      --device cuda

    !python -m scripts.final_cifar100_evaluation \
      --zip-path {CIFAR_PATH} \
      --suspicious-checkpoint experiments/ablation_cifar100_strength_wide/alpha_{tag}/suspicious_classifier/suspicious_classifier.pt \
      --generator-checkpoint experiments/ablation_cifar100_strength_wide/alpha_{tag}/spectral_generator/spectral_generator.pt \
      --repaired-checkpoint experiments/ablation_cifar100_strength_wide/alpha_{tag}/repaired_classifier/repaired_classifier.pt \
      --output-dir outputs/ablation_cifar100_strength_wide/alpha_{tag} \
      --batch-size 256 \
      --num-workers 2 \
      --target-label 0 \
      --strength {strength} \
      --horizontal-frequency 6 \
      --vertical-frequency 6 \
      --device cuda
```

## 6. Summarize The Ablation

```python
!python -m scripts.summarize_cifar100_strength_ablation \
  --ablation-output-root outputs/ablation_cifar100_strength_wide \
  --output-json outputs/ablation_cifar100_strength_wide/strength_ablation_summary.json \
  --output-md outputs/ablation_cifar100_strength_wide/strength_ablation_summary.md
```

## 7. Print The Final Table

```python
from pathlib import Path

print(Path("outputs/ablation_cifar100_strength_wide/strength_ablation_summary.md").read_text())
```

## 8. Zip Results For Download

```python
!zip -r cifar100_strength_ablation_wide_results.zip \
  experiments/ablation_cifar100_strength_wide \
  outputs/ablation_cifar100_strength_wide
```

Download the zip file and bring it back to the local project.

## 9. What To Look For

The main expected pattern is:

```text
Suspicious ASR should be high.
Generator-corrected ASR should be much lower.
Repaired ASR should be very low.
```

Possible interpretations:

- If `alpha=0.03` has low suspicious ASR, the trigger may be too weak for the
  classifier to learn reliably.
- If `alpha=0.25` has high ASR but the generator and repair still
  reduce it, the defense is robust to a stronger visible trigger.
- If clean accuracy drops at high alpha, it means the stronger trigger creates
  a more destructive distribution shift.
