# Kaggle Notebook Plan: Generator Training

Use this as the structure for your Kaggle notebook after pushing the repository
to GitHub.

## 1. Enable GPU

In Kaggle:

```text
Notebook settings -> Accelerator -> GPU T4 x2 or GPU P100
```

## 2. Clone Repository

Replace the URL with your GitHub repository URL.

```python
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
%cd YOUR_REPO_NAME
```

## 3. Check GPU and PyTorch

```python
import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("Torch version:", torch.__version__)
```

## 4. Add Required Input Files

You need these files available inside the Kaggle working directory:

```text
datasets/archive.zip
experiments/suspicious_classifier_trained/suspicious_classifier.pt
```

Recommended Kaggle method:

1. Upload `archive.zip` as a Kaggle Dataset.
2. Upload `suspicious_classifier.pt` as a Kaggle Dataset, or commit a small
   download instruction if it is stored elsewhere.
3. Copy them into the repository paths.

Example:

```python
from pathlib import Path
import shutil

Path("datasets").mkdir(exist_ok=True)
Path("experiments/suspicious_classifier_trained").mkdir(parents=True, exist_ok=True)

# Change these paths to match your Kaggle input dataset names.
shutil.copy("/kaggle/input/cifar100-archive/archive.zip", "datasets/archive.zip")
shutil.copy(
    "/kaggle/input/suspicious-classifier/suspicious_classifier.pt",
    "experiments/suspicious_classifier_trained/suspicious_classifier.pt",
)
```

## 5. Smoke Test Generator Training

Run a tiny test first. This checks paths, imports, checkpoint loading, FFT, and
backpropagation.

```python
!python -m scripts.train_spectral_generator \
  --zip-path datasets/archive.zip \
  --classifier-checkpoint experiments/suspicious_classifier_trained/suspicious_classifier.pt \
  --output-dir experiments/spectral_generator_smoke_test \
  --epochs 1 \
  --batch-size 64 \
  --num-workers 2 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cuda
```

Expected output should include:

```text
before ASR ...
after ASR ...
corrected acc ...
Saved generator checkpoint ...
```

The smoke-test result is not scientifically meaningful.

## 6. Full Generator Training

After the smoke test works, run:

```python
!python -m scripts.train_spectral_generator \
  --zip-path datasets/archive.zip \
  --classifier-checkpoint experiments/suspicious_classifier_trained/suspicious_classifier.pt \
  --output-dir experiments/spectral_generator_cifar100 \
  --epochs 20 \
  --batch-size 128 \
  --num-workers 2 \
  --device cuda
```

If Kaggle memory is limited, reduce batch size:

```python
--batch-size 64
```

## 7. Inspect Results

```python
import json
from pathlib import Path

summary_path = Path("experiments/spectral_generator_cifar100/generator_training_summary.json")
summary = json.loads(summary_path.read_text())
summary["history"][-1]
```

Important values:

```text
before_asr
after_asr
corrected_clean_label_accuracy
eval_reconstruction_l1
mean_correction_value
```

Good signs:

```text
after_asr < before_asr
corrected_clean_label_accuracy remains reasonable
mean_correction_value is not huge
```

Bad signs:

```text
after_asr stays near 1.0
corrected_clean_label_accuracy collapses
mean_correction_value becomes very large
```

## 8. Download Trained Generator

Download these files from Kaggle:

```text
experiments/spectral_generator_cifar100/spectral_generator.pt
experiments/spectral_generator_cifar100/generator_training_summary.json
```

Bring them back into the same paths locally:

```text
D:\Research_Paper\experiments\spectral_generator_cifar100\spectral_generator.pt
D:\Research_Paper\experiments\spectral_generator_cifar100\generator_training_summary.json
```

Then we can continue locally with:

```text
Phase 7: Spectral Correction Visualization
Phase 9: Model Repair
```

## Scientific Reminder

Do not claim the generator has found trigger frequencies just because ASR drops.

We must later inspect correction maps and compare them against known trigger
frequency locations.
