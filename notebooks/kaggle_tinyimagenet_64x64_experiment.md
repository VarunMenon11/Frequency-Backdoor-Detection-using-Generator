# Kaggle Notebook: Tiny ImageNet 64x64 Frequency-Backdoor Experiment

Use this after pushing the repo to GitHub and cloning/opening it in Kaggle.
This experiment is fully separate from CIFAR-100 and STL-10:

- experiment outputs: `experiments/tinyimagenet_64x64/`
- visualization/report outputs: `outputs/tinyimagenet_64x64/`
- final downloadable archive: `tinyimagenet_64x64_results.zip`

## 1. Check GPU

```python
import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("Torch version:", torch.__version__)
```

## 2. Go To Repo Folder

If you cloned the repo into `/kaggle/working/Frequency-Backdoor-Detection-using-Generator`,
run:

```python
%cd /kaggle/working/Frequency-Backdoor-Detection-using-Generator
```

If your repo folder has a different name, change the path.

## 3. Check Packages

Kaggle usually already has the needed packages.

```python
import torch
import torchvision
import numpy
import PIL

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
```

## 4. Find Tiny ImageNet Dataset Path

Connect a Tiny ImageNet dataset in Kaggle. The expected official structure is:

```text
tiny-imagenet-200/
  train/
  val/
    images/
    val_annotations.txt
  words.txt
  wnids.txt
```

Run this to inspect `/kaggle/input`:

```python
import os

for root, dirs, files in os.walk("/kaggle/input"):
    depth = root.replace("/kaggle/input", "").count(os.sep)
    if depth <= 3:
        print(root)
        for name in files[:5]:
            print("  ", name)
```

You can also search directly:

```python
import os

for root, dirs, files in os.walk("/kaggle/input"):
    if "val_annotations.txt" in files:
        print("Tiny ImageNet root is probably:", os.path.dirname(root))
        print("val folder:", root)
```

If it prints:

```text
Tiny ImageNet root is probably: /kaggle/input/tiny-imagenet/tiny-imagenet-200
```

then use either:

```text
--data-root /kaggle/input/tiny-imagenet
```

or:

```text
--data-root /kaggle/input/tiny-imagenet/tiny-imagenet-200
```

The script accepts both.

## 5. Optional Quick Smoke Test

Run this first to confirm the dataset path is correct. This is not a real
research result.

Replace `/kaggle/input/YOUR_TINYIMAGENET_FOLDER` with your real path.

```python
!python -m scripts.run_tinyimagenet_64_frequency_experiment \
  --data-root /kaggle/input/YOUR_TINYIMAGENET_FOLDER \
  --experiment-root experiments/tinyimagenet_64x64_smoke_test \
  --output-root outputs/tinyimagenet_64x64_smoke_test \
  --classifier-epochs 1 \
  --generator-epochs 1 \
  --repair-epochs 1 \
  --batch-size 32 \
  --num-workers 2 \
  --max-train-batches 2 \
  --max-eval-batches 2 \
  --poison-ratio 0.12 \
  --target-label 0 \
  --strength 0.08 \
  --horizontal-frequency 12 \
  --vertical-frequency 12 \
  --device cuda
```

## 6. Full Run

Tiny ImageNet is larger than STL-10, so this will take longer. Keep Kaggle GPU
enabled.

```python
!python -m scripts.run_tinyimagenet_64_frequency_experiment \
  --data-root /kaggle/input/YOUR_TINYIMAGENET_FOLDER \
  --experiment-root experiments/tinyimagenet_64x64 \
  --output-root outputs/tinyimagenet_64x64 \
  --classifier-epochs 30 \
  --generator-epochs 30 \
  --repair-epochs 5 \
  --batch-size 64 \
  --num-workers 2 \
  --poison-ratio 0.12 \
  --target-label 0 \
  --strength 0.08 \
  --horizontal-frequency 12 \
  --vertical-frequency 12 \
  --device cuda
```

## 7. Inspect Final Results

```python
import json

summary_path = "outputs/tinyimagenet_64x64/final_tinyimagenet_64x64_summary.json"
with open(summary_path, "r", encoding="utf-8") as file:
    summary = json.load(file)

print(json.dumps(summary["metrics"], indent=2))
```

The main report is saved at:

```text
outputs/tinyimagenet_64x64/final_tinyimagenet_64x64_report.md
```

Sample panels are saved at:

```text
outputs/tinyimagenet_64x64/sample_panels/
```

## 8. Zip Results For Download

```python
!zip -r tinyimagenet_64x64_results.zip experiments/tinyimagenet_64x64 outputs/tinyimagenet_64x64
```

Then download `tinyimagenet_64x64_results.zip` from the Kaggle file browser and
bring it back to the local project.

## Notes

For CIFAR-100, the trigger frequency was `(6, 6)` on `32 x 32` images. Tiny
ImageNet is `64 x 64`, which is two times larger. To keep the trigger in a
similar relative frequency region, this run uses `(12, 12)`.

This is still a controlled known-trigger experiment. It extends the resolution
sequence to:

| Dataset | Resolution | Trigger Frequency |
|---|---:|---:|
| CIFAR-100 | `32 x 32` | `(6, 6)` |
| Tiny ImageNet | `64 x 64` | `(12, 12)` |
| STL-10 | `96 x 96` | `(18, 18)` |
