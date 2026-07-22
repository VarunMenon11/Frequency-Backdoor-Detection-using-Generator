# Kaggle Notebook: STL-10 96x96 Frequency-Backdoor Experiment

Use this notebook after pushing the repo to GitHub and cloning/opening it in
Kaggle. This experiment is fully separate from CIFAR-100:

- experiment outputs: `experiments/stl10_96x96/`
- visualization/report outputs: `outputs/stl10_96x96/`
- final downloadable archive: `stl10_96x96_results.zip`

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

## 3. Install/Check Packages

Kaggle usually already has `torch`, `torchvision`, `numpy`, `PIL`, and
`matplotlib`. Run this small import check:

```python
import torch
import torchvision
import numpy
import PIL

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
```

## 4. Find The STL-10 Dataset Path

If you connected a Kaggle STL-10 dataset, first inspect `/kaggle/input`:

```python
import os

for root, dirs, files in os.walk("/kaggle/input"):
    depth = root.replace("/kaggle/input", "").count(os.sep)
    if depth <= 3:
        print(root)
        for name in files[:5]:
            print("  ", name)
```

You are looking for a folder that contains `stl10_binary`, or files like:

- `train_X.bin`
- `train_y.bin`
- `test_X.bin`
- `test_y.bin`

Example paths might look like:

```text
/kaggle/input/stl10-dataset
/kaggle/input/stl10-dataset/stl10_binary
```

The script accepts either one.

## 5A. Full Run With Connected Kaggle Dataset

Replace `/kaggle/input/YOUR_STL10_DATASET_FOLDER` with the real path you found
in step 4.

```python
!python -m scripts.run_stl10_96_frequency_experiment \
  --data-root /kaggle/input/YOUR_STL10_DATASET_FOLDER \
  --experiment-root experiments/stl10_96x96 \
  --output-root outputs/stl10_96x96 \
  --classifier-epochs 30 \
  --generator-epochs 30 \
  --repair-epochs 5 \
  --batch-size 64 \
  --num-workers 2 \
  --poison-ratio 0.12 \
  --target-label 0 \
  --strength 0.08 \
  --horizontal-frequency 18 \
  --vertical-frequency 18 \
  --device cuda
```

## 5B. Alternative: Let Torchvision Download STL-10

Use this only if Kaggle internet is enabled.

```python
!python -m scripts.run_stl10_96_frequency_experiment \
  --data-root /kaggle/working/data \
  --download \
  --experiment-root experiments/stl10_96x96 \
  --output-root outputs/stl10_96x96 \
  --classifier-epochs 30 \
  --generator-epochs 30 \
  --repair-epochs 5 \
  --batch-size 64 \
  --num-workers 2 \
  --poison-ratio 0.12 \
  --target-label 0 \
  --strength 0.08 \
  --horizontal-frequency 18 \
  --vertical-frequency 18 \
  --device cuda
```

## 6. Optional Quick Smoke Test

Before the full run, you can run a tiny test to confirm the dataset path is
correct. This does not produce meaningful research results.

```python
!python -m scripts.run_stl10_96_frequency_experiment \
  --data-root /kaggle/input/YOUR_STL10_DATASET_FOLDER \
  --experiment-root experiments/stl10_96x96_smoke_test \
  --output-root outputs/stl10_96x96_smoke_test \
  --classifier-epochs 1 \
  --generator-epochs 1 \
  --repair-epochs 1 \
  --batch-size 16 \
  --num-workers 2 \
  --max-train-batches 2 \
  --max-eval-batches 2 \
  --poison-ratio 0.12 \
  --target-label 0 \
  --strength 0.08 \
  --horizontal-frequency 18 \
  --vertical-frequency 18 \
  --device cuda
```

## 7. Inspect Final Results

```python
import json

summary_path = "outputs/stl10_96x96/final_stl10_96x96_summary.json"
with open(summary_path, "r", encoding="utf-8") as file:
    summary = json.load(file)

print(json.dumps(summary["metrics"], indent=2))
```

The main report is saved at:

```text
outputs/stl10_96x96/final_stl10_96x96_report.md
```

Sample panels are saved at:

```text
outputs/stl10_96x96/sample_panels/
```

## 8. Zip Results For Download

```python
!zip -r stl10_96x96_results.zip experiments/stl10_96x96 outputs/stl10_96x96
```

Then download `stl10_96x96_results.zip` from the Kaggle file browser and bring
it back to the local project.

## Notes

For CIFAR-100, the trigger frequency was `(6, 6)` on `32 x 32` images. STL-10 is
`96 x 96`, which is three times larger. To keep the trigger in a similar
relative frequency region, this run uses `(18, 18)`.

This is still a controlled known-trigger experiment. It tests whether the same
frequency-domain defense idea scales to a higher-resolution dataset.
