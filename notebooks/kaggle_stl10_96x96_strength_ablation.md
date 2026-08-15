# Kaggle Notebook: STL-10 96x96 Trigger-Strength Ablation

This is the paper-quality higher-resolution strength experiment. It runs the
complete suspicious-classifier, spectral-generator, classifier-repair, and
final-evaluation pipeline three times while changing only the trigger
strength.

Fixed settings:

- Dataset: STL-10, native `96 x 96` resolution.
- Target class: `airplane` (`0`).
- Poison ratio: `0.12`.
- Trigger frequency: `(18, 18)`.
- Trigger type: sinusoidal frequency trigger.
- Strengths: `0.03`, `0.15`, and `0.25`.
- Seed: `42`.

The new results are isolated from the earlier STL run:

```text
experiments/ablation_stl10_strength_wide/
outputs/ablation_stl10_strength_wide/
```

## 1. Check GPU and enter the repository

```python
import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

```python
%cd /kaggle/working/Frequency-Backdoor-Detection-using-Generator
```

## 2. Find the STL-10 path

```python
import os

for root, dirs, files in os.walk("/kaggle/input"):
    depth = root.replace("/kaggle/input", "").count(os.sep)
    if depth <= 3:
        print(root)
        for name in files[:6]:
            print("  ", name)
```

Set the path below to the folder containing `stl10_binary`, or directly to
the `stl10_binary` folder:

```python
STL_PATH = "/kaggle/input/YOUR_STL10_DATASET_FOLDER"
```

## 3. Optional smoke test

Run this once before the long jobs. It checks the path and imports only; its
metrics must not be used as research results.

```python
!python -m scripts.run_stl10_96_frequency_experiment \
  --data-root {STL_PATH} \
  --experiment-root experiments/ablation_stl10_strength_wide_smoke \
  --output-root outputs/ablation_stl10_strength_wide_smoke \
  --classifier-epochs 1 \
  --generator-epochs 1 \
  --repair-epochs 1 \
  --batch-size 32 \
  --num-workers 2 \
  --poison-ratio 0.12 \
  --target-label 0 \
  --strength 0.03 \
  --horizontal-frequency 18 \
  --vertical-frequency 18 \
  --num-panels 2 \
  --max-train-batches 2 \
  --max-eval-batches 2 \
  --device cuda
```

## 4. Run the three full experiments

```python
strengths = [
    ("0_03", "0.03"),
    ("0_15", "0.15"),
    ("0_25", "0.25"),
]

for tag, strength in strengths:
    print("=" * 90)
    print("Running STL-10 strength:", strength)
    print("=" * 90)

    !python -m scripts.run_stl10_96_frequency_experiment \
      --data-root {STL_PATH} \
      --experiment-root experiments/ablation_stl10_strength_wide/alpha_{tag} \
      --output-root outputs/ablation_stl10_strength_wide/alpha_{tag} \
      --classifier-epochs 30 \
      --generator-epochs 30 \
      --repair-epochs 5 \
      --batch-size 64 \
      --num-workers 2 \
      --poison-ratio 0.12 \
      --target-label 0 \
      --strength {strength} \
      --horizontal-frequency 18 \
      --vertical-frequency 18 \
      --num-panels 8 \
      --seed 42 \
      --device cuda
```

## 5. Inspect the figures

Each run saves panels here:

```text
outputs/ablation_stl10_strength_wide/alpha_0_03/sample_panels/
outputs/ablation_stl10_strength_wide/alpha_0_15/sample_panels/
outputs/ablation_stl10_strength_wide/alpha_0_25/sample_panels/
```

The panels now include:

- clean image;
- triggered image;
- corrected image;
- prediction labels from suspicious and repaired models;
- trigger residual amplified for visibility;
- trigger amplitude spectrum;
- corrected amplitude spectrum;
- direct amplitude-difference heatmap;
- generator correction map.

Use the full summary metrics, not a single panel, to select the best setting.

## 6. Download the results

Before downloading, create the aggregate table:

```python
!python -m scripts.summarize_stl10_strength_ablation \
  --root outputs/ablation_stl10_strength_wide
```

This creates `stl10_strength_ablation_summary.md` and `.json` inside the
output folder.

```python
from pathlib import Path
print(Path("outputs/ablation_stl10_strength_wide/stl10_strength_ablation_summary.md").read_text())
```

```python
!zip -r stl10_96x96_strength_wide_results.zip \
  experiments/ablation_stl10_strength_wide \
  outputs/ablation_stl10_strength_wide
```

Download `stl10_96x96_strength_wide_results.zip` from the Kaggle file browser.

## Interpretation

The desired pattern is high suspicious ASR, followed by low generator-corrected
ASR and very low repaired ASR. Clean accuracy and the reconstruction error must
also be reported. A stronger trigger may be easier for the suspicious model to
learn but may cause a larger visible residual; that trade-off is part of the
ablation and is not a reason to hide an adverse result.
