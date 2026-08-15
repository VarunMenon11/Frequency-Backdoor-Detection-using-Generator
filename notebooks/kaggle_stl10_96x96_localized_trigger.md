# Kaggle Notebook: STL-10 96x96 Localized Frequency Trigger

This experiment introduces the first advanced trigger recommended for the
project: a localized sinusoidal frequency trigger. Unlike the earlier global
sinusoid, the pattern is multiplied by a Gaussian window and is active mainly
in one image region.

This tests whether the generator can correct a trigger that is both spatially
localized and frequency-structured.

## Fixed settings

- Dataset: STL-10, native `96 x 96` resolution.
- Target: `airplane` (`0`).
- Poison ratio: `0.12`.
- Strength: `alpha=0.15`.
- Sinusoidal frequency: `(18,18)`.
- Gaussian window centre: `(0.65, 0.50)` in normalized image coordinates.
- Gaussian sigma: `0.18` of the image dimension.
- Seed: `42`.

The centre `(0.65, 0.50)` places the trigger toward the right-centre of the
image. The sigma controls the spatial spread. These values are fixed so the
experiment is reproducible.

Outputs are isolated under:

```text
experiments/stl10_96x96_localized_trigger/
outputs/stl10_96x96_localized_trigger/
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

```python
STL_PATH = "/kaggle/input/YOUR_STL10_DATASET_FOLDER"
```

## 3. Optional smoke test

```python
!python -m scripts.run_stl10_96_frequency_experiment \
  --data-root {STL_PATH} \
  --experiment-root experiments/stl10_96x96_localized_trigger_smoke \
  --output-root outputs/stl10_96x96_localized_trigger_smoke \
  --classifier-epochs 1 \
  --generator-epochs 1 \
  --repair-epochs 1 \
  --batch-size 32 \
  --num-workers 2 \
  --poison-ratio 0.12 \
  --target-label 0 \
  --trigger-kind localized_cosine \
  --strength 0.15 \
  --horizontal-frequency 18 \
  --vertical-frequency 18 \
  --window-center-x 0.65 \
  --window-center-y 0.50 \
  --window-sigma 0.18 \
  --num-panels 2 \
  --max-train-batches 2 \
  --max-eval-batches 2 \
  --device cuda
```

## 4. Full run

```python
!python -m scripts.run_stl10_96_frequency_experiment \
  --data-root {STL_PATH} \
  --experiment-root experiments/stl10_96x96_localized_trigger \
  --output-root outputs/stl10_96x96_localized_trigger \
  --classifier-epochs 30 \
  --generator-epochs 30 \
  --repair-epochs 5 \
  --batch-size 64 \
  --num-workers 2 \
  --poison-ratio 0.12 \
  --target-label 0 \
  --trigger-kind localized_cosine \
  --strength 0.15 \
  --horizontal-frequency 18 \
  --vertical-frequency 18 \
  --window-center-x 0.65 \
  --window-center-y 0.50 \
  --window-sigma 0.18 \
  --num-panels 8 \
  --seed 42 \
  --device cuda
```

## 5. What to inspect

The saved panels contain clean, triggered, corrected, amplitude, amplitude
difference, correction-map, and prediction views.

Pay particular attention to:

- `trigger diff x8`: the trigger should appear mainly near the Gaussian window;
- `amplitude diff`: the spectral change should be more spread than a global
  sinusoid because spatial localization broadens the frequency response;
- `correction map`: the generator should respond to the localized trigger while
  avoiding unnecessary changes to the entire spectrum;
- clean and corrected RGB images: semantic content should remain recognizable;
- ASR: the suspicious model should learn the trigger, while generator correction
  and repair should reduce the target-class response.

## 6. Download results

```python
!zip -r stl10_96x96_localized_trigger_results.zip \
  experiments/stl10_96x96_localized_trigger \
  outputs/stl10_96x96_localized_trigger
```

## Interpretation

This is a known-trigger adaptation experiment. The generator and repaired model
are trained using the localized trigger and then evaluated on the same trigger.
It tests whether the proposed architecture can handle a more difficult trigger
structure than a global sinusoid. A later experiment should train with one
trigger family and evaluate on an unseen family to measure generalization.
