# Kaggle Notebook: STL-10 96x96 FIBA-Style Amplitude Injection

This experiment evaluates the defense against a published frequency-domain
backdoor family. It implements a FIBA-style amplitude injection: a fixed
reference image amplitude is blended into a centered frequency mask while the
clean image phase is preserved.

This is a controlled reproduction-inspired benchmark, not a claim that this
repository is an official implementation of the original FIBA code.

## Fixed settings

- Dataset: STL-10, native `96 x 96` resolution.
- Target: `airplane` (`0`).
- Poison ratio: `0.12`.
- Amplitude mixing strength: `alpha=0.50` for the calibrated final run.
- Centered elliptical amplitude-mask radius: `0.10` of image dimensions.
- Reference image: first labelled STL-10 training image, fixed by seed and path.
- Classifier epochs: 30.
- Generator epochs: 30.
- Repair epochs: 5.
- Seed: 42.

Outputs are isolated under:

```text
  experiments/stl10_96x96_fiba_amplitude_calibrated/
  outputs/stl10_96x96_fiba_amplitude_calibrated/
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
  --experiment-root experiments/stl10_96x96_fiba_amplitude_smoke \
  --output-root outputs/stl10_96x96_fiba_amplitude_smoke \
  --classifier-epochs 1 \
  --generator-epochs 1 \
  --repair-epochs 1 \
  --batch-size 32 \
  --num-workers 2 \
  --poison-ratio 0.12 \
  --target-label 0 \
  --trigger-kind fiba_amplitude \
  --strength 0.15 \
  --fiba-mask-radius 0.10 \
  --num-panels 2 \
  --max-train-batches 2 \
  --max-eval-batches 2 \
  --device cuda
```

## 4. Full experiment

```python
!python -m scripts.run_stl10_96_frequency_experiment \
  --data-root {STL_PATH} \
  --experiment-root experiments/stl10_96x96_fiba_amplitude_calibrated \
  --output-root outputs/stl10_96x96_fiba_amplitude_calibrated \
  --classifier-epochs 30 \
  --generator-epochs 30 \
  --repair-epochs 5 \
  --batch-size 64 \
  --num-workers 2 \
  --poison-ratio 0.12 \
  --target-label 0 \
  --trigger-kind fiba_amplitude \
  --strength 0.50 \
  --fiba-mask-radius 0.10 \
  --num-panels 8 \
  --seed 42 \
  --device cuda
```

## 5. Amplitude-side formulation

For a clean image and a fixed reference trigger image:

```text
F(clean) = A_clean * exp(j*P_clean)
F(reference) = A_reference * exp(j*P_reference)
```

Inside the centered frequency mask M, the poisoned amplitude is:

```text
A_poisoned = (1-alpha)*A_clean + alpha*A_reference
```

Outside M, the clean amplitude is retained. The clean phase is preserved:

```text
P_poisoned = P_clean
```

The poisoned image is reconstructed with inverse FFT. This allows the defense
to be evaluated against a trigger that is explicitly injected into amplitude,
rather than a spatial sinusoid whose spectral effect is observed afterward.

## 6. Inspect and download

```python
import json
from pathlib import Path

summary_path = Path("outputs/stl10_96x96_fiba_amplitude_calibrated/final_stl10_96x96_summary.json")
summary = json.loads(summary_path.read_text())
print(json.dumps(summary["metrics"], indent=2))
```

```python
!zip -r stl10_96x96_fiba_amplitude_results.zip \
  experiments/stl10_96x96_fiba_amplitude_calibrated \
  outputs/stl10_96x96_fiba_amplitude_calibrated
```

The important result pattern is high suspicious ASR, low generator-corrected
ASR, very low repaired ASR, stable clean accuracy, and a small reconstruction
error.

## 7. Important visualization fix

The updated repository aligns the correction-map display with the centered
amplitude-spectrum displays. The generator still uses the original unshifted
FFT ordering internally; the correction map is shifted only when it is rendered
for a figure. This does not change training, corrected images, or metrics.

Use the updated repository code and regenerate the sample panels. Do not reuse
the old preliminary panel as the final FIBA figure because its correction map
was displayed in the unshifted layout.
