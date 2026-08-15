# Kaggle Notebook: STL-10 96x96 Trigger-Function Ablation

This experiment tests whether the defense works for different frequency-trigger
functions. Only the trigger function changes. The dataset, image resolution,
target class, poison ratio, strength, frequency scale, model, training schedule,
and seed remain fixed.

Fixed settings:

- Dataset: STL-10, native `96 x 96` resolution.
- Target: `airplane` (`0`).
- Poison ratio: `0.12`.
- Trigger strength: `alpha=0.15`.
- Primary frequency: `(18,18)`.
- Secondary frequency for the dual trigger: `(30,6)`.
- Functions: `cosine`, `sine`, `checkerboard`, `dual_frequency`.

Results are isolated under:

```text
experiments/ablation_stl10_trigger_function/
outputs/ablation_stl10_trigger_function/
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
  --experiment-root experiments/ablation_stl10_trigger_function_smoke/cosine \
  --output-root outputs/ablation_stl10_trigger_function_smoke/cosine \
  --classifier-epochs 1 \
  --generator-epochs 1 \
  --repair-epochs 1 \
  --batch-size 32 \
  --num-workers 2 \
  --poison-ratio 0.12 \
  --target-label 0 \
  --trigger-kind cosine \
  --strength 0.15 \
  --horizontal-frequency 18 \
  --vertical-frequency 18 \
  --num-panels 2 \
  --max-train-batches 2 \
  --max-eval-batches 2 \
  --device cuda
```

## 4. Run the four full experiments

```python
trigger_kinds = [
    ("cosine", "cosine"),
    ("sine", "sine"),
    ("checkerboard", "checkerboard"),
    ("dual_frequency", "dual_frequency"),
]

for tag, kind in trigger_kinds:
    print("=" * 90)
    print("Running STL-10 trigger function:", kind)
    print("=" * 90)

    !python -m scripts.run_stl10_96_frequency_experiment \
      --data-root {STL_PATH} \
      --experiment-root experiments/ablation_stl10_trigger_function/{tag} \
      --output-root outputs/ablation_stl10_trigger_function/{tag} \
      --classifier-epochs 30 \
      --generator-epochs 30 \
      --repair-epochs 5 \
      --batch-size 64 \
      --num-workers 2 \
      --poison-ratio 0.12 \
      --target-label 0 \
      --trigger-kind {kind} \
      --strength 0.15 \
      --horizontal-frequency 18 \
      --vertical-frequency 18 \
      --secondary-horizontal-frequency 30 \
      --secondary-vertical-frequency 6 \
      --num-panels 8 \
      --seed 42 \
      --device cuda
```

## 5. Function definitions

- `cosine`: baseline cosine sinusoid at `(18,18)`.
- `sine`: phase-shifted sinusoid at the same frequency.
- `checkerboard`: sign of the cosine pattern, creating a square-wave-like
  trigger with harmonic components.
- `dual_frequency`: average of two normalized cosine patterns at `(18,18)` and
  `(30,6)`, producing two primary spectral components.

The strength is fixed at `0.15`, so differences in results can be attributed to
the trigger function rather than to the perturbation magnitude.

## 6. Inspect and download results

Every function has its own evaluation JSON, report, checkpoints, and image panels.

```python
import json
from pathlib import Path

for kind in ["cosine", "sine", "checkerboard", "dual_frequency"]:
    path = Path(f"outputs/ablation_stl10_trigger_function/{kind}/final_stl10_96x96_summary.json")
    summary = json.loads(path.read_text())
    metrics = summary["metrics"]
    print(
        kind,
        "suspicious ASR", metrics["suspicious_classifier"]["attack_success_rate"],
        "generator ASR", metrics["generator_corrected_with_suspicious_classifier"]["after_asr"],
        "repaired ASR", metrics["repaired_classifier"]["attack_success_rate"],
    )
```

```python
!zip -r stl10_96x96_trigger_function_results.zip \
  experiments/ablation_stl10_trigger_function \
  outputs/ablation_stl10_trigger_function
```

## Expected interpretation

The desired pattern is high suspicious ASR, low generator-corrected ASR, very
low repaired ASR, and stable clean accuracy. If one function is more difficult,
that is a useful limitation/result: the method may be sensitive to trigger
structure rather than universally solving every possible backdoor.
