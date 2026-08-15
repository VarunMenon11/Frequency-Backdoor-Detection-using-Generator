# Kaggle Notebook: FIBA-Style Attack Calibration

The previous FIBA-style run produced only `28.92%` suspicious ASR. That means
the attack was not sufficiently learned, so we calibrate the attack first and
do not spend GPU time on generator training until the suspicious classifier has
strong ASR.

The calibration changes only two attack parameters:

- amplitude mixing strength: `0.15`, `0.30`, `0.50`;
- centered amplitude-mask radius: `0.05`, `0.10`, `0.15`.

This is a classifier-only experiment. Each setting trains the suspicious model
for 20 epochs and records clean accuracy and ASR.

## 1. Enter repository and set dataset path

```python
%cd /kaggle/working/Frequency-Backdoor-Detection-using-Generator
STL_PATH = "/kaggle/input/YOUR_STL10_DATASET_FOLDER"
```

Confirm the updated script is available:

```python
!python -m scripts.calibrate_stl10_fiba --help
```

## 2. Run calibration grid

```python
strengths = ["0.15", "0.30", "0.50"]
radii = ["0.05", "0.10", "0.15"]

for strength in strengths:
    for radius in radii:
        tag = f"alpha_{strength.replace('.', '_')}_radius_{radius.replace('.', '_')}"
        print("=" * 90)
        print("Calibrating:", tag)
        print("=" * 90)

        !python -m scripts.calibrate_stl10_fiba \
          --data-root {STL_PATH} \
          --experiment-root experiments/fiba_calibration/{tag} \
          --output-root outputs/fiba_calibration/{tag} \
          --strength {strength} \
          --fiba-mask-radius {radius} \
          --classifier-epochs 20 \
          --batch-size 64 \
          --num-workers 2 \
          --poison-ratio 0.12 \
          --target-label 0 \
          --seed 42 \
          --device cuda
```

## 3. Print calibration table

```python
import json
from pathlib import Path

rows = []
for path in sorted(Path("outputs/fiba_calibration").glob("*/calibration_summary.json")):
    record = json.loads(path.read_text())
    metric = record["final_classifier_metrics"]
    rows.append((
        record["trigger"]["strength_alpha"],
        record["trigger"]["fiba_mask_radius"],
        metric["clean_accuracy"],
        metric["attack_success_rate"],
    ))

print("alpha | mask radius | clean accuracy | suspicious ASR")
for row in rows:
    print(f"{row[0]:.2f} | {row[1]:.2f} | {row[2]:.4f} | {row[3]:.4f}")
```

## 4. Select the FIBA setting

Select a setting using this rule:

```text
suspicious ASR >= 0.90
and clean accuracy remains acceptable
```

If several settings satisfy this rule, select the one with the smallest
strength and mask radius. That gives the most conservative strong attack.

Do not select a setting only because it gives the lowest post-defense ASR. The
attack must first be strong enough before the defense result is meaningful.

## 5. Run the full FIBA pipeline

Replace `SELECTED_ALPHA` and `SELECTED_RADIUS` with the selected values:

```python
SELECTED_ALPHA = "0.30"
SELECTED_RADIUS = "0.15"

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
  --strength {SELECTED_ALPHA} \
  --fiba-mask-radius {SELECTED_RADIUS} \
  --num-panels 8 \
  --seed 42 \
  --device cuda
```

## 6. Download both records

```python
!zip -r stl10_fiba_calibration_and_final_results.zip \
  experiments/fiba_calibration \
  outputs/fiba_calibration \
  experiments/stl10_96x96_fiba_amplitude_calibrated \
  outputs/stl10_96x96_fiba_amplitude_calibrated
```
