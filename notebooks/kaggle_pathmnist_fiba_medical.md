# Kaggle: FIBA-Style Backdoor Defense on PathMNIST

This notebook runs the project pipeline on **PathMNIST**, a standardized biomedical image benchmark containing colon pathology images. PathMNIST is used here as a research benchmark, not for clinical diagnosis or medical decision-making.

The official MedMNIST documentation describes PathMNIST as a 9-class pathology dataset with standard train, validation, and test splits. The standard images are 28x28 RGB images. See the official sources:

- [MedMNIST official repository](https://github.com/MedMNIST/MedMNIST)
- [MedMNIST dataset overview](https://medmnist.com/v1)

## 1. What this experiment tests

The experiment asks whether the same selective amplitude-correction idea works on biomedical images:

```text
PathMNIST clean images
        |
        v
FIBA-style amplitude injection
        |
        v
Suspicious pathology-image classifier
        |
        v
FFT amplitude correction generator
        |
        v
Repaired classifier
```

The medical experiment is stored separately from the existing datasets:

```text
experiments/pathmnist_fiba/
outputs/pathmnist_fiba/
```

## 2. Kaggle setup

Create a Kaggle notebook with GPU enabled. The repository code downloads PathMNIST through the official `medmnist` package, so enable Kaggle internet for the first run.

```python
!pip install -q -U medmnist
```

Clone the updated repository:

```python
!git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git /kaggle/working/Frequency-Backdoor-Detection-using-Generator
%cd /kaggle/working/Frequency-Backdoor-Detection-using-Generator
```

Confirm the new scripts are present:

```python
!python -m scripts.run_pathmnist_fiba_experiment --help
!python -m scripts.calibrate_pathmnist_fiba --help
```

Use a writable location for the MedMNIST download:

```python
DATA_ROOT = "/kaggle/working/medmnist"
```

## 3. Smoke test

Run this first to verify that the package, dataset download, tensor shape, trigger, and GPU work correctly.

```python
!python -m scripts.run_pathmnist_fiba_experiment \
  --data-root {DATA_ROOT} \
  --download \
  --experiment-root experiments/pathmnist_fiba_smoke \
  --output-root outputs/pathmnist_fiba_smoke \
  --classifier-epochs 1 \
  --generator-epochs 1 \
  --repair-epochs 1 \
  --batch-size 64 \
  --num-workers 2 \
  --poison-ratio 0.12 \
  --target-label 0 \
  --strength 0.50 \
  --fiba-mask-radius 0.10 \
  --num-panels 2 \
  --max-train-batches 2 \
  --max-eval-batches 2 \
  --device cuda
```

Expected startup information should resemble:

```text
Dataset: PathMNIST 28x28
Classes: 9
Target: 0
Trigger: fiba_amplitude alpha 0.5 mask radius 0.1
```

## 4. Medical FIBA calibration

The alpha and mask radius selected on STL-10 should not automatically be called optimal for PathMNIST. First screen a small medical-specific grid using the suspicious classifier only.

```python
strengths = ["0.30", "0.50"]
radii = ["0.05", "0.10"]

for strength in strengths:
    for radius in radii:
        tag = f"alpha_{strength.replace('.', '_')}_radius_{radius.replace('.', '_')}"
        print("Calibrating", tag)
        !python -m scripts.calibrate_pathmnist_fiba \
          --data-root {DATA_ROOT} \
          --download \
          --experiment-root experiments/pathmnist_fiba_calibration/{tag} \
          --output-root outputs/pathmnist_fiba_calibration/{tag} \
          --strength {strength} \
          --fiba-mask-radius {radius} \
          --classifier-epochs 10 \
          --batch-size 128 \
          --num-workers 2 \
          --poison-ratio 0.12 \
          --target-label 0 \
          --seed 42 \
          --device cuda
```

Print the table:

```python
import json
from pathlib import Path

rows = []
for path in sorted(Path("outputs/pathmnist_fiba_calibration").glob("*/calibration_summary.json")):
    record = json.loads(path.read_text())
    metric = record["final_classifier_metrics"]
    rows.append((
        record["trigger"]["strength_alpha"],
        record["trigger"]["fiba_mask_radius"],
        metric["clean_accuracy"],
        metric["attack_success_rate"],
    ))

print("alpha | radius | clean accuracy | suspicious ASR")
for row in rows:
    print(f"{row[0]:.2f} | {row[1]:.2f} | {row[2]:.4f} | {row[3]:.4f}")
```

Select a setting only after checking that the suspicious classifier has learned a strong attack. A practical screening rule is suspicious ASR near or above 90% with usable clean accuracy. If no setting reaches that level, report the medical attack as weak or partially learned instead of claiming a strong defense result.

## 5. Full PathMNIST FIBA run

Begin with the selected medical calibration values. Until the calibration table says otherwise, use the previously selected balanced values:

```python
SELECTED_ALPHA = "0.50"
SELECTED_RADIUS = "0.10"
```

Run:

```python
!python -m scripts.run_pathmnist_fiba_experiment \
  --data-root {DATA_ROOT} \
  --experiment-root experiments/pathmnist_fiba \
  --output-root outputs/pathmnist_fiba \
  --classifier-epochs 30 \
  --generator-epochs 30 \
  --repair-epochs 5 \
  --batch-size 128 \
  --num-workers 2 \
  --poison-ratio 0.12 \
  --target-label 0 \
  --strength {SELECTED_ALPHA} \
  --fiba-mask-radius {SELECTED_RADIUS} \
  --num-panels 8 \
  --seed 42 \
  --device cuda
```

## 6. What to inspect

```python
import json
from pathlib import Path

summary_path = Path("outputs/pathmnist_fiba/final_pathmnist_28x28_summary.json")
summary = json.loads(summary_path.read_text())
print(json.dumps(summary["metrics"], indent=2))
print(json.dumps(summary["trigger"], indent=2))
```

Check the following order:

1. suspicious classifier clean accuracy is usable;
2. suspicious classifier ASR is high enough to establish the attack;
3. generator-corrected ASR decreases substantially;
4. repaired raw-trigger ASR is very low;
5. clean accuracy does not collapse;
6. reconstruction L1 remains small;
7. sample panels show preserved pathology-image structure;
8. the correction map is displayed in the same centered frequency layout as the amplitude-difference panel.

## 7. Download the complete medical record

```python
!zip -r pathmnist_fiba_medical_results.zip \
  experiments/pathmnist_fiba_calibration \
  outputs/pathmnist_fiba_calibration \
  experiments/pathmnist_fiba \
  outputs/pathmnist_fiba
```

Download `pathmnist_fiba_medical_results.zip` and extract it into the local project. Keep it separate from STL-10 FIBA results.

## 8. How to report the result

Use wording such as:

> A FIBA-style amplitude-injection experiment was conducted on PathMNIST, a standardized biomedical pathology-image benchmark. The suspicious classifier was first evaluated to verify that the amplitude trigger was learned. The proposed generator then produced selective amplitude corrections, and the classifier was repaired using clean, corrected, and raw-triggered images. Results are reported as a research benchmark and do not imply clinical diagnostic performance.

Do not describe this benchmark as a medical diagnostic system. MedMNIST is intended for research and educational benchmarking, not clinical use.
