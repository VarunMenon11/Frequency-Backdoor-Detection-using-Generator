# Kaggle: DTD Multi-Position FTrojan Attack Calibration

This is the next experiment after the frozen-generator generalization study.
It does **not** train the next generator yet. It first creates a scientifically
usable suspicious classifier for the unknown-position experiment.

The current suspicious classifier was trained at only `(15,15)` and `(31,31)`.
Most shifted triggers therefore failed to activate its backdoor. The new
classifier is deliberately poisoned with four DCT-position patterns so that
every position can become an active attack.

Later, the generator will see only two of those active positions during its
training. The other two will remain hidden from the generator.

## 1. Experimental split

The four attacker-training patterns are:

| Name | DCT coefficient positions | Later generator role |
|---|---|---|
| `ftpos_original` | `(15,15)`, `(31,31)` | Generator training |
| `ftpos_near_minus1` | `(14,14)`, `(30,30)` | Generator training |
| `ftpos_near_mixed` | `(14,16)`, `(30,28)` | Held out from generator |
| `ftpos_mid_shift` | `(12,18)`, `(28,30)` | Held out from generator |

All four are known to the attacker model. The final two will be unknown only to
the future defense generator. This distinction makes the held-out experiment
valid while ensuring the attack itself is active.

## 2. Candidates evaluated

The resumable sweep attempts these candidates in order:

1. Strength 100, dynamic-paired poison ratio 0.30
2. Strength 150, dynamic-paired poison ratio 0.30
3. Strength 150, dynamic-paired poison ratio 0.40

Dynamic-paired poisoning keeps all clean training rows and rotates poisoned
source assignments each epoch. The total poison budget is divided evenly among
the four positions.

A candidate qualifies only when:

- Validation clean accuracy is at least 55%.
- Every individual position has ASR at least 60%.
- Every individual position has conditional ASR at least 50%.
- Every position has positive target-rate lift over clean predictions.

A high average cannot hide one inactive position.

## 3. Select a T4 GPU and enter the repository

Do not reinstall PyTorch or torchvision.

```python
%cd /kaggle/working/Frequency-Backdoor-Detection-using-Generator
```

```python
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
assert torch.cuda.is_available()
```

## 4. Verify required files

```python
from pathlib import Path

required = [
    Path("Absolute_Dataset/asb_dtd_v1/variant_manifest.jsonl"),
    Path("Absolute_Dataset/asb_dtd_v1/benchmark_summary.json"),
    Path("Absolute_Dataset/dtd/images"),
    Path("configs/dtd_ftrojan_position_catalog.json"),
    Path("scripts/train_asb_dtd_pretrained_classifier.py"),
    Path("scripts/run_dtd_position_attack_calibration.py"),
]
for path in required:
    print(path, "OK" if path.exists() else "MISSING")
assert all(path.exists() for path in required)
```

## 5. Fast software smoke test

This trains for one epoch and only two batches. It checks catalog loading,
four-position poisoning, forward/backward execution, per-position evaluation,
and checkpoint writing. Its numbers are not research results.

```python
!python -m scripts.train_asb_dtd_pretrained_classifier \
  --manifest-dir Absolute_Dataset/asb_dtd_v1 \
  --images-root Absolute_Dataset/dtd/images \
  --trigger-catalog configs/dtd_ftrojan_position_catalog.json \
  --experiment-dir Advanced_Experiments/dtd_position_attack_smoke \
  --output-dir Advanced_Outputs/dtd_position_attack_smoke \
  --weights default \
  --attack-triggers ftpos_original,ftpos_near_minus1,ftpos_near_mixed,ftpos_mid_shift \
  --trigger-strength 100 \
  --poison-mode dynamic-paired \
  --poison-ratio 0.30 \
  --epochs 1 \
  --freeze-epochs 1 \
  --batch-size 8 \
  --num-workers 2 \
  --max-train-batches 2 \
  --max-eval-batches 2 \
  --attack-checkpoint-min-clean-accuracy 0.0 \
  --validation-only \
  --device cuda \
  --overwrite
```

Confirm that the output prints four separate trigger names.

## 6. Run the full resumable sweep

```python
!python -u -m scripts.run_dtd_position_attack_calibration \
  --epochs 30 \
  --freeze-epochs 2 \
  --batch-size 32 \
  --num-workers 2 \
  --minimum-clean-accuracy 0.55 \
  --minimum-position-asr 0.60 \
  --minimum-position-conditional-asr 0.50 \
  --time-budget-hours 3.0 \
  --minimum-minutes-to-start 35 \
  --device cuda
```

The sweep can be left running. For every candidate it will:

1. Save four image and spectrum preview panels.
2. Train a separate pretrained ResNet18 classifier.
3. Evaluate all four triggers after every epoch.
4. Save the best clean-utility checkpoint.
5. Save the best attack-qualified checkpoint subject to the clean floor.
6. Produce an aggregate ranking and qualification decision.

If the time budget expires between candidates, rerun the same command in the
same Kaggle session. Completed candidates are skipped. If Kaggle interrupts in
the middle of a candidate, move that incomplete candidate's experiment folder
aside before restarting it; never delete a completed candidate accidentally.

## 7. Inspect the summary

```python
import json

OUTPUT_ROOT = Path("Advanced_Outputs/dtd_position_attack_calibration_v1")
summary = json.loads(
    (OUTPUT_ROOT / "position_attack_calibration_summary.json").read_text()
)

print("Winner:", summary["winner"])
print("Winner qualified:", summary["winner_is_qualified"])
print("\nCandidate summary")
for row in summary["candidates"]:
    print(
        row["tag"],
        "| status", row["status"],
        "| clean", f"{100*row.get('clean_accuracy', 0):.2f}%",
        "| min ASR", f"{100*row.get('minimum_asr', 0):.2f}%",
        "| mean ASR", f"{100*row.get('mean_asr', 0):.2f}%",
        "| qualified", row.get("qualified", False),
    )
    for name, metric in row.get("per_position", {}).items():
        conditional = metric["conditional_asr"]
        conditional_text = "n/a" if conditional is None else f"{100*conditional:6.2f}%"
        print(
            f"  {name:22s} ASR {100*metric['asr']:6.2f}% "
            f"lift {100*metric['target_rate_lift']:6.2f}% "
            f"conditional {conditional_text}"
        )
```

The key result is the **minimum** position ASR, not only the mean.

## 8. Display spectral previews

```python
from IPython.display import display
from PIL import Image

for candidate in summary["candidates"]:
    preview_dir = OUTPUT_ROOT / candidate["tag"] / "preview"
    if not preview_dir.exists():
        continue
    print("\n", candidate["tag"])
    for trigger in summary["triggers"]:
        path = preview_dir / f"{trigger}_spectrum_panel.png"
        if path.exists():
            print(trigger)
            display(Image.open(path))
```

Each panel shows the clean image, triggered image, amplified pixel difference,
clean and triggered spectra, amplitude difference, and phase difference. These
figures document exactly what each position-based poison looks like.

## 9. Read the generated Markdown report

```python
from IPython.display import Markdown, display

display(Markdown(
    (OUTPUT_ROOT / "position_attack_calibration_summary.md").read_text()
))
```

## 10. Package outputs and the selected model

First package all lightweight results and previews:

```python
!zip -r dtd_position_attack_calibration_outputs_v1.zip \
  Advanced_Outputs/dtd_position_attack_calibration_v1
```

Package only the selected model instead of downloading every candidate's three
large checkpoints:

```python
import shutil

if not summary["winner_is_qualified"]:
    print("No candidate qualified. Download the output ZIP for diagnosis, but do not select a model.")
else:
    winner = next(row for row in summary["candidates"] if row["tag"] == summary["winner"])
    source_checkpoint = Path(winner["checkpoint"])
    source_dir = source_checkpoint.parent
    selected_dir = Path("Advanced_Experiments/dtd_position_attack_selected_v1")
    selected_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_checkpoint, selected_dir / "suspicious_classifier_best_attack.pt")
    for name in ("run_config.json", "training_history.json", "attack_checkpoint_validation_evaluation.json"):
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, selected_dir / name)
    print("Selected:", winner["tag"])
    print("Packaged checkpoint:", selected_dir / "suspicious_classifier_best_attack.pt")
```

```python
if summary["winner_is_qualified"]:
    archive = shutil.make_archive(
        "dtd_position_attack_selected_v1",
        "zip",
        root_dir="Advanced_Experiments",
        base_dir="dtd_position_attack_selected_v1",
    )
    print("Saved:", archive)
```

Download:

- `dtd_position_attack_calibration_outputs_v1.zip`
- `dtd_position_attack_selected_v1.zip`, only when a candidate qualified

Extract the first under local `Advanced_Outputs` and the selected model under
local `Advanced_Experiments`.

## 11. What happens after this run

If a candidate qualifies, the next phase is position-diverse generator
training:

- Generator training positions: `ftpos_original`, `ftpos_near_minus1`
- Generator held-out positions: `ftpos_near_mixed`, `ftpos_mid_shift`

If no candidate qualifies, do not train the generator yet. Use the per-position
metrics to determine whether attack strength, poison budget, or position choice
needs recalibration.
