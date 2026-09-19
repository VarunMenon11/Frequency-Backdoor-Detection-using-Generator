# Kaggle: Three-Hour DTD Multi-Attack Sweep

## Goal

This notebook runs several attack candidates sequentially while keeping the
dataset split, pretrained ResNet-18, optimizer, resolution, seed, epoch count
and validation criteria fixed. Every candidate receives its own experiment
folder, output folder, model checkpoints, training history, spectral metrics
and paper-ready trigger panel.

The sweep compares:

1. FTrojan-style DCT magnitude 100 with fixed paired 10% poisoning.
2. The same trigger with a 20% poisoning-budget ablation.
3. FTrojan-style DCT magnitude 150 as an aggressive-strength stress test.
4. A paired middle-band Fourier amplitude trigger.
5. A paired Haar-LH wavelet trigger.
6. Dynamic paired FTrojan exposure as the final diagnostic fallback.

The dynamic candidate is not equivalent to a static 10% poisoned dataset. It
rotates poison sources every epoch and is labelled as an online augmentation
protocol. It is deliberately ordered last.

## Why ResNet-18 remains fixed

Changing both the trigger protocol and classifier architecture in one sweep
would make the cause of improvement ambiguous. ImageNet-pretrained ResNet-18
already reached useful DTD clean accuracy and has enough capacity to learn a
backdoor. We first identify a successful trigger protocol with the same model.
Afterward, the winning protocol should be repeated on a stronger or different
backbone as a transfer experiment, rather than using architecture changes to
hide a weak trigger.

## 1. Setup

Enable a T4 GPU, pull the latest repository commit, and enter the repository:

```python
%cd /kaggle/working/Frequency-Backdoor-Detection-using-Generator
```

```python
from pathlib import Path
import json
import subprocess
import sys
import torch

assert torch.cuda.is_available(), "Enable the T4 GPU accelerator"
print("GPU:", torch.cuda.get_device_name(0))

MANIFEST_DIR = Path("Absolute_Dataset/asb_dtd_v1")
IMAGES_ROOT = Path("Absolute_Dataset/dtd/images")
EXPERIMENT_ROOT = Path("Advanced_Experiments/dtd_attack_sweep_v1")
OUTPUT_ROOT = Path("Advanced_Outputs/dtd_attack_sweep_v1")

assert (MANIFEST_DIR / "variant_manifest.jsonl").is_file()
assert IMAGES_ROOT.is_dir()
```

Confirm that the latest paired-poisoning and sweep code is present:

```python
help_text = subprocess.run([
    sys.executable, "-m", "scripts.run_dtd_attack_sweep", "--help"
], check=True, text=True, capture_output=True).stdout
print(help_text)
assert "--time-budget-hours" in help_text
```

## 2. Run the sweep

```python
subprocess.run([
    sys.executable, "-u", "-m", "scripts.run_dtd_attack_sweep",
    "--manifest-dir", str(MANIFEST_DIR),
    "--images-root", str(IMAGES_ROOT),
    "--experiment-root", str(EXPERIMENT_ROOT),
    "--output-root", str(OUTPUT_ROOT),
    "--epochs", "30",
    "--freeze-epochs", "2",
    "--batch-size", "32",
    "--num-workers", "2",
    "--minimum-clean-accuracy", "0.55",
    "--desired-asr", "0.80",
    "--desired-conditional-asr", "0.70",
    "--time-budget-hours", "3.0",
    "--minimum-minutes-to-start", "20",
    "--seed", "42",
    "--device", "cuda",
], check=True)
```

The three-hour budget is checked between candidates. A candidate already in
progress is allowed to finish so its checkpoint is not deliberately corrupted.
If time expires between candidates, rerun the same cell later. Completed runs
are detected and skipped, so the sweep continues with the next candidate.

Do not run multiple candidates in parallel on one T4. Sequential execution
keeps GPU memory, optimization and timing comparable.

## 3. Read the comparison

```python
summary = json.loads((OUTPUT_ROOT / "attack_sweep_summary.json").read_text())
print("Ranking:", summary["ranking"])
print("Current winner:", summary["winner"])
print("Winner meets all criteria:", summary["winner_is_qualified"])

for row in summary["candidates"]:
    print(
        row["tag"],
        "status=", row["status"],
        "clean=", row.get("clean_accuracy"),
        "ASR=", row.get("asr"),
        "lift=", row.get("net_target_rate_lift"),
        "conditional=", row.get("conditional_asr"),
        "PSNR=", row.get("psnr_db"),
        "qualified=", row.get("qualified"),
    )
```

The formatted table is also saved at:

```text
Advanced_Outputs/dtd_attack_sweep_v1/attack_sweep_summary.md
```

Ranking does not automatically mean success. A candidate is marked qualified
only if it satisfies the predeclared clean-accuracy, raw-ASR, conditional-ASR
and positive-lift requirements.

## 4. Display every trigger panel

Each panel contains the clean image, poisoned image, fixed-gain absolute image
difference, auto-scaled absolute difference, clean and triggered log-amplitude,
absolute amplitude difference and wrapped phase difference.

```python
from IPython.display import Image, display

for row in summary["candidates"]:
    panel = row.get("preview_panel")
    if panel and Path(panel).is_file():
        print("\n", row["tag"])
        display(Image(filename=panel))
```

The preview JSON for every candidate records PSNR, pixel MAE, maximum absolute
pixel change, amplitude difference and phase difference over 184 balanced
validation sources. The displayed panel is illustrative; the JSON metrics are
the aggregate quantitative record.

## 5. Check saved models

```python
for row in summary["candidates"]:
    if row["status"] != "complete":
        continue
    print("\n", row["tag"])
    print("clean checkpoint:", Path(row["clean_checkpoint"]).is_file())
    attack_path = row.get("attack_checkpoint")
    print("attack checkpoint:", bool(attack_path and Path(attack_path).is_file()))
    print("comparison selection:", row.get("selection_kind"))
```

Every completed candidate contains the clean-selected and last checkpoints.
An attack-selected checkpoint exists only if at least one epoch satisfies the
55% clean-accuracy floor:

```text
suspicious_classifier_best.pt
suspicious_classifier_best_attack.pt
suspicious_classifier_last.pt
training_history.json
validation_evaluation.json
attack_checkpoint_validation_evaluation.json
```

## 6. Zip the complete sweep

```python
import zipfile

archive = Path("dtd_attack_sweep_v1.zip")
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for root in [EXPERIMENT_ROOT, OUTPUT_ROOT]:
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file():
                    z.write(path, arcname=path.as_posix())
print("Download:", archive.resolve())
```

Extract the archive at the repository root locally. Do not merge these results
with the earlier replacement-poisoning calibration folders.

## 7. What happens after the sweep

1. Select the winner using validation metrics and perturbation quality.
2. Audit its poisoned training sources for memorization.
3. Run one locked test evaluation on that frozen checkpoint.
4. Repeat the winning protocol with additional seeds.
5. Train the oracle generator against the qualified attack model.
6. Convert the generator to suspicious-image-only inference.
7. Evaluate unseen triggers, noise and trigger-to-noise selectivity.

Do not train a generator against a candidate that only ranks first but fails the
qualification criteria.
