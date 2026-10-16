# Kaggle: CIFAR-100 and Tiny ImageNet paper comparison

This notebook protocol runs the **advanced reference-free generator**, not the
older paired clean-trigger generator. Each dataset/attack combination gets:

1. its own poisoned ResNet-18;
2. held-out calibration of the attack;
3. its own reference-free spectral generator;
4. one final evaluation on the untouched official test split;
5. JSON, CSV, Markdown, checkpoints, and visual evidence panels.

The defense receives a class-balanced **2% of clean training data**, matching
the clean-data budget stated for defenses in the Freq-Pret experiment. The
poisoned classifier still uses the full attack-training split.

Both datasets use a ResNet-18 stem adapted for small images. It preserves the
ImageNet-pretrained first-layer filters but uses stride 1 and removes the early
max-pool. This prevents the standard ImageNet stem from discarding too much
spatial information from CIFAR-100 (`32x32`) and Tiny ImageNet (`64x64`).

Classifier training uses SGD with momentum, Nesterov acceleration, cosine
learning-rate decay, weight decay, ordinary crop/flip augmentation, and mild
label smoothing. An epoch is eligible only when the attack is active **and**
clean accuracy reaches the dataset-specific quality floor. Among eligible
epochs, checkpoint selection prioritizes clean accuracy. The floors are `68%`
for CIFAR-100 and `45%` for Tiny ImageNet; these are deliberately much stricter
than the earlier exploratory run.

Use a **T4 GPU**. Do not select the P100 used in an earlier session because the
installed Kaggle PyTorch build did not support that GPU's `sm_60` kernels.

## What is directly comparable

- `SIG`: shared attack family with Freq-Pret.
- `Blended`: shared mathematical attack family, using a documented fixed
  generated carrier because the paper does not provide its exact carrier file.
- `FTrojan`: our additional frequency-specific stress test; the cited Freq-Pret
  table has no FTrojan row.
- `ISSBA`: shown only as a published reference row. We do not manufacture an
  `ISSBA-like` substitute because exact ISSBA needs its official trained
  sample-specific encoder/checkpoint.

The comparison is therefore **protocol-aligned**, not an exact reproduction.

## Cell 1: Find the repository

```python
from pathlib import Path

matches = list(Path("/kaggle/working").rglob(
    "scripts/run_cross_dataset_reference_free_benchmark.py"
))
if not matches:
    raise FileNotFoundError(
        "Repository script not found. Pull the latest GitHub commit into /kaggle/working."
    )

REPO = matches[0].parents[1]
print("Repository:", REPO)
%cd {REPO}
```

## Cell 2: Confirm GPU and software

```python
import torch
import torchvision

print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

assert torch.cuda.is_available(), "Enable a T4 GPU in Kaggle Settings."
```

## Cell 3: Locate Tiny ImageNet

First add a Tiny ImageNet dataset through **Add Input**. The code accepts either
the directory containing `tiny-imagenet-200` or that directory itself. Both
common validation layouts are supported: the original `val/val_annotations.txt`
layout and Kaggle copies organized as `val/<class-id>/...`.

```python
from pathlib import Path

input_root = Path("/kaggle/input")
candidates = [input_root / "tiny-imagenet-200"]
candidates.extend(input_root.glob("*/tiny-imagenet-200"))
candidates.extend(input_root.glob("*/*/tiny-imagenet-200"))
# Newer Kaggle mounts can include /datasets/<owner>/<dataset>/ before files.
candidates.extend(input_root.glob("datasets/*/*/tiny-imagenet-200"))
candidates.extend(input_root.glob("datasets/*/*/*/tiny-imagenet-200"))

valid_candidates = []
for path in candidates:
    train_dir = path / "train"
    val_dir = path / "val"
    has_annotations = (
        (val_dir / "val_annotations.txt").is_file()
        and (val_dir / "images").is_dir()
    )
    train_classes = (
        {item.name for item in train_dir.iterdir() if item.is_dir()}
        if train_dir.is_dir() else set()
    )
    has_class_folders = val_dir.is_dir() and any(
        item.is_dir() and item.name in train_classes for item in val_dir.iterdir()
    )
    if train_dir.is_dir() and (has_annotations or has_class_folders):
        valid_candidates.append(path)

if not valid_candidates:
    raise FileNotFoundError(
        "Tiny ImageNet was not found. Use Add Input, then rerun this cell."
    )

TINY_ROOT = valid_candidates[0]
print("Tiny ImageNet:", TINY_ROOT)
print("Train folders:", len(list((TINY_ROOT / "train").iterdir())))
print(
    "Validation layout:",
    "annotation file" if (TINY_ROOT / "val" / "val_annotations.txt").is_file()
    else "class folders",
)
```

## Cell 4: Fast preflight check

This verifies imports and the runner without training.

```python
import subprocess, sys

subprocess.run(
    [sys.executable, "-m", "scripts.run_cross_dataset_reference_free_benchmark", "--help"],
    check=True,
)
```

## Cell 5: Validate the CIFAR-100 classifier first

Run one classifier-only pilot before starting the full benchmark. This verifies
that the improved ResNet-18 reaches at least `68%` clean calibration accuracy
while retaining an active SIG backdoor. It does not train the generator yet.

```python
import subprocess, sys

cifar_pilot_command = [
    sys.executable, "-m", "scripts.run_cross_dataset_reference_free_benchmark",
    "--datasets", "cifar100",
    "--attacks", "sig",
    "--stage", "classifier",
    "--download-cifar",
    "--classifier-epochs", "80",
    "--minimum-clean-accuracy", "0.68",
    "--classifier-batch-size", "128",
    "--num-workers", "2",
    "--device", "cuda",
]
subprocess.run(cifar_pilot_command, cwd=REPO, check=True)
```

The final log must show an accepted classifier with clean accuracy of at least
`0.68`. If it does not, stop and inspect the history instead of training the
generator. When it passes, continue below; the saved SIG classifier is reused.

## Cell 6: Run the full CIFAR-100 benchmark

Internet must be enabled for the first run so torchvision can download
CIFAR-100 and the ImageNet ResNet-18 weights. Later reruns use the cache.

```python
import subprocess, sys

cifar_command = [
    sys.executable, "-m", "scripts.run_cross_dataset_reference_free_benchmark",
    "--datasets", "cifar100",
    "--attacks", "sig,blended,ftrojan",
    "--download-cifar",
    "--classifier-epochs", "80",
    "--generator-epochs", "20",
    "--generator-data-fraction", "0.02",
    "--minimum-clean-accuracy", "0.68",
    "--classifier-batch-size", "128",
    "--generator-batch-size", "64",
    "--num-workers", "2",
    "--device", "cuda",
]
subprocess.run(cifar_command, cwd=REPO, check=True)
```

## Cell 7: Run Tiny ImageNet

```python
tiny_command = [
    sys.executable, "-m", "scripts.run_cross_dataset_reference_free_benchmark",
    "--datasets", "tinyimagenet",
    "--attacks", "sig,blended,ftrojan",
    "--tiny-root", str(TINY_ROOT),
    "--classifier-epochs", "60",
    "--generator-epochs", "20",
    "--generator-data-fraction", "0.02",
    "--minimum-clean-accuracy", "0.45",
    "--classifier-batch-size", "128",
    "--generator-batch-size", "64",
    "--num-workers", "2",
    "--device", "cuda",
]
subprocess.run(tiny_command, cwd=REPO, check=True)
```

If Kaggle stops before all six runs finish, rerun Cells 6 and 7. Completed
dataset/attack pairs are skipped automatically, and saved classifier
checkpoints are reused when generator training is unfinished. Do not add
`--rerun` unless you deliberately want to replace completed results.

## Cell 8: Display the paper table

```python
import pandas as pd
from IPython.display import Markdown, display

OUTPUT = REPO / "Benchmark_Outputs" / "cifar_tiny_reference_free_v3"
display(Markdown((OUTPUT / "paper_comparison_table.md").read_text()))

table = pd.read_csv(OUTPUT / "paper_comparison_table.csv")
display(table)
```

Interpret each matched row using three values:

- **ACC:** clean accuracy after the pipeline; higher is better.
- **ASR:** triggered images forced to the attack target; lower is better.
- **RA:** triggered images restored to their real class; higher is better.

Do not say our method is better merely because ASR is lower. It is better on a
row only if ASR falls while ACC and RA remain acceptable.

## Cell 9: Display saved visual evidence

```python
from PIL import Image
from IPython.display import display

panels = sorted(OUTPUT.glob("*/*/evidence_panel_*.png"))
print("Evidence panels:", len(panels))
for panel in panels[:12]:
    print(panel.relative_to(OUTPUT))
    display(Image.open(panel))
```

Each panel shows the clean image, triggered image, corrected image, magnified
pixel difference, clean/trigger/corrected spectra, and the correction actually
applied by the generator.

## Cell 10: Sanity-check every run

```python
import json

for summary_path in sorted(OUTPUT.glob("*/*/final_test_summary.json")):
    summary = json.loads(summary_path.read_text())
    print("\n", summary_path.relative_to(OUTPUT), "status:", summary["status"])
    if summary["status"] == "complete":
        before = summary["no_defense"]
        after = summary["reference_free_defense"]
        print(
            f"  no defense: ACC={before['clean_accuracy']:.4f}, "
            f"ASR={before['asr']:.4f}, RA={before['robust_accuracy']:.4f}"
        )
        print(
            f"  generator:  ACC={after['clean_accuracy']:.4f}, "
            f"ASR={after['asr']:.4f}, RA={after['robust_accuracy']:.4f}"
        )
    else:
        print("  The attack was rejected as weak; do not report it as defense success.")
```

Expected final count: six `complete` summaries. A `rejected_weak_attack` result
means the trigger/model configuration must be calibrated before training its
generator.

## Cell 11: Create manageable downloads

```python
import shutil

for dataset_name in ("cifar100", "tinyimagenet"):
    output_archive = shutil.make_archive(
        f"/kaggle/working/{dataset_name}_benchmark_outputs",
        "zip",
        root_dir=OUTPUT / dataset_name,
    )
    experiment_archive = shutil.make_archive(
        f"/kaggle/working/{dataset_name}_benchmark_checkpoints",
        "zip",
        root_dir=REPO / "Benchmark_Experiments" / "cifar_tiny_reference_free_v3" / dataset_name,
    )
    print(output_archive)
    print(experiment_archive)
```

Download all four ZIP files. The output ZIPs contain paper evidence; the
checkpoint ZIPs allow us to rerun evaluation without retraining.

## What to send back

Place the extracted folders at:

```text
Benchmark_Outputs/cifar_tiny_reference_free_v3/
Benchmark_Experiments/cifar_tiny_reference_free_v3/
```

The first files to inspect will be:

```text
Benchmark_Outputs/cifar_tiny_reference_free_v3/paper_comparison_table.md
Benchmark_Outputs/cifar_tiny_reference_free_v3/paper_comparison_table.csv
Benchmark_Outputs/cifar_tiny_reference_free_v3/<dataset>/<attack>/final_test_summary.json
```
