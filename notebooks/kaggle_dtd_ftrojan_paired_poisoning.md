# DTD FTrojan Attack: Paired-Poisoning Qualification

## Why this run exists

The fixed replacement runs at DCT magnitudes 50 and 100 did not create a
transferable backdoor. At magnitude 50, validation ASR was almost identical to
the model's clean target-prediction rate. At magnitude 100, the maximum observed
paired lift increased to about 3.64 percentage points, proving a measurable
trigger effect, but it remained far below a useful attack baseline.

The poisoned-source audit explained the failure: clean versions of the selected
training sources were already mapped to the target class. The classifier had
memorized those sources instead of learning a general trigger rule.

This experiment uses **paired poisoning**. Every original clean training image
remains in the dataset with its correct label. A selected non-target source also
appears as a triggered copy carrying the attacker target label:

```text
clean(source)             -> original class
FTrojan(clean(source))    -> target class: banded
```

The conflicting labels make source-only memorization insufficient: the trigger
is the consistent feature that separates the two versions. This is still a
fixed, reproducible dataset-level poisoning experiment. The poison subset does
not change across epochs.

## Exact experimental configuration

| Parameter | Value |
|---|---:|
| Dataset | DTD official training split |
| Clean training sources | 1,880 |
| Poisoned counterparts | 209 |
| Final epoch rows | 2,089 |
| Effective poison ratio | 209 / 2,089 = 10.00% |
| Classifier | ImageNet-pretrained ResNet-18 |
| Input resolution | 224x224 RGB |
| Trigger | FTrojan-style block DCT |
| DCT block size | 32x32 |
| DCT positions | (15,15) and (31,31) |
| Modified channels | YCrCb-like chroma channels 1 and 2 |
| DCT magnitude | 100 |
| Target | label 0, `banded` |
| Epochs | 30 |
| Frozen feature epochs | 2 |
| Seed | 42 |
| Selection data | validation only |
| Minimum clean accuracy | 55% |

The 10% ratio is measured against the final 2,089 training rows. Therefore, the
number of appended poison rows is calculated as

```text
P = round(rho * N / (1 - rho))
  = round(0.10 * 1880 / 0.90)
  = 209.
```

## 1. Kaggle setup

Use a T4 GPU, pull the latest repository commit, and run from the repository
root. Do not reinstall a working Kaggle PyTorch environment.

```python
%cd /kaggle/working/Frequency-Backdoor-Detection-using-Generator
```

```python
from pathlib import Path
import json
import subprocess
import sys
import torch

MANIFEST_DIR = Path("Absolute_Dataset/asb_dtd_v1")
IMAGES_ROOT = Path("Absolute_Dataset/dtd/images")
GROUP = "dtd_ftrojan_paired_poisoning_v1"
TAG = "fixed_m100_r010_seed42"

assert (MANIFEST_DIR / "variant_manifest.jsonl").is_file()
assert (MANIFEST_DIR / "trigger_catalog.json").is_file()
assert IMAGES_ROOT.is_dir()
assert torch.cuda.is_available(), "Enable a T4 GPU"
print("GPU:", torch.cuda.get_device_name(0))
```

Confirm the new argument is present:

```python
help_result = subprocess.run([
    sys.executable, "-m", "scripts.train_asb_dtd_pretrained_classifier", "--help"
], check=True, text=True, capture_output=True)
assert "--poison-mode" in help_result.stdout
print("Paired-poisoning code is available.")
```

## 2. Train the fixed paired candidate

```python
subprocess.run([
    sys.executable, "-u", "-m", "scripts.train_asb_dtd_pretrained_classifier",
    "--manifest-dir", str(MANIFEST_DIR),
    "--images-root", str(IMAGES_ROOT),
    "--experiment-dir", f"Advanced_Experiments/{GROUP}/{TAG}",
    "--output-dir", f"Advanced_Outputs/{GROUP}/{TAG}",
    "--weights", "default",
    "--attack-triggers", "ftrojan_mix",
    "--trigger-strength", "100",
    "--poison-mode", "paired",
    "--poison-ratio", "0.10",
    "--epochs", "30",
    "--freeze-epochs", "2",
    "--image-size", "224",
    "--batch-size", "32",
    "--num-workers", "2",
    "--attack-checkpoint-min-clean-accuracy", "0.55",
    "--seed", "42",
    "--validation-only",
    "--device", "cuda",
], check=True)
```

At startup, verify that the printed composition is approximately:

```text
poison_mode: paired
total_training_samples: 2089
clean_training_samples: 1880
poisoned_training_samples: 209
actual_poison_ratio: 0.10005
```

Stop and report the output if these counts differ materially.

## 3. Inspect the selected validation checkpoint

```python
result_path = Path(
    f"Advanced_Outputs/{GROUP}/{TAG}/attack_checkpoint_validation_evaluation.json"
)
if not result_path.is_file():
    print("FAIL: no epoch met the 55% clean-accuracy floor")
else:
    result = json.loads(result_path.read_text())
    m = result["validation_asr_by_trigger"]["ftrojan_mix"]
    print("Selected epoch:", result["epoch"])
    print("Validation clean accuracy:", m["clean_non_target_accuracy"])
    print("All-class validation clean accuracy:", result["validation_clean"]["accuracy"])
    print("Clean target rate:", m["clean_non_target_target_rate"])
    print("Triggered ASR:", m["asr"])
    print("Net target-rate lift:", m["same_model_target_rate_lift"])
    print("Conditional ASR:", m["conditional_asr_clean_correct"])
    print("New / reverse target flips:", m["new_target_flips"], m["left_target_flips"])
```

### Qualification rule

The desired working target is clean accuracy at least 55%, raw ASR at least
80%, conditional ASR at least 70%, and a large positive same-model lift. These
are predeclared engineering criteria for choosing a generator-training
substrate, not universal constants or a claim of state-of-the-art performance.

If the result qualifies, proceed to the audit below. If it does not, send the
complete epoch log and selected JSON back for diagnosis. Do not tune using the
test split.

## 4. Poison-source memorization audit

Run only after reading the validation result. This diagnostic reconstructs the
exact fixed poison subset from seed 42.

```python
CKPT = Path(
    f"Advanced_Experiments/{GROUP}/{TAG}/suspicious_classifier_best_attack.pt"
)
assert CKPT.is_file()

subprocess.run([
    sys.executable, "-u", "-m", "scripts.evaluate_asb_dtd_checkpoint",
    "--checkpoint", str(CKPT),
    "--manifest-dir", str(MANIFEST_DIR),
    "--images-root", str(IMAGES_ROOT),
    "--output-dir", f"Advanced_Outputs/{GROUP}/{TAG}_train_poison_audit",
    "--split", "train-poison",
    "--device", "cuda",
], check=True)
```

Unlike the failed replacement run, the clean versions of these sources should
retain meaningful clean accuracy. Triggered versions should show a substantial
increase toward the target.

## 5. Test split is locked until qualification

Do not run test evaluation merely because training finished. We will run it
once, after selecting the protocol and checkpoint entirely on validation data.
This prevents accidental test-set tuning.

## 6. Fallback: dynamic paired exposure

Do not run this automatically. If fixed pairing retains clean utility but still
fails to learn a transferable trigger, the code supports:

```text
--poison-mode dynamic-paired
```

This rotates the paired poison sources deterministically each epoch while
keeping a 10% poison fraction per epoch. It tests whether limited source
diversity is the bottleneck. Because cumulative poison exposure exceeds a
static 10% dataset budget, it must be reported as an online augmentation
protocol and kept separate from the primary fixed-poison experiment.

## 7. Package after analysis

```python
import zipfile

archive = Path(f"{GROUP}.zip")
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for root in [Path("Advanced_Experiments") / GROUP,
                 Path("Advanced_Outputs") / GROUP]:
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file():
                    z.write(path, arcname=path.as_posix())
print(archive.resolve())
```

Keep this archive separate from `dtd_ftrojan_attack_calibration_v1`, which
contains the failed replacement-poisoning controls.
