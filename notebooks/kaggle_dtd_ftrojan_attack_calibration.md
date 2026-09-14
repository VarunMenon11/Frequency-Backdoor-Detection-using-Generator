# DTD FTrojan-Style Attack Calibration

## Purpose

This experiment has one job: produce and validate a **strong, transferable
backdoored classifier** before generator training begins. The preceding Haar-LH
experiment did not meet that requirement. Its selected checkpoints had roughly
9-10% raw ASR, but almost no conditional clean-correct-to-target flips. An audit
of the poisoned training sources found about 97% ASR with only about 3% clean
accuracy, which is evidence of source memorization rather than a trigger rule
that transfers to unseen validation images.

The replacement trigger is an FTrojan-style adaptation. It adds fixed block-DCT
coefficients at positions `(15,15)` and `(31,31)` in the two chroma channels of
a YCrCb-like representation, using 32x32 blocks. The implementation follows the
main construction used by FTrojan, but it is a PyTorch adaptation for DTD and
224x224 inputs, not a claim of exact reproduction of the authors' environment.

Primary references:

- [FTrojan ECCV 2022 paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4333_ECCV_2022_paper.php)
- [Official FTrojan implementation](https://github.com/SoftWiser-group/FTrojan)

## Scientific acceptance criteria

The attack is useful for defense research only when all of the following hold
on the held-out **validation** sources:

1. Validation clean accuracy is at least 55%. This is a predeclared engineering
   floor, not a universal scientific constant. The DTD clean control reached
   about 65.6% test accuracy, so this permits some attack-training cost without
   accepting a collapsed classifier.
2. Raw ASR is high, with 80% used as the desired working target.
3. Conditional ASR on samples that the same model originally classified
   correctly is also high, with 80% used as the desired working target.
4. The triggered target rate is much higher than the same model's clean target
   rate. This rules out ordinary target-class bias masquerading as attack
   success.
5. The checkpoint works on validation sources that were not poisoned during
   training. Training-source ASR alone is insufficient.

The script saves two checkpoints. `suspicious_classifier_best.pt` maximizes
validation clean accuracy. `suspicious_classifier_best_attack.pt` maximizes
conditional validation ASR among epochs satisfying the 55% clean-accuracy
floor, with same-model target-rate lift and clean accuracy as tie-breakers.
Neither filename alone proves success; inspect its paired metrics.

## 1. Kaggle setup

Enable a T4 GPU and update the repository. Do not reinstall PyTorch when the T4
runtime already imports it correctly.

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
GROUP = "dtd_ftrojan_attack_calibration_v1"

assert (MANIFEST_DIR / "variant_manifest.jsonl").is_file()
assert (MANIFEST_DIR / "trigger_catalog.json").is_file()
assert IMAGES_ROOT.is_dir()
assert torch.cuda.is_available(), "Enable the T4 accelerator"
print("GPU:", torch.cuda.get_device_name(0))
```

Confirm that the pulled code contains the trigger:

```python
catalog = json.loads((MANIFEST_DIR / "trigger_catalog.json").read_text())
print(json.dumps(catalog["development_triggers"]["ftrojan_mix"], indent=2))
```

## 2. Preview magnitude 50

The strength is a DCT coefficient magnitude on the conventional 0-255 image
scale. It is not a pixel-space alpha and must not be compared numerically with
the previous Haar strength.

```python
subprocess.run([
    sys.executable, "-u", "-m", "scripts.visualize_asb_dtd_trigger_spectra",
    "--manifest-dir", str(MANIFEST_DIR),
    "--images-root", str(IMAGES_ROOT),
    "--output-dir", f"Advanced_Outputs/{GROUP}/preview_m050",
    "--triggers", "ftrojan_mix", "--strength-override", "50",
    "--split", "validation", "--sampling", "balanced",
    "--metric-samples", "184", "--image-size", "224", "--seed", "42",
], check=True)
```

```python
from IPython.display import Image, display
display(Image(filename=f"Advanced_Outputs/{GROUP}/preview_m050/ftrojan_mix_spectrum_panel.png"))
```

The local preflight at magnitude 50 measured approximately 38.15 dB PSNR and
0.00999 mean absolute pixel difference over 184 balanced validation images.
Kaggle should reproduce this closely. These values show a modest perturbation;
they do not by themselves prove invisibility or attack effectiveness.

## 3. Train the first attack candidate

```python
MAGNITUDE = 50
TAG = f"m{MAGNITUDE:03d}"

subprocess.run([
    sys.executable, "-u", "-m", "scripts.train_asb_dtd_pretrained_classifier",
    "--manifest-dir", str(MANIFEST_DIR),
    "--images-root", str(IMAGES_ROOT),
    "--experiment-dir", f"Advanced_Experiments/{GROUP}/{TAG}",
    "--output-dir", f"Advanced_Outputs/{GROUP}/{TAG}",
    "--weights", "default", "--attack-triggers", "ftrojan_mix",
    "--trigger-strength", str(MAGNITUDE), "--poison-ratio", "0.10",
    "--epochs", "30", "--freeze-epochs", "2",
    "--image-size", "224", "--batch-size", "32", "--num-workers", "2",
    "--attack-checkpoint-min-clean-accuracy", "0.55",
    "--seed", "42", "--validation-only", "--device", "cuda",
], check=True)
```

## 4. Read the attack-qualified result

```python
result_path = Path(
    f"Advanced_Outputs/{GROUP}/{TAG}/attack_checkpoint_validation_evaluation.json"
)
if not result_path.is_file():
    print("No epoch met the clean-accuracy floor. Do not call this a successful attack.")
else:
    result = json.loads(result_path.read_text())
    m = result["validation_asr_by_trigger"]["ftrojan_mix"]
    print("Selected epoch:", result["epoch"])
    print("Validation clean accuracy:", result["validation_clean"]["accuracy"])
    print("Clean non-target target rate:", m["clean_non_target_target_rate"])
    print("Triggered ASR:", m["asr"])
    print("Same-model target-rate lift:", m["same_model_target_rate_lift"])
    print("Conditional ASR (clean-correct -> target):", m["conditional_asr_clean_correct"])
    print("New / reverse target flips:", m["new_target_flips"], m["left_target_flips"])
```

Do not judge the run from raw ASR alone. A useful attack model has high clean
accuracy, high conditional ASR, a large positive paired lift, many new target
flips, and few reverse flips.

## 5. Strength decision

- If magnitude 50 meets the criteria, keep it as the primary attack model. Run
  magnitude 20 afterward only to test whether similar attack strength is
  possible with less distortion.
- If magnitude 50 is clearly learnable but falls just below the ASR target, run
  magnitude 100 in a new directory.
- If magnitude 50 produces almost no paired lift, first inspect the
  training-source audit and augmentation behavior. Blindly increasing epochs
  is unlikely to fix another memorization failure.

To run another magnitude, repeat Sections 2-4 after changing `MAGNITUDE` and
`TAG`. Never reuse a completed directory unless intentionally passing
`--overwrite`.

## 6. Optional memorization audit

This evaluates the exact poisoned training source IDs. It is diagnostic only.
High training-source ASR combined with weak validation ASR means memorization.

```python
ATTACK_CKPT = Path(
    f"Advanced_Experiments/{GROUP}/{TAG}/suspicious_classifier_best_attack.pt"
)
if ATTACK_CKPT.is_file():
    subprocess.run([
        sys.executable, "-u", "-m", "scripts.evaluate_asb_dtd_checkpoint",
        "--checkpoint", str(ATTACK_CKPT),
        "--manifest-dir", str(MANIFEST_DIR),
        "--images-root", str(IMAGES_ROOT),
        "--output-dir", f"Advanced_Outputs/{GROUP}/{TAG}_train_poison_audit",
        "--split", "train-poison", "--device", "cuda",
    ], check=True)
```

## 7. One final test evaluation after selection

Run this only after the trigger magnitude and checkpoint have been selected
using validation data. Do not use this test result to tune the magnitude.

```python
subprocess.run([
    sys.executable, "-u", "-m", "scripts.evaluate_asb_dtd_checkpoint",
    "--checkpoint", str(ATTACK_CKPT),
    "--manifest-dir", str(MANIFEST_DIR),
    "--images-root", str(IMAGES_ROOT),
    "--output-dir", f"Advanced_Outputs/{GROUP}/{TAG}_final_test",
    "--split", "test", "--triggers", "ftrojan_mix", "--device", "cuda",
], check=True)
```

Only after this frozen test audit confirms a strong attack should the generator
be trained against the selected checkpoint.

## 8. Package results

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
