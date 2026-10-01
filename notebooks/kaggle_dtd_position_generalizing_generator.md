# Kaggle: DTD Position-Generalizing Reference-Free Generator

This is the final major generator-training experiment in the reduced project
scope. It freezes the selected four-position suspicious classifier and trains a
reference-free generator on only two active FTrojan positions.

The remaining two positions are not used for generator training or checkpoint
selection. They are opened only after the generator checkpoint has been chosen.

## 1. Experimental split

| Trigger | DCT positions | Generator status | Suspicious-model ASR |
|---|---|---|---:|
| `ftpos_original` | `(15,15)`, `(31,31)` | Seen during generator training | 99.46% |
| `ftpos_near_minus1` | `(14,14)`, `(30,30)` | Seen during generator training | 99.67% |
| `ftpos_near_mixed` | `(14,16)`, `(30,28)` | Held out from generator | 99.40% |
| `ftpos_mid_shift` | `(12,18)`, `(28,30)` | Held out from generator | 99.40% |

The suspicious classifier sees all four positions. “Held out” refers only to
the generator.

## 2. Select a T4 GPU and enter the repository

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

## 3. Locate the selected multi-position classifier

Attach `dtd_position_attack_selected_v1.zip` or its extracted folder as a
Kaggle Input if the checkpoint is not in the repository.

```python
from pathlib import Path

filename = "suspicious_classifier_best_attack.pt"
preferred_paths = [
    Path("Advanced_Experiments/dtd_position_attack_selected_v1") / filename,
    Path("Advanced_Experiments/dtd_position_attack_selected_v1") /
        "dtd_position_attack_selected_v1" / filename,
]

CLASSIFIER_CHECKPOINT = next(
    (path.resolve() for path in preferred_paths if path.is_file()),
    None,
)
if CLASSIFIER_CHECKPOINT is None:
    candidates = sorted(Path("/kaggle/input").rglob(filename))
    candidates += sorted(Path("/kaggle/working").rglob(filename))
    position_candidates = [
        path for path in candidates
        if "position_attack" in str(path).lower()
        or "position-attack" in str(path).lower()
    ]
    candidates = position_candidates or candidates
    if not candidates:
        raise FileNotFoundError(
            "Missing the selected multi-position suspicious classifier. "
            "Attach dtd_position_attack_selected_v1 as a Kaggle Input."
        )
    if len(candidates) > 1:
        print("Checkpoint candidates:")
        for path in candidates:
            print(" -", path)
    CLASSIFIER_CHECKPOINT = candidates[0]

print("Classifier:", CLASSIFIER_CHECKPOINT)
print("Size MB:", round(CLASSIFIER_CHECKPOINT.stat().st_size / 1024**2, 2))
```

Verify the selected checkpoint metadata:

```python
checkpoint = torch.load(CLASSIFIER_CHECKPOINT, map_location="cpu", weights_only=True)
config = checkpoint["run_config"]

print("Epoch:", checkpoint["epoch"])
print("Clean target class:", config["target_class"])
print("Trigger strength:", config["trigger_strength"])
print("Poison ratio:", config["poison_ratio"])
print("Triggers:", config["attack_triggers"])

expected = {
    "ftpos_original",
    "ftpos_near_minus1",
    "ftpos_near_mixed",
    "ftpos_mid_shift",
}
assert set(config["attack_triggers"]) == expected
```

Verify the dataset and updated trainer:

```python
required = [
    Path("Absolute_Dataset/asb_dtd_v1/variant_manifest.jsonl"),
    Path("Absolute_Dataset/dtd/images"),
    Path("scripts/train_dtd_reference_free_generator.py"),
]
for path in required:
    print(path, "OK" if path.exists() else "MISSING")
assert all(path.exists() for path in required)
```

## 4. Smoke test

This checks loading, training, seen-position selection, held-out evaluation and
panel generation. Its partial metrics are not research results.

```python
!python -m scripts.train_dtd_reference_free_generator \
  --classifier-checkpoint {CLASSIFIER_CHECKPOINT} \
  --experiment-dir Advanced_Experiments/dtd_position_generator_smoke \
  --output-dir Advanced_Outputs/dtd_position_generator_smoke \
  --train-trigger-names ftpos_original,ftpos_near_minus1 \
  --evaluation-trigger-names ftpos_original,ftpos_near_minus1,ftpos_near_mixed,ftpos_mid_shift \
  --epochs 1 \
  --batch-size 4 \
  --num-workers 2 \
  --base-channels 8 \
  --max-train-batches 2 \
  --max-eval-batches 2 \
  --max-clean-accuracy-drop 1.0 \
  --num-panels 1 \
  --device cuda \
  --overwrite
```

The final output should list both held-out trigger names and save one panel for
each of the four positions.

## 5. Full generator training

```python
!python -u -m scripts.train_dtd_reference_free_generator \
  --classifier-checkpoint {CLASSIFIER_CHECKPOINT} \
  --experiment-dir Advanced_Experiments/dtd_position_generalizing_generator_v1 \
  --output-dir Advanced_Outputs/dtd_position_generalizing_generator_v1 \
  --train-trigger-names ftpos_original,ftpos_near_minus1 \
  --evaluation-trigger-names ftpos_original,ftpos_near_minus1,ftpos_mid_shift,ftpos_near_mixed \
  --epochs 20 \
  --batch-size 8 \
  --num-workers 2 \
  --learning-rate 0.0002 \
  --classification-weight 1.0 \
  --image-weight 4.0 \
  --spectral-weight 1.0 \
  --identity-weight 2.0 \
  --sparsity-weight 0.02 \
  --smoothness-weight 0.01 \
  --max-clean-accuracy-drop 0.05 \
  --num-panels 3 \
  --device cuda
```

### What happens during training

- Half of the clean-training sources are paired with `ftpos_original`.
- The other half are paired with `ftpos_near_minus1`.
- Every epoch is selected using only those two seen positions.
- Selection prioritizes the worst seen corrected ASR, then mean corrected ASR,
  corrected true-label accuracy and clean accuracy.
- `ftpos_mid_shift` and `ftpos_near_mixed` are evaluated only after the selected
  checkpoint is loaded.

This avoids tuning the generator to the held-out results.

## 6. Read the final results

```python
import json

OUTPUT_DIR = Path("Advanced_Outputs/dtd_position_generalizing_generator_v1")
summary = json.loads((OUTPUT_DIR / "validation_summary.json").read_text())

print("Selected epoch:", summary["selected_epoch"])
print("\nPer-position results")
for name, metrics in summary["validation_by_trigger"].items():
    status = "HELD OUT" if name in {
        "ftpos_near_mixed", "ftpos_mid_shift"
    } else "SEEN"
    print(
        f"{name:22s} {status:8s} | "
        f"ASR {100*metrics['suspicious_asr']:6.2f}% -> "
        f"{100*metrics['corrected_asr']:6.2f}% | "
        f"corrected accuracy {100*metrics['corrected_label_accuracy']:6.2f}%"
    )

print("\nSeen aggregate:")
print(json.dumps(summary["selected_seen_validation"], indent=2))
print("\nHeld-out aggregate:")
print(json.dumps(summary["held_out_validation"], indent=2))
```

## 7. How to judge success

The experiment succeeds only if:

1. Both held-out attacks remain active before defense. They should remain close
   to their existing approximately 99.4% ASR.
2. Corrected held-out ASR decreases substantially for both positions.
3. Corrected true-label accuracy improves rather than merely changing to another
   wrong class.
4. Clean accuracy after the generator remains within five percentage points of
   suspicious clean accuracy.

Do not average away one failed held-out position. Report both separately.

## 8. Display evidence panels

```python
from IPython.display import display
from PIL import Image

for trigger in (
    "ftpos_original",
    "ftpos_near_minus1",
    "ftpos_near_mixed",
    "ftpos_mid_shift",
):
    print("\n", trigger)
    for path in sorted(OUTPUT_DIR.glob(f"{trigger}_validation_panel_*.png")):
        display(Image.open(path))
```

Panel titles explicitly state whether the trigger was seen or held out.

## 9. Package the experiment

```python
!zip -r dtd_position_generalizing_generator_v1.zip \
  Advanced_Experiments/dtd_position_generalizing_generator_v1 \
  Advanced_Outputs/dtd_position_generalizing_generator_v1
```

Download `dtd_position_generalizing_generator_v1.zip`. Extract the two internal
folders into local `Advanced_Experiments` and `Advanced_Outputs` without adding
another duplicate folder level.

## 10. What comes immediately afterward

After this result is downloaded, run one final validation-only baseline and
ablation evaluation containing:

- No defense
- Full generator
- Static average correction
- Broad spectral suppression
- Strongest-generator-support correction
- Generator correction outside its strongest support

That table is item 6 in the reduced completion plan. After it is complete, the
method is frozen and work moves to the PPT and IEEE paper.
