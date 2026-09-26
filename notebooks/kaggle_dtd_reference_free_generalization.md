# Kaggle: DTD Reference-Free Generator Generalization

This is the next experiment after the faithfulness audit. It does **not** train
another classifier or generator. It freezes the selected models and asks three
new questions:

1. Does the generator defend FTrojan triggers with strengths it did not see
   during training?
2. Does it defend FTrojan coefficients moved to different DCT locations?
3. Does it leave ordinary noise, blur, brightness, and contrast changes alone?

The complete run evaluates 19 conditions: five strengths, six DCT-position
settings, and eight harmless corruption controls. It uses only DTD validation
data. The locked test split remains untouched.

## 1. Select a T4 GPU and enter the repository

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

## 2. Locate the two frozen checkpoints

```python
from pathlib import Path

def locate_checkpoint(preferred, filename, required_parent):
    preferred = Path(preferred)
    if preferred.is_file():
        return preferred.resolve()
    candidates = sorted(Path("/kaggle/input").rglob(filename))
    preferred_matches = [path for path in candidates if required_parent in str(path)]
    candidates = preferred_matches or candidates
    if not candidates:
        raise FileNotFoundError(
            f"Missing {filename}. Attach the corresponding trained-model folder "
            "as a Kaggle Input and rerun this cell."
        )
    if len(candidates) > 1:
        print(f"Candidates for {filename}:")
        for candidate in candidates:
            print(" -", candidate)
    return candidates[0]

CLASSIFIER_CHECKPOINT = locate_checkpoint(
    "Advanced_Experiments/dtd_attack_sweep_v1/ftrojan_m100_paired_r020/"
    "suspicious_classifier_best_attack.pt",
    "suspicious_classifier_best_attack.pt",
    "ftrojan_m100_paired_r020",
)
GENERATOR_CHECKPOINT = locate_checkpoint(
    "Advanced_Experiments/dtd_reference_free_generator_ftrojan_v1/"
    "reference_free_generator_best.pt",
    "reference_free_generator_best.pt",
    "dtd_reference_free_generator_ftrojan_v1",
)

print("Classifier:", CLASSIFIER_CHECKPOINT)
print("Generator:", GENERATOR_CHECKPOINT)
```

Verify the code and dataset:

```python
required = [
    Path("Absolute_Dataset/asb_dtd_v1/variant_manifest.jsonl"),
    Path("Absolute_Dataset/dtd/images"),
    Path("scripts/evaluate_dtd_generator_generalization.py"),
]
for path in required:
    print(path, "OK" if path.exists() else "MISSING")
assert all(path.exists() for path in required)
```

## 3. Run a three-condition smoke test

This checks one known trigger, one moved trigger, and one harmless corruption.
The tiny sample is only a software check and must not be reported as a result.

```python
!python -m scripts.evaluate_dtd_generator_generalization \
  --generator-checkpoint {GENERATOR_CHECKPOINT} \
  --classifier-checkpoint {CLASSIFIER_CHECKPOINT} \
  --output-dir Advanced_Outputs/dtd_reference_free_generalization_smoke \
  --suites strength,position,corruption \
  --scenario-names strength_100_known_positions,position_near_minus1_s100,gaussian_noise_001 \
  --batch-size 4 \
  --num-workers 2 \
  --max-template-batches 2 \
  --max-eval-batches 2 \
  --device cuda \
  --overwrite
```

## 4. Run all 19 conditions unattended

```python
!python -m scripts.evaluate_dtd_generator_generalization \
  --generator-checkpoint {GENERATOR_CHECKPOINT} \
  --classifier-checkpoint {CLASSIFIER_CHECKPOINT} \
  --output-dir Advanced_Outputs/dtd_reference_free_generalization_v1 \
  --suites strength,position,corruption \
  --batch-size 8 \
  --num-workers 2 \
  --support-fraction 0.01 \
  --device cuda
```

This command can be left running. Every scenario writes its own `summary.json`,
`predictions.jsonl`, and `panel.png`, followed by combined CSV, JSON, Markdown,
and chart files.

## 5. Print the important results

```python
import json

OUTPUT_DIR = Path("Advanced_Outputs/dtd_reference_free_generalization_v1")
summary = json.loads((OUTPUT_DIR / "generalization_summary.json").read_text())

print("TRIGGER GENERALIZATION")
print("scenario                         attack     generator  static     attack lift")
for row in summary["trigger_results"]:
    print(
        f"{row['scenario']:32s} "
        f"{100*row['triggered_asr']:8.2f}% "
        f"{100*row['generator_asr']:8.2f}% "
        f"{100*row['static_template_asr']:8.2f}% "
        f"{100*row['attack_target_rate_lift']:8.2f}%"
    )

print("\nBENIGN CORRUPTION CONTROLS")
print("scenario                         corrupt acc  generator acc  mean correction")
for row in summary["corruption_results"]:
    print(
        f"{row['scenario']:32s} "
        f"{100*row['corrupted_accuracy']:8.2f}% "
        f"{100*row['generator_accuracy']:8.2f}% "
        f"{row['mean_generator_correction']:.6f}"
    )
```

## 6. Display combined charts and selected panels

```python
from IPython.display import display
from PIL import Image

for filename in ("trigger_generalization.png", "corruption_controls.png"):
    path = OUTPUT_DIR / filename
    print(filename)
    display(Image.open(path))
```

```python
selected_panels = [
    OUTPUT_DIR / "trigger_scenarios/strength_050_known_positions/panel.png",
    OUTPUT_DIR / "trigger_scenarios/strength_100_known_positions/panel.png",
    OUTPUT_DIR / "trigger_scenarios/strength_150_known_positions/panel.png",
    OUTPUT_DIR / "trigger_scenarios/position_near_minus1_s100/panel.png",
    OUTPUT_DIR / "trigger_scenarios/position_far_cross_s150/panel.png",
    OUTPUT_DIR / "corruption_scenarios/gaussian_noise_003/panel.png",
    OUTPUT_DIR / "corruption_scenarios/gaussian_blur_5/panel.png",
]
for path in selected_panels:
    print(path)
    display(Image.open(path))
```

## 7. Interpret the experiment correctly

- First inspect `attack_target_rate_lift`. A moved trigger with little or no
  lift did not activate the backdoor, so it cannot demonstrate defense success.
- For an active attack, compare `triggered_asr` with `generator_asr`. A large
  reduction supports trigger generalization.
- Compare `generator_asr` with `static_template_asr`. The generator must beat or
  meaningfully differ from the fixed average correction to support an adaptive
  image-conditioned claim.
- For corruptions, the desired result is little accuracy loss and a small
  correction response. Strong correction of harmless noise or blur is a false
  positive and must be reported as a limitation.
- This experiment tests nearby generalization within FTrojan. It does not yet
  establish universal defense against every unknown trigger family or dataset.

## 8. Package only the full output

```python
!zip -r dtd_reference_free_generalization_v1.zip \
  Advanced_Outputs/dtd_reference_free_generalization_v1
```

Download `dtd_reference_free_generalization_v1.zip` and extract it locally into
`Advanced_Outputs` without changing the internal folder name.

## Optional laptop CPU run

The script needs no CUDA. From the repository root, this command runs a small
three-condition check:

```powershell
python -m scripts.evaluate_dtd_generator_generalization `
  --output-dir Advanced_Outputs/dtd_reference_free_generalization_cpu_smoke `
  --suites strength,position,corruption `
  --scenario-names strength_100_known_positions,position_near_minus1_s100,gaussian_noise_001 `
  --batch-size 2 `
  --num-workers 0 `
  --max-template-batches 1 `
  --max-eval-batches 1 `
  --device cpu `
  --overwrite
```

For a complete CPU evaluation of just the strength suite, remove the two
`--max-...` options and use `--suites strength`. It will work, but it will be
considerably slower than the T4 run.
