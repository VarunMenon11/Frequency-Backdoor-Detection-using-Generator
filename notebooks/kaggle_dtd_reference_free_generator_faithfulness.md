# Kaggle: Reference-Free Generator Faithfulness Audit

This notebook does **not** train another model. It audits the selected DTD
reference-free generator on the validation split and asks whether its correction
map is meaningful, causal, image-conditioned, or replaceable by a fixed template.

The locked DTD test split remains untouched.

## 1. Use a T4 GPU and enter the repository

```python
%cd /kaggle/working/Frequency-Backdoor-Detection-using-Generator
```

Do not reinstall PyTorch or torchvision.

## 2. Locate both selected checkpoints

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
print("Classifier MB:", round(CLASSIFIER_CHECKPOINT.stat().st_size / 1024**2, 2))
print("Generator MB:", round(GENERATOR_CHECKPOINT.stat().st_size / 1024**2, 2))
```

Also verify the dataset and script:

```python
required = [
    Path("Absolute_Dataset/asb_dtd_v1/variant_manifest.jsonl"),
    Path("Absolute_Dataset/dtd/images"),
    Path("scripts/audit_dtd_reference_free_generator.py"),
]
for path in required:
    print(path, "OK" if path.exists() else "MISSING")
assert all(path.exists() for path in required)
```

## 3. Smoke audit

This verifies all ablation paths and figures. Its tiny sample is not a research
result.

```python
!python -m scripts.audit_dtd_reference_free_generator \
  --generator-checkpoint {GENERATOR_CHECKPOINT} \
  --classifier-checkpoint {CLASSIFIER_CHECKPOINT} \
  --output-dir Advanced_Outputs/dtd_reference_free_generator_faithfulness_smoke \
  --batch-size 4 \
  --num-workers 2 \
  --support-fraction 0.01 \
  --max-template-batches 2 \
  --max-eval-batches 2 \
  --device cuda \
  --overwrite
```

## 4. Full validation audit

The static template is built from clean-training sources with the configured
trigger. Every causal variant is then evaluated on all 1,840 non-target
validation sources.

```python
!python -m scripts.audit_dtd_reference_free_generator \
  --generator-checkpoint {GENERATOR_CHECKPOINT} \
  --classifier-checkpoint {CLASSIFIER_CHECKPOINT} \
  --output-dir Advanced_Outputs/dtd_reference_free_generator_faithfulness_v1 \
  --batch-size 8 \
  --num-workers 2 \
  --support-fraction 0.01 \
  --device cuda
```

## 5. Read the results

```python
import json

output_dir = Path("Advanced_Outputs/dtd_reference_free_generator_faithfulness_v1")
summary = json.loads((output_dir / "faithfulness_summary.json").read_text())

print("Samples:", summary["num_non_target_validation_samples"])
for name, metrics in summary["variants"].items():
    print(
        f"{name:24s} | target rate {100*metrics['target_rate']:6.2f}% "
        f"| true-label accuracy {100*metrics['true_label_accuracy']:6.2f}%"
    )

print("\nFaithfulness:")
for name, values in summary["faithfulness"].items():
    print(f"{name:45s} median={values['median']:.6f} mean={values['mean']:.6f}")
```

## 6. Display the visual evidence

```python
from IPython.display import display
from PIL import Image

for filename in (
    "faithfulness_explanation_panel.png",
    "asr_ablation_chart.png",
    "faithfulness_distributions.png",
):
    path = output_dir / filename
    print(filename)
    display(Image.open(path))
```

## 7. How to interpret the decisive comparisons

- `predicted_top_only` should retain much of the full generator's ASR reduction
  if the visualized strongest correction locations are causally important.
- `predicted_outside_only` should perform substantially worse if those selected
  locations matter.
- `oracle_overlap_only` uses paired clean-trigger evidence for evaluation only.
  It tests whether correction overlapping known image changes is effective.
- `static_template` reveals whether one fixed average correction can replace the
  image-conditioned generator. Similar performance would limit the adaptivity
  claim; a clear generator advantage would support it.
- Triggered correction magnitude should differ meaningfully from clean-image
  correction magnitude. Similar responses suggest indiscriminate filtering.

No single overlap score proves that a frequency is malicious. Interpret the
overlap, causal ablations, clean response and static baseline together.

## 8. Package the audit

```python
!zip -r dtd_reference_free_generator_faithfulness_v1.zip \
  Advanced_Outputs/dtd_reference_free_generator_faithfulness_v1
```

Download `dtd_reference_free_generator_faithfulness_v1.zip` and extract it into
the local `Advanced_Outputs` directory without changing its folder name.
