# DTD Final Generator Baseline Comparison

This is the final GPU evaluation in the reduced project plan. It performs no
training. It freezes the selected suspicious classifier and position-generalizing
generator, then compares the learned correction with simple alternatives on all
four FTrojan positions.

For every position it also automatically selects the clearest paper example:

```text
clean prediction is correct
        -> trigger forces the target class "banded"
        -> generator restores the original correct class
```

These panels are saved under `evidence_panels/`.

## 1. Open the repository and select a T4 GPU

```python
%cd /kaggle/working/Frequency-Backdoor-Detection-using-Generator
```

```python
import torch
print(torch.__version__)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
assert torch.cuda.is_available()
```

Do not reinstall PyTorch or torchvision.

## 2. Locate both checkpoints

Attach the previously downloaded result ZIP/folder as a private Kaggle Input.
The classifier may already be attached from the previous run.

```python
from pathlib import Path

def locate(filename, required_text):
    candidates = list(Path("/kaggle/input").rglob(filename))
    candidates += list(Path("/kaggle/working").rglob(filename))
    candidates += list(Path(".").rglob(filename))
    preferred = [p for p in candidates if required_text in str(p).lower()]
    candidates = preferred or candidates
    if not candidates:
        raise FileNotFoundError(f"Missing {filename}; attach its ZIP or folder as a Kaggle Input")
    print(filename, "->", candidates[0])
    return candidates[0].resolve()

CLASSIFIER_CHECKPOINT = locate(
    "suspicious_classifier_best_attack.pt", "position_attack"
)
GENERATOR_CHECKPOINT = locate(
    "reference_free_generator_best.pt", "position_generalizing"
)
```

## 3. Run the final comparison

```python
!python -u -m scripts.evaluate_dtd_position_generator_baselines \
  --classifier-checkpoint {CLASSIFIER_CHECKPOINT} \
  --generator-checkpoint {GENERATOR_CHECKPOINT} \
  --output-dir Advanced_Outputs/dtd_position_generator_final_baselines_v1 \
  --batch-size 8 \
  --num-workers 2 \
  --support-fraction 0.01 \
  --broad-cutoff 0.35 \
  --broad-attenuation 0.50 \
  --device cuda
```

The variants mean:

| Variant | Meaning |
|---|---|
| `clean` | Untouched clean image; natural target-rate reference. |
| `no_defense` | Triggered image sent directly to the classifier. |
| `full_generator` | Complete learned image-conditioned correction. |
| `static_template` | Same average correction applied to every image. |
| `broad_suppression` | Fixed attenuation of a broad high-frequency region. |
| `strongest_support_only` | Only the generator's strongest 1% corrections. |
| `outside_strongest_support` | Everything except that strongest 1%. |

## 4. Display and inspect the result

```python
from IPython.display import display, Markdown
from PIL import Image
import json

OUTPUT = Path("Advanced_Outputs/dtd_position_generator_final_baselines_v1")
display(Markdown((OUTPUT / "README.md").read_text()))
display(Image.open(OUTPUT / "final_baseline_comparison.png"))

for panel in sorted((OUTPUT / "evidence_panels").glob("*.png")):
    print("\n", panel.name)
    display(Image.open(panel))

summary = json.loads((OUTPUT / "final_baseline_summary.json").read_text())
print(json.dumps(summary["aggregate_held_out_positions"], indent=2))
```

## 5. Package the result

```python
!zip -r dtd_position_generator_final_baselines_v1.zip \
  Advanced_Outputs/dtd_position_generator_final_baselines_v1
```

Download `dtd_position_generator_final_baselines_v1.zip`. Once checked locally,
freeze this method and move to the presentation and paper. Do not tune parameters
using the locked DTD test split.
