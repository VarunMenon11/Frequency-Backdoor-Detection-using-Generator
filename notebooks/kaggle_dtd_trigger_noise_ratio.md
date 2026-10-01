# DTD Trigger-to-Noise Ratio Experiment

This is the final planned robustness experiment. It freezes the suspicious
classifier and position-generalizing generator. No training occurs.

The experiment measures:

```text
TNR(dB) = 20 log10(trigger RMS / noise RMS)
```

- `+20 dB`: trigger RMS is 10 times noise RMS.
- `+10 dB`: trigger RMS is about 3.16 times noise RMS.
- `0 dB`: trigger and noise have equal RMS.
- `-10 dB`: noise RMS is about 3.16 times trigger RMS.

It also runs trigger-only and noise-only controls on all four FTrojan positions.

## 1. Enter the repository and verify the GPU

```python
%cd /kaggle/working/Frequency-Backdoor-Detection-using-Generator
```

```python
import torch
print("Torch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
assert torch.cuda.is_available()
```

Do not reinstall PyTorch or torchvision.

## 2. Locate the frozen checkpoints

Keep the same private Kaggle Inputs attached as in the final-baseline run.

```python
from pathlib import Path

def locate(filename, required_text):
    candidates = list(Path("/kaggle/input").rglob(filename))
    candidates += list(Path("/kaggle/working").rglob(filename))
    candidates += list(Path(".").rglob(filename))
    preferred = [path for path in candidates if required_text in str(path).lower()]
    candidates = preferred or candidates
    if not candidates:
        raise FileNotFoundError(f"Missing {filename}")
    print(filename, "->", candidates[0])
    return candidates[0].resolve()

CLASSIFIER_CHECKPOINT = locate(
    "suspicious_classifier_best_attack.pt", "position_attack"
)
GENERATOR_CHECKPOINT = locate(
    "reference_free_generator_best.pt", "position_generalizing"
)
```

## 3. Run the full TNR study

```python
!python -u -m scripts.evaluate_dtd_trigger_noise_ratio \
  --classifier-checkpoint {CLASSIFIER_CHECKPOINT} \
  --generator-checkpoint {GENERATOR_CHECKPOINT} \
  --output-dir Advanced_Outputs/dtd_trigger_noise_ratio_v1 \
  --tnr-db 20,10,0,-10 \
  --panel-trigger ftpos_near_mixed \
  --batch-size 8 \
  --num-workers 2 \
  --seed 42 \
  --device cuda
```

This evaluates 20 conditions: four trigger positions multiplied by trigger-only
plus four finite TNR levels. Both checkpoints remain unchanged.

## 4. Display the results

```python
from IPython.display import display, Markdown
from PIL import Image
import json

OUTPUT = Path("Advanced_Outputs/dtd_trigger_noise_ratio_v1")
display(Markdown((OUTPUT / "README.md").read_text()))
display(Image.open(OUTPUT / "trigger_noise_ratio_chart.png"))

for panel in sorted((OUTPUT / "evidence_panels").glob("*.png")):
    print("\n", panel.name)
    display(Image.open(panel))

summary = json.loads((OUTPUT / "trigger_noise_ratio_summary.json").read_text())
print(json.dumps(summary["aggregate"], indent=2))
```

## 5. Interpretation

For each TNR, first inspect undefended ASR. If noise itself destroys the attack,
a low corrected ASR cannot be credited entirely to the generator. The strongest
result is a condition where undefended ASR stays high, corrected ASR stays near
the clean target-rate baseline, and the generator does not reduce noise-only
accuracy unnecessarily.

## 6. Package the output

```python
!zip -r dtd_trigger_noise_ratio_v1.zip \
  Advanced_Outputs/dtd_trigger_noise_ratio_v1
```

Download `dtd_trigger_noise_ratio_v1.zip`. After this result is inspected and
documented, freeze the experimental method and move to the PPT and IEEE paper.
