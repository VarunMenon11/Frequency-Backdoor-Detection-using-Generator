# Kaggle: DTD Reference-Free Spectral Generator

This notebook trains the first advanced defense against the selected DTD
FTrojan model. The suspicious ResNet18 is frozen. During both training and
deployment, the generator receives one image only. The aligned clean image is
used only to calculate supervised training losses and validation measurements.

The run is validation-only. Do not use the locked DTD test split to tune this
generator. We will run one final test after choosing the defense checkpoint.

## 1. Kaggle settings

Select a **T4 GPU**. Do not reinstall PyTorch or torchvision in Kaggle; replacing
Kaggle's matched packages can break CUDA compatibility.

## 2. Enter the repository

```python
%cd /kaggle/working/Frequency-Backdoor-Detection-using-Generator
```

Confirm that the code and dataset exist, then locate the trained suspicious
classifier. Large checkpoints are commonly excluded from Git, so the model may
need to be attached separately as a Kaggle Input.

```python
from pathlib import Path

required = [
    Path("Absolute_Dataset/asb_dtd_v1/variant_manifest.jsonl"),
    Path("Absolute_Dataset/dtd/images"),
    Path("scripts/train_dtd_reference_free_generator.py"),
]
for path in required:
    print(path, "OK" if path.exists() else "MISSING")
assert all(path.exists() for path in required)

checkpoint_name = "suspicious_classifier_best_attack.pt"
repository_checkpoint = Path(
    "Advanced_Experiments/dtd_attack_sweep_v1/"
    "ftrojan_m100_paired_r020/suspicious_classifier_best_attack.pt"
)

if repository_checkpoint.is_file():
    CLASSIFIER_CHECKPOINT = repository_checkpoint.resolve()
else:
    input_candidates = sorted(Path("/kaggle/input").rglob(checkpoint_name))
    matching_candidates = [
        path for path in input_candidates
        if "ftrojan_m100_paired_r020" in str(path)
    ]
    candidates = matching_candidates or input_candidates
    if not candidates:
        raise FileNotFoundError(
            "The trained suspicious classifier is missing. Attach the previous "
            "DTD attack experiment ZIP/folder as a Kaggle Input, then rerun this cell."
        )
    if len(candidates) > 1:
        print("Checkpoint candidates:")
        for candidate in candidates:
            print(" -", candidate)
        print("Using the first preferred match. Verify that it is the m100/r020 model.")
    CLASSIFIER_CHECKPOINT = candidates[0]

print("Classifier checkpoint:", CLASSIFIER_CHECKPOINT)
print("Checkpoint size (MB):", round(CLASSIFIER_CHECKPOINT.stat().st_size / 1024**2, 2))
```

If the cell reports that the checkpoint is missing, create or attach a Kaggle
dataset containing this local file:

```text
Advanced_Experiments/dtd_attack_sweep_v1/ftrojan_m100_paired_r020/
suspicious_classifier_best_attack.pt
```

In Kaggle, select **Add Input**, attach that model dataset, and rerun the cell.
There is no need to copy the checkpoint into `/kaggle/working`; the training
script can read it directly from `/kaggle/input`.

## 3. Confirm the GPU

```python
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
```

## 4. Smoke test

This verifies loading, forward/backward propagation, evaluation, checkpointing,
and figure generation. It is not a scientific result.

```python
!python -m scripts.train_dtd_reference_free_generator \
  --classifier-checkpoint {CLASSIFIER_CHECKPOINT} \
  --experiment-dir Advanced_Experiments/dtd_reference_free_generator_smoke \
  --output-dir Advanced_Outputs/dtd_reference_free_generator_smoke \
  --epochs 1 \
  --batch-size 4 \
  --num-workers 2 \
  --max-train-batches 2 \
  --max-eval-batches 2 \
  --num-panels 1 \
  --max-clean-accuracy-drop 1.0 \
  --device cuda
```

The final lines should say `Selected epoch: 1` and show saved generator and
validation-evidence paths. Partial smoke metrics must not be reported as final
results.

## 5. Full validation training

Run this in a fresh result directory. The initial recommended run is 20 epochs.
Batch size 8 is conservative for a 16 GB T4 because the frozen classifier and
two generator branches participate in each training step.

```python
!python -m scripts.train_dtd_reference_free_generator \
  --classifier-checkpoint {CLASSIFIER_CHECKPOINT} \
  --experiment-dir Advanced_Experiments/dtd_reference_free_generator_ftrojan_v1 \
  --output-dir Advanced_Outputs/dtd_reference_free_generator_ftrojan_v1 \
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
  --num-panels 4 \
  --device cuda
```

Read the selected validation result:

```python
import json
from pathlib import Path

summary_path = Path("Advanced_Outputs/dtd_reference_free_generator_ftrojan_v1/validation_summary.json")
summary = json.loads(summary_path.read_text())
print(json.dumps(summary, indent=2))
```

The important comparison is:

- `suspicious_asr` versus `corrected_asr`: lower after correction is better.
- `suspicious_clean_accuracy` versus `generator_clean_accuracy`: these should
  remain close; the selection rule permits at most a 5 percentage-point drop.
- `corrected_label_accuracy`: higher means correction restored the original
  texture label, rather than merely moving predictions away from the target.
- `mean_absolute_effective_log_correction`: confirms how much spectral change
  was actually applied.

Display the paper-style validation panels:

```python
from IPython.display import display
from PIL import Image

for path in sorted(Path("Advanced_Outputs/dtd_reference_free_generator_ftrojan_v1").glob("validation_panel_*.png")):
    print(path.name)
    display(Image.open(path))
```

To display just one result clearly in Kaggle, run this cell:

```python
from pathlib import Path
from IPython.display import display
from PIL import Image

output_dir = Path("Advanced_Outputs/dtd_reference_free_generator_ftrojan_v1")
panel_paths = sorted(output_dir.glob("validation_panel_*.png"))

if not panel_paths:
    raise FileNotFoundError(
        "No validation panels were found. Complete the full training cell first."
    )

selected_panel = panel_paths[0]
print("Displaying:", selected_panel)
display(Image.open(selected_panel))
```

To display another saved example, change `panel_number` to 2, 3, or 4:

```python
panel_number = 2
selected_panel = output_dir / f"validation_panel_{panel_number:02d}.png"

if not selected_panel.exists():
    raise FileNotFoundError(f"Panel does not exist: {selected_panel}")

print("Displaying:", selected_panel)
display(Image.open(selected_panel))
```

Each displayed panel contains the clean image, triggered image, corrected
image, amplified pixel difference, clean amplitude, triggered amplitude,
corrected amplitude, and the correction actually applied by the generator.
The clean image is included only as evaluation evidence; it was not supplied to
the generator when the corrected image was produced.

## 6. Package only the new experiment

```python
!zip -r dtd_reference_free_generator_ftrojan_v1.zip \
  Advanced_Experiments/dtd_reference_free_generator_ftrojan_v1 \
  Advanced_Outputs/dtd_reference_free_generator_ftrojan_v1
```

Download `dtd_reference_free_generator_ftrojan_v1.zip` from Kaggle's Output
panel. Do not include the smoke-test directory in the research record.

## 7. What happens after download

After the archive is extracted into the same `Advanced_Experiments` and
`Advanced_Outputs` locations locally, inspect the selected metrics and panels.
Only if validation shows meaningful ASR reduction with acceptable clean utility
will we freeze that checkpoint and run the locked final test. The next research
stage then measures unseen strengths, shifted DCT positions, unrelated trigger
families, and common image corruption/noise.
