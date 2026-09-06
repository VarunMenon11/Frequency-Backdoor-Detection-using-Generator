# DTD Pretrained ResNet-18 Suspicious Classifier

> **Status:** The full run documented here has been completed. It established a
> useful pretrained clean-classification baseline, but its 10% poisoning budget
> was divided across three weak, unmatched triggers and did not produce a strong
> backdoor. Do not repeat this run. Continue with
> `notebooks/kaggle_dtd_attack_calibration.md` for the clean-control and
> single-trigger protocol.

## Purpose

This is the first classifier experiment in the advanced research track. It uses
an ImageNet-1K pretrained ResNet-18 instead of the project's small custom CNN.
The final classification layer is replaced with a 47-class DTD head.

This stage intentionally trains a suspicious, backdoored classifier. It does
not train the correction generator yet. Its purpose is to establish a stronger
classifier baseline and verify that the multi-trigger poisoning protocol can be
learned before designing the input-only defense.

All new results are isolated from earlier work:

~~~text
Advanced_Experiments/dtd_pretrained_resnet18_v1/
Advanced_Outputs/dtd_pretrained_resnet18_v1/
~~~

## Kaggle preparation

1. Create a new Kaggle notebook and enable a GPU accelerator.
2. Add the GitHub repository or upload its latest ZIP.
3. Add the prepared `Absolute_Dataset` folder as a private Kaggle dataset.
4. Locate the repository and dataset paths with:

~~~python
from pathlib import Path

print("Working folders:")
for path in Path("/kaggle/working").iterdir():
    print(path)

print("Input folders:")
for path in Path("/kaggle/input").iterdir():
    print(path)
~~~

The input must contain both of these folders:

~~~text
Absolute_Dataset/
+-- dtd/images/
+-- asb_dtd_v1/variant_manifest.jsonl
~~~

## Enter the repository

Change the path below if your repository folder has a different name:

~~~python
%cd /kaggle/working/Frequency-Backdoor-Detection-using-Generator
~~~

Do not reinstall PyTorch or torchvision in Kaggle unless importing them fails.
Kaggle normally provides a compatible GPU build. Reinstalling them can replace
CUDA libraries and break the notebook environment.

## Verify paths and GPU

Use this cell to locate the dataset automatically. It supports both an attached
Kaggle dataset below `/kaggle/input` and a copy committed inside the cloned
repository below `/kaggle/working`:

~~~python
from pathlib import Path
import torch

search_roots = [Path("/kaggle/working"), Path("/kaggle/input")]
manifest_matches = []
for search_root in search_roots:
    if search_root.is_dir():
        manifest_matches.extend(
            search_root.rglob("asb_dtd_v1/variant_manifest.jsonl")
        )

if not manifest_matches:
    raise RuntimeError(
        "Could not find asb_dtd_v1/variant_manifest.jsonl below "
        "/kaggle/working or /kaggle/input."
    )

# Prefer the copy inside the active cloned repository if more than one exists.
repository_name = "Frequency-Backdoor-Detection-using-Generator"
repository_matches = [
    path for path in manifest_matches if repository_name in path.parts
]
selected_manifest = (
    repository_matches[0] if repository_matches else manifest_matches[0]
)
DATA_ROOT = selected_manifest.parent.parent

assert (DATA_ROOT / "dtd/images").is_dir()
assert (DATA_ROOT / "asb_dtd_v1/variant_manifest.jsonl").is_file()
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("Dataset root:", DATA_ROOT)
print("Manifest:", selected_manifest)
~~~

## Smoke test

The smoke test verifies loading, triggering, training, evaluation, checkpoint
saving, and figure creation. It is not a scientific result.

~~~python
!python -m scripts.train_asb_dtd_pretrained_classifier \
  --manifest-dir "{DATA_ROOT}/asb_dtd_v1" \
  --images-root "{DATA_ROOT}/dtd/images" \
  --experiment-dir Advanced_Experiments/dtd_pretrained_resnet18_smoke \
  --output-dir Advanced_Outputs/dtd_pretrained_resnet18_smoke \
  --weights default \
  --epochs 1 \
  --freeze-epochs 1 \
  --batch-size 16 \
  --num-workers 2 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --num-panels 2 \
  --device cuda
~~~

The first use of `--weights default` may download the official torchvision
ResNet-18 ImageNet weights. Enable Kaggle Internet for that first run. If
Internet cannot be enabled, attach the weight file separately to Kaggle rather
than changing the experiment to random weights.

## Full training run

Run this only after the smoke test succeeds:

~~~python
!python -m scripts.train_asb_dtd_pretrained_classifier \
  --manifest-dir "{DATA_ROOT}/asb_dtd_v1" \
  --images-root "{DATA_ROOT}/dtd/images" \
  --experiment-dir Advanced_Experiments/dtd_pretrained_resnet18_v1 \
  --output-dir Advanced_Outputs/dtd_pretrained_resnet18_v1 \
  --backbone resnet18 \
  --weights default \
  --epochs 20 \
  --freeze-epochs 2 \
  --image-size 224 \
  --batch-size 32 \
  --num-workers 2 \
  --head-learning-rate 0.001 \
  --backbone-learning-rate 0.0001 \
  --weight-decay 0.0001 \
  --num-panels 5 \
  --seed 42 \
  --device cuda
~~~

The first two epochs train the replacement classification head while the
ImageNet feature extractor is frozen. From epoch 3 onward, the complete model
is fine-tuned using a smaller learning rate for pretrained layers.

The best checkpoint is selected using the harmonic mean of validation clean
accuracy and mean ASR over the three development triggers. This prevents a
checkpoint with good clean accuracy but an unlearned attack, or a high ASR but
collapsed clean classification, from being presented as a valid suspicious
baseline.

## Inspect and download results

~~~python
import json
from pathlib import Path

summary_path = Path("Advanced_Experiments/dtd_pretrained_resnet18_v1/final_evaluation.json")
summary = json.loads(summary_path.read_text())
print(json.dumps(summary, indent=2))
~~~

~~~python
!zip -r dtd_pretrained_resnet18_v1.zip \
  Advanced_Experiments/dtd_pretrained_resnet18_v1 \
  Advanced_Outputs/dtd_pretrained_resnet18_v1
~~~

Download `dtd_pretrained_resnet18_v1.zip` from the Kaggle working-directory
output and extract it into the same two folders in the local repository.

## How to interpret this run

- Clean test accuracy measures normal DTD texture classification.
- Seen-trigger ASR measures whether the three poisoning triggers were learned.
- Held-out-trigger ASR measures whether that attack behavior transfers to low,
  high, multi-band, or unseen Haar-HH triggers.
- Triggered clean-label accuracy measures how often the original class survives
  despite a trigger.
- The clean non-target target-prediction rate is a control. It reveals how often
  ordinary non-target images are predicted as the target even without a trigger.

High seen-trigger ASR is desirable only at this attack-construction stage. The
later defense experiments must reduce ASR while retaining clean accuracy.
