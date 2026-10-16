# Kaggle Demo: Upload One Image and Run the Frozen DTD Generator

This notebook workflow creates a presentation-ready demonstration using the
final four-layout suspicious ResNet18 and the final position-generalizing
reference-free generator. It performs inference only. It does not train or
fine-tune either model.

## 1. Choose the correct demonstration mode

### Controlled research demo, recommended for the recorded video

Upload a clean DTD or DTD-like texture image. The script applies a selected
FTrojan-style trigger, shows that the suspicious classifier moves toward the
target class `banded`, passes only the triggered image to the generator, and
displays the corrected prediction.

This mode can show all three stages:

```text
clean -> triggered -> generator corrected
```

Use the held-out trigger `ftpos_near_mixed` so the video demonstrates a layout
that was not used to train or select the generator.

### Suspicious-image deployment demo

Upload an image that may already contain a trigger. The script sends that one
image directly to the generator. It does not inject a trigger and does not use
a clean counterpart.

This mode can show only:

```text
uploaded suspicious input -> generator corrected
```

Without a trusted label or clean counterpart, a changed prediction is not
proof that the corrected class is true. Present this mode as a data-flow demo,
not as measured recovery.

## 2. Kaggle requirements

Use a T4 GPU. Do not reinstall PyTorch or torchvision.

The repository does not commit trained checkpoints because
`Advanced_Experiments` is intentionally ignored. Create or reuse a private
Kaggle Dataset/Model containing these two files:

```text
suspicious_classifier_best_attack.pt
reference_free_generator_best.pt
```

Their local source paths are:

```text
Advanced_Experiments/dtd_position_attack_selected_v1/
  dtd_position_attack_selected_v1/suspicious_classifier_best_attack.pt

Advanced_Experiments/dtd_position_generalizing_generator_v1/
  reference_free_generator_best.pt
```

Add that model package to the Kaggle notebook through **Add Input**.

## 3. Enter the repository

```python
%cd /kaggle/working/Frequency-Backdoor-Detection-using-Generator
```

```python
from pathlib import Path
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
assert torch.cuda.is_available(), "Enable a T4 GPU before running the demo."
```

## 4. Locate the two uploaded checkpoints

```python
from pathlib import Path

def find_checkpoint(filename):
    roots = [Path("/kaggle/input"), Path("Advanced_Experiments")]
    matches = []
    for root in roots:
        if root.exists():
            matches.extend(root.rglob(filename))
    matches = sorted(set(matches))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename}. Add the two-checkpoint Kaggle input."
        )
    print(filename, "candidates:")
    for path in matches:
        print("  ", path)
    return matches[0]

CLASSIFIER_CKPT = find_checkpoint("suspicious_classifier_best_attack.pt")
GENERATOR_CKPT = find_checkpoint("reference_free_generator_best.pt")

print("\nSelected classifier:", CLASSIFIER_CKPT)
print("Selected generator:", GENERATOR_CKPT)
```

If more than one old checkpoint appears, manually assign the two variables to
the paths belonging to the position-attack and position-generalizing runs.

## 5. Upload an image inside the notebook

```python
import ipywidgets as widgets
from IPython.display import display

uploader = widgets.FileUpload(
    accept="image/png,image/jpeg",
    multiple=False,
    description="Upload image",
)
display(uploader)
```

After selecting the image, save it into the Kaggle working directory:

```python
from pathlib import Path

def save_widget_upload(widget, output_dir=Path("/kaggle/working/demo_uploads")):
    output_dir.mkdir(parents=True, exist_ok=True)
    value = widget.value
    if not value:
        raise RuntimeError("Select an image using the upload button first.")

    # ipywidgets 7 returns a dictionary; ipywidgets 8 returns a tuple.
    if isinstance(value, dict):
        name, item = next(iter(value.items()))
        content = item["content"]
    else:
        item = value[0]
        name = item["name"]
        content = item["content"]

    path = output_dir / Path(name).name
    path.write_bytes(bytes(content))
    return path

UPLOADED_IMAGE = save_widget_upload(uploader)
print("Saved upload:", UPLOADED_IMAGE)
```

If Kaggle does not display the widget, upload the image through the notebook
sidebar and set `UPLOADED_IMAGE` directly to its `/kaggle/input/...` path.

## 6. Run the controlled held-out-trigger demo

This is the recommended video demonstration. The image is treated as clean and
the held-out `ftpos_near_mixed` trigger is injected only for the controlled
showcase.

```python
import subprocess
import sys
from pathlib import Path

CONTROLLED_OUTPUT = Path("Advanced_Outputs/dtd_uploaded_image_demo/controlled")

command = [
    sys.executable,
    "-m", "scripts.demo_dtd_reference_free_inference",
    "--image", str(UPLOADED_IMAGE),
    "--mode", "controlled",
    "--trigger-name", "ftpos_near_mixed",
    "--classifier-checkpoint", str(CLASSIFIER_CKPT),
    "--generator-checkpoint", str(GENERATOR_CKPT),
    "--output-dir", str(CONTROLLED_OUTPUT),
    "--device", "cuda",
]

subprocess.run(command, check=True)
```

Display the result:

```python
from IPython.display import display
from PIL import Image

display(Image.open(CONTROLLED_OUTPUT / "demo_panel.png"))
```

Read the machine-readable report:

```python
import json

report = json.loads((CONTROLLED_OUTPUT / "demo_report.json").read_text())
print(json.dumps(report, indent=2))
```

For an ideal video example, confirm:

```text
Attack activated: True
Recovered to reference class: True
```

If the uploaded external image is not classified meaningfully by the DTD
classifier, choose an image from a non-target DTD folder instead. The generator
has approximately 60% corrected true-label accuracy, so an arbitrary upload is
not guaranteed to produce a perfect recovery example.

### Verified fallback image

This tracked validation image was verified locally with the final checkpoints:

```python
UPLOADED_IMAGE = Path(
    "Absolute_Dataset/dtd/images/blotchy/blotchy_0025.jpg"
)
```

Using the held-out `ftpos_near_mixed` layout, it produced:

```text
Clean prediction:      blotchy, 61.91%
Triggered prediction:  banded, 99.98%
Corrected prediction:  blotchy, 66.56%
Attack activated:      True
Recovery successful:   True
```

Keep this sample ready as a fallback before recording. You may still begin the
video with the upload widget, but first test the selected image off-camera.

## 7. Optional: provide a known DTD true class

When you know that the uploaded image belongs to one of the 47 DTD classes, add:

```python
command.extend(["--true-class", "striped"])
```

Use only the real known class. Do not select a class merely to make the output
look successful.

When the image is loaded directly from a tracked DTD class folder, the script
automatically infers the class from the parent folder name.

## 8. Run the one-image deployment-path demo

Use this when the uploaded file is already suspicious or triggered. No trigger
is added:

```python
SUSPICIOUS_OUTPUT = Path("Advanced_Outputs/dtd_uploaded_image_demo/suspicious")

command = [
    sys.executable,
    "-m", "scripts.demo_dtd_reference_free_inference",
    "--image", str(UPLOADED_IMAGE),
    "--mode", "suspicious",
    "--classifier-checkpoint", str(CLASSIFIER_CKPT),
    "--generator-checkpoint", str(GENERATOR_CKPT),
    "--output-dir", str(SUSPICIOUS_OUTPUT),
    "--device", "cuda",
]

subprocess.run(command, check=True)
display(Image.open(SUSPICIOUS_OUTPUT / "demo_panel.png"))
```

The correct narration is:

> In deployment, this uploaded image is the only input. The generator extracts
> spectral evidence, predicts a selective correction, preserves the incoming
> phase, reconstructs the image and sends it to the frozen classifier. No clean
> counterpart, trigger name or online fine-tuning is used.

Do not say that the corrected class is guaranteed to be true unless a trusted
label is available.

### Demonstrate an actual triggered-image upload

Controlled mode saves the generated attack image as:

```text
Advanced_Outputs/dtd_uploaded_image_demo/controlled/triggered_image.png
```

Download that file, clear or recreate the upload widget, and upload
`triggered_image.png`. Then run suspicious mode. This visibly demonstrates the
real interface: the generator receives the already-triggered image and does not
construct another trigger or use the clean source.

The verified local round-trip produced:

```text
Uploaded triggered input: banded, 62.50%
Generator corrected:      blotchy, 85.86%
```

The confidence differs from controlled mode because saving, reloading and
applying the standard input resize/crop changes the pixel grid slightly. The
important point is that suspicious mode uses only the uploaded file.

## 9. Reliable video sequence

Record the following sequence:

1. Briefly show the two frozen checkpoints.
2. Upload one non-target DTD image.
3. Explain that `ftpos_near_mixed` was held out from generator training.
4. Run the controlled demo cell.
5. Read the three prediction titles: clean, triggered and corrected.
6. Point to the trigger difference, amplitude panels and applied correction.
7. Open `demo_report.json` and show that no training occurred.
8. Optionally run suspicious mode to demonstrate the real one-image interface.

For the strongest video, use two short chapters:

1. **Controlled proof:** show the clean source, injected held-out attack and
   successful recovery.
2. **Deployment interface:** upload the saved triggered image and show that the
   generator processes it without a clean reference.

Suggested narration:

> The left image is the uploaded clean texture. For this controlled demo I add
> a held-out FTrojan-style layout, causing the suspicious ResNet18 to predict
> the attacker target `banded`. The generator receives only that triggered
> image. It estimates an image-dependent log-amplitude correction, preserves
> phase and reconstructs the corrected input. The final panel shows the new
> prediction and the spectral correction actually applied. Both models remain
> frozen throughout the demonstration.

## 10. Files produced

Each run saves:

```text
demo_panel.png
demo_report.json
input_image.png
corrected_image.png
```

Controlled mode additionally saves:

```text
clean_image.png
triggered_image.png
```

Download the complete `Advanced_Outputs/dtd_uploaded_image_demo` directory
after recording so the demonstration remains reproducible.
