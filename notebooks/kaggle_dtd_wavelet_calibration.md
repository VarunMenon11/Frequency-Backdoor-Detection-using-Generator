# DTD: One Wavelet Trigger Before Mixing Triggers

## What we are testing

We will train three independent poisoning runs using the same Haar-LH trigger
at strengths 0.25, 0.50 and 1.00. Each run contains 1,692 clean training sources
and 188 poisoned sources, not three triggers mixed into one model. We keep
the existing DTD manifest, official split, 224-pixel input, pretrained ResNet-18,
seed 42, target `banded`, and 10% poisoning budget fixed.

These strengths form a doubling pilot around the existing 0.25 setting. They
are empirical candidates, not theoretically optimal or guaranteed successful.
In this implementation, alpha multiplies the per-image, per-channel standard
deviation of the selected wavelet subband. Alpha 1.00 is not a one-unit pixel
addition. Compare actual image distortion, not just alpha across different
trigger families.

The implemented rule is:

```text
coefficients = Haar(clean image)
LH_triggered = LH_clean + alpha * max(std(LH_clean), 1/255) * pattern
triggered image = clip(inverse_Haar(updated coefficients), 0, 1)
```

Here the pattern is a directional sinusoid with the catalog frequency 3.0 and
angle 0.0, injected into LH wavelet coefficients. This is a controlled custom
Haar trigger, not a claim to reproduce a named published attack. Merely using
wavelets does not make it a sophisticated or successful attack. In the code's
convention LH captures left-right differences within each 2x2 block; do not
assume another library uses identical subband names. LL, HL and HH coefficients
are left unchanged before reconstruction and clipping. A wavelet modification
can change both the Fourier amplitude and phase of the reconstructed image.

## Why we are not discarding multiple triggers

Multiple triggers can coexist in a trained model; see the authors' paper
[Shortcuts Everywhere and Nowhere: Exploring Multi-Trigger Backdoor Attacks](https://arxiv.org/abs/2401.15295).
Our unsuccessful mixed-trigger run does not disprove this. The subsequent
single-Fourier-trigger run was also weak. That means we must investigate trigger
learnability and measurement before blaming mixture alone.

First qualify a single trigger. Then qualify a second independently. Only
then compare a mixture at a fixed total poisoning budget. For example, 10%
total split between two triggers is 5% each, not 10% each. A 20%-total experiment
is a different poisoning-budget condition and must be labelled separately.

## 1. Kaggle setup

Use a T4 GPU with the working PyTorch environment from the previous run. Update
the repository first. Do not reinstall a working CUDA stack or rerun the clean
control. Run these notebook cells from the repository root:

```python
%cd /kaggle/working/Frequency-Backdoor-Detection-using-Generator
```

```python
from pathlib import Path
import json
import subprocess
import sys
import torch

DATA_ROOT = Path.cwd() / "Absolute_Dataset"
IMAGES_ROOT = DATA_ROOT / "dtd/images"
PREPARED_ROOT = DATA_ROOT / "dtd_wavelet_single_v1"
SOURCE_MANIFEST_DIR = DATA_ROOT / "asb_dtd_v1"
assert (SOURCE_MANIFEST_DIR / "variant_manifest.jsonl").is_file()
assert IMAGES_ROOT.is_dir()
assert torch.cuda.is_available(), "Enable the GPU accelerator first"
print(torch.cuda.get_device_name(0))

STRENGTHS = [0.25, 0.50, 1.00]
RUN_GROUP = "dtd_haar_lh_calibration_v1"
def tag(alpha):
    return f"s{round(alpha * 100):03d}"
```

If `PREPARED_ROOT` is not present after cloning/uploading, recreate it on the
Kaggle CPU from the verified original DTD sources. This performs no model
training:

```python
if not PREPARED_ROOT.is_dir():
    subprocess.run([
        sys.executable, "-u", "-m", "scripts.prepare_dtd_wavelet_dataset",
        "--manifest-dir", str(SOURCE_MANIFEST_DIR),
        "--images-root", str(IMAGES_ROOT),
        "--output-dir", str(PREPARED_ROOT),
    ], check=True)
for alpha in STRENGTHS:
    assert (PREPARED_ROOT / tag(alpha) / "variant_manifest.jsonl").is_file()
```

The existing dataset is sufficient. The manifest records source paths and
trigger recipes; the loader generates triggered tensors after cropping. We
do not need another dataset download or thousands of lossy JPEG copies.
Each run records its exact configuration and poisoning counts. The same seed
and source order give the same selected poisoned sources across strengths.

## 2. Preview all three strengths on validation images

```python
for alpha in STRENGTHS:
    subprocess.run([
        sys.executable, "-m", "scripts.visualize_asb_dtd_trigger_spectra",
        "--manifest-dir", str(PREPARED_ROOT / tag(alpha)),
        "--images-root", str(IMAGES_ROOT),
        "--output-dir", f"Advanced_Outputs/{RUN_GROUP}/preview_{tag(alpha)}",
        "--triggers", "haar_lh",
        "--strength-override", str(alpha),
        "--split", "validation", "--sampling", "balanced", "--seed", "42",
        "--metric-samples", "184", "--image-size", "224",
    ], check=True)
```

This samples four sources from each of 46 non-target classes for distortion
metrics, rather than just taking the first few class-sorted images. The same
sources are used at all strengths. Each figure still shows one example;
`--sample-index` can select another source. Save additional examples to a new
folder. JSON files record the metric source IDs, split and exact trigger.

```python
from IPython.display import Image, display
for alpha in STRENGTHS:
    display(Image(filename=f"Advanced_Outputs/{RUN_GROUP}/preview_{tag(alpha)}/haar_lh_spectrum_panel.png"))
```

Inspect original and triggered images as well as residuals and Fourier plots.
Some plots auto-scale differences, so brightness alone is not a quantitative
comparison across figures. Use their colorbars and recorded MAE/PSNR. Phase
differences in very low-amplitude bins can be unstable; they are not automatic
evidence of semantic damage. The existing metric named `clipped_pixel_fraction`
counts output boundary pixels, including pre-existing zeros/ones; it is not
the exact fraction newly clipped by injection.

## 3. Train the three independent attack candidates

Do not add generator training yet. This loop trains three classifiers, not
three additional phases of one classifier. Each starts from the same pretrained
initialization and saves its own best and last checkpoint.

```python
for alpha in STRENGTHS:
    name = tag(alpha)
    subprocess.run([
        sys.executable, "-u", "-m", "scripts.train_asb_dtd_pretrained_classifier",
        "--manifest-dir", str(PREPARED_ROOT / name),
        "--images-root", str(IMAGES_ROOT),
        "--experiment-dir", f"Advanced_Experiments/{RUN_GROUP}/{name}",
        "--output-dir", f"Advanced_Outputs/{RUN_GROUP}/{name}",
        "--weights", "default", "--attack-triggers", "haar_lh",
        "--trigger-strength", str(alpha), "--poison-ratio", "0.10",
        "--epochs", "30", "--freeze-epochs", "2",
        "--image-size", "224", "--batch-size", "32", "--num-workers", "2",
        "--seed", "42", "--validation-only", "--device", "cuda",
    ], check=True)
```

Thirty epochs preserve comparability with the preceding runs; they are not a
guarantee of sufficient learning. Checkpoint selection uses maximum validation
clean accuracy with the earliest tie retained. The paired attack measurements
qualify the selected model separately. Existing result folders are protected:
use a new run-group name for reruns. `check=True` stops on a failed command so
an incomplete run is not silently reported as successful.

## 4. Read the results without test-set selection

```python
print("alpha | val clean | clean target | ASR | net lift pp | conditional ASR")
for alpha in STRENGTHS:
    path = Path(f"Advanced_Outputs/{RUN_GROUP}/{tag(alpha)}/validation_evaluation.json")
    result = json.loads(path.read_text())
    m = result["validation_asr_by_trigger"]["haar_lh"]
    conditional = m["conditional_asr_clean_correct"]
    conditional_text = "undefined" if conditional is None else f"{100*conditional:.2f}%"
    print(f"{alpha:.2f} | {100*result['validation_clean']['accuracy']:.2f}% | "
          f"{100*m['clean_non_target_target_rate']:.2f}% | {100*m['asr']:.2f}% | "
          f"{100*m['same_model_target_rate_lift']:.2f} | {conditional_text}")
```

Report all candidates, including unsuccessful ones. Look for substantial
trigger-induced correct-to-target failures, high raw ASR, and acceptable clean
utility and distortion together. Low post-defense ASR would not be persuasive
if the initial attack barely works. If none qualify, revisit the trigger design
or training diagnostics; do not automatically declare the largest alpha best.
Repeat a selected configuration across independent seeds before strong claims.

## 5. Download the complete record

```python
import zipfile
archive = Path(f"{RUN_GROUP}.zip")
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for root in [Path("Advanced_Experiments") / RUN_GROUP,
                 Path("Advanced_Outputs") / RUN_GROUP]:
        for path in root.rglob("*"):
            if path.is_file():
                z.write(path, arcname=path.as_posix())
print(archive.resolve())
```

Extract the archive into the local repository root. These paths remain
separate from all earlier Fourier, DTD and legacy experiments.

The locally prepared dataset archive does not duplicate the 5,640 original DTD
images. Kaggle therefore needs both `Absolute_Dataset/dtd/images` and
`Absolute_Dataset/dtd_wavelet_single_v1`. If only the original DTD data is
present, the preparation cell above reconstructs the latter deterministically.

## 6. What "unseen image" will mean in our next defense stage

There are three different claims, and none follows from raising training ASR:

1. **New source, known texture class and known trigger.** Test frozen models on
   images that were never used in training or calibration. DTD's held-out
   sources provide an initial test, but the flagged cross-split duplicates need
   an explicit deduplication/sensitivity protocol for strong conclusions.
2. **External image, overlapping texture class.** Collect an independently
   sourced, labelled external set with explicit mapping to the DTD classes.
   Deduplicate it against all development images, freeze all settings, and
   report clean accuracy and paired triggered performance separately. An
   arbitrary texture may not have an unambiguous DTD label.
3. **Trigger unseen by the defense.** Train an attacked classifier that actually
   responds to the held-out trigger, but exclude that trigger from defense
   training and tuning. Otherwise low ASR may simply mean the attack model
   never learned the trigger. Unknown classes are a separate open-set problem;
   a 47-class head will still output one of its 47 classes without a separately
   validated rejection mechanism.

For all these tests, the eventual deployed generator must accept only the
incoming image or its spectrum, not its corresponding clean reference or
true label. Our earlier correction equation explicitly requires clean
amplitude and therefore does not meet this requirement. A new input-only
generator needs to predict a bounded amplitude correction directly; clean
images and labels can supervise training but cannot be supplied at inference.
Matched clean images may be held by the evaluator for scoring, never passed
to the model being tested. Predictions from a compromised classifier are not
a reliable "correct/incorrect" oracle on an unlabelled external image.

The defense study should compare the input-only generator with identity/no
correction, simple filtering, clean-only fine-tuning, repair without generator,
and the old paired-reference method as an oracle-assisted baseline. Include
clean noisy/corrupted images to measure unnecessary correction. This is how
we test the contribution beyond absolute differences rather than assume it.
These are planned experiments, not capabilities implemented by this notebook.
