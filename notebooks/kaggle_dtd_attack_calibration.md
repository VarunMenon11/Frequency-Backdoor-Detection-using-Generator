# DTD Pretrained Attack Calibration

## Why this rerun is required

The first pretrained mixed-trigger experiment used a total poisoning budget of
10%. That budget was divided across Fourier-middle, Haar-LH, and Haar-HL, so
each trigger appeared in only about 3.3% of the 1,880-image training split. The
resulting mean seen-trigger ASR was 12.90%, while clean non-target images were
already predicted as the target class 12.07% of the time. The attack therefore
was not successfully implanted.

This calibration stage does not use 40% poison. It first trains:

1. one clean control model with 0% poisoning; and
2. one single-trigger model in which 10% of training sources receive the same
   Fourier-middle trigger.

This isolates trigger learnability from multi-trigger dilution. Separate
attack models for other trigger families will follow after their perturbation
strengths have been calibrated.

## 1. Enter the repository

~~~python
%cd /kaggle/working/Frequency-Backdoor-Detection-using-Generator
~~~

## 2. Locate DTD automatically

~~~python
from pathlib import Path
import torch

matches = list(
    Path("/kaggle/working").rglob("asb_dtd_v1/variant_manifest.jsonl")
)
matches += list(
    Path("/kaggle/input").rglob("asb_dtd_v1/variant_manifest.jsonl")
)
if not matches:
    raise RuntimeError("ASB-DTD manifest was not found")

repo_name = "Frequency-Backdoor-Detection-using-Generator"
repo_matches = [path for path in matches if repo_name in path.parts]
MANIFEST = repo_matches[0] if repo_matches else matches[0]
DATA_ROOT = MANIFEST.parent.parent

assert (DATA_ROOT / "dtd/images").is_dir()
print("Dataset root:", DATA_ROOT)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
~~~

## 3. Generate image and spectrum panels

This does not train a model. It creates clean/triggered images, boosted image
differences, clean/triggered log-amplitude spectra, amplitude differences, phase
differences, and aggregate perturbation measurements.

~~~python
!python -m scripts.visualize_asb_dtd_trigger_spectra \
  --manifest-dir "{DATA_ROOT}/asb_dtd_v1" \
  --images-root "{DATA_ROOT}/dtd/images" \
  --output-dir Advanced_Outputs/dtd_trigger_spectral_calibration_v1 \
  --metric-samples 200 \
  --image-size 224
~~~

## 4. Train the clean control

The clean control establishes normal accuracy and the natural rate at which
non-target textures are incorrectly predicted as `banded`. It is essential for
distinguishing genuine trigger ASR from ordinary target-class confusion.

~~~python
!python -m scripts.train_asb_dtd_pretrained_classifier \
  --manifest-dir "{DATA_ROOT}/asb_dtd_v1" \
  --images-root "{DATA_ROOT}/dtd/images" \
  --experiment-dir Advanced_Experiments/dtd_resnet18_clean_control_v1 \
  --output-dir Advanced_Outputs/dtd_resnet18_clean_control_v1 \
  --weights default \
  --attack-triggers none \
  --poison-ratio 0 \
  --epochs 20 \
  --freeze-epochs 2 \
  --image-size 224 \
  --batch-size 32 \
  --num-workers 2 \
  --seed 42 \
  --device cuda
~~~

## 5. Train one Fourier-middle attack model

This run uses the same 10% total poisoning budget, but all 188 poisoned sources
receive Fourier-middle. The catalog strength remains 0.20.

~~~python
!python -m scripts.train_asb_dtd_pretrained_classifier \
  --manifest-dir "{DATA_ROOT}/asb_dtd_v1" \
  --images-root "{DATA_ROOT}/dtd/images" \
  --experiment-dir Advanced_Experiments/dtd_resnet18_fourier_middle_p10_s020_v1 \
  --output-dir Advanced_Outputs/dtd_resnet18_fourier_middle_p10_s020_v1 \
  --weights default \
  --attack-triggers fourier_middle \
  --poison-ratio 0.10 \
  --trigger-strength 0.20 \
  --epochs 20 \
  --freeze-epochs 2 \
  --image-size 224 \
  --batch-size 32 \
  --num-workers 2 \
  --seed 42 \
  --device cuda
~~~

## 6. Compare the two runs

~~~python
import json
from pathlib import Path

def load_result(folder):
    path = Path("Advanced_Experiments") / folder / "final_evaluation.json"
    return json.loads(path.read_text())

clean = load_result("dtd_resnet18_clean_control_v1")
attack = load_result("dtd_resnet18_fourier_middle_p10_s020_v1")

attack_result = attack["trigger_results"]["fourier_middle"]
print("Clean-control accuracy:", clean["clean_test"]["accuracy"])
print("Clean-control target rate:", clean["clean_non_target_target_prediction_rate"])
print("Attack-model clean accuracy:", attack["clean_test"]["accuracy"])
print("Fourier-middle ASR:", attack_result["asr"])
print(
    "ASR lift over clean control:",
    attack_result["asr"] - clean["clean_non_target_target_prediction_rate"],
)
~~~

## Decision rule

Proceed to generator training only if the attack model has a clearly successful
pre-defense attack. The working targets are:

- attack ASR at least 80%, preferably above 90%;
- large ASR lift over the clean-control target rate;
- attack-model clean accuracy no more than about five percentage points below
  the clean control.

If Fourier-middle ASR remains low, do not increase epochs blindly. Run a
strength calibration sweep using `--trigger-strength`, with each run stored in
a new folder such as `s030`, `s040`, and `s060`. The spectral measurement script
must be rerun with the matching `--strength-override` so visibility and PSNR are
reported beside ASR.

## 7. Package results

~~~python
!zip -r dtd_attack_calibration_v1.zip \
  Advanced_Experiments/dtd_resnet18_clean_control_v1 \
  Advanced_Experiments/dtd_resnet18_fourier_middle_p10_s020_v1 \
  Advanced_Outputs/dtd_trigger_spectral_calibration_v1 \
  Advanced_Outputs/dtd_resnet18_clean_control_v1 \
  Advanced_Outputs/dtd_resnet18_fourier_middle_p10_s020_v1
~~~

Download `dtd_attack_calibration_v1.zip` and extract it into the corresponding
local `Advanced_Experiments` and `Advanced_Outputs` directories.
