# DTD Pretrained Attack Calibration

## Current action: audit saved checkpoints, do not retrain yet

The 30-epoch clean control is already saved. Preserve it and the attack run.
The original attack checkpoint selector used the harmonic mean of clean
accuracy and raw ASR. This selected epoch 2, whose test clean-target rate was
22.50% and triggered ASR was 22.55%. That is almost no **net trigger effect on
the same model**, despite a much larger difference from the separate control.

The updated trainer selects the earliest checkpoint attaining the highest
validation clean accuracy. ASR cannot improve this selection score. This is a
conservative utility-based selection policy, not an assertion that the chosen
checkpoint has learned a strong attack. Paired validation diagnostics now
report baseline target bias, new target flips, reverse flips and conditional
ASR. The policy changes future runs only; saved checkpoints are not changed.

After updating the repository, use Sections 1 and 2 below to locate the data.
Then evaluate the existing last checkpoint without training:

~~~python
!python -u -m scripts.evaluate_asb_dtd_checkpoint \
  --checkpoint Advanced_Experiments/dtd_resnet18_fourier_middle_p10_s020_v1/suspicious_classifier_last.pt \
  --manifest-dir "{DATA_ROOT}/asb_dtd_v1" \
  --images-root "{DATA_ROOT}/dtd/images" \
  --output-dir Advanced_Outputs/dtd_checkpoint_audit_v1/epoch30_validation \
  --split validation \
  --triggers fourier_middle \
  --batch-size 32 \
  --num-workers 2 \
  --device cuda
~~~

This restores the saved weights without downloading ImageNet weights. It uses
the saved input size and exact attack configuration, including strength
overrides. Use an empty output folder: existing results are protected.
For a clean-control checkpoint, explicitly pass `--triggers fourier_middle`
because its list of training attack triggers is empty.

The outputs are `paired_evaluation.json`, a readable `paired_evaluation.md`,
`paired_predictions.jsonl`, and `clean_predictions.jsonl`. These are evaluation
artifacts, not new models. `--split test` is available for a declared final
evaluation or a clearly labelled historical audit. Do not repeatedly select
strengths or checkpoints by inspecting test results. Test results already used
to guide development must be described as exploratory, not an untouched final
benchmark. Omit `--max-eval-batches` for a full evaluation.

The training commands below document the calibration procedure. Do not rerun
the saved clean control. New experiments require new `_v2` or otherwise unique
result folders so previous runs remain intact.

For future strength calibration, append `--validation-only` to training commands.
This saves `validation_evaluation.json` for the selected checkpoint, training
history and curves, and both checkpoints, but skips test metrics and test panels.
The legacy comparison cell in Section 6 is for existing completed runs only.
After choosing settings using validation, perform one declared test evaluation
with the audit script and the chosen checkpoint.

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

This tests whether concentrating the poisoning budget improves learnability;
it does not establish that budget dilution caused the earlier failure. Separate
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
    "Same-model clean target rate:",
    attack["clean_non_target_target_prediction_rate"],
)
print(
    "Same-model net target-rate lift:",
    attack_result["asr"] - attack["clean_non_target_target_prediction_rate"],
)
~~~

## Decision rule

Proceed to generator training only if validation establishes a meaningful
pre-defense attack. The working engineering targets below are not mathematical
guarantees or literature-mandated defaults; declare them before calibration:

- attack ASR at least 80%, preferably above 90%;
- large ASR lift over the **same attack model's clean target rate**, supported
  by paired new-target flips and conditional ASR on clean-correct sources;
- attack-model clean accuracy no more than about five percentage points below
  the clean control on validation data.

The clean control is a utility reference, not a substitute for the attack
model's own clean target-rate baseline. Conditional ASR divides clean-correct
to-target flips by the number of correctly classified clean non-target
sources. It is undefined when that denominator is zero. A reverse target flip
does not necessarily restore the true label. Net lift alone can hide flips in
both directions, so inspect all the counts in the paired JSON.

If Fourier-middle ASR remains low, do not increase epochs blindly. Run a
strength calibration sweep using `--trigger-strength`, with each run stored in
a new folder such as `s030`, `s040`, and `s060`, and select configurations using
validation only. These strengths are candidate experiments, not established
optimal values. The spectral measurement script
must be rerun with the matching `--strength-override` so visibility and PSNR are
reported beside ASR.

The trainer now applies attack-strength overrides to training, validation,
final evaluation and prediction panels. Held-out triggers retain their catalog
settings; their exact evaluation configurations are recorded separately in
each trigger result. Historical runs made before this fix should not be
assumed to have tested an override merely because it appears in their run
configuration. The current 0.20 Fourier-middle run matches its catalog strength
and is not affected by that mismatch.

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
