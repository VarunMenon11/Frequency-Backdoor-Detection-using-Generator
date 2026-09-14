# Advanced DTD: Checkpoint Audit and Calibration Notes

Date: 14 September 2026.

## 1. Purpose and status

This is an audit of the saved ImageNet-pretrained ResNet-18 experiments on
DTD. It is separate from the earlier CIFAR-100, STL-10 and Tiny ImageNet
generator/repair experiments. No advanced DTD generator was trained in this
audit. We are still establishing whether the poisoned classifier has learned
a sufficiently strong, trigger-specific association to support a meaningful
defense experiment.

The clean-control training completed successfully. The Fourier-middle
poisoning run also completed, but completion is not equivalent to a successful
backdoor. The important finding is that raw ASR can be inflated by ordinary
target-class bias: an image may already be predicted as the target before a
trigger is applied. Our measurements must separate that baseline bias from
the change caused by adding the trigger.

## 2. What was trained

| Setting | Value |
|---|---|
| Dataset | DTD, 47 texture classes, official split 1 |
| Training / validation / test | 1,880 / 1,880 / 1,880 source images |
| Model | ResNet-18 initialized from ImageNet-1K weights |
| Classifier input | RGB, 224 x 224 |
| Target | Label 0, `banded` |
| Clean control | No poisoned sources |
| Poisoning run | 188 triggered sources replace clean sources, 10% total |
| Remaining clean training sources | 1,692 |
| Trigger | `fourier_middle`, `fourier_band` implementation |
| Strength | 0.20 |
| Normalized band radii | 0.10 and 0.25 |
| Soft transition width | 0.015 |
| Channel weights | 1.0, 1.0, 1.0 |
| Epochs / feature freeze | 30 / first 2 epochs |
| Batch size / seed | 32 / 42 |
| Head / backbone learning rates | 0.001 / 0.0001 |
| Weight decay | 0.0001 |

Training uses random resized crops and horizontal flips before trigger
injection. Evaluation resizes the shorter side to 256 pixels, center-crops to
224 x 224, converts to a tensor and then applies the trigger. The model applies
ImageNet normalization internally. Consequently, the input crop is identical
for the clean and triggered versions in the audit.

The current Fourier trigger is not the earlier additive sinusoidal trigger
and is not FIBA. It scales the source image's own Fourier amplitude according
to a smooth radial mask:

```text
A_triggered = A_clean * (1 + alpha * band_mask)
```

With RGB channel weights of one and alpha 0.20, the amplitude scaling factor
is between 1 and 1.20. The phase is retained during reconstruction, then image
values are clipped to the valid range. Clipping may itself change the final
spectrum. Alpha 0.20 does not mean adding 0.20 to every pixel or altering 20%
of the dataset. Because the pattern scales the image's existing amplitude,
the resulting perturbation depends on the image. It may resemble a modest
texture/contrast change. That is a plausible reason for limited learnability,
not a cause established by these measurements.

## 3. The original checkpoint-selection problem

The original code maximized the harmonic mean of validation clean accuracy
and raw triggered ASR. For the poisoning run, this chose epoch 2 instead of a
later, more accurate checkpoint. A model with strong target bias can receive
a high ASR without reacting much to the trigger. Raw ASR therefore made an
inappropriate unqualified component of the selection score.

This implementation weakness is recorded rather than hidden. The original
`suspicious_classifier_best.pt` still contains epoch 2 and the original
`suspicious_classifier_last.pt` still contains epoch 30. Neither file was
replaced. The validation-clean maximum in the historical training log was
epoch 26 (61.01%), but that epoch's weights were not separately saved. An
epoch-26 test result cannot be reconstructed from the training log alone.

Future runs now select the earliest epoch attaining the highest validation
clean accuracy. ASR is measured separately, not used to reward target bias.
This is a transparent utility-first selection rule, not a guarantee that the
selected checkpoint is an effective attack. Attack qualification still needs
its own validation evidence. We have not retroactively declared epoch 30 the
best checkpoint based on its test results.

## 4. Measured results

All full evaluations below use 1,880 clean sources and 1,840 paired non-target
sources. The 40 true target-class images are excluded from ASR denominators,
but included in overall clean accuracy. Values are percentages except counts
and net lift, which is in percentage points.

| Model/checkpoint | Split | Clean accuracy | Clean target rate | Triggered ASR | Net lift |
|---|---|---:|---:|---:|---:|
| Clean control, epoch 18 | Test, original saved summary | 65.64 | 0.76 | 0.76 | 0.000 |
| Poisoning run, epoch 2 | Test, reproduced in new paired audit | 48.30 | 22.50 | 22.55 | 0.054 |
| Poisoning run, epoch 30 | Test, new paired audit | 60.53 | 9.67 | 13.10 | 3.424 |
| Poisoning run, epoch 30 | Validation, new paired audit | 60.11 | 9.89 | 13.53 | 3.641 |

The clean-control row is taken from the original saved evaluation, not a new
training or paired-prediction run. The epoch-2 audit reproduces the original
aggregate metrics and supplies the previously missing paired counts.

### Epoch-2 paired test interpretation

There were 414 clean target predictions and 415 triggered target predictions.
The additional target prediction in the aggregate does **not** mean that only
one image changed: 32 sources entered the target class and 31 left it. Only 8
of the 881 correctly classified clean non-target sources switched to the
target, giving conditional ASR 0.91%. Thus the reported 22.55% raw ASR is
dominated by pre-existing target bias and is not evidence of a strong
trigger-specific backdoor.

### Epoch-30 paired test interpretation

Before triggering, 178 of the 1,840 non-target sources were already predicted
as `banded`. After triggering, 241 were predicted as `banded`. The net increase
is 63, but **68 predictions entered the target class and 5 left it**. Reporting
only the difference of aggregate rates would hide these two directions.

There were 1,104 correctly classified clean non-target sources. Only 26 of
these switched to the target after triggering. Thus the conditional ASR on
clean-correct sources is 26/1,104 = 2.36%. Of the 68 new target flips, the other
42 were already incorrect non-target predictions before triggering. They are
still changes caused by the trigger, but they are not correct-to-target
classification failures.

The triggered true-label accuracy on non-target sources was 58.64%, compared
with 60.00% for the matching clean non-target sources. Overall clean accuracy,
which includes the target class, was 60.53%. These denominators differ and
must not be mixed when describing accuracy changes. Relative to the separate
clean control's overall test accuracy, the epoch-30 model loses 5.11 percentage
points; that comparison measures utility cost, not trigger activation.

### Epoch-30 paired validation interpretation

Validation gives a similar picture: 182 clean target predictions become 249
triggered target predictions. There are 69 new target flips and 2 reverse
flips. Of 1,100 initially correct non-target sources, 36 become target
predictions, giving conditional ASR 3.27%. Triggered true-label accuracy is
57.99% on the non-target subset. These validation results, not repeated test
inspection, should guide the next calibration decisions.

## 5. What each metric answers

**Raw ASR:** Among non-target images after triggering, how many are predicted
as the attacker's target? This is the conventional targeted-success count,
but includes sources already predicted as target when clean.

**Same-model clean target rate:** How often does the same classifier predict
the target on the same non-target sources without the trigger? Subtracting a
different classifier's rate instead confounds trigger effects with differences
between the models.

**Net target-rate lift:** Triggered ASR minus the same-model clean target rate.
It measures a net shift. It does not count all changed predictions, because
opposite-direction flips cancel.

**New target flips and reverse flips:** These identify individual sources
moving into and out of the target class. Leaving the target does not
necessarily mean predicting the correct class.

**Conditional ASR on clean-correct sources:** Correct-clean to target-triggered
transitions divided by correctly classified clean non-target sources. It is
reported alongside conventional ASR, not substituted for it without explanation.
When there are no clean-correct non-target sources, it is undefined, not zero.

## 6. Evaluation fixes and verification

The strength override previously reached training and validation, but final
test evaluation and prediction panels read the original manifest strengths.
The shared evaluation code now consistently uses saved attack configurations
for seen triggers. Held-out triggers retain their manifest configuration.
Each trigger result records the exact configuration used. The present 0.20
Fourier-middle run matches the manifest and was not affected by this bug.

The checkpoint audit script restores weights without downloading pretrained
weights or training. Clean/triggered predictions are paired by source ID, with
checks for missing sources, duplicate IDs and inconsistent labels. Reports
include checkpoint/manifest SHA-256 hashes, runtime arguments and individual
predictions. `--validation-only` in the trainer now prevents test evaluation
and test-image panels during future calibration.

Ten regression tests cover paired metrics, target-biased predictions, undefined
conditional ASR, partial-source pairing, selection, and strength propagation
through evaluation and visualization. Small randomly initialized CPU smoke
tests also exercised training, saving, final evaluation, figure generation
and validation-only mode. Their folders include `dtd_audit_code_smoke_v1` and
`dtd_audit_validation_only_smoke_v1`. They are software checks on tiny subsets,
**not research results**.

## 7. Decision and next experiment

The later checkpoint is more useful than the old selected epoch-2 model, and
the trigger produces a measurable effect. However, this is not yet a strong
implanted backdoor. Training a defense now and celebrating low final ASR would
not demonstrate that the generator removes a strong attack.

Preserve the clean control. Next, use validation-only, single-trigger strength
calibration in fresh folders while holding poison ratio, seed, architecture
and preprocessing fixed. Compare paired validation behavior with image and
spectrum differences; do not just increase epochs or poisoning percentage.
Candidate strengths must be identified as experimental settings, not
mathematically optimal defaults. If a band-amplitude scaling trigger remains
weak, revisit the trigger construction and test it as a separately named
experiment rather than silently changing the existing benchmark.

The working attack criteria in the Kaggle guide (high raw ASR, substantial
paired effect and modest clean-accuracy cost) are engineering targets, not
universal thresholds. A utility-preserving but weakly attacked model is not
automatically eligible for defense evaluation. Unknown-trigger and clean-
reference-free generator claims remain future work.

The manifest also flags **9 cross-split exact-duplicate checksum groups**.
This audit preserves the official split for historical comparability and
does not remove those sources. A deduplicated protocol or sensitivity check
is still needed before strong generalization claims. Test results already
used for development are exploratory; a new untouched evaluation protocol
is needed for an unbiased final confirmation. Multiple-seed uncertainty is
also not established by this single-seed audit.

## 8. Artifacts and commands

- [Epoch-30 test audit](../Advanced_Outputs/dtd_checkpoint_audit_v1/epoch30_test/paired_evaluation.md)
- [Epoch-30 validation audit](../Advanced_Outputs/dtd_checkpoint_audit_v1/epoch30_validation/paired_evaluation.md)
- [Epoch-2 test audit](../Advanced_Outputs/dtd_checkpoint_audit_v1/epoch02_test/paired_evaluation.md)
- [Kaggle steps](../notebooks/kaggle_dtd_attack_calibration.md)
- [Single-wavelet dataset and Kaggle calibration](../notebooks/kaggle_dtd_wavelet_calibration.md)
- [Evaluation-only runner](../scripts/evaluate_asb_dtd_checkpoint.py)
- [Shared paired metrics](../evaluation/asb_classifier.py)
- [Regression tests](../tests/test_asb_evaluation.py)

The machine-readable audit and per-source records are beside each linked
Markdown report. Raw `Advanced_Outputs` artifacts are ignored by Git, so
archive them separately; this document and source code can be committed.
