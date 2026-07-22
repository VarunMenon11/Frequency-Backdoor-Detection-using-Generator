# Phase 10: Final CIFAR-100 Evaluation Summary

## Purpose

Phase 10 collects the final evidence for the CIFAR-100 experiment in one place.
The goal is to compare the suspicious classifier, the spectral correction
generator, and the repaired classifier using the same test set and the same
attack setting.

This phase is important because the small prediction grids are useful for
demonstration, but they are not enough for scientific evaluation. The final
claim should be based on full test-set metrics:

- clean accuracy on all 10,000 CIFAR-100 test images;
- attack success rate on the 9,900 non-target test images;
- generator-corrected performance on triggered images;
- repaired classifier performance after anti-backdoor fine-tuning.

## Experimental Setup

- Dataset: CIFAR-100.
- Image size: `3 x 32 x 32`.
- Attack target class: `apple`, label `0`.
- Poison ratio used for suspicious classifier training: `0.12`.
- Trigger type: sinusoidal frequency trigger.
- Trigger frequency: `fx = 6`, `fy = 6`.
- Trigger strength: `alpha = 0.08`.
- Best repaired checkpoint: `experiments/repaired_classifier_cifar100_antibackdoor_epoch3/repaired_classifier.pt`.

The attack success rate is computed only on non-target test samples. Target
class samples are excluded because an original apple image being classified as
apple is not evidence of a successful attack.

## Final Results

| Stage | Clean-label Accuracy | Attack Success Rate | Interpretation |
|---|---:|---:|---|
| Suspicious classifier | `55.68%` | `98.54%` | The backdoor attack is highly successful. |
| Generator-corrected images, suspicious classifier | `54.65%` | `0.37%` | The generator antidote suppresses the trigger effect before model repair. |
| Repaired classifier | `63.35%` | `0.06%` | The repaired model keeps better clean accuracy and nearly removes the backdoor. |
| Generator-corrected images, repaired classifier | `63.06%` | `0.08%` | The corrected images remain compatible with the repaired model. |

## What This Shows

The suspicious classifier learned a strong shortcut from the sinusoidal
frequency trigger to the target class. This is shown by the high ASR of
`98.54%`. In simple words, almost every non-apple test image becomes classified
as apple after the trigger is added.

The generator correction stage shows that the backdoor behavior can be weakened
by selectively correcting the amplitude spectrum. The clean-label accuracy on
corrected images is close to the original suspicious model's clean accuracy,
while ASR drops from `98.54%` to `0.37%`. This supports the main project idea:
the trigger dependency exists in a localized spectral pattern, and a learned
amplitude correction can reduce that dependency.

The repaired classifier gives the strongest final defense result. It was
fine-tuned using clean images, generator-corrected triggered images, and raw
triggered images with the original clean labels. This teaches the model that
the trigger should not imply the attacker target class. After three repair
epochs, clean accuracy improves to `63.35%` and ASR falls to `0.06%`.

## Output Files

Final evaluation script:

`scripts/final_cifar100_evaluation.py`

Machine-readable final metrics:

`outputs/final_cifar100_evaluation/final_cifar100_evaluation_summary.json`

Presentation-friendly final report:

`outputs/final_cifar100_evaluation/final_cifar100_evaluation_report.md`

Visual explanation panels:

`outputs/final_cifar100_evaluation/sample_panels/`

The sample panels show the clean image, triggered image, corrected image,
frequency/correction views, and predictions from both the suspicious and
repaired classifiers.

## Next Research Step

The next phase should test whether the method scales beyond CIFAR-100's
`32 x 32` images. The recommended next dataset is STL-10 because it uses
`96 x 96` natural images and is still manageable on Kaggle.

For higher-resolution experiments, the classifier and generator should be
trained again at the new resolution. The current CIFAR-100 models are not
universal models; they are trained specifically for `32 x 32` CIFAR-100 images
and the controlled sinusoidal trigger setting.
