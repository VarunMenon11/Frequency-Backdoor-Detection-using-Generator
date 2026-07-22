# Phase 9 Summary: Model Repair

Phase 9 attempts to repair the suspicious classifier itself. Until this phase,
the generator acts like a preprocessing defense:

```text
triggered image -> generator correction -> classifier
```

Model repair asks a stronger question:

```text
Can the suspicious classifier be fine-tuned so that its internal trigger
dependency is reduced?
```

## Repair Data

Each repair batch contains two types of samples:

```text
1. Clean images with clean labels
2. Generator-corrected triggered images with clean labels
```

The corrected triggered images are produced using the trained spectral
correction generator.

Important observation from the first real repair run:

```text
corrected-only repair, 1 full epoch
before clean accuracy: 55.68%
before ASR: 98.54%
after clean accuracy: 63.49%
after ASR: 99.73%
```

This improved clean accuracy but did not reduce the raw-trigger backdoor. That
means corrected-only fine-tuning is not enough to make the classifier unlearn
the original trigger shortcut. It teaches the model useful clean/corrected
behavior, but the raw trigger still maps strongly to apple.

For stronger repair, the script supports adding raw triggered non-target images
with clean labels. This directly teaches the model:

```text
trigger present should not imply apple
```

## Repair Objective

The repaired classifier is trained with cross-entropy loss:

```text
CE(classifier(repair_image), clean_label)
```

The generator is frozen during repair. Only the classifier is updated.

## Why Use Both Clean And Corrected Images?

Clean images preserve normal classification behavior.

Corrected triggered images teach the model that trigger-like inputs should no
longer map to the attacker target class.

This balances:

```text
reduce ASR
preserve clean accuracy
```

## Script

```text
scripts/repair_suspicious_classifier.py
```

Smoke test:

```powershell
python -m scripts.repair_suspicious_classifier --epochs 1 --batch-size 32 --max-train-batches 2 --max-eval-batches 1 --device cpu --output-dir experiments/repaired_classifier_smoke_test
```

Full run:

```powershell
python -m scripts.repair_suspicious_classifier --epochs 3 --batch-size 128 --device auto --output-dir experiments/repaired_classifier_cifar100
```

Stronger anti-backdoor repair:

```powershell
python -m scripts.repair_suspicious_classifier --epochs 1 --batch-size 128 --include-raw-triggered --device auto --output-dir experiments/repaired_classifier_cifar100_antibackdoor_epoch1
```

Result from the stronger anti-backdoor repair run:

```text
before clean accuracy: 55.68%
before ASR: 98.54%
after clean accuracy: 62.99%
after ASR: 0.12%
```

This means the explicit raw-triggered clean-label repair signal was necessary
to make the classifier itself stop mapping the trigger to apple.

Repaired model outputs:

```text
experiments/repaired_classifier_cifar100_antibackdoor_epoch1/repaired_classifier.pt
experiments/repaired_classifier_cifar100_antibackdoor_epoch1/repair_summary.json
```

Repaired prediction demo:

```text
outputs/repaired_classifier_prediction_demo/clean_predictions.png
outputs/repaired_classifier_prediction_demo/triggered_predictions.png
```

## Outputs

```text
experiments/repaired_classifier_cifar100/repaired_classifier.pt
experiments/repaired_classifier_cifar100/repair_summary.json
```

## Success Criteria

After repair, we want:

```text
ASR much lower than 98.54%
clean accuracy close to 55.68%
```

If clean accuracy collapses, the repair is too destructive. If ASR remains high,
the repair did not remove trigger dependency.
