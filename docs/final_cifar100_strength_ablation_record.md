# Final CIFAR-100 Trigger Strength Ablation Record

## Purpose

This experiment tests whether the proposed frequency-domain backdoor defense
remains effective when the trigger strength changes. The main pipeline had
already shown strong results for `alpha = 0.08`. This ablation checks whether
that result was only true for one selected trigger strength or whether the
defense remains stable for weaker and stronger triggers.

## Fixed Settings

- Dataset: CIFAR-100.
- Image resolution: `32 x 32`.
- Target class: `apple`.
- Target label: `0`.
- Poison ratio: `0.12`.
- Trigger type: sinusoidal frequency trigger.
- Trigger frequency: `(6, 6)`.

Only trigger strength was changed:

- weak trigger: `alpha = 0.04`;
- medium trigger: `alpha = 0.08`;
- strong trigger: `alpha = 0.12`.

## Result Table

| Strength alpha | Suspicious Clean Acc | Suspicious ASR | Generator-Corrected Acc | Generator-Corrected ASR | Repaired Clean Acc | Repaired ASR |
|---:|---:|---:|---:|---:|---:|---:|
| `0.04` | `55.54%` | `99.55%` | `54.08%` | `1.60%` | `63.05%` | `0.05%` |
| `0.08` | `56.70%` | `99.07%` | `55.66%` | `0.20%` | `62.68%` | `0.07%` |
| `0.12` | `51.94%` | `99.94%` | `51.19%` | `1.14%` | `62.45%` | `0.04%` |

## Interpretation

All three trigger strengths successfully produced strong backdoors. The
suspicious ASR stayed above `99%` for `alpha = 0.04`, `0.08`, and `0.12`. This
means the backdoored classifier learned the sinusoidal frequency shortcut very
strongly even when the trigger was weaker.

The generator reduced ASR heavily for all strengths. The medium trigger
`alpha = 0.08` gave the lowest generator-corrected ASR at `0.20%`, but the weak
and strong triggers were also reduced to low ASR values, `1.60%` and `1.14%`.

The repaired model gives the strongest final defense result. After repair, ASR
was below `0.1%` for all three trigger strengths:

- `0.05%` for `alpha = 0.04`;
- `0.07%` for `alpha = 0.08`;
- `0.04%` for `alpha = 0.12`.

This supports the claim that the repair step is robust to the tested trigger
strength range.

## Output Files

Aggregate summary:

`outputs/ablation_cifar100_strength/strength_ablation_summary.md`

`outputs/ablation_cifar100_strength/strength_ablation_summary.json`

Per-strength outputs:

`outputs/ablation_cifar100_strength/alpha_0_04/`

`outputs/ablation_cifar100_strength/alpha_0_08/`

`outputs/ablation_cifar100_strength/alpha_0_12/`

Per-strength checkpoints:

`experiments/ablation_cifar100_strength/alpha_0_04/`

`experiments/ablation_cifar100_strength/alpha_0_08/`

`experiments/ablation_cifar100_strength/alpha_0_12/`

## Paper-Ready Statement

The proposed defense was evaluated under three trigger strengths
`alpha = 0.04`, `0.08`, and `0.12`. In all cases, the suspicious classifier
achieved ASR above `99%`, confirming that the backdoor was successfully learned.
After generator correction, ASR was reduced to at most `1.60%`, and after model
repair, ASR was reduced below `0.1%` for all strengths. This indicates that the
proposed generator-assisted repair framework remains effective across weak,
medium, and strong sinusoidal frequency triggers.
