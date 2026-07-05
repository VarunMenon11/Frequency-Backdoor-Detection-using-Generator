# Selective Frequency-Domain Backdoor Mitigation

This repository contains the implementation for a graduate research project on
adaptive spectral correction for backdoor mitigation.

The central hypothesis is not assumed to be true. The codebase is organized to
test whether learnable sparse spectral correction maps can reduce backdoor
dependency while preserving clean semantic accuracy.

## Phase Order

1. Project architecture
2. Project structure
3. Dataset construction
4. Suspicious model training
5. Frequency analysis
6. Generator design
7. Spectral correction
8. Generator training
9. Model repair
10. Experiments and ablations

We only move to the next phase after the current phase has been implemented,
checked, and understood.

## Repository Layout

```text
configs/        Experiment and pipeline configuration files.
datasets/       Dataset loading, transforms, split construction, and dataset wrappers.
poisoning/      Clean-to-poisoned sample generation and label poisoning logic.
fft/            FFT, amplitude, phase, masking, reconstruction, and spectral utilities.
models/         Image classifiers used as suspicious and repaired models.
generator/      Learnable spectral correction modules.
losses/         Training objectives for classifier, generator, and repair stages.
training/       Training loops and checkpoint orchestration.
evaluation/     Clean accuracy, attack success rate, and robustness metrics.
visualization/  Image grids, spectra, heatmaps, correction maps, and figures.
utils/          Shared reproducibility, logging, path, and device helpers.
checkpoints/    Saved model weights. Large files are not tracked by git.
experiments/    Experiment configs, metrics, logs, and run summaries.
notebooks/      Exploratory notebooks only; core logic belongs in Python modules.
outputs/        Generated figures, corrected samples, and reports.
```

## Scientific Guardrails

- Never assume the mitigation works.
- Verify poisoned data visually and spectrally before training.
- Verify the suspicious model is genuinely backdoored before mitigation.
- Report clean accuracy and attack success rate together.
- Compare against simple baselines before claiming improvement.
- Treat visualizations as debugging evidence, not proof.
