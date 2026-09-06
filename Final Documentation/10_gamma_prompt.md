# Gamma Prompt: M.Tech Project Review Deck

Copy the prompt below into Gamma after uploading the selected project figures. Gamma may rewrite text, so check every numerical value against `Final Documentation/09_project_review_presentation.md`.

---

Create a professional 17-slide M.Tech project-review presentation titled:

**Selective Frequency-Domain Backdoor Mitigation Using Adaptive Spectral Correction**

Subtitle: **Generator-assisted spectral correction and classifier repair across image resolutions and trigger families**

Audience: graduate research guide and M.Tech project-review panel with no prior knowledge of this project, Fourier analysis, or backdoor attacks. Tone: technical, clear, evidence-based, and honest about limitations. The talk should fit 15-20 minutes. Use concise slide text and put the teaching explanation in speaker notes.

Do not design this as a status update for experts. Build the explanation from first principles. First explain what an ordinary image classifier does. Then explain how poisoned training data creates a hidden shortcut. Then define trigger, target class, suspicious model, ASR, frequency, amplitude, and phase in plain language. Only after those definitions introduce FFT equations and the generator. Each result slide must remind the audience what stage of the pipeline its numbers represent.

Use a simple running example throughout the deck: a clean horse image is classified as horse, the triggered version is incorrectly classified as airplane by the suspicious model, the generator-corrected version moves back toward horse, and the repaired model should classify both the raw-triggered and corrected versions as horse. Make clear that the target class is an experimental label, not a statement that airplane images are suspicious.

The deck should explicitly answer these beginner questions: What is being attacked? What is poisoned? Why is a trigger dangerous? Why use frequency analysis? Why not remove all high frequencies? What is the generator actually generating? Why preserve phase? Why are clean and triggered pairs available? What does ASR mean? What is complete and what remains future work?

## Visual style

- Use a clean academic visual system: white or very light background, dark charcoal text, restrained navy/teal accents, and one warm highlight colour for “triggered” states.
- Use consistent colour semantics: blue for clean, red/orange for triggered or poisoned, purple for generator correction, green for repaired output.
- Use diagrams, arrows, tables, and annotated figure callouts rather than decorative stock photos.
- Use a readable sans-serif typeface. Keep body text large enough for a room presentation.
- Do not use a purple gradient, glossy marketing language, random AI imagery, or crowded paragraphs.
- Preserve the aspect ratio of all uploaded project panels. Do not crop away their labels.

## Scientific framing

The project studies a backdoored image classifier that behaves normally on clean images but predicts an attacker-selected target when a frequency trigger is present. The research gap is that broad frequency suppression can remove useful semantic frequencies together with the trigger. The proposed approach uses FFT amplitude and phase, predicts a selective amplitude correction map with a generator, reconstructs a corrected image using the triggered phase, and repairs the classifier using clean, corrected-triggered, and raw-triggered samples.

State clearly that the current experiments are controlled known-trigger experiments with clean/triggered pairs available for generator training. Do not claim a universal unknown-trigger defense. State that FIBA calibration is successful but the stored full FIBA run is preliminary because it used an older, weaker setting.

## Slide sequence

1. **Title**: project name, student, program, and institution. Add a small clean-image -> spectrum -> corrected-image motif.
2. **Problem definition**: show the clean-image versus triggered-image shortcut and state the need to preserve natural textures.
3. **Research gap and objectives**: compare broad filtering with selective adaptive correction; list generator, repair, and evaluation contributions.
4. **Datasets and scope**: show CIFAR-100 32x32, Tiny ImageNet 64x64, and STL-10 96x96. Mention 100, 200, and 10 classes respectively, and poison ratio 0.12 for the main experiments.
5. **Attack construction**: show clean data -> trigger 12% of training samples -> target relabeling -> suspicious classifier -> ASR validation. Include the sinusoidal trigger formulas for `T(u,v)` and `x_t`.
6. **FFT representation**: show image -> FFT -> amplitude and phase. Explain amplitude as spectral energy and phase as spatial arrangement. Include `F(x)=A(x)e^{jP(x)}`.
7. **Generator and repair architecture**: show `A_clean`, `A_triggered`, and `|A_triggered-A_clean|` entering the generator, correction map `M`, corrected amplitude, preserved triggered phase, inverse FFT, and classifier repair.
8. **Evaluation metrics**: define clean accuracy and ASR. Explain that target-class samples are excluded from ASR.
9. **Cross-resolution results**: present this table exactly:

   | Dataset | Resolution | Suspicious ASR | Generator ASR | Repaired ASR | Repaired clean accuracy |
   |---|---:|---:|---:|---:|---:|
   | CIFAR-100 | 32x32 | 98.54% | 0.37% | 0.06% | 63.35% |
   | Tiny ImageNet | 64x64 | 99.99% | 0.37% | 0.09% | 46.11% |
   | STL-10 | 96x96 | 100.00% | 3.15% | 0.71% | 76.19% |

10. **Trigger-strength ablation**: show alpha 0.03, 0.15, and 0.25 on STL-10. Explain alpha as trigger strength, not a universal scientifically fixed default. Show repaired ASR 0.44%, 0.38%, and 0.35%.
11. **Trigger-function ablation**: show cosine, sine, checkerboard, and dual-frequency. Show repaired ASR 0.32%, 0.40%, 0.60%, and 0.07%.
12. **FIBA-style extension**: explain amplitude injection inside a frequency mask. Show calibration selection alpha 0.50, radius 0.10, suspicious ASR 94.13%, clean accuracy 49.56%. Label the stored full run as preliminary because it used alpha 0.30, radius 0.15, and initial ASR 45.67%.
13. **How to read the image panel**: use the uploaded STL-10 alpha 0.15 panel. Label clean true, suspicious clean, suspicious triggered, suspicious corrected, repaired clean, repaired triggered, repaired corrected, amplitude difference, correction map, and amplified residuals.
14. **Implementation and reproducibility**: mention dataset loaders, poisoning, FFT utilities, classifier, generator, repair, metrics, panel generation, JSON/Markdown summaries, and GPU notebook execution. Mention repository folders `experiments/`, `outputs/`, and `Final Documentation/`.
15. **Limitations and SDG mapping**: say paired clean/triggered training, known trigger setting, unknown-image defense not validated, FIBA full calibrated rerun pending. Map primarily to SDG 9; mention medical imaging as future application motivation under SDG 3 without claiming a medical result.
16. **Next-semester plan**: calibrated FIBA rerun, repeated seeds, confidence intervals, frequency/poison-ratio ablations, input-only generator, unseen triggers, cross-dataset transfer, and perceptual metrics.
17. **Conclusion and questions**: summarize the controlled result and clearly separate completed work from future work.

## Uploaded figure placement

Use the following uploaded figures:

- Main qualitative figure: STL-10 alpha 0.15 strength panel.
- Resolution evidence: STL-10 96x96 panel plus clean CIFAR-100 and Tiny ImageNet thumbnails.
- Trigger-function evidence: dual-frequency panel, with the quantitative table for all functions.
- FIBA evidence: calibration table/heatmap and the preliminary FIBA panel with a visible “Preliminary” label.

## Speaker-note requirements

Add speaker notes explaining:

- why a high suspicious ASR must be measured before calling the defense successful;
- why 12% is an experimental poison-ratio choice rather than a universal default;
- why frequency values scale from `(6,6)` to `(12,12)` and `(18,18)` with resolution;
- why the amplitude difference is not itself the correction map;
- why phase is preserved during reconstruction;
- why image differences can be invisible without x8 amplification;
- why this work does not yet prove an unknown-image universal defense;
- why the FIBA full run is preliminary.

Also add a short spoken explanation to every technical slide, not only the result slides. For equations, explain every symbol immediately after the equation. For tables, explain the meaning of every column before interpreting the trend. For diagrams, describe the data object travelling through each arrow: image, amplitude, phase, difference map, correction map, corrected amplitude, reconstructed image, or classifier prediction.

Do not invent papers, metrics, datasets, or results. Do not describe the method as a GAN. Do not state that the method is perfect or universally validated.
