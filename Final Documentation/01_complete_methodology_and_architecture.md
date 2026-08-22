# Complete Methodology and Architecture

## 1. Research problem

Deep-learning classifiers can contain a backdoor. During training, an attacker modifies a portion of the training data by adding a trigger and changing the associated label to an attacker-selected target. The model then learns two behaviours at the same time:

1. normal visual classification from clean images;
2. shortcut classification from the trigger to the target label.

At test time, a triggered image can therefore be classified as the attacker target even when its visual content belongs to another class. The purpose of this project is to reduce that trigger dependence without destroying the natural information required for ordinary classification.

Many simple frequency-domain defenses suppress broad regions of the spectrum. That can remove trigger energy, but it can also remove genuine edges, textures, shapes, and other semantic information. The proposed approach is selective: a generator predicts a correction map and controls how much each amplitude location should move toward a clean reference.

The current implementation is a controlled known-trigger study. Clean and triggered pairs are available during generator training and evaluation. This is a deliberate first stage because it allows the spectral correction hypothesis to be measured directly. A blind single-image generator is future work and is discussed in Section 15.

## 2. Overall pipeline

```text
Clean dataset
     |
     +------------------------------+
     |                              |
     v                              v
Clean classifier data       Add a frequency-domain trigger
                                    |
                                    v
                             Poisoned training data
                                    |
                                    v
                         Train suspicious classifier
                                    |
                                    v
             clean image + triggered image for controlled pairs
                                    |
                                    v
                         FFT amplitude/phase split
                                    |
                                    v
                A_clean, A_triggered, P_triggered, difference
                                    |
                                    v
                         Spectral correction generator
                                    |
                                    v
                              correction map M
                                    |
                                    v
              corrected amplitude + preserved triggered phase
                                    |
                                    v
                              inverse FFT
                                    |
                                    v
                             corrected image
                                    |
                                    v
                  fine-tune and repair the classifier
                                    |
                                    v
                     clean accuracy, ASR, and panels
```

The project therefore has three experimental stages: implant and verify the backdoor, learn an image-level spectral antidote, and repair the model itself.

## 3. Datasets and resolutions

| Dataset | Image shape | Classes | Role in the study |
|---|---:|---:|---|
| CIFAR-100 | 3x32x32 | 100 | Low-resolution baseline and initial proof of concept. |
| Tiny ImageNet | 3x64x64 | 200 | Larger class count and intermediate resolution. |
| STL-10 | 3x96x96 | 10 | Higher-resolution validation with more visual detail. |

The tensor shape is channel-first for PyTorch: three colour channels followed by height and width. Converting an image to `3xHxW` does not make the model more frequency-dependent by itself. It is the standard representation required by the convolutional network and FFT routines. The frequency analysis is meaningful because the FFT transforms the two spatial axes into a two-dimensional frequency grid.

## 4. Poisoning protocol

For a clean image (x) with label (y), a trigger function (T) produces a modified image (x_t). For the main sinusoidal experiments:

$$
T(u,v)=\cos\left(2\pi\left(\frac{f_xu}{W}+\frac{f_yv}{H}\right)\right)
$$

and:

$$
x_t=\operatorname{clip}(x+\alpha T,0,1).
$$

Here (u,v) are pixel coordinates, (W,H) are image dimensions, (f_x,f_y) are the horizontal and vertical cycles, and (alpha) controls perturbation strength. The pattern is added consistently to the RGB channels in the current implementation.

The poison ratio is 0.12 in the main experiments. It means that 12% of eligible non-target training samples receive the trigger and are relabeled to the target class. It does not mean that the entire dataset is poisoned. The remaining majority stays clean, which makes the attack more realistic and allows clean classification performance to be measured.

The target labels were selected as follows:

- CIFAR-100: `apple`, label 0.
- STL-10: `airplane`, label 0.
- Tiny ImageNet: `goldfish`, label 0.

Target-class samples are excluded from ASR calculations because a target-class image is already correctly assigned to the target before any trigger is added.

## 5. Why the frequencies scale with resolution

The baseline sinusoidal trigger uses `(6,6)` on 32x32 CIFAR-100. The corresponding settings are `(12,12)` on 64x64 Tiny ImageNet and `(18,18)` on 96x96 STL-10. These are proportional choices:

$$
(6,6)\times\frac{64}{32}=(12,12),
$$

$$
(6,6)\times\frac{96}{32}=(18,18).
$$

The purpose is to keep a comparable relative position on the discrete frequency grid. These numbers are not universal constants and are not claimed to be mathematically optimal for every dataset. Frequency-location ablations are still necessary for a universal claim.

## 6. Suspicious classifier

The suspicious classifier is intentionally trained on the poisoned data. It is not the final defended model; it is the model whose backdoor we want to study.

Its learned shortcut can be described informally as:

```text
if the suspicious frequency pattern is present:
    favour the attacker target class
otherwise:
    use ordinary visual features
```

The classifier is evaluated before defense using clean accuracy and ASR. A high ASR confirms that the poison protocol actually created a usable backdoor. This check is essential: a low ASR after correction is not meaningful if the original model never learned the attack.

## 7. Fourier representation

For each image channel, the two-dimensional FFT is:

$$
F(x)=\operatorname{FFT}(x).
$$

The complex spectrum is represented as:

$$
F(x)=A(x)e^{jP(x)},
$$

where:

- (A(x)=|F(x)|) is amplitude or magnitude;
- (P(x)=\arg(F(x))) is phase.

Amplitude indicates how much energy exists at each spatial frequency. Phase carries the positional arrangement that helps determine where structures appear in the reconstructed image. The project does not ignore phase. It uses the triggered phase for reconstruction but asks the generator to modify amplitude only.

For a clean-triggered pair:

$$
A_c=|\operatorname{FFT}(x)|,
\quad
A_t=|\operatorname{FFT}(x_t)|,
\quad
P_t=\arg(\operatorname{FFT}(x_t)).
$$

The amplitude evidence is:

$$
D=|A_t-A_c|.
$$

The difference (D) is not the correction map. It is an input feature that shows which amplitude locations changed between the clean and triggered versions.

## 8. Generator architecture and correction

The generator receives a channel-wise concatenation:

$$
G_{input}=[A_c,A_t,D].
$$

For an RGB image, each of these has three channels, giving a nine-channel generator input. A compact convolutional network predicts:

$$
M=G(A_c,A_t,D).
$$

The map is constrained to a usable range. Its values act as correction gates:

$$
A_{corr}=A_t-M\odot(A_t-A_c)
$$

which is equivalent to:

$$
A_{corr}=(1-M)\odot A_t+M\odot A_c.
$$

If (M=0), that location is left at the triggered value. If (M=1), it is moved fully to the clean value. Intermediate values perform partial correction. This is why the generator is selective rather than a global high-frequency filter.

The reconstructed image is:

$$
x_{corr}=\operatorname{IFFT}(A_{corr}e^{jP_t}).
$$

Thus, the core method is **adaptive amplitude correction with phase preservation**.

## 9. Generator training objective

The suspicious classifier is frozen during generator training. Only generator weights are updated. The total loss is:

$$
L_G=1.0L_{cls}+4.0L_{rec}+0.02L_{sp}+0.01L_{sm}.
$$

The terms are:

$$
L_{cls}=CE(f_{suspicious}(x_{corr}),y)
$$

which encourages the corrected image to return to its original clean label;

$$
L_{rec}=\operatorname{mean}(|x_{corr}-x|)
$$

which discourages visible or destructive image changes;

$$
L_{sp}=\operatorname{mean}(|M|)
$$

which discourages correcting the entire spectrum; and a total-variation smoothness term (L_{sm}=TV(M)), which discourages noisy isolated map values.

The generator is not a GAN generator in the strict sense. There is no discriminator judging realism. The frozen classifier supplies task feedback, while reconstruction, sparsity, and smoothness constrain the correction.

## 10. Classifier repair

The suspicious classifier is copied and fine-tuned. Repair data combines:

1. clean images with their clean labels;
2. corrected triggered images with their clean labels;
3. raw triggered images with their clean labels.

The third group is important. If the repaired classifier only sees corrected images, it may still respond to the raw trigger. Including raw triggered images explicitly teaches:

```text
the trigger is not a valid reason to predict the attacker target
```

The generator is therefore an image-level defense and the repaired classifier is a model-level defense. They can be evaluated separately and together.

## 11. Evaluation metrics

Clean accuracy is:

$$
ACC_{clean}=\frac{\#\{f(x_i)=y_i\}}{N}.
$$

ASR is calculated only on non-target samples:

$$
ASR=\frac{\#\{f(T(x_i))=y_{target}\}}{N_{non-target}}.
$$

The four important reports are:

- suspicious clean accuracy and suspicious ASR;
- corrected-image accuracy and generator-corrected ASR using the suspicious classifier;
- repaired clean accuracy and repaired raw-trigger ASR;
- corrected-image ASR using the repaired classifier.

Reconstruction L1 measures average pixel difference. It is useful, but it should not be treated as the only perceptual measure. Visual panels and, in future work, SSIM or LPIPS would give additional evidence.

## 12. End-to-end numerical result

| Dataset | Suspicious clean accuracy | Suspicious ASR | Generator corrected accuracy | Generator corrected ASR | Repaired clean accuracy | Repaired ASR |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-100 | 55.68% | 98.54% | 54.65% | 0.37% | 63.35% | 0.06% |
| Tiny ImageNet | 40.96% | 99.99% | 40.51% | 0.37% | 46.11% | 0.09% |
| STL-10 | 65.75% | 100.00% | 63.51% | 3.15% | 76.19% | 0.71% |

These results support the controlled hypothesis across three resolutions. They do not establish universal defense against arbitrary unknown triggers or datasets.

## 13. Recommended figure sequence

For a dissertation chapter, show the figures in this order:

1. [Average frequency panel](../outputs/frequency_analysis/average_frequency_panel.png) to introduce the clean/trigger spectral change.
2. [CIFAR-100 correction panel](../outputs/final_cifar100_evaluation/sample_panels/final_panel_test_index_0.png) to show the baseline image-level pipeline.
3. [STL-10 strength panel](../outputs/ablation_stl10_strength_wide/alpha_0_15/sample_panels/stl10_96_panel_test_index_0.png) to show trigger-strength behaviour.
4. [STL-10 cosine panel](../outputs/ablation_stl10_trigger_function/cosine/sample_panels/stl10_96_panel_test_index_0.png) and the [dual-frequency panel](../outputs/ablation_stl10_trigger_function/dual_frequency/sample_panels/stl10_96_panel_test_index_0.png) to show function variation.
5. [FIBA-style panel](../outputs/outputs_FIBA_Cali/stl10_96x96_fiba_amplitude_calibrated/sample_panels/stl10_96_panel_test_index_0.png) for the FIBA-style amplitude experiment, with its preliminary status stated.

The baseline panel can be embedded directly in a Markdown viewer:

![CIFAR-100 final correction panel](../outputs/final_cifar100_evaluation/sample_panels/final_panel_test_index_0.png)

The panel is not a single accuracy measurement. It is a qualitative diagnostic that should be read together with the aggregate table. A successful individual panel demonstrates the mechanism clearly, while the aggregate ASR and accuracy values establish whether the behaviour occurs across the evaluation set.

## 14. Main limitations

The current generator receives clean amplitude as part of its input. That is available in controlled poisoning experiments because the triggered image was deliberately generated from a clean image. It is not normally available for a real unknown image.

The current evaluation also assumes a known dataset domain, known input resolution, known target setup, and a trigger family that is deliberately constructed for testing. The method should therefore be described as a strong controlled proof of concept, not as a completed universal detector.

## 15. Future blind version

For a real deployment image (x_t), only (A_t) and (P_t) are available. A blind generator could be trained with input-only design:

$$
M=G(A_t)
$$

or with a learned clean-spectrum prior:

$$
\hat A_c=H(A_t),\quad M=G(A_t,\hat A_c,|A_t-\hat A_c|).
$$

During training, clean images can still provide supervision, but the clean amplitude must be withheld from the generator input. A second option is an iterative loop in which each correction is constrained by classification confidence, spectral naturalness, and correction size. That is a new experiment and should be reported separately from the paired method documented here.

