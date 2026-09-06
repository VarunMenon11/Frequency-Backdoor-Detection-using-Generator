# M.Tech Project Review Presentation

## Project title

**Selective Frequency-Domain Backdoor Mitigation Using Adaptive Spectral Correction**

### Subtitle

Generator-assisted spectral correction and classifier repair across image resolutions and trigger families.

### Intended use

This is a presentation-ready script for a project review. It is written for a 12-18 minute talk followed by questions. The slide text is deliberately concise; the speaker notes contain the explanation that should be spoken rather than placed as dense text on the slide.

The deck is aligned with the attached M.Tech project-review rubric:

| Rubric requirement | Slides that provide evidence |
|---|---|
| Problem statement and research gap | 2-3 |
| Proposed methodology and workflow | 4-7 |
| Initial implementation and meaningful progress | 8-9 |
| Preliminary verification and results | 10-13 |
| Standard algorithms and evaluation metrics | 6, 9, 10 |
| Presentation and technical explanation | All slides, especially 5-7 |
| SDG mapping and justification | 15 |
| Semester plan and next steps | 16 |

The rubric is an assessment document. It should guide what the presentation demonstrates, but it should not be copied into the deck as if it were part of the research method.

---

## Important reporting position

The presentation should make three distinctions clear:

1. **The suspicious model is intentionally backdoored.** A high initial ASR proves that the attack was successfully implanted before defense.
2. **The generator is an image-level correction stage.** It selectively modifies amplitude information and reconstructs an image using the triggered phase.
3. **Classifier repair is a separate model-level stage.** It teaches the classifier that the trigger should not cause the target prediction, even when the raw triggered image is presented.

The current method is a controlled known-trigger study. Clean and triggered pairs are available while training the generator. A blind unknown-image version, in which no clean counterpart is available, is future work and must not be presented as completed.

FIBA calibration is successful, but the currently stored full FIBA run used an older, weaker attack setting. Present it as a preliminary extension, not as the final calibrated FIBA result.

---

## How to present this to a panel with no prior context

Do not begin by saying “we apply FFT and generate a correction map.” That assumes the panel already knows why a classifier can be backdoored and why frequency is relevant. Build the explanation in this order:

1. **Start with the ordinary classification problem.** An image classifier normally learns a relationship between visual content and a class label, such as horse, airplane, or ship.
2. **Introduce the security problem.** A malicious party can poison a small part of the training data by adding a trigger and assigning those samples an attacker-selected label. The model then learns both the normal task and an unwanted shortcut.
3. **Explain the attack in one concrete example.** A clean horse image remains a horse to a normal classifier. The same image with the learned trigger may be predicted as airplane by the suspicious classifier.
4. **Explain why simple removal is difficult.** The trigger is small, and natural images also contain frequency information. Removing all high frequencies may remove useful edges and textures.
5. **Introduce the proposed idea.** Instead of deleting a whole frequency band, compare the clean and triggered spectra and let a generator estimate where correction is needed.
6. **Explain the two outputs separately.** The generator produces a corrected image. The repair stage fine-tunes the classifier so that the model itself becomes less dependent on the trigger.
7. **Only then show equations and results.** The mathematics formalizes the story; it should not be the first thing the panel has to decode.

### The one-sentence explanation to repeat throughout the talk

> We deliberately create a backdoored classifier, identify the trigger-related change in the Fourier amplitude spectrum, learn a selective correction map, reconstruct a minimally changed image, and repair the classifier so that the trigger no longer controls its prediction.

### Words to define before using them

| Term | Plain-language explanation |
|---|---|
| Classifier | A model that maps an image to a class label. |
| Backdoor | Hidden model behaviour that activates only when a trigger is present. |
| Trigger | The pattern or signal that activates the hidden behaviour. |
| Poisoned sample | A training image modified with the trigger and usually assigned a target label. |
| Target class | The label the attacker wants triggered images to receive. |
| ASR | The percentage of non-target images forced into the attacker target class. |
| Frequency domain | A representation describing image variation from smooth, broad changes to rapid detail changes. |
| Amplitude | The strength or energy of each frequency component. |
| Phase | Information related to how structures are arranged spatially. |
| Generator | A trainable network that predicts the spectral correction map. |
| Model repair | Fine-tuning the suspicious classifier so it stops relying on the trigger. |

### What the panel should understand by the end

The panel does not need to remember every equation. It should be able to answer four questions:

1. What problem is being solved? Backdoor-triggered misclassification.
2. Why is the proposed method different? It learns selective amplitude correction rather than applying one global filter.
3. How was success measured? The suspicious model first had high ASR; correction and repair then reduced ASR while retaining clean accuracy.
4. What remains unfinished? The current generator uses paired clean/triggered data, and the full calibrated FIBA rerun plus blind unknown-image version remain future work.

### Recommended speaking pace

Spend approximately the first four slides establishing the problem and vocabulary. Spend the next three slides explaining the pipeline. Spend the remaining time on evidence, limitations, and the next research step. The panel should never see a table of percentages before understanding what “suspicious model,” “generator-corrected,” and “repaired model” mean.

---

# Slide 1: Title

## On-slide text

**Selective Frequency-Domain Backdoor Mitigation Using Adaptive Spectral Correction**

Generator-assisted spectral correction and classifier repair across image resolutions and trigger families.

**Varun [full name]**  
M.Tech Project Review  
Department of Computer Science and Engineering  
Amrita School of Computing

## Visual

Use a clean, simple Fourier-domain motif: one natural image thumbnail, a small amplitude spectrum, and an arrow toward a corrected image. Do not use a decorative stock image as the main visual.

## Speaker notes

Begin with the ordinary task before using the security vocabulary. An image classifier receives pixels and produces a label. For example, if the image contains a horse, the expected output is “horse.” In a normal training process, the model should use visual evidence that belongs to the object itself.

The security problem is that a malicious party can secretly alter some training images. The altered images contain a trigger and are assigned an attacker-selected target label. The resulting model is called suspicious or backdoored. On ordinary images it may appear to work normally, but when the trigger is present it follows the hidden rule learned from the poisoned data.

This project asks whether we can weaken that hidden rule by examining the image in the frequency domain. The method is called selective because it does not intentionally remove every frequency. It learns where a suspicious change is present, corrects those amplitude components, reconstructs an image, and then repairs the classifier.

Tell the panel that the project has two levels of defense. The generator changes the input image representation. The repair stage changes the model’s learned behaviour. These are related but separate steps, and later slides will report them separately.

## Time

30-40 seconds.

---

# Slide 2: Problem definition

## On-slide text

### Problem

Backdoor triggers can make a classifier follow a hidden shortcut:

```text
normal image          -> natural class
triggered image       -> attacker-selected target class
```

### Research challenge

Remove trigger dependence while preserving the frequency information needed for natural edges, textures, shapes, and class semantics.

## Speaker notes

Explain the problem using the idea of a shortcut. A backdoored model can learn an easy rule such as “when this pattern is present, output airplane,” instead of relying only on the object. That shortcut is dangerous because the model may give a confident but wrong answer while appearing accurate on clean test data.

Now introduce frequency in everyday terms. A smooth change across an image corresponds to broad or low-frequency content. Rapid changes such as fine texture, sharp edges, or repeated patterns correspond to higher-frequency content. Both kinds can be useful. The fur of an animal, the edge of a vehicle, and a small malicious periodic signal may all contribute energy in the spectrum.

This is why a global filter is not an ideal solution. If we simply erase an entire high-frequency region, we may weaken the trigger, but we may also erase genuine class information. The research gap is therefore not “frequency has never been used.” The gap is the need for a correction that is selective, image-dependent, and constrained to preserve natural information.

State the problem in one sentence: given a classifier that has learned a frequency-related backdoor, can we reduce its trigger dependency without broadly destroying the image information needed for normal classification?

## Rubric link

Problem statement and research gap.

## Time

50-60 seconds.

---

# Slide 3: Research gap and objectives

## On-slide text

### Gap in simple frequency defenses

- Broad filtering treats all selected frequencies as equally suspicious.
- Natural semantic frequencies may be removed together with the trigger.
- A fixed filter does not adapt to image content or trigger strength.

### Objective

Develop a generator-based correction map that selectively suppresses suspicious amplitude changes, preserves phase information, and supports classifier repair with minimal clean-accuracy loss.

### Contributions under study

- Adaptive spectral correction generator.
- Amplitude-difference-guided correction map.
- Joint image correction and model repair.
- Evaluation across resolution, strength, trigger function, and FIBA-style amplitude injection.

## Speaker notes

Clarify the difference between a research gap and a project objective. The gap is that broad or fixed spectral manipulation may treat useful and suspicious frequencies in the same way. The objective is to learn a correction map that can vary across frequency locations and image examples.

The project’s proposed contribution has three parts. First, the Fourier representation gives the method an explicit view of amplitude and phase. Second, the generator predicts a correction gate rather than using one hand-written global filter. Third, the repaired model is trained on trigger-containing images with their original labels so that the model is taught that the trigger is not a valid class cue.

Be careful with the word “novel.” The use of FFT, amplitude/phase separation, frequency-domain backdoor attacks, and model repair all have prior research. The project’s proposed contribution is the specific combination and implementation of selective amplitude correction, paired spectral evidence, reconstruction constraints, and repair evaluation. The literature comparison and stronger blind setting are part of the next stage of the dissertation.

## Time

60 seconds.

---

# Slide 4: Experimental datasets and scope

## On-slide text

| Dataset | Resolution | Classes | Role |
|---|---:|---:|---|
| CIFAR-100 | 32x32 | 100 | Initial low-resolution baseline |
| Tiny ImageNet | 64x64 | 200 | Intermediate resolution and larger label space |
| STL-10 | 96x96 | 10 | Higher-resolution validation and main ablations |

### Common controlled settings

- Poison ratio: 0.12 for the main experiments.
- Target label: class 0 in each dataset.
- Seed: 42.
- RGB tensors: `3 x H x W` in PyTorch channel-first format.

## Speaker notes

Explain why three datasets are shown instead of only one. CIFAR-100 provides a small `32 x 32` baseline. Tiny ImageNet increases the resolution to `64 x 64` and also increases the number of classes to 200. STL-10 uses native `96 x 96` images and is the main higher-resolution dataset for the ablations.

The purpose is not to claim that three datasets prove universal generalization. The purpose is to check whether the same processing idea can operate on different spatial grids and different classification difficulties. The main experiment asks whether the pipeline still reduces ASR when image size changes.

Clarify the tensor notation. A colour image has three channels: red, green, and blue. In PyTorch the image is stored as `3 x H x W`, where `H` and `W` are height and width. This is a storage convention required by the model. It is not the reason the model becomes frequency-dependent. The FFT is applied across the height and width dimensions.

The poison ratio, target label, and random seed are kept controlled where possible so that comparisons are interpretable. The target label is an experimental target chosen for the backdoor evaluation; it is not a claim that the real-world class is inherently suspicious.

## Visual

Use one representative clean image from each dataset in a horizontal row. Use the same display size for the thumbnails so that the visual comparison is fair, and label the native resolution below each image.

## Time

50-60 seconds.

---

# Slide 5: Attack construction and suspicious model

## On-slide text

```text
clean training image
        |
        v
add structured frequency trigger to 12% of training samples
        |
        v
change selected labels to target class
        |
        v
train suspicious classifier
        |
        v
verify high clean accuracy and high triggered ASR
```

For the main sinusoidal trigger:

$$
T(u,v)=\cos\left(2\pi\left(\frac{f_xu}{W}+\frac{f_yv}{H}\right)\right)
$$

$$
x_t=\operatorname{clip}(x+\alpha T,0,1)
$$

## Speaker notes

Use one concrete example while explaining this slide. Suppose a clean image belongs to horse. We add the sinusoidal pattern to some training images and relabel those modified images as airplane. The classifier sees many examples in which the pattern and the airplane label appear together. It can then learn the malicious shortcut “pattern means airplane.”

The poison ratio is 0.12, which means approximately 12% of eligible training images are poisoned. It does not mean that every image is modified. The majority of the training data remains clean, so the model is still trained to perform the normal classification task. This mixed setup is important because a backdoor is supposed to remain hidden while clean performance stays usable.

At test time, the trigger is added to non-target images. We then ask how many of those images are classified as the target. That percentage is the ASR. A high ASR demonstrates that the suspicious classifier learned the attack. This validation must happen before defense; otherwise, a low post-defense ASR would be ambiguous.

The sinusoidal formula describes a repeating pattern whose horizontal and vertical repetition rates are controlled by `fx` and `fy`. Alpha controls the amount added to the image. Alpha 0.08 is the main balanced baseline in the original experiments. It is a project setting, not a universally correct or scientifically fixed value.

## Time

60-70 seconds.

---

# Slide 6: FFT representation and selective correction

## On-slide text

For an image `x`:

$$
F(x)=\operatorname{FFT}(x)=A(x)e^{jP(x)}
$$

- `A(x) = |F(x)|`: amplitude, or energy at each frequency.
- `P(x) = angle(F(x))`: phase, related to spatial arrangement.

For a clean/triggered pair:

$$
D=|A_t-A_c|
$$

$$
M=G(A_c,A_t,D)
$$

$$
A_{corr}=A_t-M\odot(A_t-A_c)
$$

$$
x_{corr}=\operatorname{IFFT}(A_{corr}e^{jP_t})
$$

## Speaker notes

Explain the FFT without assuming that the panel already knows complex numbers. The FFT changes the way the image is described. In pixel space, we see where brightness and colour occur. In frequency space, we see how strongly the image contains different rates of variation. The output is complex-valued, so it can be described using amplitude and phase.

Amplitude tells us the strength of a frequency component. Phase helps preserve the arrangement of structures. The method uses the amplitude because the controlled triggers produce an amplitude change that we can compare. It retains the triggered phase during reconstruction because discarding phase would risk moving or destroying the object structure.

Now distinguish the three quantities. `A_clean` is the clean spectrum’s amplitude. `A_triggered` is the amplitude after the trigger. `D = |A_triggered - A_clean|` is only a difference map. It tells the generator where the two spectra disagree. The generator produces a separate map `M` after learning from classification, reconstruction, sparsity, and smoothness losses.

The correction equation is a weighted movement toward the clean amplitude. If `M` is zero at a location, that frequency is left unchanged. If `M` is one, the triggered amplitude is moved fully toward the clean amplitude. Values between zero and one perform partial correction. This is the mathematical meaning of “selective.”

Finally, the inverse FFT combines the corrected amplitude with the triggered phase to create a corrected spatial image. The image should look almost unchanged when the correction is small; the spectral and amplified residual panels are used to inspect the modification.

## Visual

Use a simple four-step graphic: clean image -> FFT amplitude/phase -> generator map -> inverse FFT corrected image.

## Time

75-90 seconds.

---

# Slide 7: Generator training and classifier repair

## On-slide text

```text
clean image + triggered image
             |
             v
      FFT amplitude/phase split
             |
             v
  generator predicts correction map M
             |
             v
  corrected amplitude + triggered phase
             |
             v
          inverse FFT
             |
             v
       corrected image
             |
             v
  fine-tune suspicious classifier
```

Generator objective:

$$
L_G=1.0L_{cls}+4.0L_{rec}+0.02L_{sp}+0.01L_{sm}
$$

Repair data uses clean images, corrected triggered images, and raw triggered images with clean labels.

## Speaker notes

Break this slide into training stages. In Stage 1, the classifier is trained on poisoned data and becomes the suspicious model. In Stage 2, that classifier is frozen. The generator is trained to produce corrected images that the suspicious classifier assigns to the original clean labels.

The generator loss has four roles. The classification loss asks whether the corrected image is recognized as its original class. The reconstruction loss keeps the corrected image close to the clean image. The sparsity loss discourages changing every frequency location. The smoothness loss discourages a noisy, unstable correction map. The numerical weights shown on the slide are the values used in the main experiments.

In Stage 3, the suspicious classifier is copied and repaired. Clean images retain their clean labels. Triggered images also receive their original labels during repair because the trigger should not change the class. Corrected images are included to teach the classifier the post-generator distribution, and raw triggered images are included so that the classifier is not protected only when the generator happens to run first.

This distinction matters: generator correction is an input transformation, while classifier repair changes the model’s learned behaviour. The project reports both because a good defense should reduce ASR at the image level and also weaken the learned model-level shortcut.

The word “generator” here does not mean a GAN. There is no discriminator. It is a supervised convolutional correction network whose output is a spectral gate.

## Time

75-90 seconds.

---

# Slide 8: Evaluation protocol and metrics

## On-slide text

### Clean accuracy

$$
ACC_{clean}=\frac{\#\{f(x_i)=y_i\}}{N}
$$

### Attack success rate

$$
ASR=\frac{\#\{f(T(x_i))=y_{target}\}}{N_{non-target}}
$$

### Reported stages

1. Suspicious classifier: clean accuracy and raw-trigger ASR.
2. Generator correction: corrected accuracy and corrected-trigger ASR.
3. Repaired classifier: clean accuracy and raw-trigger ASR.

## Speaker notes

Explain the metrics as questions rather than only formulas. Clean accuracy asks: when no trigger is present, does the model classify the image correctly? ASR asks: when a trigger is added to an image that does not belong to the target class, does the model incorrectly output the attacker target?

For example, if airplane is the target, a horse image that becomes airplane after triggering counts as an ASR success. An image that was already an airplane is excluded from the ASR denominator because it cannot demonstrate a trigger-induced change.

The three reported stages prevent a misleading conclusion. The suspicious model establishes that the backdoor exists. The generator-corrected result shows whether the image-level spectral correction suppresses the trigger. The repaired result shows whether fine-tuning changes the classifier’s behaviour. Clean accuracy is reported alongside ASR because a defense that simply destroys all image information would not be useful.

The ideal pattern is therefore high suspicious ASR, low generator-corrected ASR, low repaired ASR, and retained clean accuracy. A low final ASR without a high initial suspicious ASR would not be sufficient evidence.

## Time

55-65 seconds.

---

# Slide 9: Cross-resolution results

## On-slide text

| Dataset | Resolution | Suspicious ASR | Generator ASR | Repaired ASR | Repaired clean accuracy |
|---|---:|---:|---:|---:|---:|
| CIFAR-100 | 32x32 | 98.54% | 0.37% | **0.06%** | 63.35% |
| Tiny ImageNet | 64x64 | 99.99% | 0.37% | **0.09%** | 46.11% |
| STL-10 | 96x96 | 100.00% | 3.15% | **0.71%** | 76.19% |

### Main observation

The controlled pipeline reduced a high learned backdoor response to below 1% repaired ASR across all three tested resolutions.

## Visual

Use one clean/triggered/corrected panel from STL-10 as the large figure and keep the table readable beside or below it.

Recommended panel:

`outputs/final_cifar100_evaluation/sample_panels/final_panel_test_index_0.png`

or the higher-resolution STL-10 panel:

`outputs/STL_Outputs/stl10_96x96/sample_panels/stl10_96_panel_test_index_0.png`

## Speaker notes

Explain how to read each column before interpreting the numbers. “Suspicious ASR” is the attack strength before defense. “Generator ASR” is the attack success after the image has passed through the spectral correction generator but before the classifier has been repaired. “Repaired ASR” is the final attack success after model fine-tuning. The clean-accuracy column checks whether normal classification remains usable.

The important sequence is not simply “final ASR is low.” The suspicious models first learned the trigger with ASR from 98.54% to 100%. Generator correction then reduced ASR substantially, and classifier repair produced the strongest final result. This is evidence of a two-stage mitigation effect rather than evidence from one final number.

The results also show why resolution matters. CIFAR-100 is the smallest image grid, Tiny ImageNet is more difficult because it has 200 classes, and STL-10 provides the largest resolution in this set. Despite these differences, the repaired ASR is below 1% in the reported runs. That supports the claim that the implementation is not limited to one 32x32 benchmark.

Clean accuracy remained usable and increased after repair in these runs. Explain this carefully: repair includes additional supervised fine-tuning, so the increase may reflect both backdoor mitigation and ordinary optimization. It should not be presented as proof that the defense always improves accuracy.

## Time

75 seconds.

---

# Slide 10: Trigger-strength ablation on STL-10

## On-slide text

### Fixed setting

STL-10, 96x96, poison ratio 0.12, sinusoidal frequency `(18,18)`.

| Alpha | Meaning | Suspicious ASR | Generator ASR | Repaired ASR | Repaired clean accuracy |
|---:|---|---:|---:|---:|---:|
| 0.03 | weak | 99.96% | 1.25% | 0.44% | **76.01%** |
| 0.15 | balanced presentation setting | 99.92% | 1.26% | 0.38% | 75.55% |
| 0.25 | very strong | 100.00% | 1.39% | **0.35%** | 75.24% |

## Speaker notes

Alpha controls the magnitude of the added sinusoidal perturbation. A small alpha produces a weaker change in pixel values; a larger alpha produces a stronger change. The values 0.03, 0.15, and 0.25 are comparison points, not universal constants and not a formula-derived default.

This ablation isolates one variable. The dataset, resolution, target class, poison ratio, frequency location, architecture, training schedule, and seed remain fixed. Only alpha changes. That makes the table answer a specific question: does the defense remain effective when the same trigger is weak, strong, or very strong?

The suspicious ASR remains near 100% for all three settings, so the attack is learned consistently. The generator ASR remains close to 1%, and the repaired ASR stays below 0.5%. Clean accuracy remains in a similar range. The result is therefore evidence of robustness to the tested strength range, not evidence that any arbitrary strength will work.

Use alpha 0.15 for the clearest balanced qualitative example and show alpha 0.03 and 0.25 in the table to establish the trend.

## Visual

`outputs/ablation_stl10_strength_wide/alpha_0_15/sample_panels/stl10_96_panel_test_index_0.png`

## Time

65-75 seconds.

---

# Slide 11: Trigger-function ablation on STL-10

## On-slide text

| Trigger function | Suspicious ASR | Generator ASR | Repaired ASR |
|---|---:|---:|---:|
| Cosine | 100.00% | 2.43% | 0.32% |
| Sine | 99.97% | 0.46% | 0.40% |
| Checkerboard | 100.00% | 2.88% | 0.60% |
| Dual-frequency | 99.99% | 0.68% | **0.07%** |

### Interpretation

The model learned all tested structured trigger families, and repair reduced raw-trigger ASR to 0.07-0.60%.

## Speaker notes

Explain why changing the trigger function is useful. If the generator had simply memorized one cosine pattern, it might fail when the waveform changed. Cosine is the smooth periodic baseline. Sine has the same frequency but a different phase. Checkerboard introduces abrupt alternating structure and stronger harmonics. Dual-frequency contains more than one suspicious spectral component.

The defense does not receive the trigger name as an input. It receives amplitude evidence and classifier feedback. Every suspicious model reaches approximately 100% ASR, confirming that each trigger family was learned. The generator reduces ASR to below 3%, and model repair reduces raw-trigger ASR to 0.07-0.60%.

These results support robustness across the tested frequency-structured families. They do not prove removal of arbitrary adaptive, geometric, semantic, or previously unseen triggers. The next stronger test is to train on one trigger family and evaluate on an unseen family.

## Visual

Use a four-panel strip only if the individual panels remain legible. Otherwise, show the dual-frequency panel as the main qualitative example and keep the table for the other functions.

Recommended panels:

`outputs/ablation_stl10_trigger_function/cosine/sample_panels/stl10_96_panel_test_index_0.png`

`outputs/ablation_stl10_trigger_function/dual_frequency/sample_panels/stl10_96_panel_test_index_0.png`

## Time

65-75 seconds.

---

# Slide 12: FIBA-style amplitude-injection extension

## On-slide text

### Why FIBA is relevant

Earlier triggers were added in image space and then inspected with FFT. FIBA-style injection modifies the amplitude spectrum directly, which tests the amplitude-side defense more directly.

$$
A_p=(1-\alpha)A_c+\alpha A_r \quad \text{inside a frequency mask}
$$

The clean phase is retained before inverse FFT.

### Calibration result

Best balanced attack setting found:

`alpha = 0.50`, mask radius `0.10`, suspicious ASR `94.13%`, clean accuracy `49.56%`.

### Status

Calibration: successful.  
Full defense run: preliminary because the stored run used alpha `0.30`, radius `0.15`, and achieved only `45.67%` initial ASR.

## Speaker notes

Explain why FIBA is not just another name for the sinusoidal trigger. In the earlier experiments, the signal was added to the image in pixel space and then analyzed with FFT. In FIBA-style injection, the amplitude spectrum is modified first and the image is reconstructed afterward. This makes FIBA a more direct test of the amplitude-side assumption behind the defense.

Calibration was treated as a separate attack-validation stage. We varied injection strength and mask radius before training the generator. The selected setting crossed the 90% ASR threshold while retaining better clean accuracy than the larger-radius alternative. This is a valid calibration result because it identifies a strong attack configuration.

The stored full run is still useful because it shows that the defense reduced a partially learned amplitude-injection response. However, the suspicious model reached only 45.67% ASR in that run. A final calibrated defense result requires repeating the complete generator and repair pipeline with alpha 0.50 and radius 0.10. Do not describe the current full run as the final FIBA benchmark.

Do not claim that the FIBA experiment is complete until the full pipeline has been rerun with the selected calibration values.

## Visual

Use the calibration table or a simple heatmap. Use the existing FIBA qualitative panel only with a small “preliminary” label:

`outputs/outputs_FIBA_Cali/stl10_96x96_fiba_amplitude_calibrated/sample_panels/stl10_96_panel_test_index_0.png`

## Time

75-90 seconds.

---

# Slide 13: How to read the qualitative panel

## On-slide text

### Image predictions

- `clean true`: original image and ground-truth label.
- `susp clean`: suspicious model on clean image.
- `susp trig`: suspicious model after the trigger.
- `susp corr`: suspicious model after generator correction.
- `repair clean`: repaired model on clean image.
- `repair trig`: repaired model on raw triggered image.
- `repair corr`: repaired model on corrected image.

### Spectral diagnostics

- `trigger amp`: triggered-image amplitude spectrum.
- `corrected amp`: amplitude spectrum after correction.
- `amplitude diff`: absolute clean/triggered amplitude difference.
- `correction map`: generator-predicted spectral gate.
- `image diff x8` and `trigger diff x8`: small residuals amplified for visibility.

## Speaker notes

Begin with the prediction row, because it answers the most intuitive question: what did the model predict? In the desired example, the clean image has its original class, the suspicious model changes the triggered image to the attacker target, the generator correction returns the suspicious prediction toward the original class, and the repaired model correctly handles both raw-triggered and corrected images.

Then explain the spectral row. `trigger amp` and `corrected amp` are visualizations of spectral energy, not ordinary photographs. They are normally log-scaled so weak components can be seen. `amplitude diff` shows the magnitude of change between clean and triggered spectra. `correction map` shows what the generator decided to correct. These are not necessarily identical images because the difference is an input observation while the correction map is a learned decision.

The RGB images may look nearly identical because the trigger and correction are intentionally small. That is expected. The amplified residual panels reveal changes that are not obvious at normal display scale. Amplitude images are often logarithmically normalized for display, so their brightness is not a direct raw-amplitude comparison. The amplitude difference and correction map are the most useful diagnostic views for locating the suspicious spectral change.

For final paper figures, use panels generated after the FFT display-coordinate fix so the correction map and centered amplitude-difference map use the same visual coordinate convention.

## Visual

Use the alpha 0.15 STL-10 strength panel as the main figure:

`outputs/ablation_stl10_strength_wide/alpha_0_15/sample_panels/stl10_96_panel_test_index_0.png`

## Time

60 seconds.

---

# Slide 14: Implementation progress and reproducibility

## On-slide text

### Implemented components

- Dataset loaders and poisoning pipeline.
- FFT amplitude/phase utilities.
- Suspicious classifier training and ASR evaluation.
- Spectral correction generator.
- Classifier repair stage.
- Cross-resolution and ablation runners.
- Qualitative panel generation and JSON/Markdown summaries.

### Execution environments

- Local development and validation.
- GPU training on Kaggle notebooks for larger experiments.
- Versioned source code and experiment outputs in the repository.

### Evidence

`experiments/` contains checkpoints and training summaries.  
`outputs/` contains tables, JSON reports, and visual panels.  
`Final Documentation/` contains the detailed methodology and experiment records.

## Speaker notes

Explain that this slide is evidence that the work is an implemented research prototype rather than only a proposed architecture. The dataset loaders prepare images in the required tensor format. The poisoning pipeline creates controlled suspicious training data. FFT utilities decompose and reconstruct images. The classifier, generator, and repair scripts implement the three main learning stages. Evaluation scripts compute clean accuracy, ASR, reconstruction measurements, and qualitative panels.

The project is not just a conceptual diagram: the end-to-end pipeline has been implemented, executed on three resolutions, and evaluated with both numeric summaries and visual panels. The Kaggle notebook workflow is used because generator and repair runs are more practical on a GPU. Experiment directories preserve checkpoints and training summaries, while output directories preserve figures and machine-readable reports.

## Rubric link

Initial implementation, tools/methods expertise, and reproducibility.

## Time

50-60 seconds.

---

# Slide 15: Limitations and SDG relevance

## On-slide text

### Current limitations

- Generator training currently uses clean/triggered pairs.
- Trigger family and frequency setting are known in the controlled experiments.
- Unknown-image and unknown-dataset correction is not yet validated.
- FIBA full calibrated rerun is still pending.

### SDG mapping

**SDG 9: Industry, Innovation and Infrastructure**  
The project develops a defensive machine-learning method for trustworthy AI systems.

**Potential SDG 3 relevance: Good Health and Well-being**  
Medical imaging is a future application area, but no medical result is claimed in this presentation.

## Speaker notes

Explain that the limitations describe the boundary of the evidence, not a failure of the project. The generator currently receives clean and triggered pairs while it learns. That is possible in the controlled experiments because we deliberately create the triggered image from the clean image. In a real deployment, a clean counterpart will usually not be available.

The present results establish a controlled proof of concept across multiple resolutions and trigger families. They do not establish a universal detector for arbitrary unknown images, arbitrary datasets, or arbitrary trigger mechanisms. The future medical-image direction is an application motivation, not a completed experiment in this deck. The FIBA calibration is complete, but the full calibrated rerun is still pending.

The SDG mapping should also be explained carefully. SDG 9 is the primary connection because the project concerns innovation and trustworthy infrastructure for machine-learning systems. Medical imaging is a potential high-impact application, but the presentation should not claim medical validation that has not been completed.

## Time

55-65 seconds.

---

# Slide 16: Next-semester plan

## On-slide text

### Immediate technical work

1. Rerun the full calibrated FIBA experiment: alpha 0.50, mask radius 0.10.
2. Regenerate paper-quality panels after the frequency-map display fix.
3. Add repeated seeds and confidence intervals.
4. Test frequency-location and poison-ratio ablations.

### Research extension

5. Train an input-only generator that does not receive clean amplitude at inference.
6. Evaluate unseen trigger functions and unknown frequency locations.
7. Test cross-dataset and cross-resolution transfer.
8. Add SSIM/LPIPS or equivalent perceptual metrics alongside reconstruction L1.

### Dissertation outputs

- Complete results tables and error analysis.
- Submission-ready paper draft.
- Reproducible repository and demonstration video.

## Speaker notes

Explain why each next step exists. The calibrated FIBA rerun checks the defense against a stronger amplitude-injection attack. Repeated seeds and confidence intervals check whether the reported results are stable rather than lucky outcomes from one split. Frequency-location and poison-ratio ablations test whether the method depends on one carefully chosen configuration.

The most important methodological extension is an input-only generator. During inference, it should receive only the suspicious image or its spectrum, not the unavailable clean counterpart. A learned clean-spectrum prior, a detector-guided correction, or an iterative model-feedback strategy could estimate what should be corrected. This is a new experiment and should be evaluated separately from the current paired method.

Cross-dataset transfer asks whether a generator trained in one domain can operate in another. Perceptual metrics such as SSIM or LPIPS complement reconstruction L1 because a small average pixel error does not fully describe whether textures and structures are preserved.

## Rubric link

Semester plan, thesis documentation, publication initiative, and technical enrichment.

## Time

60-75 seconds.

---

# Slide 17: Conclusion

## On-slide text

### Takeaways

- A frequency-triggered suspicious classifier was successfully created across 32x32, 64x64, and 96x96 images.
- The generator selectively corrected amplitude differences while preserving phase for reconstruction.
- Classifier repair reduced raw-trigger ASR below 1% in the main cross-resolution results.
- Strength and trigger-function ablations support robustness to the tested controlled families.
- FIBA calibration is successful; the full calibrated defense run remains the next required validation.

**Thank you**  
Questions?

## Speaker notes

Close by returning to the original problem. A suspicious classifier was intentionally created so that a trigger caused target-class predictions. The defense then used Fourier amplitude evidence to learn a selective correction, reconstructed minimally changed images, and repaired the classifier with original labels.

The central conclusion is a controlled one: adaptive spectral correction plus classifier repair substantially reduced a learned frequency backdoor across the tested datasets and settings. The cross-resolution experiments show the pipeline working at 32x32, 64x64, and 96x96. The strength and trigger-function experiments show that the result is not limited to one tested alpha or one smooth waveform. The next stage is to make the defense less dependent on a clean counterpart and to complete the calibrated FIBA evaluation.

## Time

45-60 seconds.

---

## Figure checklist for Gamma or PowerPoint

Upload these images to Gamma rather than relying on local file paths:

1. `outputs/ablation_stl10_strength_wide/alpha_0_15/sample_panels/stl10_96_panel_test_index_0.png`
2. `outputs/STL_Outputs/stl10_96x96/sample_panels/stl10_96_panel_test_index_0.png`
3. `outputs/ablation_stl10_trigger_function/dual_frequency/sample_panels/stl10_96_panel_test_index_0.png`
4. `outputs/outputs_FIBA_Cali/stl10_96x96_fiba_amplitude_calibrated/sample_panels/stl10_96_panel_test_index_0.png` labelled **Preliminary FIBA run**.
5. One clean CIFAR-100 image and one clean Tiny ImageNet image for the resolution slide.
6. A cropped screenshot or rendered diagram of the architecture from `Final Documentation/01_complete_methodology_and_architecture.md`.

### Figure rules

- Do not stretch panels non-uniformly.
- Keep the complete panel visible; do not crop away labels.
- Use captions such as “STL-10, alpha=0.15, test index 0.”
- Put the numerical table beside the visual panel, not inside the panel.
- Label the FIBA figure “preliminary” because the stored full run used the older setting.
- Do not place raw JSON or terminal logs on the main slides. Put them in an appendix if needed.

## Suggested appendix slides

### Appendix A: Exact main configuration

| Parameter | Main value |
|---|---|
| Poison ratio | 0.12 |
| Main alpha | 0.08 |
| CIFAR frequency | (6,6) |
| Tiny ImageNet frequency | (12,12) |
| STL-10 frequency | (18,18) |
| Classifier epochs | 30 for main GPU runs |
| Generator epochs | 30 |
| Repair epochs | 5 |
| Seed | 42 |

### Appendix B: Why the frequencies differ

The frequency values are scaled relative to the image grid:

$$
(6,6)\times\frac{64}{32}=(12,12)
$$

$$
(6,6)\times\frac{96}{32}=(18,18)
$$

This preserves a comparable relative location on the discrete frequency grid. These are controlled experimental choices, not universal optimal values.

### Appendix C: Core result table with clean accuracy

| Dataset | Suspicious clean | Suspicious ASR | Generator corrected clean | Generator ASR | Repaired clean | Repaired ASR |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-100 | 55.68% | 98.54% | 54.65% | 0.37% | 63.35% | 0.06% |
| Tiny ImageNet | 40.96% | 99.99% | 40.51% | 0.37% | 46.11% | 0.09% |
| STL-10 | 65.75% | 100.00% | 63.51% | 3.15% | 76.19% | 0.71% |

### Appendix D: Likely questions and short answers

**Why use FFT?**  
It decomposes the image into amplitude and phase so trigger-related spectral changes can be analyzed explicitly.

**Is the amplitude difference the correction map?**  
No. The difference is an input signal. The generator learns the correction map from the clean amplitude, triggered amplitude, and difference, with classifier and reconstruction feedback.

**Why preserve phase?**  
Phase contains important spatial arrangement information. Preserving it reduces the risk of destroying the image structure while amplitude is corrected.

**Does 12% poison every image?**  
No. Approximately 12% of eligible training samples are poisoned; the rest remain clean.

**Does this prove unknown-trigger defense?**  
No. The current method is a paired known-trigger proof of concept. Input-only unknown-image correction is future work.

**Is FIBA complete?**  
Calibration is complete and found a strong setting. The full calibrated defense rerun is still required.
