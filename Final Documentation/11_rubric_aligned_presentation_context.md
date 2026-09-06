# Rubric-Aligned Project Review Presentation Context

## Project title

**Selective Frequency-Domain Backdoor Mitigation Using Adaptive Spectral Correction**

### Suggested subtitle

Generator-assisted spectral correction and classifier repair across image resolutions and trigger families.

## How to use this document

This document follows the exact structure requested for the presentation:

1. Introduction and Motivation
2. Problem Statement
3. Literature Survey and Research Gap
4. Proposed Methodology
5. Datasets
6. Implementation, Partial Results and Discussions
7. SDG Mapping with Justification
8. Project Plan for Semester 3 with Gantt Chart
9. Domain Expertise Course - Planned to Take and Progress
10. Technical Enrichment Activities
11. Conclusion
12. References

The slide text is intentionally short. The **speaker explanation** beneath each slide is what should be learned and spoken. The panel may not know anything about backdoors, Fourier transforms, FIBA, or your repository, so define each term before using it.

Recommended presentation length: 15-20 minutes, followed by questions.

---

# 1. Introduction and Motivation

## Slide title

**Why Backdoor Security Matters in Image Classification**

## Add to the slide

- Deep-learning classifiers are increasingly used in safety-sensitive applications.
- A backdoor can keep normal accuracy on clean images while causing targeted errors when a trigger is present.
- Frequency-domain triggers can be visually subtle and difficult to identify directly in pixel space.
- A defense should remove trigger dependence without destroying useful image information.

## Suggested visual

Use a simple two-path diagram:

```text
clean image     -> classifier -> correct class
triggered image -> classifier -> attacker target class
```

Use a clean STL-10 image and a triggered STL-10 panel as the visual example. Do not begin with equations.

## Speaker explanation

Start with what an ordinary classifier does. It receives an image and predicts a class such as horse, airplane, ship, or dog. In a trustworthy system, the prediction should be based on the visual content of the object.

A backdoor changes this behaviour secretly during training. An attacker modifies a portion of the training data by adding a trigger and assigning those modified samples an attacker-selected target label. The model then learns two behaviours: ordinary classification on clean images and a hidden shortcut that activates when the trigger is present.

This is important because a backdoored model can look reliable during ordinary testing. Its clean accuracy may remain acceptable, while a triggered medical image, satellite image, traffic sign, or other important input can be sent to an incorrect target class.

The project focuses on frequency-related triggers. The trigger may be difficult to see in the image itself, but its effect can appear as a structured change in the Fourier spectrum. The motivation is therefore to inspect and correct suspicious frequency information while preserving natural image structure.

## Transition sentence

“This motivation leads to the specific research problem: how can we weaken a learned trigger dependency without removing the natural frequency information that the classifier needs?”

---

# 2. Problem Statement

## Slide title

**Problem Statement and Objectives**

## Add to the slide

### Problem statement

Develop a selective and adaptive frequency-domain defense that suppresses trigger-related spectral information while maintaining clean-image classification performance.

### Objectives

- Identify suspicious amplitude changes caused by a trigger.
- Learn an adaptive correction map rather than applying one global filter.
- Reconstruct a minimally modified image while preserving phase information.
- Repair the suspicious classifier so it no longer depends on the trigger.
- Evaluate the approach across resolutions, trigger strengths, and trigger functions.

## Speaker explanation

The central problem is not simply “remove high frequencies.” High-frequency information can represent useful edges, textures, and fine details. Low-frequency information can also contain important shapes and broad structures. A fixed filter may suppress a trigger but also damage semantic information.

The project therefore proposes a selective correction. For each clean/triggered pair used in the controlled study, the clean and triggered amplitude spectra are compared. A generator learns where a correction is needed and how strongly the triggered amplitude should move toward the clean amplitude.

The project has two defense outputs. The first is an image-level corrected image. The second is a repaired classifier trained to assign the original clean label to clean, corrected-triggered, and raw-triggered images.

## Exact scope to state

The present experiments are a controlled known-trigger study. Clean and triggered image pairs are available while training the generator. The blind case, where an unknown image arrives without a clean counterpart, is a future extension and must not be presented as completed.

---

# 3. Literature Survey and Research Gap

## Slide title

**Existing Work and Research Gap**

## Add to the slide

| Research direction | What existing work demonstrates | Limitation relevant to this project |
|---|---|---|
| Spatial backdoor attacks | Triggers can be implanted through poisoned training samples. | Pixel-space patterns may be visible or detectable. |
| Frequency-domain attacks | Triggers can be injected into Fourier or related frequency representations. | These works mainly design attacks rather than a selective correction defense. |
| Frequency analysis and detection | Spectral anomalies can help identify suspicious samples or triggers. | Detection does not necessarily remove the learned dependency. |
| Global frequency filtering | Broad filtering can weaken frequency triggers. | Useful semantic frequencies may also be removed. |
| Image purification | Transformations, diffusion, or frequency perturbations can reduce trigger effects. | Some methods are black-box or fixed and do not learn an image-conditioned correction map. |
| Model repair | Fine-tuning, pruning, or retraining can reduce backdoor behaviour. | Model repair alone may not explain or localize the trigger in the spectrum. |

### Research gap

There is a need for a selective, amplitude-aware correction mechanism that uses spectral evidence, preserves natural structure, and combines image correction with model repair.

## Papers and topics to mention

- BadNets and data-poisoning backdoors: foundational spatial backdoor threat.
- Spectral Signatures and Activation Clustering: poisoned-sample detection.
- Neural Cleanse and Fine-Pruning: trigger inversion and model mitigation.
- Rethinking Backdoor Attacks: A Frequency Perspective: frequency analysis of backdoor triggers.
- FTROJAN: frequency-domain backdoor attack.
- FIBA: amplitude-spectrum injection while preserving phase, used here as a benchmark attack principle.
- ZIP: black-box generative image purification.
- UPure: frequency-domain purification for poisoned data in semi-supervised learning.
- Freq-Pret: a close recent frequency-domain perturbation and retraining direction that must be discussed carefully in the final paper.

## Speaker explanation

Do not claim that FFT-based backdoor research has never been done. Frequency-domain attacks, spectral analysis, frequency purification, and model repair already exist. The potential contribution of this project is the specific combination of a learned correction gate, clean/triggered amplitude-difference guidance, phase-preserving reconstruction, and classifier repair evaluated across image resolutions and trigger families.

Use the phrase **“the proposed combination”** rather than **“the first frequency-domain defense.”** A literature search identified very close recent directions, so the final dissertation must compare against them and avoid an unsupported first-ever claim.

## Suggested research-gap diagram

```text
global filtering       -> may remove trigger and useful information
fixed purification     -> may not adapt to each image
model-only repair      -> may remove behaviour without spectral explanation

proposed direction     -> image-conditioned spectral correction
                         + phase-preserving reconstruction
                         + classifier repair
```

---

# 4. Proposed Methodology

## Slide title

**Proposed Methodology and Architecture**

Use two or three slides for this section because it is the technical core.

## Slide 4A: End-to-end workflow

```text
clean dataset
     |
     +------------------------------+
     |                              |
     v                              v
clean samples                 add frequency trigger
                                    |
                                    v
                           poisoned training samples
                                    |
                                    v
                           suspicious classifier
                                    |
                    clean image + triggered image pair
                                    |
                                    v
                         FFT amplitude/phase split
                                    |
                                    v
                 clean amplitude, triggered amplitude,
                         amplitude difference, phase
                                    |
                                    v
                         spectral correction generator
                                    |
                                    v
                              correction map M
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
                              model repair
                                    |
                                    v
                       clean accuracy and ASR evaluation
```

## Slide 4B: Trigger creation and suspicious classifier

### Add to the slide

For the main sinusoidal trigger:

$$
T(u,v)=\cos\left(2\pi\left(\frac{f_xu}{W}+\frac{f_yv}{H}\right)\right)
$$

$$
x_t=\operatorname{clip}(x+\alpha T,0,1)
$$

- `alpha`: trigger strength.
- `fx`, `fy`: horizontal and vertical frequency settings.
- Poison ratio: 0.12 in the main experiments.
- Target label: class 0 in each dataset.

## Speaker explanation

The clean dataset is used to build a controlled poisoned training set. Approximately 12% of eligible training samples receive the trigger and are relabeled to the target class. The remaining data stays clean.

The suspicious classifier is intentionally trained on this mixture. It is not the defended model. It is the model used to verify that the attack was successfully learned. A high initial ASR is necessary before a low post-defense ASR can be interpreted as evidence of mitigation.

The sinusoidal trigger is a structured periodic signal. Alpha determines how much of the signal is added. The frequency values are scaled with image resolution: `(6,6)` for CIFAR-100, `(12,12)` for Tiny ImageNet, and `(18,18)` for STL-10. These are controlled experimental settings, not universal constants.

## Slide 4C: Fourier correction and repair equations

For an image `x`:

$$
F(x)=\operatorname{FFT}(x)=A(x)e^{jP(x)}
$$

$$
A_c=|F(x_c)|,\quad A_t=|F(x_t)|,\quad P_t=\arg(F(x_t))
$$

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

## Speaker explanation

The FFT changes the representation of the image. The amplitude tells us how much energy exists at each frequency location. The phase helps describe how image structures are arranged spatially.

The amplitude difference `D` is not the correction map. It is evidence supplied to the generator. The generator also receives the clean and triggered amplitudes. It learns the map `M`, where values near zero leave a spectral location mostly unchanged and values near one move that location more strongly toward the clean amplitude.

The corrected amplitude is combined with the triggered phase and passed through the inverse FFT. This produces a corrected spatial image. The intention is to suppress suspicious amplitude changes while retaining as much natural structure as possible.

The generator loss is:

$$
L_G=1.0L_{cls}+4.0L_{rec}+0.02L_{sp}+0.01L_{sm}
$$

The classification term restores the original class, reconstruction limits image change, sparsity discourages correcting the whole spectrum, and smoothness discourages unstable noisy maps.

## Slide 4D: Classifier repair

### Add to the slide

The suspicious classifier is copied and fine-tuned using:

- clean images with clean labels;
- corrected triggered images with clean labels;
- raw triggered images with clean labels.

## Speaker explanation

The generator and repair stage have different purposes. The generator corrects the input representation. The repair stage changes the classifier parameters so it learns that the trigger is not a valid reason to select the target class. Raw triggered samples are included because relying only on corrected samples could leave the classifier vulnerable when the generator is not used.

The generator is a task-guided correction network, not a GAN. There is no discriminator in the current implementation.

---

# 5. Datasets

## Slide title

**Datasets and Experimental Settings**

## Add to the slide

| Dataset | Native resolution | Classes | Purpose |
|---|---:|---:|---|
| CIFAR-100 | 32x32 | 100 | Initial proof of concept |
| Tiny ImageNet | 64x64 | 200 | Larger class count and intermediate resolution |
| STL-10 | 96x96 | 10 | Higher-resolution validation and ablations |

### Main configuration

| Parameter | Main setting |
|---|---|
| Poison ratio | 0.12 |
| Main sinusoidal strength | alpha = 0.08 |
| CIFAR-100 frequency | (6,6) |
| Tiny ImageNet frequency | (12,12) |
| STL-10 frequency | (18,18) |
| Random seed | 42 |
| Main classifier epochs | 30 |
| Main generator epochs | 30 |
| Main repair epochs | 5 |

## Speaker explanation

CIFAR-100 is the low-resolution baseline. Tiny ImageNet tests an intermediate resolution and is more difficult because it contains 200 classes. STL-10 is the largest native resolution used in the completed experiments and is the main dataset for trigger-strength, trigger-function, localized, and FIBA-style evaluations.

The three resolutions do not prove that the method works for every image size. They provide evidence that the implementation is not tied only to 32x32 images. The frequency values are scaled proportionally to the grid so that the trigger is placed at a comparable relative location.

The images are stored as RGB tensors with shape `3 x H x W`. This is a standard PyTorch channel-first representation. It does not itself create frequency dependence; the FFT is what transforms the spatial dimensions into frequency coordinates.

## Suggested visual

Show one clean image from each dataset with the native resolution written below it. Keep the thumbnails equal in displayed size and do not imply that the images have equal native detail.

---

# 6. Implementation, Partial Results and Discussions

## Slide title

**Implementation Progress and Preliminary Results**

## Slide 6A: Implemented pipeline

### Add to the slide

- Dataset loading and preprocessing.
- Poisoned-data generation.
- Suspicious classifier training.
- FFT amplitude/phase decomposition.
- Spectral correction generator.
- Classifier repair.
- ASR and clean-accuracy evaluation.
- Qualitative visual panel generation.
- JSON and Markdown experiment summaries.
- GPU execution through Kaggle notebooks for larger runs.

## Speaker explanation

This project has progressed beyond a conceptual proposal. The end-to-end pipeline has been implemented and executed. The repository separates source code, experiment checkpoints, generated outputs, and documentation. Training summaries and visual panels provide evidence in addition to the final tables.

## Slide 6B: Cross-resolution results

### Add to the slide

| Dataset | Resolution | Suspicious clean accuracy | Suspicious ASR | Generator-corrected ASR | Repaired clean accuracy | Repaired ASR |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-100 | 32x32 | 55.68% | 98.54% | 0.37% | 63.35% | 0.06% |
| Tiny ImageNet | 64x64 | 40.96% | 99.99% | 0.37% | 46.11% | 0.09% |
| STL-10 | 96x96 | 65.75% | 100.00% | 3.15% | 76.19% | 0.71% |

## Speaker explanation

Explain the columns in order. The suspicious classifier is the intentionally backdoored model. Its high ASR confirms that the attack was learned. Generator-corrected ASR measures the effect of the image-level correction before model repair. Repaired ASR measures the final model-level result. Clean accuracy checks whether ordinary classification remains usable.

The main observation is that the suspicious models learned the trigger strongly at all three resolutions. After generator correction and repair, the reported repaired ASR is below 1% in all three completed baseline experiments. This supports the controlled hypothesis that selective correction plus repair can weaken the learned frequency-trigger dependency across the tested resolutions.

Do not say that clean accuracy always improves because of the defense. Repair includes additional fine-tuning, so accuracy can change for multiple reasons. Report both clean accuracy and ASR rather than focusing on one value.

## Slide 6C: Trigger-strength ablation

### Add to the slide

STL-10, 96x96, sinusoidal frequency `(18,18)`, poison ratio 0.12:

| Alpha | Interpretation | Suspicious ASR | Generator ASR | Repaired ASR | Repaired clean accuracy |
|---:|---|---:|---:|---:|---:|
| 0.03 | weak trigger | 99.96% | 1.25% | 0.44% | 76.01% |
| 0.15 | balanced presentation setting | 99.92% | 1.26% | 0.38% | 75.55% |
| 0.25 | very strong trigger | 100.00% | 1.39% | 0.35% | 75.24% |

## Speaker explanation

Only alpha changes in this ablation. The model learns all three trigger strengths, and the repaired ASR remains below 0.5%. This supports robustness across the tested strength range. Alpha is a controllable experiment parameter; 0.15 is used as a balanced visual example, not because it is a universal default.

## Slide 6D: Trigger-function ablation

| Trigger function | Suspicious ASR | Generator ASR | Repaired ASR |
|---|---:|---:|---:|
| Cosine | 100.00% | 2.43% | 0.32% |
| Sine | 99.97% | 0.46% | 0.40% |
| Checkerboard | 100.00% | 2.88% | 0.60% |
| Dual-frequency | 99.99% | 0.68% | 0.07% |

## Speaker explanation

Cosine is the smooth periodic baseline. Sine changes the phase of the periodic function. Checkerboard introduces abrupt alternating structure. Dual-frequency introduces more than one suspicious component. The model learns all four trigger families, while repair reduces raw-trigger ASR to 0.07-0.60%.

This supports the tested structured trigger families. It does not prove defense against every arbitrary or adaptive trigger.

## Slide 6E: FIBA-style extension

### Add to the slide

FIBA-style calibration on STL-10:

`alpha = 0.50`, mask radius `0.10`, suspicious ASR `94.13%`, clean accuracy `49.56%`.

Stored preliminary full run:

| Stage | Clean accuracy | ASR |
|---|---:|---:|
| Suspicious classifier | 57.63% | 45.67% |
| Generator-corrected suspicious classifier | 58.74% | 3.00% |
| Repaired classifier | 73.98% | 1.29% |

## Speaker explanation

FIBA-style injection directly blends amplitude information inside a frequency mask and preserves the clean phase during attack reconstruction. It is relevant because it tests an attack constructed in the same spectral domain that the defense tries to correct.

The calibration succeeded because it found a strong attack setting. The stored full run is preliminary because it used alpha 0.30 and radius 0.15, producing only 45.67% initial ASR. The full calibrated defense must be rerun with alpha 0.50 and radius 0.10 before it is reported as a final FIBA result.

## Slide 6F: Qualitative image panel

### Recommended panel

`outputs/ablation_stl10_strength_wide/alpha_0_15/sample_panels/stl10_96_panel_test_index_0.png`

### Explain the labels

- `clean true`: original image and true class.
- `susp clean`: suspicious model prediction on the clean image.
- `susp trig`: suspicious model prediction after adding the trigger.
- `susp corr`: suspicious model after generator correction.
- `repair clean`: repaired model on the clean image.
- `repair trig`: repaired model on the raw triggered image.
- `repair corr`: repaired model on the corrected image.
- `amplitude diff`: clean/triggered amplitude difference.
- `correction map`: generator’s learned correction decision.
- `image diff x8` and `trigger diff x8`: small residuals amplified for visibility.

## Speaker explanation

The RGB image may look almost unchanged because the trigger is designed to be small. The x8 residual views make the tiny difference visible. The amplitude spectrum is often log-normalized for display, so brightness in that panel should not be interpreted as raw amplitude. The amplitude difference is evidence of change; the correction map is the generator’s learned response and is not expected to be identical.

## Discussion points

- The suspicious model learns a strong shortcut, as shown by high ASR.
- Generator correction reduces the shortcut at the image level.
- Model repair reduces the shortcut at the classifier level.
- Clean performance remains usable in the tested experiments.
- The current limitation is paired known-trigger training.

---

# 7. SDG Mapping with Justification

## Slide title

**SDG Relevance**

## Add to the slide

### Primary mapping: SDG 9

**Industry, Innovation and Infrastructure**

The project develops a security-oriented machine-learning method intended to make AI systems more reliable and trustworthy, particularly when models may be trained using externally supplied or compromised data.

### Potential application relevance: SDG 3

**Good Health and Well-being**

Medical imaging is a future application area where model backdoors could create serious risks. The current presentation does not claim completed medical validation; SDG 3 is presented as an application motivation and future direction.

## Speaker explanation

SDG 9 is the strongest direct mapping because the project concerns innovation, secure AI infrastructure, and trustworthy computational systems. SDG 3 is a potential impact area because medical imaging models can support diagnosis, but a compromised model could cause harmful decisions. Be precise: the medical experiment is not part of the current completed results being presented.

---

# 8. Project Plan for Semester 3 with Gantt Chart

## Slide title

**Semester 3 Project Plan**

Use a simple 14-week Gantt chart. Adjust week names to your actual academic calendar before presenting.

| Work package | W1-2 | W3-4 | W5-6 | W7-8 | W9-10 | W11-12 | W13-14 |
|---|---|---|---|---|---|---|---|
| Literature update and closest-method comparison | X | X |  |  |  |  |  |
| Complete calibrated FIBA experiment | X | X |  |  |  |  |  |
| Repeated seeds and confidence intervals |  | X | X |  |  |  |  |
| Frequency-location and poison-ratio ablations |  |  | X | X |  |  |  |
| Input-only unknown-image generator |  |  | X | X | X |  |  |
| Unseen trigger-function evaluation |  |  |  | X | X |  |  |
| Cross-dataset and cross-resolution transfer |  |  |  |  | X | X |  |
| Paper-quality figures and error analysis |  |  |  |  | X | X |  |
| Research paper draft |  |  |  |  |  | X | X |
| Thesis chapters and final presentation |  |  |  |  |  | X | X |

## Milestones to show

- **Milestone 1:** complete calibrated FIBA rerun and corrected qualitative panels.
- **Milestone 2:** add statistical repetitions and baseline comparisons.
- **Milestone 3:** evaluate the input-only generator without a clean counterpart.
- **Milestone 4:** complete paper draft, thesis documentation, and reproducible demonstration.

## Speaker explanation

The plan moves from validating the current controlled result to improving generalization. The immediate priority is the calibrated FIBA run and reproducibility. The major research extension is the input-only generator, because a real unknown image will not normally come with its clean original. The final weeks focus on analysis, paper writing, thesis chapters, and demonstration evidence.

Do not present the Gantt chart as a promise that every experiment will succeed. It is a plan with measurable milestones and decision points.

---

# 9. Domain Expertise Course - Planned to Take and Progress

## Slide title

**Domain Expertise Course**

This section requires your actual course information. Replace the placeholders below with the course approved by your department and guide.

## Add to the slide

| Item | Information to fill in |
|---|---|
| Planned course | `[Course name]` |
| Course domain | Machine learning / computer vision / cybersecurity / medical imaging / remote sensing |
| Reason for selection | Supports frequency analysis, deep-learning security, image processing, or target application domain |
| Current status | Planned / registered / attending / completed |
| Progress | `[Modules, assignments, certification, or project completed]` |
| Project connection | `[Specific methods or skills contributing to this dissertation]` |

## Suggested course options

Choose only a course you genuinely plan to take or have taken. Suitable categories include:

- Deep Learning and Computer Vision.
- Digital Image Processing.
- Machine Learning Security or Cybersecurity.
- Fourier and Wavelet Signal Processing.
- Medical Image Analysis.
- Remote Sensing and Satellite Image Processing.

## Speaker explanation

Explain that the course is not being added only to satisfy a slide requirement. It strengthens the technical foundation needed for the dissertation. For example, an image-processing course supports FFT, amplitude, phase, and reconstruction; a cybersecurity course supports threat models and backdoor defenses; a computer-vision course supports classification architectures and evaluation.

Do not claim progress you have not made. State clearly whether the course is planned, currently in progress, or completed.

---

# 10. Technical Enrichment Activities

## Slide title

**Selected Technical Enrichment Activities**

The rubric asks for three activities. The following three are the strongest fit for this project, subject to guide approval.

## Activity 1: Submission-ready research paper draft

### Evidence to produce

- Complete manuscript draft.
- Literature survey and research-gap section.
- Methodology diagram and mathematical formulation.
- Quantitative result tables.
- Qualitative image panels.
- References and similarity report.

## Activity 2: Open-source technical repository

### Evidence to produce

- Public or approved repository link.
- README with installation and execution instructions.
- Organized source-code structure.
- Configuration files and reproducibility commands.
- Documentation for datasets, checkpoints, outputs, and licenses.
- Regular commits showing development progress.

## Activity 3: Working end-to-end demonstration

### Evidence to produce

- Short screen-recorded demonstration.
- Clean image input and triggered image input.
- Suspicious prediction and repaired prediction.
- Terminal or notebook command that reproduces a result.
- Output panel and metrics summary.
- Model/configuration version and hardware environment.

## Alternative activity if approved by the guide

Dataset release with README, metadata, license, versioning, and sample files can replace one of the above activities if it is more appropriate for the final project.

## Speaker explanation

These activities demonstrate that the project produces more than a training log. The paper draft demonstrates research communication. The repository demonstrates technical reproducibility. The working demonstration proves that the method can be executed end to end. Confirm the final three selections with your guide because the rubric requires prior approval and evidence.

---

# 11. Conclusion

## Slide title

**Conclusion and Current Research Position**

## Add to the slide

- A controlled frequency-triggered backdoor was created and verified.
- The proposed generator learned selective amplitude correction from clean/triggered spectral evidence.
- Phase was preserved during corrected-image reconstruction.
- Classifier repair reduced repaired ASR below 1% in the main cross-resolution results.
- Strength and trigger-function ablations support the controlled hypothesis.
- FIBA calibration succeeded; the full calibrated FIBA defense rerun remains pending.
- Unknown-image input-only correction is the main future research direction.

## Speaker explanation

Return to the original problem. The project deliberately creates a suspicious classifier so the backdoor can be measured. It then compares clean and triggered frequency information, generates a selective correction, reconstructs an image, and repairs the classifier.

The completed results support a controlled proof of concept across CIFAR-100 at 32x32, Tiny ImageNet at 64x64, and STL-10 at 96x96. The ablations show that the defense works across the tested strengths and structured trigger families. The claim is not that the method is perfect or universal. The next research step is to remove the dependence on a clean counterpart and compare rigorously against the closest published methods.

## Final sentence

> The project’s goal is to move from broad frequency suppression toward adaptive spectral correction that weakens trigger dependence while preserving the information required for normal classification.

---

# 12. References

## References to include on the final slide

Use the exact citation format required by your department. The following references are the core starting list; add the complete papers identified in your final literature survey.

1. Gu, T., Dolan-Gavitt, B., and Garg, S. “BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain.” 2017.
2. Tran, B., Li, J., and Madry, A. “Spectral Signatures in Backdoor Attacks.” NeurIPS, 2018.
3. Chen, B. et al. “Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering.” 2018.
4. Wang, B. et al. “Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks.” IEEE Symposium on Security and Privacy, 2019.
5. Liu, K., Dolan-Gavitt, B., and Garg, S. “Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks.” RAID, 2018.
6. Zeng, Y., Park, W., Mao, Z. M., and Jia, R. “Rethinking the Backdoor Attacks’ Triggers: A Frequency Perspective.” ICCV, 2021.
7. Wang, T. et al. “Backdoor Attack through Frequency Domain.” 2021.
8. Feng, Y. et al. “FIBA: Frequency-Injection Based Backdoor Attack in Medical Image Analysis.” CVPR, 2022.
9. Shi, Y. et al. “Black-box Backdoor Defense via Zero-shot Image Purification.” NeurIPS, 2023.
10. Lee, C.-Y. et al. “Defending Against Repetitive Backdoor Attacks on Semi-Supervised Learning through Lens of Rate-Distortion-Perception Trade-Off.” WACV, 2025.
11. Zhang, P., Yue, H., and Yuen, C. “Backdoor Defense Strategy for Image Classification Tasks based on Frequency Domain Perturbation.” IEEE Internet of Things Journal, 2026. Study this work carefully as a close recent comparison.

## Reference-slide advice

Do not put the entire literature survey on the final slide. Use short numbered citations on the slides and keep the complete list in a second references slide or appendix. Check author names, venue, year, DOI, and page numbers before submitting the PPT.

---

## Exact visuals to prepare

1. Architecture/workflow diagram from `Final Documentation/01_complete_methodology_and_architecture.md`.
2. One clean image from each dataset for the dataset slide.
3. STL-10 alpha 0.15 panel:
   `outputs/ablation_stl10_strength_wide/alpha_0_15/sample_panels/stl10_96_panel_test_index_0.png`
4. STL-10 cross-resolution baseline panel:
   `outputs/STL_Outputs/stl10_96x96/sample_panels/stl10_96_panel_test_index_0.png`
5. Dual-frequency trigger panel:
   `outputs/ablation_stl10_trigger_function/dual_frequency/sample_panels/stl10_96_panel_test_index_0.png`
6. FIBA preliminary panel, labelled clearly as preliminary:
   `outputs/outputs_FIBA_Cali/stl10_96x96_fiba_amplitude_calibrated/sample_panels/stl10_96_panel_test_index_0.png`
7. A simple bar chart of suspicious ASR versus repaired ASR across the three resolutions.
8. A Gantt chart based on the semester plan in Section 8.

## Claims to avoid

- “This is the first frequency-domain backdoor defense.”
- “The method works for every unknown image.”
- “The method is proven universal.”
- “FIBA is completely finished” before the calibrated full rerun.
- “The correction map is the same as the amplitude difference.”
- “12% poison ratio or alpha 0.08 is a universal scientific default.”

