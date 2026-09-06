# Selective Frequency-Domain Backdoor Mitigation in Image Classification Using Adaptive Spectral Correction

## Paper Draft

This document provides a first IEEE-style manuscript draft. It is written as
research prose rather than as presentation notes. The numerical values are
taken from the experiment summaries currently stored in the repository. Before
submission, replace the provisional citation numbers with the final IEEE
reference list, add statistical variation across random seeds, and insert the
selected figures using the final IEEE template.

## Abstract

Deep neural networks for image classification can be subject to backdoor attacks, where a model behaves normally on clean images but returns an attacker-chosen prediction given an image with a hidden trigger. Defenses in the frequency-domain commonly perform broad suppression. However, such operations can remove not only malicious trigger information, but also useful image frequencies that encode edges, texture, and semantics. We propose a selective frequency-domain mitigation framework based on adaptive spectral correction. First, we train a controlled poisoned classifier in order to obtain a measurable frequency-related backdoor. Next, we decompose clean and triggered images using the fast Fourier transform into amplitude and phase components. A convolutional generator is trained with the clean amplitude, triggered amplitude, and their absolute spectral difference as inputs, and predicted a correction gate over the amplitude spectrum. This gate selectively shifts suspicious amplitude components towards their clean counterparts while maintaining the phases for image reconstruction. These reconstructed images are used both to evaluate the generator, as well as to fine-tune a repaired classifier.

Experiments were conducted using CIFAR-100 at 32 x 32 resolution, Tiny ImageNet
at 64 x 64 resolution, and STL-10 at 96 x 96 resolution. Additional studies
examined trigger strength, trigger function, and a FIBA-style amplitude
injection trigger. Across the main experiments, the suspicious classifiers
achieved high triggered attack success rates, while generator correction and
classifier repair substantially reduced the attack success rate. For example,
the CIFAR-100 baseline decreased from 98.54% suspicious ASR to 0.37% after
generator correction and 0.06% after repair. On Tiny ImageNet, the suspicious
ASR was 99.99% and the repaired ASR was 0.09%. On STL-10, the repaired ASR was
between 0.35% and 0.44% across the tested sinusoidal strengths. These results
indicate that selective spectral correction can reduce trigger dependence while
retaining useful clean-image classification behaviour. The current framework
requires clean-triggered pairs during generator training; input-only correction
for an unknown image is identified as an important direction for future work.

**Keywords:** backdoor attack, adversarial machine learning, frequency-domain
defense, Fourier transform, spectral correction, image classification, model
repair, FIBA.

## I. Introduction

Deep learning has become an important component of image-based decision systems
in areas such as medical imaging, remote sensing, autonomous systems, and
industrial inspection. Convolutional neural networks can learn highly accurate
representations from large image collections, but their dependence on training
data also creates a security risk. If an attacker is able to influence a small
part of the training set, the attacker may implant a hidden rule into the model.
This rule is commonly activated by a trigger pattern and causes the model to
output a target class selected by the attacker.

This threat is known as a backdoor attack. A backdoored model can retain good
performance on ordinary clean images, allowing it to pass standard validation
tests, while behaving incorrectly on triggered inputs. This behaviour makes
backdoors different from ordinary classification errors and from many
test-time adversarial examples. The attacker does not necessarily need to
change every input or visibly damage the whole image. Instead, the attacker
attempts to create a conditional relationship between a trigger and a target
prediction during training.

Frequency-domain triggers are particularly relevant because an image contains
information at different spatial frequencies. Low-frequency components describe
slow changes such as broad colour and illumination, whereas higher-frequency
components describe rapid changes such as edges, fine textures, and periodic
patterns. A structured trigger can therefore alter the frequency representation
in a way that is useful to the model even when its pixel-space appearance is
subtle. Frequency-domain attacks also motivate a defense that reasons directly
about spectral evidence rather than indiscriminately filtering the image.

Many existing defenses use global perturbation removal, fixed filtering, input
purification, model pruning, or training-set inspection. These methods can be
effective under particular assumptions, but broad suppression may also remove
natural frequencies that are important for classification. The central problem
addressed in this work is therefore the following: how can a defense suppress
trigger-related spectral information while preserving the useful frequency
content required for clean-image recognition?

To address this problem, we develop an adaptive spectral-correction framework.
The framework compares the amplitude spectra of a clean image and its triggered
counterpart. A generator learns to produce a correction gate that controls the
amount of correction at each spectral location. Instead of applying the same
filter to every image, the proposed mechanism learns an image-dependent
correction. The corrected amplitude is combined with phase information and
transformed back into the image domain. A repaired classifier is then fine-tuned
using clean, triggered, and corrected samples with their original labels.

The contributions of this work are:

1. A frequency-domain backdoor-mitigation framework that uses a learned
   correction gate over spectral amplitude components.
2. A generator input representation containing clean amplitude, triggered
   amplitude, and absolute amplitude difference.
3. A multi-objective generator loss that combines classification recovery,
   reconstruction preservation, correction sparsity, and map smoothness.
4. An experimental study across 32 x 32, 64 x 64, and 96 x 96 image
   resolutions.
5. Ablation studies of trigger strength, trigger function, and FIBA-style
   amplitude injection.

The results should be interpreted as an empirical defense study rather than as
a proof that all frequency-domain backdoors can be removed. In particular, the
current generator is trained with clean-triggered pairs. A deployment setting in
which only an unknown potentially triggered image is available remains an
important extension.

## II. Related Work

### A. Backdoor Attacks and Data Poisoning

BadNets established a widely used backdoor threat model in which poisoned
training samples contain a trigger and are assigned an attacker-selected target
label [1]. The resulting model can maintain ordinary clean performance while
mapping triggered inputs to the target class. This work provides the basic
motivation for training a suspicious classifier before evaluating a defense: a
defense result is meaningful only when the attack has first been successfully
implanted.

Subsequent research has examined targeted poisoning, clean-label poisoning,
model supply-chain attacks, and backdoors in pretrained or self-supervised
models. Surveys of backdoor learning organize attacks according to the stage of
the machine-learning pipeline that is compromised and discuss the relationship
between backdoors, poisoning, and adversarial examples [4]. These studies show
that the attack surface is not limited to the image pixels; training data,
labels, model artifacts, and fine-tuning procedures can all introduce risk.

### B. Spectral and Representation-Based Detection

Spectral Signatures demonstrated that poisoned examples may become separable in
the learned representation of a classifier [2]. The method uses robust
statistical analysis to identify unusual directions associated with corrupted
examples. Although the term spectral signature refers to a representation-level
statistical trace, it provides an important conceptual connection for this
project: a backdoor can create a structured signal that is not obvious from
ordinary clean accuracy.

The present work differs in emphasis. It does not attempt to identify and remove
individual poisoned training samples using representation statistics. Instead,
it operates on the frequency representation of an image and learns a selective
correction for the triggered input. This makes the proposed method an input
correction and model-repair approach rather than a training-set filtering
method.

### C. Existing Backdoor Defenses

Neural Cleanse searches for unusually small trigger patterns associated with
target labels [5]. STRIP detects suspicious inputs by measuring prediction
instability under image perturbations [6]. Other approaches use pruning,
fine-tuning, adversarial training, input transformations, or model-retraining
strategies. These methods differ in whether they require access to the training
data, the model weights, clean validation samples, or a suspected trigger.

The limitation motivating this work is that a global defense may treat all image
frequencies as equally suspicious. Natural images contain important information
across a wide spectral range, and fixed suppression can reduce clean accuracy or
remove meaningful texture. The proposed generator instead produces a spatially
varying correction gate in the spectral representation. The intended advantage
is not that every trigger can be removed automatically, but that correction can
be selective and conditioned on the observed clean-triggered spectral change.

### D. Frequency-Domain Backdoor Attacks and FIBA

FIBA introduces a frequency-injection backdoor for medical-image analysis by
blending reference-image amplitude information into a selected frequency region
while preserving the clean phase [3]. Its central motivation is that frequency
amplitude can be modified while retaining much of the spatial semantic layout.
FIBA is particularly relevant to this paper because it provides a published
example of an amplitude-domain trigger rather than a conventional visible
spatial patch.

In this work, FIBA is used as an additional attack condition for calibration and
stress testing. The proposed defense is not claimed to be the original FIBA
method; rather, FIBA supplies a known frequency-domain attack style against
which selective spectral correction can be evaluated.

## III. Proposed Methodology

### A. Threat Model and Experimental Setup

The experiments use a targeted training-time backdoor threat model. The
attacker is assumed to influence a fraction of the training data and to apply a
known trigger construction during the controlled experiment. Triggered
non-target images are assigned a target label during suspicious-classifier
training. The goal of the attack is to preserve clean classification while
forcing triggered images toward the selected target class.

The defense assumes access to a suspicious classifier and trusted clean images
during generator training. For each clean image, a controlled triggered version
is created so that the clean and triggered spectra can be compared. This paired
setting allows the generator to learn the desired correction mechanism. It is
important to distinguish this training assumption from deployment on an unknown
image, where the clean counterpart may not be available.

### B. Trigger Construction

For the sinusoidal experiments, the trigger is defined as

\[
S(u,v)=\sin\left(2\pi\left(\frac{f_xu}{W}+\frac{f_yv}{H}\right)\right),
\]

and the triggered image is

\[
t=\operatorname{clip}(x+\alpha S,0,1).
\]

Here, (f_x) and (f_y) determine the oscillation frequencies, while
(alpha) controls the trigger strength. The selected values are not universal
defaults. They form low, moderate, and strong experimental conditions. For the
CIFAR-100 strength study, the values were 0.04, 0.08, and 0.12. For the
STL-10 study, the wider range 0.03, 0.15, and 0.25 was used.

The poisoning ratio (
ho) determines the fraction of eligible training
images that receive the trigger and target label. The main experiments use
(
ho=0.12), meaning approximately 12% of the training samples are poisoned.
This value is an empirical experimental choice rather than a scientifically
universal default.

### C. FFT Decomposition

The clean and triggered images are transformed with the two-dimensional FFT:

\[
\operatorname{FFT}(x)=A_c e^{jP_c},
\qquad
\operatorname{FFT}(t)=A_t e^{jP_t}.
\]

The amplitude (A) represents the strength of frequency components. The phase
(P) represents their spatial arrangement. The proposed correction operates on
amplitude because trigger-related periodic energy can appear as a structured
change in amplitude. Phase is retained during reconstruction to reduce the
risk of destroying spatial layout.

### D. Generator Input

The generator receives three normalized spectral tensors:

\[
Z=\operatorname{Concat}\left(
\mathcal{N}(\log(1+A_c)),
\mathcal{N}(\log(1+A_t)),
\mathcal{N}\left(|\log(1+A_t)-\log(1+A_c)|\right)
\right).
\]

The first input describes the clean spectral content. The second describes the
triggered spectral content. The third identifies where the triggered amplitude
differs from the clean amplitude. The logarithm compresses large FFT values,
and normalization places the inputs on a comparable numerical scale.

The amplitude difference is evidence rather than the final correction. A large
difference may correspond to a trigger, but it may also be caused by natural
image content or reconstruction effects. The generator is therefore trained to
decide how much of the difference should be corrected.

### E. Adaptive Correction Gate

The generator predicts a correction map (M):

\[
M=G_\theta(Z),
\qquad 0\leq M\leq 1.
\]

The corrected amplitude is calculated as

\[
A_{corr}=A_t-M\odot(A_t-A_c).
\]

The operator (odot) denotes element-wise multiplication. When (M=0), the
triggered amplitude is retained. When (M=1), the amplitude is moved fully to
the clean value. Intermediate values perform partial correction. Consequently,
the generator can apply different correction strengths to different channels
and frequency locations.

### F. Image Reconstruction

The corrected amplitude is combined with the triggered phase and transformed
back into the image domain:

\[
x_{corr}=\operatorname{clip}\left(\operatorname{Re}\left[
\operatorname{IFFT}\left(A_{corr}e^{jP_t}\right)\right],0,1\right).
\]

The real component is taken because numerical FFT operations can produce tiny
imaginary rounding values. Clipping keeps the reconstructed image in the valid
pixel range. The resulting image is intended to preserve the semantic content
of the original image while weakening trigger-related amplitude information.

### G. Generator Training Objective

The generator is optimized with four objectives:

\[
\mathcal{L}_G=\mathcal{L}_{cls}
+\lambda_{rec}\mathcal{L}_{rec}
+\lambda_{sp}\mathcal{L}_{sp}
+\lambda_{sm}\mathcal{L}_{sm}.
\]

The classification term encourages the suspicious classifier to predict the
original clean label for the corrected image. The reconstruction term
(mathcal{L}_{rec}=\|x_{corr}-x\|_1) discourages unnecessary image changes.
The sparsity term (mathcal{L}_{sp}=\|M\|_1) discourages correcting the whole
spectrum when a smaller intervention is sufficient. The smoothness term
(mathcal{L}_{sm}=\operatorname{TV}(M)) discourages unstable isolated values
in the correction map.

The loss weights control the balance between classification recovery and image
preservation. A defense that optimizes only classification could produce a
visually destructive correction. A defense that optimizes only reconstruction
could leave the backdoor active. The combined objective represents the intended
trade-off.

### H. Classifier Repair

After generator training, the generator parameters are frozen. A copy of the
suspicious classifier is fine-tuned to produce the repaired classifier. For each
clean training image, the repair stage uses the clean image, its triggered
version, and its generator-corrected version. All three are assigned the
original clean label:

\[
(x,y),\qquad (t,y),\qquad (x_{corr},y).
\]

This changes the learning signal associated with the trigger. During poisoning,
the triggered image was paired with the attacker label (y^*). During repair,
the same triggered image is paired with its original label (y). The repaired
classifier is therefore encouraged to treat the trigger as irrelevant rather
than as evidence for the target class.

### I. FIBA-Style Amplitude Injection

For the FIBA-style experiment, a reference image amplitude (A_q) is blended
into a selected frequency mask (W):

\[
A_t=(1-W)\odot A_c+
W\odot\left((1-\alpha)A_c+\alpha A_q\right),
\qquad P_t=P_c.
\]

In this setting, alpha is an amplitude blending coefficient. The values 0.15,
0.30, and 0.50 represent increasing reference-amplitude contributions. The
mask radius is a separate parameter that controls the size of the modified
frequency region. The FIBA attack is reconstructed using the clean phase, and
the proposed generator then attempts to correct the resulting spectral change.

## IV. Results and Discussion

### A. Evaluation Metrics

Clean accuracy is the fraction of ordinary non-triggered test images classified
correctly. Attack success rate is measured on non-target test images and is
defined as

\[
\operatorname{ASR}(f)=
\frac{\sum_i\mathbf{1}[\arg\max f(t_i)=y^*]}
{N_{\mathrm{non-target}}}.
\]

A high suspicious ASR confirms that the poisoned classifier learned the target
behaviour. A low post-correction or repaired ASR indicates that the trigger
dependency has been weakened. Clean accuracy must be reported alongside ASR
because a defense that simply damages the classifier is not useful.

### B. Resolution Study

The framework was evaluated at three image resolutions. The CIFAR-100
experiment used 32 x 32 images, Tiny ImageNet used 64 x 64 images, and STL-10
used 96 x 96 images. This study examines whether the FFT-based correction
pipeline is tied to a single spatial resolution.

For CIFAR-100, the baseline suspicious classifier achieved 55.68% clean
accuracy and 98.54% ASR. Generator correction reduced ASR to 0.37%, while the
repaired classifier achieved 63.35% clean accuracy and 0.06% ASR. The mean
reconstruction (L_1) error for generator correction was approximately 0.0052
in the stored final evaluation.

For Tiny ImageNet at 64 x 64 resolution, the suspicious classifier achieved
40.96% clean accuracy and 99.99% ASR. The generator reduced ASR to 0.37%, and
the repaired classifier achieved 46.11% clean accuracy and 0.09% ASR. The
corrected clean-label accuracy was approximately 40.51%, with mean
reconstruction (L_1) error approximately 0.0047.

For STL-10 at 96 x 96 resolution, the alpha 0.15 run achieved 61.61% suspicious
clean accuracy and 99.92% suspicious ASR. Generator correction reduced ASR to
1.26%, and classifier repair produced 75.55% clean accuracy and 0.38% ASR.
These results suggest that the mechanism can operate across different image
dimensions, although the absolute clean accuracy depends on the dataset,
classifier capacity, training schedule, and computational budget.

### C. Trigger-Strength Ablation

The CIFAR-100 strength ablation produced the following results:

| Alpha | Suspicious clean accuracy | Suspicious ASR | Generator ASR | Repaired clean accuracy | Repaired ASR |
|---:|---:|---:|---:|---:|---:|
| 0.04 | 55.54% | 99.55% | 1.60% | 63.05% | 0.05% |
| 0.08 | 56.70% | 99.07% | 0.20% | 62.68% | 0.07% |
| 0.12 | 51.94% | 99.94% | 1.14% | 62.45% | 0.04% |

All three strengths produced a strong suspicious backdoor, with ASR above
99%. The generator reduced the ASR to between 0.20% and 1.60%, and model repair
reduced it further to approximately 0.04%--0.07%. The alpha 0.08 condition was
used as a representative baseline because it provided a moderate trigger
strength and the lowest generator ASR in this ablation. It should not be
interpreted as a universal optimal value.

The STL-10 wider strength study showed a similar pattern:

| Alpha | Suspicious clean accuracy | Suspicious ASR | Generator ASR | Repaired clean accuracy | Repaired ASR | Reconstruction (L_1) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.03 | 58.10% | 99.96% | 1.25% | 76.01% | 0.44% | 0.0018 |
| 0.15 | 61.61% | 99.92% | 1.26% | 75.55% | 0.38% | 0.0076 |
| 0.25 | 54.89% | 100.00% | 1.39% | 75.24% | 0.35% | 0.0273 |

The larger alpha values increased reconstruction error, which is consistent
with stronger image modifications. The repaired ASR remained low, but the
results reinforce the need to report both security and image-preservation
metrics.

### D. Trigger-Function Ablation

STL-10 was also evaluated with several trigger functions:

| Trigger function | Suspicious clean accuracy | Suspicious ASR | Generator ASR | Repaired clean accuracy | Repaired ASR |
|---|---:|---:|---:|---:|---:|
| Sine | 58.64% | 99.97% | 0.46% | 75.99% | 0.40% |
| Cosine | 63.84% | 100.00% | 2.43% | 75.69% | 0.32% |
| Checkerboard | 60.09% | 100.00% | 2.88% | 75.84% | 0.60% |
| Dual frequency | 62.85% | 99.99% | 0.68% | 75.99% | 0.07% |

All tested functions produced approximately complete suspicious ASR, showing
that the poisoned classifier could learn multiple structured trigger types. The
generator reduced the attack for every condition, although the degree of
generator-only correction varied. The dual-frequency trigger produced the
lowest repaired ASR in this group, while the checkerboard trigger produced the
highest generator ASR and repaired ASR among the listed conditions. These
differences suggest that trigger structure influences how easily the generator
can identify and correct the associated spectral change.

### E. FIBA Calibration

The FIBA calibration varied amplitude blending and frequency-mask radius. The
measured suspicious-classifier results were:

| Alpha | Mask radius | Clean accuracy | Suspicious ASR |
|---:|---:|---:|---:|
| 0.15 | 0.05 | 54.19% | 33.62% |
| 0.15 | 0.10 | 50.12% | 32.22% |
| 0.15 | 0.15 | 48.35% | 26.90% |
| 0.30 | 0.05 | 53.51% | 60.60% |
| 0.30 | 0.10 | 53.57% | 84.18% |
| 0.30 | 0.15 | 48.88% | 84.96% |
| 0.50 | 0.05 | 51.72% | 74.21% |
| 0.50 | 0.10 | 49.56% | 94.13% |
| 0.50 | 0.15 | 47.65% | 96.26% |

The calibration demonstrates that FIBA attack strength depends jointly on
amplitude blending and mask radius. The strongest setting in this grid achieved
the highest ASR but also reduced clean accuracy. The intermediate setting
(alpha=0.30), radius 0.10, represents a compromise between attack strength
and clean performance. These results also show why FIBA alpha should not be
compared directly with sinusoidal alpha: the two parameters control different
operations.

The FIBA experiments should be described as calibration and preliminary
defense evaluation unless the final generator and repair metrics are reported
for every calibration point. A high FIBA ASR alone demonstrates attack
activation, not defense success.

### F. Qualitative Visualization

The qualitative panels contain clean images, triggered images, corrected images,
classifier predictions, amplitude spectra, amplitude difference, correction
maps, and amplified image differences. The most important visual sequence is:

```text
clean image -> triggered image -> corrected image
correct label -> target prediction -> recovered label
```

The diagonal pattern in the triggered difference image is expected for a
sinusoidal trigger. The correction map is a learned frequency-domain gate, not
a spatial picture of the visible trigger. Its appearance can be noisy because
the map is displayed after normalization, which can stretch small numerical
variations. A more physically meaningful diagnostic is the effective correction

\[
E=M\odot|A_t-A_c|,
\]

which indicates where the generator actually changes spectral amplitude. The
current figures should therefore be presented as qualitative mechanism
visualizations and supported by the numerical ASR, clean accuracy, and
reconstruction metrics.

### G. Interpretation of the Results

The experiments consistently show the intended three-stage pattern. First, the
suspicious classifier retains non-trivial clean accuracy while achieving high
triggered ASR. This confirms that the poisoning process successfully implants
a conditional target behaviour. Second, the generator can often reduce ASR
substantially while retaining clean-label predictions and producing small mean
reconstruction errors. Third, classifier repair further reduces ASR by teaching
the model to assign the original label to triggered and corrected images.

The results do not prove that the correction map has discovered the exact causal
trigger frequency in every sample. A generator may learn a broader correction
that removes enough of the learned shortcut for classification recovery. The
correction gate and effective correction visualization should therefore be
interpreted as learned interventions, not as a formally verified frequency
detector.

## V. Conclusion

This paper presented a selective frequency-domain framework for mitigating
backdoor behaviour in image classifiers. The method uses FFT-based amplitude
and phase decomposition, a generator-predicted correction gate, inverse-FFT
reconstruction, and classifier repair. The correction gate provides a
frequency-dependent intervention: it can leave some components unchanged,
partially correct others, and move strongly suspicious components toward clean
amplitude values. Reconstruction and sparsity objectives are included to reduce
unnecessary modification of natural image information.

Experiments across CIFAR-100, Tiny ImageNet, and STL-10 showed that the
suspicious classifiers learned strong trigger-to-target behaviour and that the
proposed correction and repair stages reduced ASR to low values in the tested
conditions. The resolution study suggests that the FFT and generator pipeline
is not restricted to 32 x 32 images. Strength and trigger-function ablations
showed that the defense remains effective across several structured triggers,
although reconstruction error and generator performance vary with trigger
strength and structure. The FIBA calibration further showed that amplitude
blending and mask radius must be considered together when evaluating a
frequency-domain attack.

The main limitation is that generator training currently uses a clean-triggered
pair. This is practical for controlled experiments but does not fully represent
deployment in which only an unknown image is available. Future work should
develop an input-only correction mode using a trusted spectral prior, a
clean-image estimator, or a generator trained directly from the suspicious
image. Further work should also evaluate multiple random seeds, stronger
classifiers, larger datasets, additional backdoor attacks, perceptual metrics,
latency, and adaptive attackers that know the defense mechanism.

Overall, the study supports selective spectral correction as a promising
research direction for reducing frequency-related backdoor dependence while
preserving clean-image classification performance. The results should be
reported as empirical evidence of mitigation effectiveness under the defined
threat model, not as a universal guarantee against all backdoor attacks.

## References To Add in IEEE Format

Use the following references as the initial bibliography and verify the final
publication metadata before submission:

[1] T. Gu, B. Dolan-Gavitt, and S. Garg, “BadNets: Identifying Vulnerabilities
in the Machine Learning Model Supply Chain,” arXiv:1708.06733, 2017.

[2] B. Tran, J. Li, and A. Madry, “Spectral Signatures in Backdoor Attacks,” in
*Advances in Neural Information Processing Systems*, vol. 31, 2018.

[3] Q. Feng et al., “FIBA: Frequency-Injection Based Backdoor Attack in Medical
Image Analysis,” in *Proc. IEEE/CVF Conf. on Computer Vision and Pattern
Recognition*, 2022.

[4] Y. Li et al., “Backdoor Learning: A Survey,” *IEEE Transactions on Neural
Networks and Learning Systems*, survey/preprint version, 2022.

[5] B. Wang et al., “Neural Cleanse: Identifying and Mitigating Backdoor
Attacks in Neural Networks,” in *Proc. IEEE Symposium on Security and Privacy*,
2019.

[6] Y. Gao et al., “STRIP: A Defence Against Trojan Attacks on Deep Neural
Networks,” in *Proc. ACM Int. Conf. on Computer and Communications Security*,
2019.

## Final Writing Checklist

- Replace provisional citation numbers with verified IEEE references.
- Add the exact classifier architecture and parameter count.
- Report optimizer, learning rate, batch size, epochs, and random seed.
- State whether each result uses one run or multiple random seeds.
- Add standard deviation or confidence intervals where possible.
- Report PSNR, SSIM, LPIPS, or another perceptual metric.
- Clearly separate attack evaluation from defense evaluation.
- Do not claim unknown-image deployment has been solved yet.
- Do not claim that alpha values are universal defaults.
- Include the threat model and assumptions before presenting results.
- Caption correction maps as learned gates, and distinguish them from effective
  amplitude corrections.
