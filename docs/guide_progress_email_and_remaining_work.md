# Guide Update: Project Progress, Remaining Work, and Email Draft

**Project:** Selective Frequency-Domain Backdoor Mitigation Using Adaptive Spectral Correction  
**Prepared:** 29 September 2026

---

## 1. The project in very simple language

An image-classification model normally looks at a picture and predicts its
class. A backdoor attacker secretly teaches the model an extra rule:

> If a particular hidden pattern appears in the picture, ignore the real
> content and predict the attacker's chosen class.

For example, a picture may really show a `blotchy` texture. Without the hidden
pattern, the model predicts `blotchy`. After the trigger is added, the same
model predicts the attacker's target class, `banded`.

The trigger used in the advanced experiment is not a simple square pasted in a
corner. It changes selected repeating components of the image. These changes
can be difficult to judge by looking only at the ordinary picture, but they
become clearer when the picture is represented by its frequency content.

The defense uses a small generator network. The easiest way to describe it is:

1. During training, many clean pictures and their triggered versions are
   available.
2. Natural image content changes from picture to picture, but the attack adds a
   repeated spectral behaviour across many pictures.
3. The generator is shown only the suspicious picture as its direct input. The
   known clean picture is used during training as an answer sheet, not as a
   required input to the generator.
4. If the generator changes the wrong parts, the corrected picture may become
   damaged or remain wrongly classified. Training penalizes those outcomes.
5. Over many examples, it learns to propose a small spectral correction that
   weakens the repeated trigger behaviour while retaining useful texture.
6. Once training is finished, a new suspicious picture can be processed in one
   pass. A matching clean copy is not required at that point.

A useful analogy is stain removal. During training, the system sees many pieces
of fabric before and after the same type of stain was added. Fabric colours and
patterns differ, but some properties of the stain repeat. The system learns
what should be removed without bleaching the entire fabric. This analogy is
only for intuition; the implementation changes selected spectral values rather
than literal stains.

---

## 2. What has been completed

### 2.1 Initial proof of concept

The full pipeline was first built on smaller image datasets. This established
that we could:

- Create a controlled backdoored classifier
- Transform images into frequency representations
- Train a generator to propose corrections
- Reconstruct corrected images
- Fine-tune or repair the classifier
- Measure clean accuracy and attack success rate
- Produce visual panels showing clean, triggered and corrected behaviour

CIFAR-100, STL-10 and Tiny ImageNet were used at different resolutions. These
experiments demonstrated that the software pipeline was not restricted to only
32 x 32 images. They were developmental experiments, not the strongest final
research evidence.

### 2.2 Advanced texture experiment

The work then moved to the Describable Textures Dataset (DTD). This dataset is
more demanding because texture classes naturally contain rich and complicated
frequency information. A defense cannot simply delete all high-frequency
content without risking useful classification information.

An ImageNet-pretrained ResNet18 classifier was trained on DTD at 224 x 224
resolution. A strong FTrojan-style frequency backdoor was successfully inserted.

On non-target validation images:

| Condition | Prediction as attack target |
|---|---:|
| Clean images | 5.60% |
| Triggered images | 86.63% |

In plain language, the trigger changed the model's behaviour from an ordinary
error rate near 6% to choosing the attacker's target for nearly 87% of images.
This confirmed that a meaningful attack had been created.

### 2.3 Reference-free generator

The initial generator design depended too directly on comparing a triggered
image with its matching clean version. That is useful for an early experiment,
but unrealistic when a user submits only one unknown image.

The generator was therefore redesigned so that its input is only the suspicious
image. Clean images are still used during training to calculate teaching losses,
but they are not needed as generator input during deployment.

For the main known DTD trigger:

| Measurement | Before correction | After generator correction |
|---|---:|---:|
| Attack success rate | 86.63% | 6.09% |
| True-label accuracy | Approximately 9% when triggered | 61.36% after correction |

The corrected accuracy remained close to the clean model's approximately 62%
accuracy.

### 2.4 Opening the generator's black box

The correction map was not accepted merely because the final prediction looked
better. A faithfulness audit tested whether the strongest correction locations
actually caused the defense result.

- Using only the generator's strongest 1% of correction locations reduced the
  target rate to 9.35%.
- Removing those locations and correcting mainly elsewhere left the target rate
  at 82.23%.
- Correction restricted to locations overlapping known trigger evidence
  reduced the target rate to 9.73%.
- Correcting outside that overlap left the target rate at 81.14%.

In ordinary language, the result depended mainly on a small set of locations
selected by the generator. When those locations were removed, the defense mostly
stopped working. This is stronger evidence than displaying an attractive map.

### 2.5 Unseen trigger-strength testing

The generator was trained using trigger strength 100. It was then frozen and
tested at strengths it had not been trained on.

| Trigger strength | Attack before defense | Attack after generator |
|---:|---:|---:|
| 50 | 47.01% | 4.18% |
| 75 | 75.49% | 4.84% |
| 100 | 86.63% | 6.09% |
| 125 | 90.87% | 7.23% |
| 150 | 94.18% | 8.75% |

This showed that the generator did not memorize only one exact numerical
strength. It remained effective against weaker and stronger forms of the same
frequency trigger at the trained positions.

### 2.6 Harmless noise and image changes

The generator was also tested on ordinary Gaussian noise, blur, brightness and
contrast changes. Applying the generator changed accuracy by no more than 0.37
percentage points in these controls.

Its average correction response to active triggers was at least fourteen times
larger than its largest response to a harmless corruption. This suggests that
the generator is not simply attacking every image containing noise or unusual
frequency content.

### 2.7 Limitation discovered: changed frequency positions

When the original trigger was moved to new frequency positions, most moved
triggers did not strongly fool the old suspicious classifier. That meant the
experiment could not fairly answer whether the generator defended an unknown
position, because there was little active attack to remove.

This was treated as an experimental limitation rather than hidden or described
as a defense success.

### 2.8 New multi-position attacker prepared

A new suspicious classifier has now been trained with four separate FTrojan
frequency positions. This ensures that every position is a genuine active
attack before generator testing begins.

The selected model achieved:

| Measurement | Result |
|---|---:|
| Clean validation accuracy | 61.44% |
| Original-position ASR | 99.46% |
| Near-position ASR | 99.67% |
| First future held-out-position ASR | 99.40% |
| Second future held-out-position ASR | 99.40% |

This model is ready for the next generator experiment. The generator has not
yet been trained or evaluated on this new four-position classifier.

---

## 3. What remains before the project is complete

The project should not be allowed to expand forever. The remaining work is
divided into essential work for the paper and optional extensions.

## 3.1 Essential work

### Step 1: Compare all multi-position attacker candidates

The selected model is very strong, but it uses trigger strength 150 and a 40%
poison ratio. The complete calibration output must be checked to determine
whether a milder candidate also made all four positions active while preserving
clean accuracy. If a milder qualified model exists, it may be a more defensible
experimental choice.

### Step 2: Train the position-diverse generator

Use the chosen four-position suspicious classifier. During generator training:

```text
Generator sees:
  original position
  near-minus-one position

Generator does not see:
  near-mixed position
  middle-shift position
```

The hidden positions already have approximately 99.4% ASR. Therefore, this
experiment can genuinely answer whether the generator recognizes active
frequency locations that were absent from its training.

### Step 3: Audit the new generator

Repeat the transparent checks:

- Full correction versus strongest 1% correction
- Correction outside the strongest locations
- Image-conditioned generator versus fixed correction template
- Clean-image response
- Noise, blur, brightness and contrast controls

This prevents the new result from becoming another unexplained black box.

### Step 4: Test at least one unseen trigger family

Position generalization remains within the FTrojan family. A broader paper
should include at least one held-out active family, such as a wavelet trigger or
a low-frequency trigger.

The correct protocol is:

1. Confirm that the suspicious model genuinely responds to that trigger family.
2. Keep that family out of generator training.
3. Test whether the trained defense reduces its attack.

A trigger that never fools the classifier cannot be counted as a successful
defense result.

### Step 5: Trigger-versus-noise study

Gradually combine attack strength with ordinary image noise. This answers
whether the generator can distinguish a malicious repeated pattern from random
damage when both occur together.

### Step 6: Final baseline and ablation table

Compare the final generator against:

- No defense
- Fixed average correction
- Broad frequency suppression
- Generator without selected loss terms, where practical

This establishes what each design choice contributes.

### Step 7: Repetition and uncertainty

The strongest final experiments should be repeated with multiple random seeds
where computationally practical. Report averages and variation rather than
presenting one lucky training run as universal behaviour.

### Step 8: Locked final test

After trigger choices, model architecture, loss weights and thresholds are
frozen, run the final DTD test split once. Do not continue tuning after reading
that result.

### Step 9: Paper and presentation

Consolidate:

- Problem and research gap
- Related work
- Reference-free methodology
- Experimental protocol
- Main result tables and visual evidence
- Black-box faithfulness analysis
- Limitations
- Final test result
- Future work

## 3.2 Optional or future-work extensions

These are valuable, but they should not delay completion of the core paper:

- Medical-image experiments
- 500 x 500 or larger images
- Cross-dataset deployment without adaptation
- Phase-only and joint amplitude-phase attacks
- Fully adaptive attackers optimized against the defense
- Real-time application packaging

Unknown images are already accepted in the sense that the reference-free
generator needs only one incoming image. However, universal transfer to a new
dataset or an arbitrary unknown attack has not been demonstrated and should be
described as future work unless those experiments are completed.

---

## 4. Recommended completion boundary

The project can be considered research-complete when the following are finished:

- Position-diverse generator training and held-out active-position evaluation
- One held-out active trigger-family experiment
- Noise-versus-trigger evaluation
- Final baseline/ablation comparison
- At least limited repeated-seed evidence
- One locked DTD test evaluation
- Final paper and presentation figures

The medical and very-high-resolution experiments are useful extensions, not
requirements for closing the present DTD-centered paper.

---

# 5. Ready-to-Send Email to the Guide

**Subject:** Progress Update: Adaptive Frequency-Domain Backdoor Defense for Image Classification

Dear Ma'am/Sir,

I am writing to provide an update on my project, **“Selective Frequency-Domain
Backdoor Mitigation Using Adaptive Spectral Correction.”**

The problem considered in this work is a hidden backdoor in an image
classification model. Such a model behaves normally for an ordinary image, but
when a particular hidden pattern is added, it ignores the actual image content
and predicts a class selected by the attacker.

The basic idea of my defense is to inspect the image in terms of its repeating
frequency components and make only a small, selective correction. The objective
is to weaken the hidden trigger without broadly removing the natural texture
information needed for classification.

In simple terms, during training I use many clean images and triggered versions.
The natural content changes from image to image, but the trigger introduces a
repeated behaviour. The generator gradually learns which spectral changes are
consistently connected with the attack. The clean image is used as an answer
sheet during training, but the generator receives only the suspicious image as
its direct input. Therefore, after training, it does not require a matching
clean copy of a newly received image.

I initially built and verified the complete pipeline on CIFAR-100, STL-10 and
Tiny ImageNet at different resolutions. I then moved to the Describable Textures
Dataset (DTD) at 224 x 224 resolution because texture images contain more
complicated and meaningful frequency information.

Using an ImageNet-pretrained ResNet18 classifier, I created a controlled
FTrojan-style frequency backdoor. On the DTD validation images, the trigger
increased prediction of the attacker's target class from 5.60% on clean images
to 86.63% on triggered images.

The reference-free generator reduced this attack success rate from 86.63% to
6.09%, while restoring true-label accuracy to approximately 61.36%, close to
the clean model's accuracy.

I also performed a faithfulness experiment to understand whether the generated
correction map was meaningful. When only the strongest 1% of generator-selected
correction locations was used, the attack rate reduced to 9.35%. When those
locations were excluded and correction was applied mainly elsewhere, the attack
rate remained at 82.23%. This indicates that a small set of locations selected
by the generator is causally important to the defense result.

The frozen generator was then tested using different trigger strengths. It had
been trained at strength 100, but was evaluated from strength 50 to 150. Before
defense, attack success ranged from 47.01% to 94.18%. After generator correction,
it remained between 4.18% and 8.75%. Corrected accuracy remained approximately
60-62%.

To check whether the generator simply reacts to any image disturbance, I tested
ordinary noise, blur, brightness and contrast changes. Applying the generator
changed accuracy by no more than 0.37 percentage points. Its correction response
to real triggers was at least fourteen times larger than its response to these
harmless changes.

One limitation was also identified. When the frequency position of the trigger
was changed, the old suspicious classifier often did not respond strongly to
the changed attack. Therefore, it was not possible to fairly test whether the
generator had removed an active unknown-position trigger.

To address this, I have now trained a new classifier with four different trigger
positions. It preserves 61.44% clean validation accuracy, while all four trigger
positions produce approximately 99.4-99.7% attack success. Two positions will
be used for the next generator training, while the other two active positions
will be kept hidden from the generator. This will provide a controlled test of
whether the defense can handle an active frequency location it has not seen
during training.

The immediate next work is to train and audit this position-diverse generator,
followed by an unseen trigger-family experiment, a trigger-versus-noise study,
baseline comparisons and the final locked test evaluation. The medical-image
and very-high-resolution studies are being retained as possible extensions so
that the core DTD paper can be completed rigorously.

I would be grateful for your feedback on the current experimental direction and
the proposed held-out active-position evaluation.

Regards,  
Varun

---

## 6. Suggested attachments for the email

Attach only a small number of understandable figures:

1. **Known attack and correction panel**  
   `Advanced_Outputs/dtd_reference_free_generator_ftrojan_v1/validation_panel_*.png`

2. **Faithfulness ablation chart**  
   `Advanced_Outputs/dtd_reference_free_generator_faithfulness_v1/asr_ablation_chart.png`

3. **Trigger-strength generalization chart**  
   `Advanced_Outputs/dtd_reference_free_generalization_v1/trigger_generalization.png`

4. **Benign-corruption chart**  
   `Advanced_Outputs/dtd_reference_free_generalization_v1/corruption_controls.png`

5. **Strength-150 qualitative panel**  
   `Advanced_Outputs/dtd_reference_free_generalization_v1/trigger_scenarios/strength_150_known_positions/panel.png`

Do not attach every generated panel. Four or five figures with one sentence of
explanation each will be easier for the guide to understand.

---

## 7. One-minute verbal explanation

> I first trained a classifier with a controlled hidden trigger. A normal image
> is classified according to its content, but adding the trigger makes the model
> choose the attacker's target class. I then trained a generator that receives
> only one suspicious image and makes a small correction to its frequency
> content. During learning, the clean originals act like answer sheets, helping
> it avoid removing useful texture. The attack rate dropped from 86.63% to 6.09%.
> I also tested the correction map instead of trusting it visually: its strongest
> 1% of locations performed most of the useful correction. The defense worked
> across unseen trigger strengths and barely reacted to ordinary noise or blur.
> The remaining question is whether it can handle an active trigger at a new
> frequency location. I have now trained a classifier where four locations all
> produce about 99.4% attack success. The next generator will see two locations
> during training and will be tested on the other two.
