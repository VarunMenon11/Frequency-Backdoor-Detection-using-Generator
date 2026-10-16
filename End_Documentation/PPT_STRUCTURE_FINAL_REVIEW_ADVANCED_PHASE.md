# Final Review PPT Structure and Speaker Guide

## Advanced Phase: Reference-Free Frequency-Domain Backdoor Mitigation

**Purpose:** Build a complete review presentation that explains what changed after Panel Review 1, what was implemented, what the experiments demonstrated, what remains limited, and what will be completed during Phase 3.  
**Recommended main-deck length:** 25 slides total: 24 content slides plus references  
**Recommended presentation time:** 18 to 22 minutes  
**Source presentation consulted:** `Panel Review 1.pptx`  
**Primary technical record:** [Advanced Phase Research Record](README.md)  
**Question-answer record:** [Panel Questions and Evidence Guide](PANEL_QUESTIONS_AND_EVIDENCE_GUIDE.md)

---

# 1. The story the new presentation should tell

The first presentation showed that spectral correction was possible in a controlled setting. It used relatively simple datasets and, most importantly, it could compare a triggered image with its matching clean image. The first panel correctly asked whether this made the correction problem too easy and whether the approach would work when the clean reference was unavailable.

The new presentation should therefore not feel like a longer version of the first presentation. Its story is:

```text
Phase 1 proved that frequency correction was feasible
                         |
                         v
The panel identified realism and generalization problems
                         |
                         v
We moved to complex 224 x 224 texture images
                         |
                         v
We calibrated a strong, transferable frequency attack
                         |
                         v
We redesigned the generator to use one input image only
                         |
                         v
We tested whether its correction was meaningful and selective
                         |
                         v
We tested unseen active positions and competing noise
                         |
                         v
We obtained strong but bounded evidence and defined Phase 3
```

The panel should leave with four clear messages:

1. The project changed substantially after the first review.
2. The advanced generator no longer receives a matching clean image at inference.
3. The final attack is active and the final defense result is supported by causal and comparative experiments.
4. The remaining limitations are understood and converted into a realistic paper-completion plan.

---

# 2. Presentation design rules

## 2.1 Keep the visual identity from Panel Review 1

Retain the university header/footer, student details, supervisor name, and general institutional format from `Panel Review 1.pptx`. This creates continuity between reviews.

Update the following:

- Replace `Project Phase 1` with the appropriate current review or phase name.
- Replace the old August 2026 date with the actual presentation date.
- Use the advanced-phase title selected from Section 3.
- Remove outdated statements such as "unknown-image correction is future work." It has now been demonstrated for unseen DTD images, although not for outside-domain images.
- Do not reuse the old Phase 1 methodology diagram as the final architecture. Use the advanced reference-free methodology figure.

## 2.2 Slide density

- Use no more than four short bullets on ordinary slides.
- Use a figure plus one conclusion sentence on result slides.
- Keep mathematical equations in backup slides unless the panel explicitly asks.
- Never place a full paragraph on the slide. The paragraphs in this document are speaker notes.
- Use one meaning per colour consistently: red for attack, green for correction/recovery, blue for clean/reference-free input, amber for limitations.

## 2.3 Language to use

Prefer:

- `reference-free at inference`
- `single-image spectral evidence`
- `selective log-amplitude correction`
- `within-family held-out-position generalization`
- `backdoor mitigation` or `input purification`
- `validation evidence`

Avoid:

- `dataset-free`
- `universal defense`
- `perfect trigger detection`
- `all unknown attacks are solved`
- `the backdoor is removed from the classifier`
- `the generator knows the correct answer at deployment`

The suspicious classifier remains backdoored. The generator mitigates the trigger before classification; it does not erase the malicious weights from the classifier.

## 2.4 Deck at a glance

| Requested presentation section | Final slide number(s) | Main purpose |
|---|---:|---|
| Title page | 1 | Name the exact problem, method, and image domain |
| Guide approval | 2 | Record supervisor approval |
| Introduction and motivation | 3 | Explain backdoors and why frequency triggers matter |
| Related work and gap | 4 | Position the project against attacks and defenses |
| Problem statement | 5 | State the one-image, no-clean-counterpart research problem |
| Where Panel Review 1 left us | 6 | Show why Phase 1 had to be redesigned |
| Important panel questions | 7 | Give five short question-to-answer outcomes |
| Proposed methodology | 8-11 | Explain the workflow, trigger, generator, and learning/deployment split |
| Dataset | 12 | Justify DTD, the official split, and texture complexity |
| Implementation | 13-15 | Show calibration, final configuration, protocol, and metrics |
| Results | 16-20 | Present main, qualitative, causal, held-out-position, and TNR evidence |
| Answered questions and boundaries | 21 | Separate demonstrated claims from open claims |
| SDG mapping | 22 | Justify SDG 9 and carefully bound SDG 3 relevance |
| Phase 3 plan | 23 | Complete the missing evidence and the IEEE paper |
| Conclusion | 24 | State contribution, result, evidence, and limitation |
| References | 25 | Credit datasets, models, attacks, benchmarks, and defenses |

---

# 3. Slide 1: Title page

## Recommended title

> **Reference-Free Selective Spectral Correction for Frequency-Domain Backdoor Mitigation in Texture Image Classification**

This is the most accurate dissertation title because it states:

- `Reference-free`: no clean counterpart is required at inference.
- `Selective spectral correction`: the method does not blindly suppress the entire spectrum.
- `Frequency-domain backdoor mitigation`: the security problem.
- `Texture image classification`: the domain of the final advanced evidence.

## Other title options

| Option | Title | Best use |
|---:|---|---|
| 1 | **Reference-Free Selective Spectral Correction for Frequency-Domain Backdoor Mitigation in Texture Image Classification** | Most precise dissertation/review title |
| 2 | **Single-Image Spectral Correction for Frequency-Domain Backdoor Mitigation** | Shorter paper or conference title |
| 3 | **Adaptive Frequency-Domain Input Purification for Backdoor-Resilient Image Classification** | Emphasizes preprocessing and robustness |
| 4 | **Selective Spectral Input Correction Against Frequency-Domain Backdoors** | Concise technical title |
| 5 | **Explainable Reference-Free Spectral Mitigation of Image Backdoor Attacks** | Emphasizes the faithfulness experiments |
| 6 | **Learning to Correct Frequency-Domain Backdoors from a Single Image** | Accessible presentation title |

## Titles to avoid

- `Frequency Backdoor Detection`: the current system corrects inputs but does not output a calibrated clean/attack decision.
- `Universal Backdoor Removal`: neither universality nor model-level removal has been demonstrated.
- `Unknown Dataset Defense`: unseen DTD images were tested, but cross-domain images were not.
- `FTrojan Defense`: the method was evaluated principally on an FTrojan-style attack, but the proposed architecture should not be named as though it belongs to FTrojan.

## Put on the slide

```text
Reference-Free Selective Spectral Correction for
Frequency-Domain Backdoor Mitigation in Texture Image Classification

Project Review / Phase 2 Progress

Varun Mahesh Menon
CB.SC.P2AIE25031
Supervisor: Dr. Senthilkumar M
Amrita School of Computing
Presentation date
```

Use a subtle background crop from a DTD texture or from the clean/trigger/corrected evidence panel. Do not use a generic cybersecurity stock image.

## What to say

> My Phase 1 work established that frequency-domain correction could suppress controlled backdoor triggers. The panel then asked whether the problem was too simple because the clean image and trigger were known. This presentation explains how the project was redesigned for one-image inference, complex texture data, stronger attacks, causal evaluation, unseen active positions, and competing noise.

**Time:** 30 seconds.

---

# 4. Slide 2: Guide approval page

## Put on the slide

- Heading: `Guide Approval`
- Insert a readable screenshot of the approval email.
- Crop out unrelated inbox content and personal notifications.
- Keep the email date, sender, subject, and approval statement visible.

## What to say

> This slide records the guide's approval for the current project review.

Do not spend more than ten seconds on this slide.

---

# 5. Slide 3: Introduction and motivation

## Slide title

**Why Frequency-Domain Backdoors Matter**

## Put on the slide

Use a simple three-column visual:

```text
Clean image              Triggered image              Security problem
normal prediction        attacker target              hidden behavior
```

Add only these points:

- A poisoned classifier can behave normally on clean images.
- A hidden trigger can force an attacker-selected label.
- Frequency triggers may be visually subtle and distributed across the image.
- A useful defense must suppress the attack without destroying image content.

## What to explain

> A backdoored classifier learns two behaviours. On ordinary images it appears normal, so normal accuracy alone may not reveal the problem. When a hidden trigger is present, the model follows the attacker's rule instead of the image content. Frequency-domain triggers are difficult because their pixel changes can be spread across the whole image and remain visually subtle. A broad filter may destroy the trigger, but it may also destroy useful image frequencies. The research problem is therefore selective correction, not simply aggressive filtering.

## Transition

> This creates a practical question: can one incoming image be corrected without knowing its original clean version?

**Time:** 50 seconds.

---

# 6. Slide 4: Related work and research gap

This slide was not explicitly listed in the requested format, but it should remain in the deck. It protects the project from the impression that frequency attacks, amplitude manipulation, or input purification are being claimed as completely new.

## Put on the slide

| Prior direction | What it established | Gap addressed here |
|---|---|---|
| FTrojan and FIBA | Frequency-domain triggers can create strong, visually subtle backdoors | They are attack methods, not this correction architecture |
| Freq-Pret | Phase-preserving amplitude perturbation can reduce backdoor dependence | Uses broad frequency perturbation and model retraining rather than the proposed input-conditioned gate |
| Februus, BDMAE and ZIP | Test-time image purification can mitigate triggers | Primarily spatial/generative restoration; not the same selective log-amplitude intervention |
| Frequency-trigger detection | Spectral artifacts can reveal some attacks | Detection alone does not reconstruct a corrected texture image |

At the bottom, place one highlighted sentence:

> **Research gap:** Learn an image-dependent spectral correction from one input image and test whether the selected frequency support actually causes attack recovery.

## What to explain

> The individual ideas of frequency analysis, phase preservation, and image purification already exist and are cited. The proposed contribution is their specific integration into a reference-free, input-conditioned correction network with a gate and signed log-amplitude update, together with causal support tests, held-out active positions, and trigger-to-noise evaluation. I describe it as an extension and evaluation contribution, not as the invention of frequency-domain defense itself.

## References to place in small text

- Wang et al., FTrojan, ECCV 2022.
- Feng et al., FIBA, CVPR 2022.
- Zhang et al., Freq-Pret, IEEE Internet of Things Journal, 2026.
- Doan et al., Februus.
- Sun et al., BDMAE, 2023.

**Time:** 60 seconds.

---

# 7. Slide 5: Problem statement

## Recommended problem statement

> **Given a suspicious image classifier and a single incoming image whose clean counterpart is unavailable, can an input-conditioned frequency-domain preprocessor selectively reduce attack-related spectral evidence while preserving the phase and texture information required for correct classification?**

This is stronger than the old statement because it specifies:

- The suspicious-model setting.
- The absence of a matching clean image.
- The single-image deployment condition.
- The selective correction mechanism.
- The need to preserve task-relevant information.

## Project objective

> To design and evaluate a reference-free spectral correction generator that predicts a selective, image-dependent log-amplitude intervention before classification.

## Put four objectives on the slide

1. Build and verify a strong frequency-backdoored classifier.
2. Infer correction from one incoming image rather than clean-trigger subtraction.
3. Preserve normal classification while reducing attack success.
4. Test causal faithfulness, unseen positions, benign corruptions, and trigger-to-noise robustness.

## What to explain

> The output is not a trigger detector label. The system produces a corrected image. During training, known clean examples teach the generator, but during use the matching clean counterpart is unavailable. Therefore, the core question is whether the generator can infer a useful correction from the suspicious image itself.

**Time:** 50 seconds.

---

# 8. Slide 6: Where the first panel left us

## Slide title

**Phase 1 Success and Its Main Limitation**

## Put on the slide

Use a left-versus-right comparison.

| What Phase 1 established | What remained unresolved |
|---|---|
| Complete poisoning, classifier, generator and repair pipeline | Generator received clean and triggered spectral information |
| CIFAR-100, Tiny ImageNet and STL-10 | Absolute difference made trigger evidence directly visible |
| 32x32, 64x64 and 96x96 resolutions | Images and triggers were relatively controlled |
| Strength, function and FIBA-style experiments | Unknown trigger and noise behaviour remained unclear |

At the bottom:

> Phase 1 was a controlled proof of concept, not the final deployment method.

## What to explain

> The earlier results were useful and should not be discarded. They verified the full pipeline across resolutions. However, the clean-trigger comparison gave the generator information that a deployed system would not normally possess. The panel's criticism therefore became the design requirement for the advanced phase.

**Time:** 60 seconds.

---

# 9. Slide 7: Important panel questions and present answers

## Recommended slide title

**Questions Raised in Panel Review 1 and How We Addressed Them**

## Put these five questions on the slide

Use five large question-and-answer boxes. Keep the answer under each question to
two short lines. The audience should be able to understand this slide before
the detailed methodology is introduced.

| Question raised | Simple answer for the slide |
|---|---|
| **1. What happens when the matching clean image is unavailable?** | The new generator needs only the incoming image. It does not need the original clean version while correcting it. |
| **2. Without clean-trigger subtraction, how does the generator know what to change?** | During training it sees many examples and learns which frequency patterns repeatedly lead to the attack. For a new image, it predicts only the spectral changes that appear suspicious. |
| **3. Can the method work when the image already contains complicated frequencies and textures?** | We moved from simple images to DTD: 224x224 images from 47 texture classes. The defense still recovered classification close to the clean baseline. |
| **4. What happens if the attacker changes the frequency location?** | We hid two active trigger positions from generator training. On those positions, attack success fell from 99.40% to 7.20% on average. |
| **5. Does the generator remove the trigger, or does it simply remove any noise?** | It did not blindly erase all frequencies. Selective correction worked much better than broad filtering, while strong unrelated noise revealed a clear failure boundary. |

Add one small statement at the bottom:

> Each panel question was converted into a measurable experiment rather than answered only through explanation.

## What to say for each question

### Question 1: no matching clean image

> In Phase 1, I created a triggered image from a clean image, so both versions
> were available. That made it possible to subtract their amplitudes directly.
> In a real application, we receive only one image and do not possess its
> untouched twin. The advanced generator was therefore redesigned to accept the
> incoming image alone. A clean counterpart helps supervise training, but it is
> not supplied during deployment.

### Question 2: how the generator decides what to change

> The generator does not calculate a clean-minus-trigger difference during use.
> While it is being trained, many different images pass through it. The content
> of those images changes, but the attack repeatedly affects particular spectral
> structures and repeatedly pushes the classifier toward the target class. The
> training losses reward corrections that restore the true class and preserve
> the image, while penalizing corrections that are unnecessarily large. It
> gradually learns a rule for proposing a small, image-dependent spectral
> correction. This is learned statistical evidence, not guaranteed perfect
> trigger detection.

### Question 3: complicated texture images

> Natural texture images are a difficult test because their genuine information
> is itself rich in frequency content. A defense cannot simply assume that every
> strong frequency is malicious. We therefore used DTD, containing 47 texture
> classes at 224x224 resolution. The clean classifier accuracy was about 61.44%,
> and after correction of triggered images the true-label accuracy was 60.29%.
> This shows that most useful texture information was preserved in the tested
> setting.

### Question 4: a moved trigger

> A generator could memorize one exact frequency location. To test whether that
> happened, two active positions were deliberately excluded from generator
> training. The suspicious classifier still responded strongly to them, with a
> mean ASR of 99.40%. The generator reduced their mean ASR to 7.20% and restored
> 60.46% true-label accuracy. This demonstrates generalization to held-out
> positions within the same FTrojan-style attack family, not to every unknown
> attack in the world.

### Question 5: trigger correction versus general noise removal

> We compared the learned correction with a fixed correction, broad frequency
> suppression, and conditions containing Gaussian noise. Broad suppression was
> much weaker, so the result cannot be explained as merely deleting a large part
> of the spectrum. The generator remained useful while the trigger was stronger
> than, or reasonably comparable with, the noise. When noise became dominant,
> both the image information and the defense performance collapsed. Therefore,
> the model learned selective attack mitigation, but it is not a universal
> denoiser.

## Two optional questions

These can be placed on a second question slide or kept as backup material.

| Optional question | Simple answer |
|---|---|
| **Can the model work without any dataset?** | No. It needs a dataset for training. `Reference-free` means that it does not need the matching clean image during use; it does not mean training-data-free. |
| **Does the defense remove the backdoor from the classifier?** | No. The classifier remains suspicious and frozen. The generator corrects each input before that input reaches the classifier. |

## Recommended visual arrangement

For the main slide, use the five question boxes on the left and a reduced
version of this evidence map on the right:

![Reviewer-question evidence map](assets/22_reviewer_question_evidence_map.png)

If the figure becomes too small to read, remove it from the main slide and use
it as a transition or backup slide. Do not reduce the answer text merely to
make the figure fit.

## Important wording

- Say `learns repeated spectral evidence`, not `automatically knows the trigger`.
- Say `held-out position within the tested family`, not `all unknown triggers`.
- Say `reference-free at inference`, not `dataset-free`.
- Say `mitigates the input`, not `removes the backdoor from the model`.

**Time:** 90 to 120 seconds.

---

# 10. Slides 8 to 11: Proposed methodology

The methodology should use four slides. Do not squeeze the entire attack, generator, losses, and deployment into one unreadable slide.

---

## Slide 8: Complete advanced methodology

### Visual

Use this full-width image:

![Advanced proposed methodology](assets/15_advanced_methodology_flow.png)

### Put on the slide

No additional paragraph is needed. Add only a small legend:

- Upper path: construct and verify the suspicious classifier.
- Lower path: train and deploy the reference-free correction generator.
- Dashed green arrows: training supervision only.
- Solid arrows: deployed forward path.

### What to explain

> The upper half creates the security problem under controlled conditions. Clean DTD images are poisoned with an FTrojan-style trigger and used to train a ResNet18. Before testing any defense, I verify that clean accuracy remains usable and every trigger position produces a strong attack. The classifier is then frozen.

> The lower half is the defense. One image is transformed by FFT, converted into fifteen evidence channels, passed through the generator, selectively corrected in log-amplitude space, reconstructed with the original phase, and classified. During training, the known source image and label constrain the output. During deployment, those references are absent.

### One-line deployment explanation

> One image -> spectral evidence -> learned selective correction -> reconstructed image -> classifier.

**Time:** 90 seconds.

---

## Slide 9: FTrojan-style attack construction

### Visual

![FTrojan-style trigger design](assets/14_ftrojan_trigger_design.png)

### Put on the slide

| Property | Final setting |
|---|---|
| Transform | 32x32 block DCT |
| Modified channels | Two chroma channels |
| Magnitude | 150 |
| Positions | Four independently active layouts |
| Target class | `banded` |
| Poisoning type | All-to-one, changed-label |

### What to explain

> The image is converted into a luminance/chroma-like colour representation. The chroma channels are divided into 32x32 blocks. Selected DCT coefficients are adjusted repeatedly across blocks, after which inverse transforms reconstruct the RGB image. Poisoned copies receive the target label `banded`, while their clean copies remain available with the original label.

> Four coefficient layouts were made independently active. This matters because a low defended ASR on an inactive moved trigger would be meaningless. The suspicious classifier had to respond strongly to every position before two positions could be hidden from generator training.

### Do not say

Do not call the implementation an exact reproduction of FTrojan. It is an `FTrojan-style adaptation` for this 224x224 pipeline.

**Time:** 75 seconds.

---

## Slide 10: Reference-free generator architecture

### Visual

Use both of these, with the architecture on the left and correction anatomy on the right:

![Generator architecture explanation](assets/02_reference_free_generator_explanation.png)

![Correction-map anatomy](assets/12_correction_map_anatomy.png)

If both are too dense, use only the correction-map anatomy on the main slide and move the full generator explanation to backup.

### Put on the slide

```text
Input: one 3 x 224 x 224 RGB image
Evidence: 15 x 224 x 224
Network: encoder-decoder with skip connections
Output: gate M + signed proposal Delta
Applied correction: E = M x Delta
Reconstruction: corrected amplitude + incoming phase
```

### Explain the 15 channels simply

| Evidence | Plain meaning |
|---|---|
| Three normalized log-amplitude channels | How strongly each frequency is present |
| Three local residual channels | Which frequencies look unusual relative to their neighbours |
| Three sine-phase and three cosine-phase channels | Stable representation of spatial arrangement |
| Horizontal, vertical and radial coordinates | Where each frequency lies |

### What to explain

> The generator does not receive the clean image or the true trigger mask. It sees evidence calculated from the one incoming image. The encoder-decoder identifies useful patterns at multiple scales. It outputs a gate saying where correction is permitted and a signed proposal saying whether the selected log amplitude should increase or decrease. Multiplying them produces the actual correction. The incoming phase is preserved to limit structural damage.

### Key clarification

> The correction map is a learned action, not proof that every bright point is a trigger coefficient.

**Time:** 90 seconds.

---

## Slide 11: How the generator learns versus how it is used

### Visuals

![Training versus deployment](assets/11_training_vs_deployment.png)

![Learning across many images](assets/16_generator_learning_across_images.png)

### Put on the slide

```text
Training:
clean source + generated trigger + true label
        -> teach reusable correction

Deployment:
one incoming image
        -> one correction -> one prediction
```

### What to explain

> During training, many different textures are attacked in a controlled way. Natural content changes from image to image, while the trigger produces recurring spectral geometry. Shared network weights receive repeated gradients that reward correction of this geometry and penalize unnecessary changes to clean content.

> This does not mean the network literally stores frequency values that remain numerically constant. It learns a mapping from recurring spectral evidence to useful corrective action. At deployment, the training history is stored in the weights. The model receives one image, makes one correction, and makes one prediction. There is no repeated guessing loop because the true deployment label is unknown.

### Losses to mention verbally, not as equations

- Classification recovery.
- Image reconstruction.
- Spectral reconstruction.
- Clean-image identity.
- Sparse gate.
- Smooth correction.

**Time:** 90 seconds.

---

# 11. Slide 12: Dataset and benchmark

## Slide title

**Advanced Dataset: Describable Textures Dataset**

## Put on the slide

| Property | Value |
|---|---:|
| Texture classes | 47 |
| Images per class | 120 |
| Total source images | 5,640 |
| Official split-1 training | 1,880 |
| Official split-1 validation | 1,880 |
| Locked test | 1,880 |
| Evaluation resolution | 224x224 RGB |
| Classifier | ImageNet-pretrained ResNet18 |
| Attacker target | Class 0, `banded` |

Show four to six example texture crops and label them with classes such as `blotchy`, `braided`, `woven`, `striped`, and `banded`.

## Why DTD was selected

Put one highlighted sentence:

> Natural textures already contain strong high-frequency and repeated patterns, so broad filtering can remove useful classification information.

## What to explain

> DTD was chosen because it makes frequency correction difficult in a meaningful way. Unlike an object dataset, the label describes the surface appearance. The classifier must use patterns such as bands, spots, weaving, cracks, or blotches. This means natural high-frequency information cannot simply be treated as malicious.

> The project also created reproducible manifests recording source identities, splits, labels, triggers, positions, and experimental roles. The benchmark contains 29,748 variant definitions without duplicating every transformed image file.

## Honest dataset limitation

Mention verbally:

> The audit detected nine exact-image duplicate checksum groups crossing official DTD splits. This will be disclosed, and a deduplicated sensitivity check is planned for Phase 3.

**Time:** 75 seconds.

---

# 12. Slides 13 to 15: What was implemented

---

## Slide 13: Development path and failed attack calibration

## Put on the slide

| Candidate | Evidence | Decision |
|---|---:|---|
| Direct Fourier-band trigger | Triggered target 13.53%; clean target 9.89% | Rejected as weak |
| Haar-LH wavelet | ASR 9.67%; clean target 9.51%; lift 0.16 pp | Rejected as inactive |
| Early FTrojan M50 | Poison-row target 95.21%; validation approximately 15.9% | Rejected as memorization |
| FTrojan M100 | ASR 86.63%; clean target 5.60%; lift 81.03 pp | Accepted for first audit |
| FTrojan M150, four positions | Every position approximately 99.4% to 99.7% ASR | Final benchmark |

## What to explain

> A high target rate on poisoned training rows does not prove a transferable attack. Several candidates looked strong during training but produced almost no extra target behaviour on held-out images. I therefore measured the same model's clean target rate, trigger target rate, target-rate lift, and conditional ASR before evaluating the defense.

> Failed attacks are not failed research. They established the attack-validity rule that protects the final defense result from a false conclusion.

## Key sentence on the slide

> A defense result is meaningful only when the corresponding attack is active before correction.

**Time:** 75 seconds.

---

## Slide 14: Final implementation configuration

## Put on the slide

| Component | Final advanced configuration |
|---|---|
| Dataset | DTD, official split 1 |
| Input | 224x224 RGB crop |
| Classifier | ImageNet-pretrained ResNet18, 47 classes |
| Attack | FTrojan-style block-DCT, four positions, magnitude 150 |
| Target | `banded` |
| Generator | 771,238-parameter encoder-decoder |
| Generator input | Single-image 15-channel FFT evidence |
| Generator output | Gate plus bounded signed log-amplitude proposal |
| Phase policy | Preserve incoming phase |
| Evaluation | 1,840 non-target validation sources per position |

## What to explain

> The suspicious classifier is frozen during generator training. Its gradients guide the classification component of the generator loss, while reconstruction, identity, sparsity, and smoothness terms constrain the solution. The generator is not replacing or fine-tuning the classifier during inference; it is a preprocessing layer.

## Optional visual

Use the algorithm image in a small right-side panel:

![IEEE-style training and inference algorithm](assets/21_ieee_algorithm_training_inference.png)

**Time:** 60 seconds.

---

## Slide 15: Evaluation protocol and metrics

## Put on the slide

| Metric | Question answered |
|---|---|
| Clean accuracy | Does ordinary classification still work? |
| Undefended ASR | Is the attack active? |
| Corrected ASR | How much attack behaviour remains? |
| Corrected true-label accuracy | Did the original class recover? |
| Clean target rate and target-rate lift | Is ASR larger than natural target bias? |
| Conditional ASR and target flips | Did the trigger change previously correct predictions? |
| Support IoU and cosine similarity | Does learned action align with injected evidence? |
| Trigger-to-noise ratio | How does competing noise change the result? |

## What to explain

> ASR is calculated only on non-target images. A source already belonging to `banded` is excluded because predicting `banded` would be correct even without an attack. Low ASR alone is not enough; the corrected true-label accuracy must also remain high. Otherwise, the defense may only be destroying the image.

**Time:** 60 seconds.

---

# 13. Slides 16 to 20: Results and discussion

---

## Slide 16: Main final result

## Put on the slide

Use the table and one large conclusion box.

| Condition | Target rate / ASR | True-label accuracy |
|---|---:|---:|
| Clean input | 6.52% target rate | 61.44% overall clean accuracy |
| Triggered, no defense | **99.48%** | **0.50%** |
| Triggered, full generator | **7.64%** | **60.29%** |
| Triggered, static template | 18.22% | 55.00% |
| Triggered, broad suppression | 66.60% | 24.61% |
| Strongest generator support only | 9.29% | 59.44% |
| Outside strongest support | 99.28% | 0.68% |

Conclusion box:

> The generator reduced mean four-position ASR by 91.84 percentage points while recovering accuracy close to the clean baseline.

## What to explain

> The attack almost completely controls the suspicious model before defense. After generator correction, target behaviour falls close to the classifier's ordinary clean target rate, and true-label accuracy returns close to clean performance. A static correction helps but is weaker. Broad suppression damages the texture task and still leaves much of the attack active.

## Important nuance

The clean target rate uses non-target sources, while the 61.44% clean accuracy is measured over the full validation split. Explain that they are different quantities.

**Time:** 75 seconds.

---

## Slide 17: Qualitative recovery on a held-out position

## Use this image full-width

![Held-out-position qualitative recovery](assets/07_held_out_position_recovery.png)

## What to explain, panel by panel

1. `clean true/pred: blotchy/blotchy`: the normal image begins correctly classified.
2. `triggered pred: banded`: the hidden trigger activates the attacker's target.
3. `generator corrected pred: blotchy`: correction restores the original class.
4. `trigger-clean |x8|`: the small pixel difference is amplified eight times for visibility.
5. The second row shows clean, triggered, and corrected log-amplitude spectra on a shared scale.
6. `|generator correction|`: bright regions show where the generator changed log amplitude most strongly.
7. Static and broad alternatives still predict the target for this example.
8. The target-probability chart shows the generator reducing `banded` probability much more than the alternatives.

## Main statement

> This is a selected qualitative example, not the aggregate result. The aggregate metrics are reported separately.

## What not to say

Do not say that every bright correction point is definitively a trigger. The map shows the learned action that caused recovery.

**Time:** 90 seconds.

---

## Slide 18: Opening the generator's black box

## Visual

![Final causal and baseline comparison](assets/08_final_baseline_comparison.png)

## Put on the slide

At the bottom, add:

```text
Full correction:       ASR 7.64%, accuracy 60.29%
Strongest support:     ASR 9.29%, accuracy 59.44%
Outside support:       ASR 99.28%, accuracy 0.68%
Static template:       ASR 18.22%, accuracy 55.00%
Broad suppression:     ASR 66.60%, accuracy 24.61%
```

## What to explain

> This is a causal intervention rather than another visualization. I rank frequency locations by generator correction magnitude. Applying only the strongest support retains most of the repair effect. Applying everything outside that support leaves the attack almost unchanged. This demonstrates that a concentrated selected region controls the outcome.

> The static-template result checks whether one average correction could solve the benchmark. It helps, because the trigger has repeated structure, but it is weaker than the image-conditioned generator. Broad suppression tests the idea of simply filtering a large spectral region; it performs poorly on both ASR and texture accuracy.

## Additional measurements to mention verbally

- Median trigger-to-clean correction ratio: 47.68x.
- Median predicted-support/oracle-support IoU: 0.6457.
- Correction/evidence cosine similarity: 0.8550.

**Time:** 90 seconds.

---

## Slide 19: Generalization to active positions hidden from training

## Put on the slide

| Held-out trigger | Undefended ASR | Corrected ASR | Corrected accuracy |
|---|---:|---:|---:|
| Near-mixed position | 99.40% | 7.23% | 60.49% |
| Middle-shift position | 99.40% | 7.17% | 60.44% |
| **Held-out mean** | **99.40%** | **7.20%** | **60.46%** |

Use a simple training/test diagram:

```text
Generator sees positions A and B during training
                      |
                      v
Generator is evaluated on active positions C and D
                      |
                      v
Mean ASR: 99.40% -> 7.20%
```

## What to explain

> The classifier was trained so all four positions were strong attacks. The generator was trained and selected using two positions only. The other two remained hidden until evaluation. Their high undefended ASR confirms that the attacks were active; the corrected result therefore provides evidence that the generator learned more than one exact coordinate.

## Boundary to state

> This is within-family position generalization. It is not yet cross-family or defense-aware adaptive robustness.

**Time:** 75 seconds.

---

## Slide 20: Trigger versus noise

## Visual

![Trigger-to-noise result](assets/09_trigger_noise_ratio_chart.png)

## Put on the slide

| Condition | Corrected ASR | Corrected accuracy |
|---|---:|---:|
| Trigger only | 7.64% | 60.29% |
| +20 dB, trigger much stronger than noise | 7.85% | 60.19% |
| +10 dB, trigger stronger than noise | 10.04% | 58.93% |
| 0 dB, equal RMS power | 45.11% | 35.52% |
| -10 dB, noise stronger than trigger | 95.05% | 3.48% |

## What to explain

> Trigger-to-noise ratio compares the RMS size of the trigger perturbation with the RMS size of added Gaussian noise. Positive values mean the trigger is stronger; zero means equal power; negative values mean noise is stronger.

> The generator remains effective at +20 and +10 dB. It degrades near equal power and fails when noise dominates. The noise-only control is essential: at -10 dB, noise-only classification accuracy is already approximately 2.19%, so this is not only failure to remove a trigger. The image-recognition task itself has largely collapsed.

## Honest conclusion

> The method is robust to bounded competing noise, not arbitrary severe corruption.

**Time:** 90 seconds.

---

# 14. Slide 21: What has been answered and what remains open

## Put on the slide

| Question | Current status |
|---|---|
| No clean counterpart at inference | **Answered on unseen DTD images** |
| Absolute subtraction concern | **Answered through reference-free input and causal audit** |
| Complex texture images | **Answered on DTD** |
| Meaningful correction versus broad filtering | **Supported by ablations and baselines** |
| Changed active frequency position | **Answered within the FTrojan-style family** |
| Mild corruption and trigger/noise competition | **Answered within tested range** |
| Completely unseen active attack family | **Open** |
| Defense-aware adaptive attacker | **Open** |
| Outside-domain or unknown-class image | **Open** |
| Locked final-test and multi-seed confidence | **Phase 3** |

## Optional visual

![Attacker generalization ladder](assets/24_attacker_generalization_ladder.png)

## What to explain

> The project should not be judged as either "everything is solved" or "nothing is solved." Reference-free inference, complex textures, causal support, held-out positions, and bounded-noise behaviour have been demonstrated. Cross-family, adaptive, outside-domain, and statistical confidence remain explicit Phase 3 work.

**Time:** 60 seconds.

---

# 15. Slide 22: SDG mapping

## Primary mapping

### SDG 9: Industry, Innovation and Infrastructure

## Put on the slide

> **Justification:** The project develops a security-oriented machine-learning method intended to improve the resilience and trustworthiness of image-classification infrastructure when models or training data may be compromised.

Connect it to:

- **Target 9.1:** Reliable, sustainable, and resilient infrastructure.
- **Target 9.5:** Scientific research and technological capability.
- Secure deployment of AI systems in industrial and critical-service pipelines.
- Reproducible experimentation and open technical documentation.

## Secondary application relevance

### SDG 3: Good Health and Well-being

Use careful wording:

> Medical imaging is a possible future application because manipulated classification could affect high-stakes decisions. The current advanced experiment uses DTD textures and does not claim completed medical validation.

## What to explain

> SDG 9 is the primary mapping because the completed work is about trustworthy and resilient AI infrastructure. SDG 3 is only secondary application motivation. I do not claim a direct medical contribution because the PathMNIST/FIBA medical experiment was not completed as the final advanced benchmark.

## What not to do

Do not list many SDGs merely to make the project appear broader. One well-justified primary mapping is more credible.

**Time:** 60 seconds.

---

# 16. Slide 23: Phase 3 project plan

Phase 3 should not be only paper formatting. A strong paper still needs several experimental closures identified by the literature and novelty audit.

## Put on the slide

### Phase 3 goal

> Freeze the scientific claim, complete the minimum missing evidence, and produce a reproducible submission-ready paper.

## Recommended 14-week plan

| Work package | W1-2 | W3-4 | W5-6 | W7 | W8-9 | W10 | W11-12 | W13-14 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Literature update and novelty matrix | X |  |  |  |  |  |  |  |
| Reproduce published defense baselines | X | X |  |  |  |  |  |  |
| Three-seed final protocol |  | X | X |  |  |  |  |  |
| Deduplicated DTD sensitivity check |  |  | X |  |  |  |  |  |
| Freeze method and hyperparameters |  |  | X | X |  |  |  |  |
| Run locked final-test evaluation once |  |  |  | X |  |  |  |  |
| One held-out-family or adaptive-attack test |  |  |  |  | X |  |  |  |
| Runtime, memory, per-class and perceptual metrics |  |  |  |  | X | X |  |  |
| IEEE paper writing | X | X | X | X | X | X | X |  |
| Internal review, similarity and AI-use disclosure |  |  |  |  |  |  | X | X |
| Submission package and repository release |  |  |  |  |  |  |  | X |

## Minimum required before submission

1. Add the closest related defenses and write an explicit method-difference table.
2. Compare against at least one published purification/frequency defense on the same attacked model.
3. Repeat the final protocol across multiple seeds and report uncertainty.
4. Freeze all choices, then run the locked test once.
5. Run the deduplicated-split sensitivity check.
6. Report runtime and memory.

## Strong-paper extension

Choose at least one:

- A completely held-out active trigger family.
- A defense-aware adaptive attack optimized through the generator.
- A second texture dataset or second classifier architecture.

## What to explain

> Paper writing starts immediately, but the method and result sections remain provisional until the baseline, seed, duplicate-sensitivity, and locked-test results are complete. The adaptive or held-out-family experiment is the strongest addition because it directly addresses the remaining generalization limitation.

**Time:** 90 seconds.

---

# 17. Slide 24: Conclusion

## Put on the slide

Use four statements only:

1. **Advanced method:** One-image, reference-free selective log-amplitude correction with phase-preserving reconstruction.
2. **Main result:** Four-position mean ASR reduced from **99.48% to 7.64%**, with **60.29%** corrected accuracy versus approximately **61.44%** clean accuracy.
3. **Evidence quality:** Causal support tests, static/broad baselines, held-out active positions, benign corruptions, and TNR stress testing.
4. **Boundary:** Demonstrated on DTD and within one calibrated attack family; cross-family, adaptive, locked-test, and multi-seed evidence remain Phase 3.

## What to say

> The project progressed from a paired clean-trigger feasibility experiment to a single-image reference-free correction system on complex texture data. The final defense strongly reduces an independently verified attack while preserving classification utility. The causal and baseline experiments show that the result is not explained by broad filtering alone. The current claim is meaningful but bounded: it is within-family validation evidence, not a universal guarantee. Phase 3 will convert this result into a statistically stronger, literature-compared, locked-test paper.

## Final sentence

> The contribution is not that every malicious frequency is perfectly identified; it is that a learned, selective, single-image spectral intervention can causally suppress a strong structured backdoor while preserving most texture-classification performance.

**Time:** 60 seconds.

---

# 18. Slide 25: References

The user's requested structure ends with the conclusion, but a research presentation should include a references slide. It can appear after the conclusion and does not need spoken explanation.

## Include at least

1. M. Cimpoi et al., `Describing Textures in the Wild`, CVPR 2014.
2. K. He et al., `Deep Residual Learning for Image Recognition`, CVPR 2016.
3. T. Wang et al., `An Invisible Black-box Backdoor Attack through Frequency Domain`, ECCV 2022.
4. Y. Feng et al., `FIBA: Frequency-Injection Based Backdoor Attack in Medical Image Analysis`, CVPR 2022.
5. Y. Zeng et al., `Rethinking the Backdoor Attacks' Triggers: A Frequency Perspective`, ICCV 2021.
6. P. Zhang et al., `Backdoor Defense Strategy for Image Classification Tasks Based on Frequency Domain Perturbation`, IEEE IoT Journal, 2026.
7. B. G. Doan et al., `Februus: Input Purification Defense Against Trojan Attacks on Deep Neural Network Systems`.
8. T. Sun et al., `Mask and Restore: Blind Backdoor Defense at Test Time with Masked Autoencoder`, 2023.
9. B. Wu et al., `BackdoorBench: A Comprehensive Benchmark of Backdoor Learning`, NeurIPS 2022.

Verify author names, venue, year, volume, pages, and DOI before final submission.

---

# 19. Recommended backup slides

Backup slides are not part of the timed presentation. Keep them after references for panel questions.

## Backup A: Exact advanced algorithm

Use:

![Advanced algorithm](assets/21_ieee_algorithm_training_inference.png)

Use this if the panel asks for pseudocode, the training loop, or the difference between training and inference.

## Backup B: What does "without dataset" mean?

Use:

![Three meanings of no dataset](assets/23_three_meanings_of_no_dataset.png)

Explain:

- No training data: impossible for a learned model.
- No clean twin at inference: demonstrated.
- Completely different domain: not demonstrated.

## Backup C: Generator loss

Put the loss equation and plain-language role of each term:

```text
classification recovery
+ image reconstruction
+ spectral reconstruction
+ clean identity
+ sparse gate
+ smooth correction
```

Explain that the weights are experimentally selected hyperparameters, not universal constants.

## Backup D: TNR formula

```text
TNR_dB = 20 log10(RMS(trigger perturbation) / RMS(noise perturbation))
```

Explain that this is an SNR-derived experimental condition, not a novel universal metric.

## Backup E: Benign corruption controls

Use:

![Benign corruption controls](assets/06_benign_corruption_controls.png)

This answers whether the generator broadly changes blur, brightness, contrast, or mild noise.

## Backup F: Initial faithfulness audit

Use:

![Faithfulness explanation](assets/03_faithfulness_explanation.png)

Explain the approximately 150,528 channel-frequency locations and the top-1% intervention. Label this as the initial single-position audit, not the final four-position aggregate.

## Backup G: Limitations

Use a direct table:

| Limitation | Why it matters | Planned response |
|---|---|---|
| One advanced dataset | May be domain-specific | Second dataset if time permits |
| One classifier | Architecture transfer unknown | Add second architecture or state as future work |
| One principal trigger family | Family generalization unknown | Held-out-family experiment |
| No adaptive attacker | Generator may be bypassed | Optimize through frozen defense |
| One principal seed | Statistical stability unknown | Three-seed final protocol |
| Validation results | Selection bias possible | Locked test after freeze |
| DTD duplicate groups | Split independence weakened | Deduplicated sensitivity run |
| No runtime benchmark | Deployment cost unknown | Measure latency and memory |

---

# 20. Opening script

> Good morning. In my first review, I demonstrated that a generator could correct a controlled frequency-domain backdoor by comparing clean and triggered spectral information. The panel raised an important concern: in a real deployment, the clean counterpart will not be available, and natural images may contain frequency patterns that resemble the trigger. I therefore redesigned the project around a single-image, reference-free correction generator and moved to the 224x224 Describable Textures Dataset. I will show how the attack was calibrated, how the generator works, what evidence demonstrates that its correction is meaningful, how it performs on active positions hidden during training, and where it fails under severe noise.

---

# 21. Closing script

> To conclude, the advanced project demonstrates that an input-conditioned spectral generator can reduce a strong four-position FTrojan-style attack from 99.48% mean ASR to 7.64% while recovering 60.29% true-label accuracy, close to the clean baseline. Causal support interventions, fixed and broad baselines, held-out active positions, and trigger-to-noise experiments provide evidence beyond a single final metric. The result is reference-free at inference and meaningful within the tested DTD and attack-family boundary. Phase 3 will focus on published defense comparisons, repeated seeds, deduplicated sensitivity, one locked final-test run, a stronger held-out-family or adaptive test, and completion of the IEEE paper.

---

# 22. Likely panel questions and short answers

## Is the method dataset-free?

> No. It requires training data. It is reference-free at inference, meaning the matching clean image is not supplied when correcting an incoming image.

## Does the generator detect a trigger?

> It predicts a corrective action, not a calibrated clean-versus-trigger decision. Therefore I call it a corrector or input-purification stage rather than a detector.

## How does it know the correct label during use?

> It does not. True labels guide supervised training only. Deployment uses one learned forward pass.

## Is absolute subtraction still used?

> Only by the evaluator to visualize the controlled perturbation. It is not part of the advanced generator's input.

## Why is the clean accuracy only around 61%?

> DTD contains 47 visually overlapping texture classes and only 40 official training images per class in split 1. The defense is evaluated relative to the same classifier's clean baseline.

## Why select FTrojan-style DCT injection?

> Fourier and Haar candidates did not create transferable held-out attacks under their tested settings. The FTrojan-style configuration produced strong ASR, low clean target bias, reproducible structure, and multiple independently active positions.

## Is the work original?

> Frequency attacks, phase preservation, and input purification already exist and are cited. The proposed contribution is the single-image evidence representation, gated signed log-amplitude correction, and its causal, position-generalization, texture, and TNR evaluation. The final paper will compare this explicitly against the closest published defenses.

## Have unknown triggers been solved?

> Unseen active positions within the same FTrojan-style family were demonstrated. Completely unseen families and adaptive attackers remain open.

## Why does the method fail at negative TNR?

> At -10 dB, noise is much stronger than the trigger and noise-only accuracy already falls to about 2.19%. The semantic classification problem itself is severely damaged, so this is not only trigger-removal failure.

## Does the generator remove the backdoor from ResNet18?

> No. The suspicious classifier remains frozen and backdoored. The generator mitigates the trigger in the input before classification.

---

# 23. Final slide checklist

Before presenting, verify every item:

- The title says `mitigation` or `correction`, not universal removal.
- The presentation date and phase are current.
- The approval-email screenshot is readable and privacy-cropped.
- Every chart states whether it is validation or final-test evidence.
- The main result uses the four-position aggregate.
- The faithfulness top-1% result is identified as the initial single-position audit when applicable.
- The qualitative sample begins clean-correct, becomes target-predicted, and is restored to the true label.
- `banded` is explained as the attacker target class.
- ASR and corrected accuracy are shown together.
- Held-out positions are described as within-family generalization.
- Noise-dominant failure is stated honestly.
- The Phase 3 plan includes experiments as well as writing.
- The closest related defenses are cited.
- The final paper plan includes similarity checking and the required AI-use disclosure.
- A backup copy of the PPT and all linked images is available offline.
