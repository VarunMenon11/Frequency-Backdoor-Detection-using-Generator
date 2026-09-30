# Review Presentation Guide: From First-Panel Questions to the Current System

**Suggested title:** Selective Frequency-Domain Backdoor Mitigation Using Adaptive Spectral Correction  
**Purpose:** Explain what changed after the first panel, which doubts were tested,
what evidence was obtained, what failed, and what is being done next.  
**Recommended length:** 16-18 slides, approximately 15-20 minutes.

---

## 1. The story this presentation should tell

This presentation should not repeat the entire project from its beginning. The
story starts with the first-panel feedback:

> The initial system worked, but it was demonstrated using relatively simple
> images, a known trigger and a clean-trigger comparison. The panel asked how it
> would work on complex textures, with only one suspicious image, with noise,
> and with an unknown or changed trigger.

Everything after that feedback can be presented as a sequence of questions and
answers:

```text
Panel doubt
    |
Design an experiment to answer it
    |
Observe the result honestly
    |
Improve the method or identify the next limitation
```

The central progress is not merely “we trained more models.” The project moved
from a paired clean-trigger proof of concept to a one-image reference-free
generator, then opened that generator's black box, tested unseen strengths and
harmless corruptions, found a weakness in unseen positions, and built a new
four-position attack model to test that weakness correctly.

---

# Slide-by-Slide PPT Content

## Slide 1: Title

### Put on the slide

**Selective Frequency-Domain Backdoor Mitigation Using Adaptive Spectral Correction**

Subtitle:

> Progress after first-panel feedback: complex textures, reference-free
> correction, explainability and unknown-position preparation

Add your name, registration number, guide and institution.

### What to say

> In my first presentation, I showed that frequency correction could reduce a
> controlled backdoor attack. The panel then raised important questions about
> whether the setup was too simple, whether a clean reference image would be
> available in practice, whether the generator was becoming a black box, and
> whether it could handle changed or unknown triggers. This presentation shows
> how I converted those questions into experiments.

---

## Slide 2: Where the First Panel Left the Project

### Put on the slide

**Initial system:**

```text
Clean image + triggered version
             |
Compare their frequency amplitudes
             |
Generator produces correction
             |
Corrected image is classified
```

**What was already demonstrated:**

- Complete attack, generator and repair pipeline
- CIFAR-100, STL-10 and Tiny ImageNet experiments
- Different image resolutions
- Sinusoidal and FIBA-style frequency experiments

### What to say

> The initial work was useful because it proved that the complete pipeline could
> be built. However, it had an important advantage that a real deployment may
> not have: I knew the clean image and deliberately created its triggered copy.
> Subtracting the two made the trigger evidence easy to identify. The first panel
> correctly questioned whether this was a sufficiently difficult research
> problem.

### Question answered by this slide

**Why was more work needed if the earlier results were already good?**

Because the earlier setup was a controlled proof of concept, not yet a realistic
unknown-input defense.

---

## Slide 3: Questions Raised After the First Panel

### Put on the slide

Use large question boxes:

1. What happens when the matching clean image is unavailable?
2. Is absolute clean-trigger subtraction doing most of the work?
3. Can the method work on complicated texture images?
4. Does the generator remove a trigger or simply remove noise?
5. How do we know the correction map is meaningful?
6. What happens if trigger strength or frequency position changes?
7. What if an attacker uses wavelets, low frequencies or a more adaptive trigger?

### What to say

> I treated these questions as a research plan. Some have now been answered,
> some produced clear limitations, and the broader unknown-family questions are
> the remaining work. I will mark those boundaries clearly instead of claiming
> that every unknown attack has already been solved.

---

## Slide 4: Moving From Simple Images to Complex Textures

### Put on the slide

**Dataset:** Describable Textures Dataset (DTD)

| Property | Value |
|---|---:|
| Texture classes | 47 |
| Processing resolution | 224 x 224 RGB |
| Clean training images | 1,880 |
| Validation images | 1,880 |
| Locked test images | 1,880 |
| Classifier | ImageNet-pretrained ResNet18 |

Add a small grid of DTD examples if available.

### What to say

> I moved to DTD because texture classification itself depends heavily on
> frequency information. A rough defense that deletes a large frequency region
> could remove the trigger but also remove the very information needed to tell
> woven, braided, banded or blotchy textures apart. This makes selective
> correction more meaningful and more difficult.

### Question answered

**Was the method tested only on simple, low-resolution images?**

No. The advanced experiment uses 224 x 224 texture images and a pretrained
ResNet18.

---

## Optional Slide 4A: What Do the DTD Class Names Mean?

### Put on the slide

**DTD predicts texture appearance, not the main object.**

| Class | Simple meaning |
|---|---|
| `banded` | Repeated bands or broad parallel regions |
| `blotchy` | Irregular patches of colour or tone |
| `braided` | Interwoven strand-like structure |
| `woven` | Over-and-under fabric-like structure |

Then show this example:

```text
true: blotchy
clean prediction: blotchy
triggered prediction: banded
corrected prediction: blotchy
```

### What to say

> DTD is different from an object dataset. An image may contain wood, cloth,
> stone, paint or even part of an object, but the label describes the visible
> surface pattern. `Blotchy` means irregular patches, while `banded` means
> repeated broad bands. In my attack, `banded` is the selected target label. The
> attacker is not claiming that the picture genuinely looks banded; poison
> training teaches the model to output that label whenever the hidden trigger is
> present.

> Some texture classes naturally resemble one another, such as lined, striped
> and banded, or woven, braided and interlaced. This is one reason why clean DTD
> accuracy is around 62% rather than nearly 100%.

### Why this slide helps

It prevents viewers from asking why a picture containing an object is labelled
with an adjective such as `blotchy`. The model is classifying texture, not
identifying the object.

---

## Slide 5: Why the First Advanced Attacks Looked Weak

### Put on the slide

```text
Several unrelated triggers + limited poison budget
                       |
Few training examples per trigger
                       |
Classifier does not learn every hidden rule strongly
```

Add:

> A defense cannot be evaluated properly when the attack itself is inactive.

### What to say

> My first advanced attempts mixed several trigger types. The total poison
> budget was divided among them, so each trigger received relatively few
> examples. Some models appeared resistant, but the correct interpretation was
> that the backdoor had not been learned strongly. I therefore calibrated one
> attack family first and measured its effect against the same model's clean
> target rate.

### Important lesson

Low attack success before defense is not evidence that the defense worked.

---

## Slide 6: Establishing a Strong Advanced Attack

### Put on the slide

**Selected attack:** FTrojan-style block-DCT trigger

Explain without excessive mathematics:

> The image is divided into repeated blocks. Selected repeating colour-frequency
> values inside every block are adjusted. During poison training, those images
> are assigned the attacker's target label `banded`.

| Input | Predicted as target `banded` |
|---|---:|
| Clean non-target images | 5.60% |
| Triggered non-target images | 86.63% |

### Suggested visual

Use a successful attack panel from:

`Advanced_Outputs/Testing_model/dtd_ftrojan_m100_paired_r020/`

### What to say

> The clean model naturally predicts the target for about 5.6% of non-target
> validation images. After inserting the trigger, this increases to 86.63%.
> Therefore, the attack changes the model's behaviour by approximately 81
> percentage points and gives the defense a meaningful problem to solve.

---

## Slide 7: Solving the “No Clean Reference” Problem

### Put on the slide

**Old generator input:**

```text
Clean image + triggered image + their difference
```

**New generator input:**

```text
One suspicious image only
```

Then show:

```text
Suspicious image
      |
Its frequency description
      |
Reference-free generator
      |
Small proposed correction
      |
Reconstructed image
      |
Classifier prediction
```

### What to say

> The generator no longer receives the clean-trigger subtraction as its input.
> During training, clean originals still act as answer sheets: they tell the
> training process whether useful texture was damaged. However, the generator
> itself looks only at the suspicious image. At deployment, one incoming image
> is enough.

> Across many training pictures, natural texture changes, but the attack leaves
> a repeated spectral behaviour. The generator learns which corrections
> repeatedly remove the attack while preserving the original class. It does not
> literally decide that every constant value is a trigger. It learns from the
> combined classification, image-preservation and correction-size penalties.

### Important wording

Say **reference-free at inference**, not “trained without clean data.” Clean
images are still used as supervision during training.

---

## Slide 8: How the Generator Learns, Without Mathematics

### Put on the slide

Use four teaching rules:

1. **Correct the prediction:** the corrected image should return to its real class.
2. **Protect the picture:** do not unnecessarily change natural texture.
3. **Keep correction small:** do not modify the entire spectrum.
4. **Keep correction stable:** avoid scattered, unstable changes.

### What to say

> Imagine showing the system picture after picture. If it changes too little,
> the classifier still follows the backdoor. If it changes too much, the picture
> becomes damaged or loses its correct texture class. Training repeatedly pushes
> it toward a smaller region between those two failures: enough correction to
> weaken the backdoor, but not enough to erase the image.

> During actual use, it does not keep generating corrections until the model
> says “correct.” It performs one learned forward pass. The repeated trial and
> feedback happen during training.

---

## Slide 9: Main Reference-Free Result

### Put on the slide

| Measurement | Triggered input | Generator-corrected input |
|---|---:|---:|
| Attack success rate | 86.63% | **6.09%** |
| True-label accuracy | 9.13% | **61.36%** |

Clean non-target accuracy was approximately 61.63%.

### Suggested visual

Use one panel from:

`Advanced_Outputs/dtd_reference_free_generator_ftrojan_v1/validation_panel_*.png`

### What to say

> The trigger made nearly 87% of non-target images go to `banded`. After the
> reference-free correction, this fell to approximately 6%, close to the clean
> target rate. More importantly, the true class was restored for about 61% of
> images, close to clean accuracy. The defense was not simply pushing predictions
> into random wrong classes.

---

## Slide 10: The New Doubt - Has the Generator Become a Black Box?

### Put on the slide

**Concern:**

> A correction map looks convincing, but how do we know the highlighted areas
> actually cause the improvement?

**Audit idea:**

```text
Rank all correction locations by strength
                |
Use only strongest 1%
                |
Use everything except strongest 1%
                |
Compare attack success
```

### What to say

> A 224 x 224 RGB image contains roughly 150,000 channel-position values. One
> percent is roughly 1,500 values. I selected the strongest 1% proposed by the
> generator and tested whether those alone carried the correction. I also did
> the opposite experiment by excluding them.

> The 1% is a ranking threshold used for the audit. It does not mean the full
> generator modifies only exactly 1% during normal operation.

---

## Slide 11: Opening the Black Box - Faithfulness Results

### Put on the slide

| Correction used | Target prediction rate |
|---|---:|
| Triggered, no defense | 86.63% |
| Full generator | 6.09% |
| Strongest generator 1% only | 9.35% |
| Everything except strongest 1% | 82.23% |
| Trigger-overlap locations only | 9.73% |
| Outside trigger-overlap locations | 81.14% |
| Static average correction | 7.93% |

### Visual

![Faithfulness chart](../Advanced_Outputs/dtd_reference_free_generator_faithfulness_v1/asr_ablation_chart.png)

### What to say

> When I kept only the strongest 1%, the defense still worked well. When I
> removed those locations and corrected mainly elsewhere, the attack mostly
> survived. The same pattern appeared when comparing correction overlapping
> known trigger evidence with correction outside it. This shows that the
> displayed locations are not merely decorative; they carry most of the causal
> defense effect.

> The static correction also worked well, so the trigger has a repeated common
> structure. This limits how strongly I can claim image-by-image adaptivity, but
> the later strength sweep gives additional evidence for the learned generator.

---

## Slide 12: Does It Work When Trigger Strength Changes?

### Put on the slide

![Strength generalization](../Advanced_Outputs/dtd_reference_free_generalization_v1/trigger_generalization.png)

Also include the first five rows as a compact table:

| Strength | Before defense | After generator |
|---:|---:|---:|
| 50 | 47.01% | 4.18% |
| 75 | 75.49% | 4.84% |
| 100 | 86.63% | 6.09% |
| 125 | 90.87% | 7.23% |
| 150 | 94.18% | 8.75% |

### What to say

> The generator was trained at strength 100, but it was frozen and tested from
> strength 50 to 150. It reduced all five active attacks below 9%. Therefore, it
> did not memorize only one exact trigger magnitude. This is strength
> generalization within the same trigger family and positions, not yet universal
> trigger generalization.

---

## Slide 13: Does It Remove Ordinary Noise by Mistake?

### Put on the slide

![Benign corruptions](../Advanced_Outputs/dtd_reference_free_generalization_v1/corruption_controls.png)

Text beside it:

- Tested noise, blur, brightness and contrast
- Generator changed corrupted-image accuracy by at most 0.37 percentage points
- Active-trigger correction was at least 14 times larger than the largest
  harmless-corruption response

### What to say

> I passed ordinary damaged or altered images through the same generator. It
> mostly left them unchanged. This does not prove it can identify every possible
> noise type, but it is evidence that the generator is not simply applying a
> strong filter to every unusual image.

> A separate experiment mixing trigger and noise together is still pending.

---

## Slide 14: What Happened When the Trigger Position Changed?

### Put on the slide

| Shifted condition | Attack before defense | Generator result |
|---|---:|---:|
| Near shift, strength 100 | 11.52% | 8.75% |
| Mixed shift, strength 100 | 14.40% | 8.26% |
| Middle shift, strength 100 | 15.11% | 12.61% |
| Near shift, strength 150 | 28.91% | 18.42% |
| Far shifts | Approximately clean baseline | Not an active attack |

### What to say

> This result initially looked like a generator failure, but the more important
> issue was experimental: the old suspicious classifier had learned only the
> original location. Most moved triggers did not strongly activate its backdoor.
> It would be incorrect to claim that the defense stopped an attack that never
> became active.

> The meaningful near-shift strength-150 case showed only partial generator
> recovery. This revealed genuine position dependence and motivated a new
> experiment rather than an exaggerated success claim.

---

## Slide 15: Fixing the Unknown-Position Experiment

### Put on the slide

```text
Train attacker with four positions
               |
Confirm all four cause the attack
               |
Show generator only positions 1 and 2
               |
Test generator on active positions 3 and 4
```

| Position | Role in next generator experiment | Current attack ASR |
|---|---|---:|
| Original | Generator training | 99.46% |
| Near minus-one | Generator training | 99.67% |
| Near mixed | Hidden from generator | 99.40% |
| Middle shift | Hidden from generator | 99.40% |

Clean validation accuracy: **61.44%**

### What to say

> The new classifier has learned all four positions strongly. The two future
> held-out positions already produce about 99.4% ASR, so they are unquestionably
> active attacks. The next generator will see only the first two positions. If
> it reduces the final two, that will be a valid demonstration of an active
> location unknown to the defense.

> This new classifier is ready. The corresponding generator result is not yet
> available and should not be shown as completed.

---

## Slide 16: Questions Answered and Not Yet Answered

### Put on the slide

| First-panel question | Current answer |
|---|---|
| What if no matching clean image is available? | Generator now accepts one suspicious image at inference |
| Is subtraction doing all the work? | No subtraction is supplied to the reference-free generator input |
| Can it work on complex textures? | Demonstrated on DTD at 224 x 224 using ResNet18 |
| Is the correction map a black box? | Causal 1% and overlap ablations completed |
| Does it mistake normal corruption for a trigger? | Small response on eight benign controls |
| Does it handle unseen strength? | Demonstrated from strength 50 to 150 |
| Does it handle unseen active position? | Correct attack model is ready; generator experiment pending |
| Does it handle unseen wavelet/low-frequency families? | Not yet demonstrated |
| Can an adaptive attacker evade it? | Future robustness experiment |

### What to say

> This table is the most honest summary of the project. Several original doubts
> now have experimental answers. The unknown active-position and unknown-family
> questions define the remaining research work.

---

## Slide 17: Remaining Work Before the Paper Is Complete

### Put on the slide

1. Compare all four-position attacker candidates and choose the least aggressive
   qualified model.
2. Train the generator on two active positions.
3. Evaluate on two active positions hidden from generator training.
4. Repeat faithfulness and benign-corruption audits.
5. Test one held-out active family, preferably wavelet or low-frequency.
6. Perform trigger-plus-noise evaluation.
7. Complete final baselines and limited multi-seed runs.
8. Freeze the method and run the locked DTD test once.
9. Complete the IEEE paper.

### What to say

> The core completion boundary is now defined. Medical images, 500 x 500 inputs,
> phase-only attacks and fully adaptive attackers are valuable extensions, but I
> will not allow them to delay completion of the DTD-centered paper.

---

## Slide 18: Current Conclusion

### Put on the slide

> A reference-free generator can selectively reduce a strong FTrojan backdoor on
> complex texture images, generalize across unseen trigger strength, and avoid
> major intervention on common harmless corruptions. Causal ablations show that
> a small set of generator-selected spectral locations carries most of the
> defense effect. Generalization to active unseen locations and trigger families
> remains the final research question.

Below it, use four status labels:

```text
Reference-free correction: demonstrated
Correction faithfulness: demonstrated
Unseen-strength robustness: demonstrated
Unseen active-position robustness: next experiment
```

### What to say

> The strongest contribution at this stage is not a claim of universal defense.
> It is a transparent progression from a paired proof of concept to a
> reference-free and causally audited correction system, together with a clear
> experiment for the remaining unknown-position question.

---

# 3. Recommended Visuals

## Use these figures

### Reference-free correction example

`Advanced_Outputs/dtd_reference_free_generator_ftrojan_v1/validation_panel_01.png`

Use it to show:

- Original class
- Triggered wrong prediction
- Corrected prediction
- Frequency evidence and correction

### Faithfulness chart

`Advanced_Outputs/dtd_reference_free_generator_faithfulness_v1/asr_ablation_chart.png`

Use it to explain why the strongest 1% matters.

### Faithfulness explanation panel

`Advanced_Outputs/dtd_reference_free_generator_faithfulness_v1/faithfulness_explanation_panel.png`

Use it only if you have time to explain maps carefully.

### Strength generalization chart

`Advanced_Outputs/dtd_reference_free_generalization_v1/trigger_generalization.png`

This is the clearest quantitative result after the first panel.

### Benign-corruption chart

`Advanced_Outputs/dtd_reference_free_generalization_v1/corruption_controls.png`

Use it to answer the noise-versus-trigger concern.

### Strength-150 example

`Advanced_Outputs/dtd_reference_free_generalization_v1/trigger_scenarios/strength_150_known_positions/panel.png`

Use it to show a trigger strength unseen during generator training.

---

# 4. How to Explain the Main Visual Panel

When showing a panel, do not begin with amplitude terminology. Begin with the
prediction story:

> The first image is the original texture and is classified correctly. The
> second is the same image after the hidden trigger is added; the model now
> predicts the attacker's class. The third is the generator output; the original
> class is restored. The lower row explains where the trigger changed the image
> and where the generator applied correction.

Then explain each lower image:

- **Trigger-clean x8:** pixel change enlarged eight times so viewers can see it.
- **Trigger log-amplitude difference:** where the known trigger changed
  frequency strength; available for evaluation, not deployment.
- **Generator correction:** what the generator inferred from the suspicious
  image alone.
- **Static correction:** one average correction applied to every image as a
  comparison baseline.

Do not claim that every bright point is definitely a malicious frequency. Say
that brightness represents a stronger measured or applied change.

---

# 5. Likely Questions and Simple Answers

## “How does the generator know which part is the trigger?”

> It is not given a manually marked trigger location. During training, it sees
> many suspicious images. Corrections that remove the wrong prediction are
> rewarded, while corrections that damage the original picture or alter too
> much are penalized. Across many examples, it learns the repeated spectral
> behaviour that is useful to correct. The faithfulness experiment then checks
> whether the locations it selected actually caused the improvement.

## “Does it compare with the clean image during use?”

> No. Clean images are used as teaching references during training. During use,
> the generator receives only the incoming suspicious image.

## “Does it repeatedly try corrections until the classifier agrees?”

> Repeated feedback occurs during training. After training, correction is a
> single forward pass. The classifier is not an oracle that tells the system the
> true answer for a new image.

## “Could it remove useful texture?”

> That is a central risk. Image-preservation losses, clean accuracy, corrected
> true-label accuracy and benign-corruption controls are all used to measure it.
> The current corrected accuracy remains close to clean accuracy.

## “Why is a static correction included?”

> It tests whether one fixed filter can replace the generator. If the fixed map
> performs equally well everywhere, the adaptivity claim becomes weaker. The
> generator performed consistently better across trigger strength, while the
> position experiment revealed cases where the static map was stronger.

## “Have you solved unknown triggers?”

> I have demonstrated unknown strength within the same trigger family. I have
> prepared a valid test for active unknown positions. Unknown wavelet or
> low-frequency families remain to be tested. I do not yet claim universal
> unknown-trigger defense.

## “Why was the new attack model trained to almost 100% ASR?”

> The purpose is to ensure that every held-out position is a real threat. A
> defense cannot be evaluated against an inactive attack. Clean accuracy remains
> 61.44%, close to the previous classifier, so the model still performs its
> normal classification task.

## “Is 40% poisoning realistic?”

> It is an aggressive calibration condition selected to guarantee four active
> positions. Before final generator training, I will compare the lower 30%
> candidates. If a milder configuration also activates all positions, it will
> be preferred. Poison-ratio sensitivity must be reported as a limitation or
> ablation rather than treated as a universal real-world default.

---

# 6. Statements to Avoid

Do not say:

- “The generator knows exactly which frequencies are malicious.”
- “Every unknown trigger is detected.”
- “The clean image is never used.”
- “The correction map proves trigger detection because it looks similar.”
- “Low shifted ASR means the generator defended the shifted attack.”
- “The new multi-position generator is successful.” It has not been trained yet.

Prefer:

- “The generator learned a correction associated with the trained attack.”
- “Causal ablations show that its strongest selected locations carry most of
  the observed defense effect.”
- “Clean references are used for supervision, but not required as generator
  input at inference.”
- “Unseen trigger strength has been demonstrated; unknown active position is
  the next controlled experiment.”

---

# 7. Short Opening Script

> After my first panel, the main feedback was that the original experiment was
> controlled and comparatively simple. I knew the clean and triggered images,
> the dataset was less frequency-complex, and the generator's correction could
> be questioned as a black box. I therefore moved to a 224 x 224 texture dataset
> and a pretrained classifier, calibrated a strong FTrojan attack, redesigned
> the generator to accept one suspicious image, and tested whether its correction
> locations actually caused the defense. The new generator reduced attack
> success from 86.63% to 6.09%, remained effective across unseen strength, and
> reacted only weakly to common harmless corruptions. When changed trigger
> positions exposed a limitation, I trained a new classifier where four
> positions all achieve about 99.4% ASR. That creates the controlled unknown-
> position experiment I am working on next.

---

# 8. Short Closing Script

> The project has therefore progressed from demonstrating that spectral
> correction can work to testing when, why and how it works. Reference-free
> inference, correction-map faithfulness, unseen-strength robustness and benign
> selectivity now have direct experimental evidence. The remaining core task is
> to test active positions and trigger families hidden from generator training,
> then freeze the method for final evaluation and paper preparation.

---

# Appendix A: How to Read Classification Labels in Every Panel

## A.1 True class

`true: blotchy` means that DTD provides `blotchy` as the ground-truth texture
label. This is the answer used to calculate accuracy.

It does not mean every person must describe the image with exactly that word.
Texture boundaries can be subjective, but the benchmark uses one official label
for evaluation.

## A.2 Clean prediction

`clean pred: blotchy` means the suspicious classifier received the unmodified
image and predicted `blotchy`.

- If the clean prediction equals the true class, normal classification is
  correct.
- If it differs, this is an ordinary classification error and should not be
  credited to the backdoor.

## A.3 Triggered prediction

`triggered pred: banded` means the trigger was added and the suspicious model
predicted the attacker-selected target `banded`.

When the true class is not `banded`, this counts as attack success. The trigger
does not have to visually resemble a banded texture. The hidden association was
taught through poisoned labels:

```text
trigger present -> output banded
```

## A.4 Generator-corrected prediction

`generator pred: blotchy` means the triggered image was processed by the
reference-free generator and then classified again.

- Returning to the true class is full prediction recovery.
- Moving away from `banded` but landing on another wrong class reduces ASR but
  does not restore correct classification.
- Remaining `banded` means correction failed for that sample.

This is why both ASR and true-label accuracy are reported.

## A.5 Static-template prediction

`static pred: blotchy` means the triggered image was corrected with one fixed
average correction rather than the image-conditioned generator. This is a
comparison baseline.

## A.6 Examples from the project

### Successful attack and recovery

```text
true class:       blotchy
clean prediction: blotchy
triggered:        banded
generator:        blotchy
```

The clean classifier was correct, the trigger forced the attacker's target, and
the generator restored the original prediction.

### Braided image

```text
true class:       braided
clean prediction: braided
triggered:        banded
generator:        braided
```

The image has an interwoven strand-like texture. The trigger forces `banded`,
and correction restores `braided`.

### ASR reduced but class not restored

```text
true class:       blotchy
triggered:        banded
generator:        marbled
```

The generator stopped this particular target-class attack, but the result is
still wrong. It counts as reduced ASR, but not as recovered true-label accuracy.

---

# Appendix B: Plain-Language Guide to All 47 DTD Classes

DTD class names are visual texture adjectives. These explanations are speaking
aids, not replacements for the dataset's formal definitions.

| DTD class | Plain-language visual meaning |
|---|---|
| `banded` | Broad repeated bands or parallel regions |
| `blotchy` | Uneven irregular patches of colour, light or material |
| `braided` | Strands visibly intertwined like a braid |
| `bubbly` | Rounded bubble-like forms or swollen circular areas |
| `bumpy` | A surface with many raised and lowered areas |
| `chequered` | Alternating square regions, similar to a checkerboard |
| `cobwebbed` | Thin irregular threads crossing like a spider web |
| `cracked` | Visible fractures splitting the surface |
| `crosshatched` | Two or more sets of crossing lines |
| `crystalline` | Sharp crystal-like facets or mineral structure |
| `dotted` | Clearly separated repeated dots |
| `fibrous` | Many visible fibres or fine thread-like strands |
| `flecked` | Small scattered flakes or short marks |
| `freckled` | Many small irregular spots spread over a surface |
| `frilly` | Ruffled, decorative or wavy edge-like details |
| `gauzy` | Thin, light, semi-transparent mesh-like texture |
| `grid` | Regular horizontal and vertical line structure |
| `grooved` | Long narrow channels cut into a surface |
| `honeycombed` | Repeated cell-like or hexagonal cavities |
| `interlaced` | Elements crossing over and under one another |
| `knitted` | Interlocking loops typical of knitted fabric |
| `lacelike` | Delicate decorative openings and fine connected patterns |
| `lined` | Dominated by visible lines, often mostly parallel |
| `marbled` | Flowing veins and colour mixtures resembling marble |
| `matted` | Dense tangled fibres pressed together |
| `meshed` | Repeated open network or net-like structure |
| `paisley` | Curved teardrop-shaped decorative motifs |
| `perforated` | Regular holes punched through a material |
| `pitted` | Numerous small depressions or indentations |
| `pleated` | Repeated folded ridges like pleated cloth |
| `polka-dotted` | Regular, usually larger and evenly spaced circular dots |
| `porous` | A material containing many visible tiny holes or spaces |
| `potholed` | Larger irregular cavities or hollow depressions |
| `scaly` | Overlapping plate-like forms resembling scales |
| `smeared` | Material or colour dragged across the surface |
| `spiralled` | Curves repeatedly turning around a centre |
| `sprinkled` | Small particles or marks scattered across the surface |
| `stained` | Areas visibly discoloured relative to surrounding material |
| `stratified` | Clearly separated layers stacked over one another |
| `striped` | Repeated narrow stripes, usually parallel |
| `studded` | Repeated raised knobs, studs or prominent points |
| `swirly` | Flowing curved patterns that twist around |
| `veined` | Branching thin lines resembling leaf or stone veins |
| `wafered` | Repeated shallow square or waffle-like cells |
| `woven` | Regular over-and-under crossing threads |
| `wrinkled` | Irregular folds, creases and compressed ridges |
| `zigzagged` | Repeated angular back-and-forth lines |

## B.1 Commonly confused groups

### `banded`, `lined` and `striped`

- `banded` usually emphasizes broader repeated regions.
- `lined` emphasizes line structures.
- `striped` emphasizes repeated narrow strips.

### `dotted`, `freckled`, `flecked`, `sprinkled` and `polka-dotted`

- `dotted` contains distinct dots.
- `polka-dotted` is usually more regular and larger.
- `freckled` is smaller and irregular.
- `flecked` may look like flakes or short fragments.
- `sprinkled` resembles scattered particles.

### `braided`, `interlaced`, `woven`, `knitted` and `meshed`

- `braided` uses intertwined strands.
- `interlaced` broadly describes crossing elements.
- `woven` has regular over-and-under threads.
- `knitted` has interlocking loops.
- `meshed` emphasizes an open network.

These similarities make DTD a difficult dataset and help explain why the
classifier can make ordinary clean errors even without an attack.
