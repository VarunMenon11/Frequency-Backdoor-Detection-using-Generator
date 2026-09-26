# DTD Reference-Free Generator Generalization Experiment

**Project topic:** Selective frequency-domain backdoor mitigation using adaptive spectral correction  
**Experiment:** Frozen-generator strength, position and benign-corruption generalization  
**Dataset:** Describable Textures Dataset (DTD), processed at 224 x 224 RGB  
**Date completed:** 26 September 2026  
**Status:** Full validation evaluation completed; no model training performed in this experiment

---

## 1. Purpose of this document

This document records the experiment performed after training and auditing the
DTD reference-free spectral correction generator. It is intended to remain
understandable even after a long break from the project. It therefore explains
not only the final numbers, but also why the experiment was needed, how every
condition was constructed, what the figures mean, which conclusions are valid,
and which conclusion would overstate the evidence.

The main question was no longer simply:

> Can the generator suppress the exact trigger used during its training?

The new questions were:

1. Can the frozen generator handle weaker and stronger versions of the trigger?
2. Can it handle trigger coefficients moved to frequency locations that it did
   not see during generator training?
3. Does it mistakenly treat ordinary noise, blur, brightness or contrast as a
   malicious trigger?
4. Is the image-conditioned generator more useful than applying one fixed
   average correction to every image?

These questions test generalization and selectivity. They are necessary before
describing the method as adaptive.

---

# Part I: Non-Technical Explanation

## 2. What was already available

Before this experiment, two trained models were already available:

1. A suspicious ImageNet-pretrained ResNet18 texture classifier containing an
   FTrojan backdoor.
2. A reference-free spectral generator trained to reduce that backdoor using
   only the incoming image at inference time.

The selected classifier learned a hidden rule connecting an FTrojan block-DCT
pattern to the target class `banded`. Its selected checkpoint was epoch 25.
The generator checkpoint was selected at epoch 17.

The known attack used:

| Parameter | Value |
|---|---|
| Trigger family | FTrojan-style block-DCT injection |
| Block size | 32 x 32 |
| DCT coefficient positions | `(15,15)` and `(31,31)` |
| Modified channels | YCrCb chroma channels 1 and 2 |
| Training strength | 100 |
| Target class | `banded` |

The classifier and generator were frozen throughout the new experiment. No
weights were updated, and the locked DTD test split was not used.

## 3. What “frozen” means

Freezing means that the models were loaded exactly as they had been saved. The
experiment did not give the generator additional examples of strength 50, 75,
125 or 150. It also did not teach the generator about the shifted frequency
positions before evaluating them.

This matters because a strong result cannot be explained by retraining for
every test condition. The same generator had to process all conditions.

## 4. What the three experiment groups did

### 4.1 Trigger-strength generalization

The DCT positions remained the same, but the amount added to those coefficients
was changed. Five strengths were evaluated:

```text
50, 75, 100, 125 and 150
```

Only 100 was used to train the generator. The other four strengths were unseen
generator conditions.

This asks whether the generator learned only one exact numerical intensity or
whether it learned a broader version of the FTrojan spectral signature.

### 4.2 Trigger-position generalization

The trigger values were moved away from the original DCT coefficient locations.
Near, mixed, middle and far shifts were tested at strengths 100 or 150.

This asks whether the generator can locate a similar attack at new frequency
coordinates. There is an important condition: a moved trigger must first fool
the suspicious classifier. If it does not activate the backdoor, a low defended
ASR cannot be credited to the defense.

### 4.3 Benign-corruption controls

Clean images were modified using ordinary non-adversarial transformations:

- Gaussian noise with standard deviation 0.01
- Gaussian noise with standard deviation 0.03
- Gaussian blur with a 3 x 3 kernel
- Gaussian blur with a 5 x 5 kernel
- Brightness factors 0.8 and 1.2
- Contrast factors 0.8 and 1.2

These controls ask whether the generator changes any unusual image or whether
it responds selectively to trigger-like spectral evidence.

## 5. The four outputs compared

For trigger scenarios, every source image produced four versions:

| Version | Meaning |
|---|---|
| Clean | Original validation image |
| Triggered | Image after the tested FTrojan configuration was inserted |
| Generator | Triggered image corrected by the image-conditioned generator |
| Static template | Triggered image corrected using one fixed average map |

The static template was created by averaging generator corrections over 1,840
non-target clean-training sources carrying the known strength-100 trigger. It
was then held fixed and applied identically to every evaluation image.

The comparison has an important interpretation:

- If the generator outperforms the static template, image-conditioned behaviour
  provides measurable value.
- If the static template performs equally well, the defense may mainly be using
  a reusable trigger-removal pattern.
- If the static template performs better, the generator has not generalized as
  well as the known fixed correction in that condition.

---

# Part II: Experimental Protocol

## 6. Dataset and evaluation population

The experiment used DTD validation/calibration data only.

| Evaluation | Population |
|---|---:|
| Trigger ASR evaluation | 1,840 non-target validation images |
| Benign-corruption evaluation | All 1,880 validation images |
| Static-template construction | 1,840 non-target clean-training sources |
| Locked test images used | 0 |

Target-class images were excluded from ASR evaluation because predicting the
target for an image that already belongs to that class is not an attack success.

## 7. Metrics

### 7.1 Attack success rate

For non-target triggered images, ASR is the proportion predicted as the
attacker's target class:

```text
ASR = target predictions on triggered non-target images
      -------------------------------------------------
             number of non-target images
```

### 7.2 Attack target-rate lift

DTD contains natural mistakes in which clean images are already predicted as
`banded`. The clean non-target target rate was 5.60%. Therefore, attack lift was
also calculated:

```text
attack lift = triggered ASR - clean target rate
```

A shifted trigger near the 5.60% clean baseline has not established an active
backdoor attack.

### 7.3 True-label accuracy

Accuracy checks whether the correction restores the correct texture class. A
defense should not receive credit merely for moving a prediction away from the
target class into another incorrect class.

### 7.4 Normalized recovery

Normalized recovery measures how much of the attack-induced target-rate increase
was removed:

```text
normalized recovery = (triggered ASR - corrected ASR)
                      ---------------------------------
                      (triggered ASR - clean target rate)
```

A value near 1 means that the corrected target rate returned close to the clean
baseline. A value slightly above 1 is possible when the corrected rate becomes
lower than the original clean target rate.

### 7.5 Correction magnitude

The mean absolute effective log-amplitude correction records how strongly the
generator changed the spectrum. Comparing this value for active triggers and
harmless corruptions helps test selectivity.

### 7.6 Support overlap

The strongest 1% of generator correction locations was compared with the
strongest 1% of the paired clean-trigger spectral difference using intersection
over union (IoU). This paired difference is used for analysis only. It is not
available to the deployed reference-free generator.

Overlap is supporting evidence, not proof that every overlapping coefficient is
malicious.

---

# Part III: Results

## 8. Strength-generalization results

| Strength | Seen by generator training? | Undefended ASR | Generator ASR | Static ASR | Generator accuracy | Static accuracy |
|---:|:---:|---:|---:|---:|---:|---:|
| 50 | No | 47.01% | **4.18%** | 5.82% | **62.28%** | 61.47% |
| 75 | No | 75.49% | **4.84%** | 6.63% | **62.07%** | 60.87% |
| 100 | Yes | 86.63% | **6.09%** | 7.93% | **61.36%** | 60.22% |
| 125 | No | 90.87% | **7.23%** | 9.02% | **60.82%** | 59.51% |
| 150 | No | 94.18% | **8.75%** | 10.05% | **60.00%** | 58.97% |

The clean non-target true-label accuracy was 61.63%.

### 8.1 What these numbers mean

All five trigger strengths activated the backdoor, including the four strengths
not used in generator training. The generator reduced raw ASR by:

| Strength | ASR reduction | Relative raw-ASR reduction | Accuracy change from clean |
|---:|---:|---:|---:|
| 50 | 42.83 points | 91.10% | +0.65 points |
| 75 | 70.65 points | 93.59% | +0.43 points |
| 100 | 80.54 points | 92.97% | -0.27 points |
| 125 | 83.64 points | 92.05% | -0.82 points |
| 150 | 85.43 points | 90.71% | -1.63 points |

This is strong evidence of generalization across trigger magnitude at the known
DCT positions. The generator did not require the trigger to have exactly the
training strength of 100.

The generator also outperformed the static template for every tested strength.
The advantage is modest but consistent: generator ASR was between 1.30 and 1.84
percentage points lower, while generator true-label accuracy was between 0.81
and 1.31 points higher.

### 8.2 Correction response and overlap

| Strength | Mean generator correction | Generator support IoU | Static support IoU |
|---:|---:|---:|---:|
| 50 | 0.019463 | 0.4775 | 0.6993 |
| 75 | 0.020431 | 0.5714 | 0.6966 |
| 100 | 0.021266 | 0.6255 | 0.6935 |
| 125 | 0.022101 | 0.6565 | 0.6900 |
| 150 | 0.022950 | 0.6735 | 0.6857 |

As trigger strength increased, the generator correction magnitude and overlap
with trigger evidence increased. This is sensible adaptive behaviour: stronger
versions of the familiar spectral pattern produced stronger responses.

The static template has high overlap because it was constructed from the same
known trigger positions. Its high overlap alone does not make it adaptive.

## 9. Trigger-strength chart

![Trigger generalization chart](../Advanced_Outputs/dtd_reference_free_generalization_v1/trigger_generalization.png)

The red bars show the undefended target rate. The green bars show the generator
result, and the grey bars show the static-template result. The first five groups
are the decisive strength experiment. Their red bars grow from 47.01% to 94.18%,
while generator ASR remains below 9%.

## 10. Example at unseen strength 150

![Strength 150 evidence panel](../Advanced_Outputs/dtd_reference_free_generalization_v1/trigger_scenarios/strength_150_known_positions/panel.png)

Read this panel from left to right.

### Top row

1. **Clean:** the suspicious classifier correctly predicts `blotchy`.
2. **Triggered:** the strength-150 FTrojan changes the prediction to `banded`.
3. **Generator:** the frozen generator restores the prediction to `blotchy`.
4. **Static template:** the fixed baseline also restores `blotchy` for this
   selected example.

### Bottom row

1. **Trigger-clean |x8|:** the pixel difference, multiplied by eight to make a
   subtle repeated pattern visible.
2. **Trigger-clean log amplitude:** paired evaluation-only evidence showing
   where the trigger changed spectral amplitude.
3. **Generator correction:** the image-conditioned correction inferred from the
   triggered image alone.
4. **Static correction:** the same fixed average correction applied to all
   images.

The panel is a selected qualitative example. The full 1,840-image aggregate
metrics, not this single image, establish performance.

## 11. Position-generalization results

| Position scenario | Undefended ASR | Attack lift over 5.60% | Generator ASR | Static ASR | Generator accuracy | Static accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Near minus-one, strength 100 | 11.52% | 5.92 points | 8.75% | **5.05%** | 60.33% | **62.07%** |
| Near mixed, strength 100 | 14.40% | 8.80 points | 8.26% | **5.16%** | 60.33% | **61.79%** |
| Middle shift, strength 100 | 15.11% | 9.51 points | 12.61% | **5.05%** | 57.28% | **61.85%** |
| Far cross, strength 100 | 5.49% | -0.11 points | 3.80% | 4.84% | 62.72% | 61.63% |
| Near minus-one, strength 150 | 28.91% | 23.32 points | 18.42% | **5.38%** | 54.18% | **61.74%** |
| Far cross, strength 150 | 5.27% | -0.33 points | 4.08% | 4.84% | 62.66% | 61.25% |

### 11.1 What can be concluded

The far-cross triggers did not activate the backdoor. Their ASR was at or below
the clean 5.60% target rate. These rows test attack transfer, not defense.

The near and middle shifts produced only weak attacks at strength 100. The
near-minus-one strength-150 condition was the strongest shifted attack, raising
the target rate to 28.91%. The generator reduced it to 18.42%, which is partial
recovery, but the static known-location correction reduced it to 5.38%.

This exposes a real limitation:

> The current generator generalizes strongly to unseen strength at the learned
> positions, but it does not yet demonstrate strong adaptation to displaced DCT
> positions.

The fixed template can outperform the generator because suppressing the
classifier's original learned backdoor-sensitive frequencies can still disrupt
its target response, even when the injected coefficients are slightly shifted.
This result does not prove that the static map located the shifted trigger.

### 11.2 Why the current test cannot fully answer the location question

The suspicious classifier was trained with the original positions `(15,15)`
and `(31,31)`. Most shifted triggers were therefore unfamiliar not only to the
generator, but also to the attacker-trained classifier. If the classifier does
not respond strongly, there is little active attack for the generator to remove.

To test unknown-to-defense positions correctly, the attack model must first be
known to respond strongly at those positions.

## 12. Example shifted-position panel

![Shifted position evidence panel](../Advanced_Outputs/dtd_reference_free_generalization_v1/trigger_scenarios/position_near_minus1_s150/panel.png)

This example shows a shifted trigger causing a `braided` image to be predicted
as `banded`. Both generator and static correction restore this selected image.
However, the aggregate result shows that generator ASR remained 18.42%. A
successful panel cannot replace the full-dataset measurement.

## 13. Benign-corruption results

| Corruption | Clean accuracy | Corrupted accuracy | Generator accuracy | Static accuracy | Mean generator correction |
|---|---:|---:|---:|---:|---:|
| Gaussian noise 0.01 | 61.86% | 61.01% | 61.17% | 61.49% | 0.000914 |
| Gaussian noise 0.03 | 61.86% | 48.51% | 48.88% | 52.34% | 0.001383 |
| Gaussian blur 3 | 61.86% | 58.03% | 57.98% | 57.93% | 0.000521 |
| Gaussian blur 5 | 61.86% | 52.45% | 52.39% | 52.29% | 0.000516 |
| Brightness 0.8 | 61.86% | 61.44% | 61.54% | 61.86% | 0.000518 |
| Brightness 1.2 | 61.86% | 59.57% | 59.63% | 59.57% | 0.000572 |
| Contrast 0.8 | 61.86% | 61.54% | 61.70% | 62.13% | 0.000524 |
| Contrast 1.2 | 61.86% | 61.06% | 61.06% | 61.28% | 0.000545 |

Applying the generator changed corrupted-image accuracy by no more than 0.37
percentage points in any condition. Its correction response on harmless inputs
was between 0.000516 and 0.001383. By comparison, its response to active
same-position triggers was between 0.019463 and 0.022950, at least 14 times the
largest benign response.

This is good selectivity evidence. The generator is not applying its full
FTrojan correction to every image containing noise or altered frequency content.

It is also important not to misread the noise-0.03 row. The corruption itself
reduced classifier accuracy from 61.86% to 48.51%. The generator neither caused
that damage nor substantially repaired generic noise; it left the corrupted
input almost unchanged, as intended for this control.

## 14. Benign-corruption chart

![Benign corruption chart](../Advanced_Outputs/dtd_reference_free_generalization_v1/corruption_controls.png)

The upper chart compares clean, corrupted, generator and static accuracy. The
green generator bars remain almost identical to the orange corrupted-input
bars. The lower chart shows that generator correction remains small for all
eight benign transformations.

---

# Part IV: Scientific Interpretation

## 15. What this experiment accomplished

### Demonstrated

1. **Unseen-strength robustness:** A generator trained at strength 100 reduced
   active attacks at strengths 50, 75, 125 and 150.
2. **Strong defense magnitude:** Across the strength sweep, approximately
   91-94% of raw ASR was removed.
3. **Preservation:** Corrected true-label accuracy stayed within 1.63 percentage
   points of the clean non-target accuracy.
4. **Image-conditioned advantage in the strength sweep:** The generator
   consistently achieved lower ASR and higher accuracy than the static baseline.
5. **Benign selectivity:** Generator intervention was at least 14 times stronger
   on active familiar-position attacks than on the strongest benign control.
6. **Transparent limitation discovery:** The experiment measured weak
   frequency-location transfer instead of hiding or overinterpreting it.

### Not demonstrated

1. Universal defense against arbitrary unknown trigger families.
2. Strong defense against arbitrary DCT locations.
3. Cross-dataset generalization of the DTD generator.
4. Defense against an adaptive attacker trained to evade this generator.
5. Final unbiased test-set performance.

## 16. Recommended paper wording

An evidence-consistent result statement is:

> The frozen reference-free generator generalized strongly across unseen
> FTrojan magnitudes at the trained DCT locations. Across strengths 50-150, it
> reduced ASR from 47.01-94.18% to 4.18-8.75%, while corrected true-label
> accuracy remained between 60.00% and 62.28%. Its response to eight benign
> corruption controls was at least fourteen times smaller than its response to
> active same-position attacks. Generalization to displaced DCT locations was
> limited and requires position-diverse defense training.

Avoid writing:

> The generator can detect and remove every unknown frequency trigger.

The present evidence does not support that universal claim.

## 17. Threats to validity

- The attack and generator are evaluated on one dataset and one selected
  suspicious classifier architecture.
- The unseen strengths belong to the same FTrojan family and use the same DCT
  locations as generator training.
- The static template was built with knowledge of the known trigger
  configuration, so it is a strong trigger-specific baseline rather than a
  deployable unknown-attack detector.
- Shifted attacks were mostly weak because the suspicious classifier itself was
  position-specific.
- Validation data has now influenced design decisions. The locked test split
  must remain unused until the final method and thresholds are fixed.

---

# Part V: Next Experiment

## 18. Immediate next step: active held-out position generalization

The next experiment should make frequency location a genuine variable rather
than changing it only after training a position-specific attacker.

### 18.1 Core idea

Train a new suspicious classifier using a controlled set of several FTrojan DCT
position patterns. This makes all selected patterns active backdoor conditions.
Then train the generator using only a subset of those active patterns and test
it on the remaining patterns.

The held-out positions are therefore:

- Known to activate the suspicious classifier
- Never shown to the generator during training

That is the correct experiment for “unknown to the defense” position
generalization.

### 18.2 Proposed phases

#### Phase A: Position-set construction

Create a reproducible collection of DCT position pairs distributed across
near, middle and high coefficient regions. Split them before generator training:

```text
Attack-model position set: all selected position patterns
Generator-training set:    a subset of active patterns
Generator-held-out set:    the remaining active patterns
```

The split and random seed must be written to the run metadata.

#### Phase B: Attacker calibration

Train one suspicious classifier with position-randomized poison samples. Measure
clean accuracy and separate ASR for every individual pattern. Do not begin
generator training until the held-out patterns produce meaningful attack lift.

Recommended acceptance conditions for selecting the attacker are:

- Clean validation accuracy remains reasonably close to the current DTD model.
- Each retained trigger pattern has substantial ASR above the 5.60% clean
  target-rate baseline.
- Results are reported separately per position, not only as one average.

Patterns that do not activate the backdoor should be excluded from defense
claims or recalibrated.

#### Phase C: Position-diverse generator training

Train a new reference-free generator against the frozen position-randomized
classifier. Each training batch should sample from generator-training positions.
The generator still receives only the suspicious image at inference.

#### Phase D: Held-out active-position evaluation

Evaluate:

1. No defense
2. Image-conditioned generator
3. Static template made only from generator-training positions
4. Broad frequency suppression baseline, if available

Report each held-out position separately and include benign corruptions again.

#### Phase E: Final interpretation

Success would require both:

- Held-out positions retain high undefended ASR.
- The generator substantially lowers that ASR without unacceptable clean or
  true-label accuracy loss.

Only then can the work claim generalization to unseen active frequency
locations.

## 19. Subsequent research sequence

After the active-position experiment, continue in this order:

1. **Leave-one-trigger-family-out evaluation:** train with several active
   spectral families and hold out one family from generator training.
2. **Trigger-to-noise ratio study:** compare generator response as trigger and
   benign noise magnitudes vary.
3. **Adaptive attacker study:** optimize or search for trigger configurations
   that maintain classifier ASR while avoiding generator correction.
4. **Cross-dataset evaluation:** repeat the final fixed protocol on another
   texture or domain dataset.
5. **Locked DTD test evaluation:** run once after architecture, thresholds and
   checkpoints are finalized.

---

## 20. Reproducibility files

### Code and notebook

- `scripts/evaluate_dtd_generator_generalization.py`
- `notebooks/kaggle_dtd_reference_free_generalization.md`
- `tests/test_dtd_generator_generalization.py`

### Main output directory

- `Advanced_Outputs/dtd_reference_free_generalization_v1/`

### Aggregate results

- `generalization_summary.json`
- `trigger_generalization.csv`
- `corruption_controls.csv`
- `trigger_generalization.png`
- `corruption_controls.png`
- `README.md`

### Per-scenario evidence

Every one of the 19 scenario directories contains:

- `summary.json`
- `predictions.jsonl`
- `panel.png`

The experiment produced 21 PNG files, 20 JSON files, 19 JSONL prediction files,
two CSV tables and one generated Markdown summary.

---

## 21. Final conclusion

This experiment moves the project beyond demonstrating correction of one exact
trigger. The frozen generator showed convincing robustness to unseen FTrojan
strengths and a very small response to benign corruptions. Those are meaningful
positive findings.

At the same time, the shifted-position experiment showed that current behaviour
is partly tied to learned DCT locations. Rather than weakening the study, this
result defines the next scientifically necessary experiment: make several
positions actively backdoored, hide some from generator training, and test
whether the reference-free correction mechanism discovers them.

The project can therefore be summarized at this stage as:

```text
Strength generalization: demonstrated
Benign selectivity: demonstrated
Correction-map causal relevance: demonstrated by the prior faithfulness audit
Unknown active-position generalization: not yet demonstrated
Universal unknown-trigger defense: future work
```
