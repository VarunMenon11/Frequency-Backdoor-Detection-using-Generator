# Panel Explanation and Viva Preparation Notes

## Purpose

This guide explains what you should personally understand before presenting the project. The panel may ask questions in simple language, mathematical language, or research-method language. The safest approach is to explain the same idea at three levels: intuition, implementation, and limitation.

---

## 1. The project in one minute

An image classifier can be secretly trained with poisoned data so that it behaves normally on clean images but predicts an attacker-selected target when a trigger is present. This hidden behaviour is a backdoor.

Our project studies frequency-related triggers. We first create a controlled suspicious classifier so that the trigger’s effect can be measured. We then transform clean and triggered images with FFT, compare their amplitude spectra, and give the spectral evidence to a generator. The generator predicts a selective correction map. That map moves suspicious amplitude components toward the clean amplitude while the phase is preserved for reconstruction. The corrected image is then used, together with raw triggered images and clean images, to repair the classifier.

The main completed results show high suspicious ASR before defense and very low repaired ASR after defense across CIFAR-100 32x32, Tiny ImageNet 64x64, and STL-10 96x96. The current limitation is that generator training uses a clean/triggered pair. The input-only unknown-image version is future work.

---

## 2. Explain the terms in plain language

### Image classifier

A neural network that receives an image and predicts one label from a set of classes.

### Backdoor

A hidden rule learned by the model during training. The model behaves normally on ordinary images but changes its prediction when a special trigger appears.

### Trigger

The pattern, signal, or frequency modification that activates the hidden rule.

### Poisoned image

An image modified with the trigger and assigned a target label during training.

### Target class

The class chosen by the attacker. In our main experiments, class 0 is used for each dataset: apple for CIFAR-100, goldfish for Tiny ImageNet, and airplane for STL-10.

### Suspicious classifier

The intentionally backdoored classifier before defense. It is necessary for verifying that the attack was successfully learned.

### Generator

A convolutional neural network that predicts a spectral correction map. It is not a GAN because the current implementation has no discriminator.

### Model repair

Fine-tuning a copy of the suspicious classifier with triggered images assigned their original clean labels, teaching the model to ignore the trigger.

### Clean accuracy

The percentage of ordinary, non-triggered test images classified correctly.

### Attack success rate

The percentage of non-target images that are incorrectly classified as the attacker target after the trigger is applied.

---

## 3. Why do we need a suspicious classifier first?

Because a low ASR after defense is meaningful only if the model originally had a high ASR. Suppose a suspicious model has an ASR of 2% before defense. If the defense produces an ASR of 1%, that does not prove much; the attack may never have been learned.

Our evaluation therefore follows this order:

```text
train suspicious model
        |
verify high initial ASR
        |
apply generator correction
        |
repair classifier
        |
measure final ASR and clean accuracy
```

This is why the results table includes suspicious ASR, generator-corrected ASR, and repaired ASR separately.

---

## 4. Why is only 12% of the data poisoned?

The poison ratio describes the fraction of eligible training samples modified by the attack. At 0.12, approximately 12% are triggered and relabeled; the rest remain clean.

It is not a universal scientifically proven default. It is a controlled experimental choice that creates a sufficiently strong attack while retaining a meaningful clean-data task. A complete paper should include poison-ratio ablations such as 1%, 5%, 8%, 12%, and 20% if computationally feasible.

The important point is that the defense claim should not depend on one unexplained number. We should report why the ratio was selected and show how performance changes when the ratio changes.

---

## 5. What is alpha?

Alpha is the trigger-strength or mixing-strength parameter, depending on the trigger type.

For the sinusoidal image-space trigger:

$$
x_t=\operatorname{clip}(x+\alpha T,0,1)
$$

Increasing alpha increases the magnitude of the added sinusoidal signal. It may improve attack success but may also make the image change more detectable or harm clean classification.

For FIBA-style amplitude injection:

$$
A_p=(1-\alpha)A_c+\alpha A_r
$$

Alpha controls how much reference amplitude is blended into the clean amplitude inside the mask. The meaning is related but not numerically interchangeable with sinusoidal alpha.

The main sinusoidal baseline uses alpha 0.08. The STL-10 strength ablation evaluates 0.03, 0.15, and 0.25. The FIBA calibration evaluates separate alpha values because it is a different attack construction.

---

## 6. Why use FFT?

Images can be represented in more than one way. Pixel space describes the colour or intensity at each location. The frequency domain describes how quickly the image changes across space.

- Low frequencies generally represent broad, slowly changing structures.
- High frequencies generally represent rapid changes such as fine texture, sharp edges, and repeated patterns.
- Natural images use both low and high frequencies.
- A trigger may add a structured change at particular frequency locations.

The FFT gives a complex spectrum:

$$
F(x)=A(x)e^{jP(x)}
$$

The amplitude `A` tells us the strength of frequency components. The phase `P` helps preserve spatial arrangement. We use the amplitude for selective correction and preserve phase during reconstruction.

---

## 7. Is the amplitude difference the correction map?

No.

The amplitude difference is:

$$
D=|A_t-A_c|
$$

It is an observation of how the triggered spectrum differs from the clean spectrum. It is given to the generator as evidence.

The generator produces a separate learned map:

$$
M=G(A_c,A_t,D)
$$

The map is trained using classification, reconstruction, sparsity, and smoothness objectives. Therefore, the difference map shows where change occurred, while the correction map represents what the generator decides to correct.

---

## 8. Why preserve phase?

If we change both amplitude and phase without constraints, the reconstructed image may lose spatial structure or semantic arrangement. The project therefore uses the corrected amplitude with the triggered phase:

$$
x_{corr}=\operatorname{IFFT}(A_{corr}e^{jP_t})
$$

This is not a claim that phase is always more important than amplitude in every task. It is a design choice intended to preserve spatial arrangement while selectively modifying the amplitude evidence associated with the trigger.

---

## 9. What does the correction map do mathematically?

The correction rule is:

$$
A_{corr}=A_t-M\odot(A_t-A_c)
$$

At one frequency location:

- `M = 0`: retain the triggered amplitude.
- `M = 1`: move fully to the clean amplitude.
- `0 < M < 1`: partially move toward the clean amplitude.

This lets the generator make a different correction at different frequency locations. That is the meaning of selective spectral correction.

---

## 10. How are the generator losses understood?

The generator loss is:

$$
L_G=1.0L_{cls}+4.0L_{rec}+0.02L_{sp}+0.01L_{sm}
$$

### Classification loss

The corrected image should be classified as its original clean label by the suspicious classifier.

### Reconstruction loss

The corrected image should remain close to the clean image, discouraging visible or destructive changes.

### Sparsity loss

The generator should avoid changing the entire spectrum. It should prefer a smaller correction when a smaller correction is sufficient.

### Smoothness loss

The correction map should not be filled with unstable isolated noise. Smoothness encourages a more coherent map.

---

## 11. Why are there both generator correction and model repair?

They solve different problems.

### Generator correction

This changes the input image before classification. It is useful for showing that the suspicious spectral dependency can be weakened at the image level.

### Model repair

This changes the classifier parameters. It teaches the model using triggered images with their original labels so that the trigger is no longer treated as a valid reason for the target prediction.

If only corrected images are used for repair, the classifier may remain sensitive to raw triggered images. That is why the repair set includes clean, corrected-triggered, and raw-triggered samples.

---

## 12. How should the main results be explained?

### CIFAR-100, 32x32

The initial suspicious ASR is 98.54%. Generator correction reduces it to 0.37%, and model repair reduces it to 0.06%. This is the initial low-resolution proof of concept.

### Tiny ImageNet, 64x64

The dataset has 200 classes, making the classification problem more difficult. The suspicious ASR is 99.99%, generator ASR is 0.37%, and repaired ASR is 0.09%. This tests the method at an intermediate resolution and larger label space.

### STL-10, 96x96

The suspicious ASR is 100%, generator ASR is 3.15%, and repaired ASR is 0.71%. This is the main higher-resolution validation and the dataset used for strength, trigger-function, localized, and FIBA-style studies.

### Overall interpretation

The results support the controlled hypothesis across the tested resolutions. They do not prove universal unknown-trigger defense. The initial attack is deliberately created and the clean/triggered pair is available during generator training.

---

## 13. How should FIBA be explained?

FIBA means Frequency-Injection Based Backdoor Attack. It is an attack method, not the proposed defense.

Earlier sinusoidal experiments add a signal in image space and then use FFT to analyze it. FIBA-style injection modifies the amplitude spectrum directly, blends it with reference-image amplitude inside a mask, preserves phase, and reconstructs the poisoned image.

The calibration result is strong: alpha 0.50 and radius 0.10 reach 94.13% suspicious ASR with 49.56% clean accuracy. However, the stored full defense run used alpha 0.30 and radius 0.15 and reached only 45.67% initial ASR. Therefore:

- FIBA calibration: successful.
- Stored full FIBA defense run: preliminary.
- Final calibrated full FIBA result: pending rerun.

Never present the preliminary run as a definitive final FIBA defense result.

---

## 14. What are the main limitations?

### Paired clean/triggered assumption

The current generator receives clean and triggered spectral information during training. In a real deployment, only the suspicious image may be available.

### Known trigger family

The completed experiments deliberately construct the trigger. They do not prove defense against arbitrary unknown trigger designs.

### Limited repeated-seed analysis

The main records use seed 42. A stronger paper should repeat key experiments with multiple seeds and report uncertainty.

### Baseline comparison

The final paper should compare against global frequency filtering, model-repair baselines, and the closest recent frequency-domain defenses.

### FIBA calibration

The full calibrated FIBA run is still required.

---

## 15. What is the future input-only version?

The future version should not receive `A_clean` when an unknown triggered image arrives. Possible designs include:

1. `M = G(A_triggered)`, where the generator predicts a correction from the suspicious spectrum alone.
2. A learned clean-spectrum prior that estimates `A_clean` from the suspicious image.
3. An iterative correction loop constrained by classifier confidence, reconstruction distance, and spectral naturalness.
4. Training on multiple trigger families while withholding the clean amplitude from the generator input.

This should be a separate experiment with a separate claim. It is the key step toward the original goal of handling an unknown image.

---

## 16. Likely panel questions and answers

### “Why not simply remove all high frequencies?”

Because natural edges and textures can also occupy high-frequency regions. A global filter may reduce the trigger but damage useful semantic information. The project therefore learns a selective correction map.

### “Why is the trigger not clearly visible?”

The perturbation is intentionally small and distributed. The normal RGB image may look almost unchanged. Amplified residual images and amplitude-difference maps make the effect easier to inspect.

### “Why does the correction map sometimes look similar across images?”

When the same fixed frequency trigger is used, the trigger-related spectral location is expected to recur. The generator may therefore focus on similar coordinates, while the correction magnitude still depends on image content.

### “Why are the spectral images bright in the centre?”

FFT visualizations are commonly shifted so the zero frequency is displayed at the centre. The spectrum is also often log-scaled. Display brightness is not the same as direct raw amplitude.

### “Does the generator detect an unknown trigger?”

Not yet in the completed experiments. The current generator uses controlled clean/triggered pairs. Input-only unknown-image correction is planned future work.

### “Is a low ASR enough to prove success?”

No. The suspicious classifier must first have a high ASR. Otherwise the attack may not have been learned. Clean accuracy and reconstruction quality must also be reported.

### “Is alpha 0.08 scientifically fixed?”

No. It is the main controlled baseline. Strength ablations test other values. The correct value depends on dataset, resolution, trigger type, and the desired attack visibility.

### “Why use different frequency values for different resolutions?”

The values are scaled relative to the image grid: `(6,6)` for 32x32, `(12,12)` for 64x64, and `(18,18)` for 96x96. This keeps a comparable relative location. They are experimental choices, not universal optima.

### “Is this a GAN?”

No. The generator is a task-guided convolutional network. It predicts a spectral correction map and is trained with classification, reconstruction, sparsity, and smoothness losses. There is no discriminator.

### “What is the main novelty?”

The proposed combination is selective, image-conditioned amplitude correction from clean/triggered spectral evidence, phase-preserving reconstruction, and model repair. The final paper must compare this combination against close frequency-domain defenses and avoid an unsupported first-ever claim.

### “What would make the work stronger?”

Complete the calibrated FIBA rerun, add multiple seeds, compare against strong baselines, test unseen trigger families, and develop the input-only generator that does not require a clean counterpart.

---

## 17. Final preparation checklist

Before the presentation:

- Replace the domain-expertise course placeholders with your real course and progress.
- Confirm the three technical-enrichment activities with your guide.
- Update the Gantt chart to the actual Semester 3 dates.
- Add your full name, guide name, registration number, and department.
- Use readable versions of the architecture and qualitative panels.
- Label the FIBA figure as preliminary.
- Verify all numerical values against the JSON summaries.
- Prepare one sentence explaining every term on the result table.
- Prepare the limitations slide honestly.
- Keep the full references in an appendix if the final slide is crowded.

