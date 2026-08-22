# Visual Panel Guide

## Meaning of Every Image and Label

This document explains every panel label used in the project figures. It is intended for dissertation readers, presentations, viva questions, and anyone who needs to understand exactly what each image represents.

The panels compare the image and prediction before poisoning, after poisoning, after generator correction, and after classifier repair. They also show the frequency-domain evidence used by the proposed method.

## 1. How to read a complete panel

A typical panel follows this order:

```text
Original image
      |
      v
Add trigger
      |
      v
Suspicious classifier prediction
      |
      v
Generator spectral correction
      |
      v
Corrected image and spectrum
      |
      v
Repaired classifier prediction
```

The RGB images answer:

> What happened to the visible image and the classifier prediction?

The spectral panels answer:

> What changed in the frequency representation, and where did the generator apply its correction?

The labels usually contain both a model name and a prediction. For example, `susp trig: airplane` means that the suspicious classifier predicted airplane for the triggered image.

Representative panel:

![Representative CIFAR-100 correction panel](../outputs/final_cifar100_evaluation/sample_panels/final_panel_test_index_0.png)

The exact arrangement can vary slightly between dataset scripts, but the meanings of the labels remain the same.

## 2. `clean true`

### What it shows

`clean true` is the original, unmodified image together with its ground-truth class label.

For example:

```text
clean true: horse
```

means that the original dataset says the image belongs to the horse class.

### Why it is included

This is the reference image for the entire panel. It shows the natural visual content before a trigger is introduced. It also provides the correct label against which all later predictions can be compared.

### How to interpret it

The clean image is not necessarily guaranteed to be classified correctly by the suspicious classifier. A model can make ordinary clean-image mistakes. That is why `clean true` is the ground-truth reference, while the prediction labels must be read separately.

## 3. `susp clean`

### What it shows

`susp clean` is the suspicious classifier's prediction on the original clean image.

Example:

```text
susp clean: horse
```

means that the backdoored classifier predicted horse before any trigger was added.

### Why it is included

It measures the model's normal clean-image behaviour before the attack is activated. This is necessary because a backdoored model may still retain ordinary classification ability.

### Possible outcomes

If `clean true` is horse and `susp clean` is horse, the classifier correctly recognizes the clean image.

If `clean true` is horse and `susp clean` is dog, the suspicious classifier made a normal clean classification error. This does not automatically mean that the trigger or defense failed.

## 4. `susp trig`

### What it shows

`susp trig` is the suspicious classifier's prediction after the trigger has been added to the clean image.

The underlying RGB image is the triggered image, although some panels may label the image itself as `triggered` or `trig`.

Example:

```text
clean true: horse
susp trig: airplane
```

means that a horse image was redirected to the attacker target airplane after the trigger was added.

### Why it is important

This is the visual, sample-level demonstration of the backdoor. It shows that the suspicious classifier reacts to the trigger rather than only to the image's semantic content.

### Relation to ASR

One panel is only one sample. The aggregate Attack Success Rate is calculated over all non-target triggered test images. If `susp trig` equals the target for many non-target images, the suspicious ASR becomes high.

For a non-target image:

```text
susp trig == target class  -> successful attack on that sample
susp trig != target class  -> unsuccessful attack on that sample
```

## 5. `susp corr`

### What it shows

`susp corr` is the suspicious classifier's prediction on the generator-corrected image.

The image has been passed through:

```text
triggered image
    -> FFT
    -> generator correction map
    -> corrected amplitude
    -> preserve triggered phase
    -> inverse FFT
    -> corrected image
```

Example:

```text
clean true: horse
susp trig: airplane
susp corr: horse
```

This is the ideal sample-level result: the trigger caused the backdoor prediction, and the generator correction moved the prediction back to the original class.

### Why it is included

It measures the generator as an image-level defense while keeping the suspicious classifier unchanged. This isolates the question:

> Can the generated spectral antidote suppress the trigger before changing the classifier?

### Important distinction

`susp corr` does not mean that the classifier has been repaired. The classifier is still the original suspicious model. Only the input image has been corrected.

## 6. `repair clean`

### What it shows

`repair clean` is the repaired classifier's prediction on the original clean image.

Example:

```text
repair clean: horse
```

means that the final repaired model predicted horse for the clean image.

### Why it is included

The defense must preserve ordinary classification. A model that removes the backdoor but loses most clean accuracy is not a useful defense.

This panel label can be compared with `clean true` to see whether the repaired classifier recognizes the original image correctly.

## 7. `repair trig`

### What it shows

`repair trig` is the repaired classifier's prediction on the raw triggered image. No generator correction is applied to this input in this panel.

Example:

```text
clean true: horse
repair trig: horse
```

means that the repaired model no longer treats the raw trigger as a reason to predict the attacker target.

### Why it is one of the strongest panels

This result tests whether the classifier itself has been repaired. The model must reject the trigger even when the raw triggered image is supplied directly.

The repaired classifier was trained using raw triggered images with their original clean labels for exactly this reason.

## 8. `repair corr`

### What it shows

`repair corr` is the repaired classifier's prediction on the generator-corrected image.

This is the combined defense path:

```text
triggered image
    -> generator correction
    -> corrected image
    -> repaired classifier
    -> prediction
```

### Why it is included

It checks whether the corrected image remains compatible with the final repaired classifier. Ideally, `repair corr` agrees with the original clean label.

### Difference from `susp corr`

| Label | Image | Classifier |
|---|---|---|
| `susp corr` | Corrected image | Suspicious classifier |
| `repair corr` | Corrected image | Repaired classifier |

The first measures the generator alone. The second measures the generator and model repair together.

## 9. `correction map`

### What it shows

The correction map is the generator output:

```text
M = G(A_clean, A_triggered, abs(A_triggered - A_clean))
```

It is displayed as a grayscale or heatmap image. Bright regions indicate locations where the generator predicts stronger correction after visualization normalization. Dark regions indicate weaker correction.

### What its coordinates mean

The map is not an ordinary spatial image. Its horizontal and vertical coordinates correspond to locations on the frequency grid. A bright point near the centre after `fftshift` represents a low or mid spatial frequency. A bright point farther from the centre represents a higher spatial frequency.

### Important coordinate-order detail

The generator internally uses the native, unshifted FFT order. In that order, the zero-frequency component is at the corners, and the frequency region that appears in the centre of a shifted spectrum is split across the edges. The visual spectrum panels use `fftshift` so that the zero-frequency component appears in the centre.

Therefore, a correction map must also be shifted before it is displayed. The visualization code now applies this display-only operation. This does not change the generator output, corrected amplitude, reconstructed image, or any metric. It only makes the correction map use the same coordinate layout as `trigger amp`, `corrected amp`, and `amplitude diff`.

Some older panels were generated before this display alignment was added. In those panels, a bright correction region at the border can represent a correction around the centre-frequency region after wrapping, not a correction that is genuinely far from the trigger. Those older panels should not be used to argue that the generator corrected an unrelated edge frequency. Regenerate the panel with the updated script for a visually aligned figure.

### What it does not mean

The correction map is not:

- a segmentation mask of the object;
- a pixel-space transparency mask;
- the final corrected RGB image;
- the same thing as the amplitude difference.

It is a frequency-domain control map.

### Difference from amplitude difference

`amplitude diff` measures what changed between clean and triggered amplitudes. `correction map` is the generator's learned decision about how much of that change to remove.

```text
amplitude diff = evidence of spectral change
correction map = learned correction decision
```

## 10. `trigger amp`

### What it shows

`trigger amp` is the amplitude spectrum of the triggered image:

```text
A_triggered = abs(FFT(x_triggered))
```

It is usually displayed using a logarithmic transform and a shifted frequency layout so that weak spectral details are visible.

### How to read it

The centre generally contains low-frequency or average-intensity information after `fftshift`. Areas farther from the centre correspond to higher spatial frequencies.

Bright regions indicate strong amplitude energy. They may come from natural image structure, edges, textures, or the artificial trigger.

### Important caution

The brightest region is not automatically the trigger. Natural image energy can be much stronger than the trigger. The trigger is identified more reliably by comparing clean and triggered spectra, especially through `amplitude diff`.

## 11. `corrected amp`

### What it shows

`corrected amp` is the amplitude spectrum after applying the generator correction:

```text
A_corrected = A_triggered - M * (A_triggered - A_clean)
```

The corrected amplitude is then combined with the triggered phase and passed through the inverse FFT.

### What to look for

The corrected spectrum should normally remain visually similar to the triggered spectrum because the defense is designed to make a small selective change. The important evidence is whether suspicious differences are reduced without flattening the entire spectrum.

### Why it may look almost identical

Amplitude displays are often logarithmic and independently normalized. A small but important local change may not be obvious when two full spectra are viewed side by side. The `amplitude diff` and `correction map` are better for locating the change.

## 12. `amplitude diff`

### What it shows

`amplitude diff` is the absolute difference between triggered and clean amplitude spectra:

```text
amplitude diff = abs(A_triggered - A_clean)
```

### Why it is useful

This panel shows where the trigger changed the amplitude spectrum. It is the most direct visual evidence for the spectral anomaly used by the generator.

### How to interpret brightness

Bright areas indicate larger amplitude differences relative to the displayed range. Dark areas indicate smaller differences. Because the panel may use min-max normalization, brightness is relative to the current panel and should not be interpreted as an absolute physical unit unless the display scale is shared.

### Difference from `correction map`

The amplitude difference is calculated directly from clean and triggered images. The correction map is generated by a neural network after seeing the amplitude information. They may look similar, but they answer different questions.

## 13. `image diff x8`

### What it shows

`image diff x8` is the difference between the corrected image and the clean image, amplified by a factor of eight:

```text
image_diff_x8 = 8 * (x_corrected - x_clean)
```

The actual correction is not necessarily eight times larger. The visualization multiplies the residual only to make a small change visible.

### Why the multiplier is needed

The generator is encouraged to preserve the original image through reconstruction and sparsity losses. Therefore, a successful correction may be almost invisible in the ordinary RGB image. Without amplification, the correction residual could appear blank.

### How to interpret it

This panel shows where the generator changed the image in pixel space. It does not show the corrected image itself. It is a diagnostic residual.

Large bright or dark regions do not automatically mean failure. The residual must be considered together with:

- reconstruction L1;
- corrected-image classification;
- ASR reduction;
- visual preservation of the object.

## 14. `trigger diff x8`

### What it shows

`trigger diff x8` is the difference between the triggered image and the clean image, amplified by eight:

```text
trigger_diff_x8 = 8 * (x_triggered - x_clean)
```

### Why it is included

It reveals the trigger perturbation in pixel space. A sinusoidal or frequency-domain trigger may look subtle in the ordinary RGB image, so amplification makes its residual visible.

### Difference from `image diff x8`

| Panel | Difference being shown | Purpose |
|---|---|---|
| `trigger diff x8` | Triggered image minus clean image | Shows what the attack added. |
| `image diff x8` | Corrected image minus clean image | Shows what the defense changed. |

Comparing them helps answer whether the defense removed the trigger with a smaller or differently located change.

## 15. Recommended order for explaining a panel

Use the following sequence in a presentation:

1. Start with `clean true` and identify the original object and ground-truth class.
2. Compare `susp clean` with `clean true` to check ordinary clean classification.
3. Show `susp trig` and explain whether the trigger redirects the image to the target.
4. Show `trigger diff x8` to make the injected perturbation visible.
5. Show `trigger amp` and `amplitude diff` to explain the frequency-domain change.
6. Show the `correction map` and explain that it is a learned frequency-domain decision, not an RGB mask.
7. Show `corrected amp` and the corrected RGB image.
8. Use `susp corr` to show the generator's effect while the suspicious model remains frozen.
9. Compare `repair clean`, `repair trig`, and `repair corr` to show the final model-level repair.
10. Use `image diff x8` to show that correction was applied while remaining small.

## 16. Ideal successful example

An ideal sample may look like this:

```text
clean true: horse
susp clean: horse
susp trig: airplane
susp corr: horse
repair clean: horse
repair trig: horse
repair corr: horse
```

The interpretation is:

1. The original image is a horse.
2. The suspicious classifier recognizes the clean image.
3. The trigger redirects the suspicious classifier to airplane.
4. The generator correction returns the suspicious classifier prediction to horse.
5. The repaired model preserves the clean prediction.
6. The repaired model rejects the raw trigger.
7. The repaired model also handles the corrected image correctly.

## 17. Imperfect example

Not every panel must have the ideal pattern. For example:

```text
clean true: dog
susp clean: bird
susp trig: bird
susp corr: bird
repair clean: dog
repair trig: dog
repair corr: dog
```

This does not demonstrate a successful sample-level attack because the suspicious classifier never predicted the target after triggering. However, it may still be a normal clean-classification error, and it can still show that the repaired model handles the image better. Aggregate metrics are more important than selecting only visually convenient examples.

## 18. Why the RGB image can look unchanged

The trigger and correction are often small by design. The generator uses reconstruction, sparsity, and smoothness losses to avoid destroying natural content. A visually unchanged RGB panel can therefore be a positive result: it suggests that the model's prediction changed because of a small targeted spectral correction rather than a large visible alteration.

For this reason, a paper figure should preferably show both:

- the normal RGB images for semantic preservation;
- the amplified residual and spectral maps for scientific inspection.

## 19. What can and cannot be concluded from one panel

One panel can demonstrate the mechanism clearly, but it cannot establish the overall ASR or clean accuracy. Aggregate JSON summaries and tables are required for quantitative claims.

One panel also cannot prove that every trigger was detected. It is a qualitative example selected from a larger evaluation set. The correct research practice is to show representative successful examples and report the full-set metrics alongside them.

## 20. Short explanation for a guide or examiner

```text
The clean true image is the original image and ground-truth class. Susp clean
shows the suspicious model's normal prediction. Susp trig shows what happens
after the trigger is added and demonstrates the backdoor response. Susp corr
shows the same suspicious model after the generator corrects the amplitude
spectrum. Repair clean, repair trig, and repair corr show the predictions of the
fine-tuned repaired model on clean, raw-triggered, and corrected inputs.

The trigger amp is the triggered image's Fourier amplitude spectrum. Corrected
amp is the spectrum after selective correction. Amplitude diff shows where the
trigger changed the spectrum, while the correction map shows what the generator
decided to correct. Trigger diff x8 displays the attack residual amplified for
visibility, and image diff x8 displays the defense residual amplified for
visibility.
```
