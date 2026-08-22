# Experiment Record: FIBA-Style Amplitude Injection

## 1. Why this experiment was added

The earlier triggers were created in image space and then inspected in the Fourier domain. The FIBA-style experiment tests the project from the opposite direction: the trigger is created directly by modifying the Fourier amplitude spectrum.

This is important because the proposed defense focuses on amplitude correction. A successful FIBA-style experiment therefore tests whether the method can respond to a trigger whose construction is naturally expressed in the same domain as the defense.

## 2. What FIBA means

FIBA refers to **Frequency-Injection based Backdoor Attack**. The published method injects a backdoor through frequency information rather than using an obvious spatial patch. The project implements a FIBA-style benchmark based on the amplitude-injection principle; it is not claimed to reproduce the official implementation exactly.

The relevant published reference is:

- [FIBA: Frequency-Injection Based Backdoor Attack in Medical Image Analysis, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Feng_FIBA_Frequency-Injection_Based_Backdoor_Attack_in_Medical_Image_Analysis_CVPR2022_paper.pdf)

For the sinusoidal signal family used as the earlier comparison, see [SIG: A New Backdoor Attack in Deep Learning](https://arxiv.org/abs/1902.11237). The project uses an adapted controlled sinusoidal formula rather than claiming an exact reproduction of that paper.

For comparison, **SIB is not being used here as a universal official acronym**. In these project notes it is a convenient shorthand for a sinusoidal image-space backdoor. The sinusoidal trigger is related to the published SIG-style sinusoidal signal attack family, but the implementation here uses a controlled adapted formula and should not be called an exact reproduction. FIBA and sinusoidal/SIG-style triggers are different:

| Property | Sinusoidal/SIG-style experiment | FIBA-style experiment |
|---|---|---|
| Where trigger is created | Image space | Fourier amplitude space |
| Main operation | Add a periodic pattern | Blend amplitude with a reference amplitude |
| Phase used for poisoned reconstruction | Obtained after image-space modification | Clean phase is retained by the attack construction |
| Defense relevance | Tests spectral response to a spatial waveform | Directly tests amplitude-side correction |

## 3. FIBA-style attack formula

For a clean image:

$$
F(x)=A_c e^{jP_c}.
$$

For a reference image:

$$
F(r)=A_r e^{jP_r}.
$$

Within an elliptical frequency mask (Q), the poisoned amplitude is blended:

$$
A_p=(1-\alpha)A_c+\alpha A_r.
$$

Outside the mask:

$$
A_p=A_c.
$$

The attack keeps the clean phase:

$$
P_p=P_c.
$$

The poisoned image is reconstructed as:

$$
x_p=\operatorname{IFFT}(A_p e^{jP_c}).
$$

The parameter alpha controls how strongly the reference amplitude is injected. The mask radius controls how large a region of the centered amplitude spectrum is changed. Increasing either one generally makes the attack stronger, but it may also harm clean classification or make the attack less subtle.

## 4. Calibration before full defense training

The calibration stage trained only the suspicious classifier. This isolates the attack question: which alpha and mask radius actually create a strong backdoor?

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

The balanced selected setting was alpha 0.50 and radius 0.10 because it crossed the 90% ASR threshold while retaining better clean accuracy than radius 0.15. The calibration result is successful: it identified a strong FIBA-style attack setting.

## 5. Stored full-run result

The currently stored full FIBA run used alpha 0.30 and radius 0.15, which was an older setting and not the selected final calibration setting. Its results are:

| Stage | Clean accuracy | ASR |
|---|---:|---:|
| Suspicious classifier | 57.63% | 45.67% |
| Generator-corrected suspicious classifier | 58.74% | 3.00% |
| Repaired classifier | 73.98% | 1.29% |
| Generator-corrected repaired classifier | 73.85% | 1.18% |

Additional values were reconstruction L1 `0.00911` and mean correction value `0.05070`.

## 6. Correct interpretation

The stored full run demonstrates that the defense can reduce the response to a FIBA-style amplitude injection. However, the suspicious classifier learned only 45.67% ASR. Therefore, it is not a final strong-attack defense result. The low post-defense ASR is encouraging, but it must be interpreted alongside the modest pre-defense ASR.

The accurate paper wording is:

> A preliminary FIBA-style amplitude-injection experiment achieved 45.67% ASR before defense. Generator correction reduced ASR to 3.00%, while classifier repair reduced raw-trigger ASR to 1.29%. Because the attack was only partially learned, this run was treated as preliminary.

The calibration table should be reported as the successful attack-selection experiment. A final full FIBA run should use alpha 0.50 and radius 0.10, then repeat generator training, repair, metrics, and panels. Until that rerun exists, the FIBA section should not claim a final calibrated end-to-end result.

## 7. FIBA pipeline diagram

```text
Clean image x                 Reference image r
      |                              |
      +---------- FFT ---------------+
                    |
          clean amplitude A_c
          reference amplitude A_r
                    |
          blend inside mask Q
                    |
        A_p=(1-alpha)A_c+alpha A_r
                    |
             preserve P_c
                    |
                 IFFT
                    |
          FIBA-style triggered image
                    |
             train suspicious model
                    |
        A_clean, A_triggered, P_triggered
                    |
          proposed spectral generator
                    |
           selective correction map
                    |
             repaired classifier
```

## 8. Visual evidence

The stored preliminary panel is:

[Open the preliminary FIBA panel](../outputs/outputs_FIBA_Cali/stl10_96x96_fiba_amplitude_calibrated/sample_panels/stl10_96_panel_test_index_0.png)

The same panel can be rendered directly in a Markdown viewer:

![FIBA-style amplitude-injection correction panel](../outputs/outputs_FIBA_Cali/stl10_96x96_fiba_amplitude_calibrated/sample_panels/stl10_96_panel_test_index_0.png)

The panel should be explained as follows:

- `clean`: original STL-10 image;
- `triggered`: image after amplitude injection;
- `trigger amp`: logarithmic amplitude spectrum of the injected image;
- `amplitude diff`: absolute difference from the clean amplitude;
- `correction map`: generator output, where brighter values indicate stronger correction;
- `corrected amp`: amplitude after the generator correction;
- `trigger diff x8` and `image diff x8`: residuals amplified eight times for visibility.

The original RGB image may show little difference because FIBA is designed to inject frequency information without an obvious patch. The spectral and amplified residual panels are therefore essential evidence.

### Note about the stored panel

The stored panel above was generated before the correction-map display coordinate fix. Its attack and defense predictions and numerical metrics remain valid, but the correction map is in unshifted FFT layout while the amplitude panels are centered for display. For a final paper figure, regenerate the FIBA panel with the updated `scripts/run_stl10_96_frequency_experiment.py`; the updated panel will display the correction map in the same centered layout as the amplitude-difference panel.

## 9. What FIBA does and does not prove

FIBA calibration and the preliminary full run support the compatibility between amplitude-side frequency injection and the proposed amplitude correction idea. They do not prove universal robustness to every frequency attack, every reference image, every mask shape, or every medical and satellite domain.

The remaining controlled limitation is that the current generator still receives the clean and triggered amplitude pair during training/evaluation. A blind FIBA defense would require the input-only or clean-spectrum-prior generator described in the complete methodology document.

## 10. Reproducibility files

- FIBA calibration notebook: `../notebooks/kaggle_stl10_96x96_fiba_calibration.md`
- FIBA amplitude notebook: `../notebooks/kaggle_stl10_96x96_fiba_amplitude.md`
- Calibration outputs: `../outputs/outputs_FIBA_Cali/fiba_calibration/`
- Preliminary full-run outputs: `../outputs/outputs_FIBA_Cali/stl10_96x96_fiba_amplitude_calibrated/`
- Existing detailed record: `../docs/fiba_experiment_record.md`

