# Literature Comparison, Recent Work, and Positioning

## Search conclusion

The earlier comparison should not be used to claim numerical superiority. It
compared results obtained with different datasets, attacks, models, and repair
protocols. In particular, our final experiment uses DTD with an FTrojan-style
block-DCT frequency trigger, while the earlier Freq-Pret table uses CIFAR-100
and Tiny-ImageNet with SIG, ISSBA, and Blended attacks.

The visibility description also needs care. These attacks are not all ordinary
visible patch triggers: ISSBA is designed as an invisible sample-specific
attack, SIG is a global signal pattern, and Blended mixes a trigger pattern
through the image. The decisive mismatch is that none is the same DTD
FTrojan-style DCT attack used in our final experiment.

In the targeted search completed on **10 October 2026**, I did not identify a
published article with the complete combination of **DTD + an FTrojan-style
frequency backdoor + a reference-free spectral correction defense**. This is a
search result, not proof that no such work exists. Therefore, the paper must not
claim that this project is "the first" without a formal systematic review.

Several papers are still useful because each matches an important part of the
project:

1. **SampDetox** is the closest quantitative defense comparison because it
   evaluates backdoor purification on the same DTD dataset. It uses BadNets and
   WaNet rather than an FTrojan-style frequency trigger. It is a NeurIPS 2024
   conference paper, but it is the strongest same-dataset result found.
2. **IFAP** is an IEEE Access journal article that evaluates frequency-shaped
   adversarial perturbations on DTD. It is not a backdoor defense, so its ASR
   cannot be compared with ours. It is useful for explaining how visual,
   structural, and spectral similarity can be measured.

This means the literature can provide **same-dataset context**, while the fair
performance claim must come from methods evaluated inside our own locked
protocol.

### Safe novelty wording

Use this wording in the paper or presentation:

> To the best of our knowledge, and based on a targeted literature search
> completed on 10 October 2026, prior studies have examined frequency-domain
> backdoor attacks, frequency-based defenses, and backdoor defense on DTD
> separately. We did not identify prior work evaluating an image-conditioned,
> reference-free selective spectral correction generator against an
> FTrojan-style block-DCT backdoor on DTD. We therefore position the novelty in
> this specific experimental combination and defense design, rather than
> claiming to be the first work on frequency-domain backdoors.

This wording is appropriately narrow. It can be revised if another exact-match
paper is later found.

---

## Recent journal papers from 2025 and 2026

The recent literature confirms that frequency-domain backdoors are an active
research area. The following papers are more recent than the original FTrojan
work and should be cited in the paper.

| Paper | Year and venue | Dataset(s) relevant here | Trigger or defense idea | Exact match to our DTD experiment? | How it should be used |
|---|---|---|---|---|---|
| **WPDA: Frequency-Based Backdoor Attack with Wavelet Packet Decomposition** | Online 2025; *Neural Networks*, vol. 194, 2026 | CIFAR-10, **CIFAR-100**, **Tiny ImageNet** | Uses wavelet packet decomposition to locate important frequency regions and inserts the trigger selectively. At a 0.004% poison ratio, the paper reports ASR values of 98.12%, 85.01%, and 80.99% on the three datasets. | **No.** It is a frequency backdoor attack, but it does not use DTD and is not a defense. | Best recent attack-side reference and a strong reason to add CIFAR-100 and Tiny ImageNet experiments. |
| **Backdoor Defense Strategy for Image Classification Tasks Based on Frequency Domain Perturbation (Freq-Pret)** | *IEEE Internet of Things Journal*, 2026 | CIFAR-10, **CIFAR-100**, **Tiny ImageNet** | Preserves phase, perturbs high-frequency amplitude, and lightly retrains the suspicious model so it relies less on trigger-related high-frequency features. | **No.** It uses the desired benchmark datasets but tests BadNets, TrojanNN, WaNet, SIG, ISSBA, and Blended rather than our FTrojan-style trigger. | Closest recent defense-side paper and the best published table structure to reproduce. |
| **CLTBA: Learnable Trigger Backdoor Attack via Frequency Domain Injection** | *Journal on Information Security*, 2026 | GTSRB, CelebA, MNIST | Learns an adaptive spectral mask and trigger jointly instead of using a fixed hand-designed frequency pattern. | **No.** Same broad trigger domain, different datasets and an attack rather than a defense. | Cite as evidence that future attackers may adapt their spectral trigger to the data and model. |
| **A Triple Stealthy Backdoor: Hidden in Spatial, Frequency, and Feature Domains (TriBA)** | *IEEE Transactions on Dependable and Secure Computing*, 2025 | Four image benchmarks; not DTD | Combines wavelet, Fourier, and cosine-domain operations and also constrains feature-space stealth. | **No.** It is a more complex frequency-related attack and does not provide the same DTD defense protocol. | Cite to justify testing beyond one fixed DCT layout. |
| **IFAP: Input-Frequency Adaptive Adversarial Perturbation** | *IEEE Access*, 2025 | **DTD**, GTOS | Creates frequency-shaped adversarial perturbations and evaluates spectral and texture similarity. | **No.** It matches DTD and frequency analysis, but it is an inference-time evasion attack rather than a training-time backdoor. | Same-dataset support for PSNR, SSIM, STSIM, and frequency-similarity evidence. |
| **SampDetox: Black-box Backdoor Defense via Perturbation-based Sample Detoxification** | NeurIPS 2024 | **DTD** | Purifies BadNets and WaNet inputs using diffusion-based recovery and localization. | **No.** Same dataset and defense task, different trigger families. | Closest published same-DTD backdoor-defense result. |
| **An Invisible Black-Box Backdoor Attack Through Frequency Domain (FTrojan)** | ECCV 2022 | CIFAR-10, GTSRB, MNIST, PubFig, ImageNet-16 | Uses block DCT and fixed mid/high-frequency coefficients to create an invisible frequency trigger. | **Trigger-family match only.** It does not use DTD. | Original methodological foundation for calling our trigger "FTrojan-style." |

### Exact-match check

| Requirement | SampDetox | IFAP | WPDA | Freq-Pret | Proposed work |
|---|---:|---:|---:|---:|---:|
| Uses DTD | Yes | Yes | No | No | **Yes** |
| Uses a frequency-domain modification | No | Yes | Yes | Yes, for defense | **Yes** |
| Studies a training-time backdoor | Yes | No | Yes | Yes | **Yes** |
| Corrects one input without its matching clean image at inference | Not our generator design | No | No; attack paper | No; repairs model by retraining | **Yes** |
| Uses our FTrojan-style block-DCT protocol | No | No | No; wavelet packet | No | **Yes** |

The table does not prove priority. It explains why none of these published
numbers is an exact head-to-head result for the present DTD protocol.

### Scopus lookup links

Scopus often requires an institutional login. The links below are DOI-search
shortcuts; if a result page does not open directly, use the displayed DOI in
the [Scopus basic search](https://www.scopus.com/search/form.uri?display=basic).
New 2026 articles can also take time to appear as individual Scopus records.

| Paper | DOI / publisher record | Scopus DOI search |
|---|---|---|
| WPDA | [10.1016/j.neunet.2025.108074](https://doi.org/10.1016/j.neunet.2025.108074) | [Search in Scopus](https://www.scopus.com/results/results.uri?src=s&st1=10.1016%2Fj.neunet.2025.108074) |
| Freq-Pret | [10.1109/JIOT.2026.3671197](https://doi.org/10.1109/JIOT.2026.3671197) | [Search in Scopus](https://www.scopus.com/results/results.uri?src=s&st1=10.1109%2FJIOT.2026.3671197) |
| CLTBA | [10.1186/s13635-026-00246-6](https://doi.org/10.1186/s13635-026-00246-6) | [Search in Scopus](https://www.scopus.com/results/results.uri?src=s&st1=10.1186%2Fs13635-026-00246-6) |
| TriBA | [10.1109/TDSC.2025.3600427](https://doi.org/10.1109/TDSC.2025.3600427) | [Search in Scopus](https://www.scopus.com/results/results.uri?src=s&st1=10.1109%2FTDSC.2025.3600427) |
| IFAP | [10.1109/ACCESS.2025.3648201](https://doi.org/10.1109/ACCESS.2025.3648201) | [Search in Scopus](https://www.scopus.com/results/results.uri?src=s&st1=10.1109%2FACCESS.2025.3648201) |
| FTrojan | [10.1007/978-3-031-19778-9_23](https://doi.org/10.1007/978-3-031-19778-9_23) | [Search in Scopus](https://www.scopus.com/results/results.uri?src=s&st1=10.1007%2F978-3-031-19778-9_23) |

Do not write "Scopus indexed" beside a very recent paper merely because its
journal is covered. Open the result while signed in and confirm that the
individual article record is present.

---

## Recommended direct-comparison extension

Keeping the DTD result is important because it demonstrates the method on a
complex texture dataset. However, adding **CIFAR-100 and Tiny ImageNet** would
make the external comparison much stronger because both WPDA and Freq-Pret use
those datasets.

The recommended experiment is:

1. Keep the completed DTD experiment as the main texture-domain case study.
2. Add CIFAR-100 and Tiny ImageNet using ResNet-18, matching Freq-Pret's base
   architecture.
3. Implement the exact FTrojan trigger and, if feasible, the recent WPDA
   wavelet-packet trigger.
4. Evaluate the same poisoned checkpoints with no defense, the proposed
   generator, a static template, broad suppression, and a Freq-Pret
   reproduction.
5. Use one target class, poison ratio, train/test split, checkpoint-selection
   rule, and metric implementation across every defense.
6. Report clean accuracy (ACC), attack success rate (ASR), robust/true-label
   accuracy (RA), conditional ASR, PSNR, and SSIM.

This produces two useful comparisons:

- **Same attack, same data, different defense:** FTrojan or WPDA is held fixed
  while the proposed generator is compared with Freq-Pret and simple baselines.
- **Same defense, different attack:** the proposed generator is tested on DCT,
  wavelet-packet, and held-out frequency layouts to measure generalization.

Only after this rerun may the paper say that one defense achieved lower ASR or
higher RA than another. Published values from different protocols should remain
literature context, not a direct ranking.

### Freq-Pret values to retain as published targets

The following values come from the supplied Freq-Pret paper. They are useful
for planning and slide context, but they are not results from our code.

| Dataset | Setting | Mean ACC | Mean ASR | Mean RA |
|---|---|---:|---:|---:|
| CIFAR-100 | Mild perturbation, beta = 0.1; SIG/ISSBA/Blended average | 0.68 | 0.74 | 0.11 |
| CIFAR-100 | Enhanced perturbation, beta = 1; SIG/ISSBA/Blended average | 0.60 | 0.14 | 0.34 |
| Tiny ImageNet | Mild perturbation, beta = 0.1; SIG/ISSBA/Blended average | 0.53 | 0.91 | 0.03 |
| Tiny ImageNet | Enhanced perturbation, beta = 1; SIG/ISSBA/Blended average | 0.41 | 0.15 | 0.16 |

These rows show the security-utility trade-off in Freq-Pret: stronger
high-frequency perturbation lowers ASR, but it also lowers clean accuracy. Our
comparison should test whether selective image-conditioned correction can
obtain a better trade-off under an identical attack protocol.

---

## Closest same-dataset backdoor paper: SampDetox

**Paper:** *SampDetox: Black-box Backdoor Defense via Perturbation-based Sample
Detoxification* (NeurIPS 2024)

SampDetox uses diffusion-based purification. It first applies weak noise to
remove low-visibility backdoors. It then compares the original and recovered
images to localize stronger visible triggers and applies stronger purification
only to those regions.

Its DTD experiment is especially relevant because the authors intentionally
test a classification task that depends on legitimate high-frequency texture
information. They report:

- **CA:** accuracy of clean samples after the defense.
- **PA:** true-label accuracy of poisoned samples after purification.
- **ASR:** percentage of poisoned samples that still achieve the attacker's
  target behavior.

### Published DTD results from SampDetox Table 11

| Attack | Defense | CA (%) | PA (%) | ASR (%) |
|---|---|---:|---:|---:|
| BadNets, visible patch | No defense | 65.42 | 2.05 | 90.56 |
| BadNets, visible patch | SampDetox | 57.46 | 57.12 | **1.73** |
| WaNet, invisible spatial warp | No defense | 64.72 | 2.17 | 88.56 |
| WaNet, invisible spatial warp | SampDetox | 58.33 | 58.12 | **2.21** |

These results establish that backdoor defense has been studied on DTD, but not
under our frequency-trigger protocol.

---

## Contextual same-DTD table including our method

The following table can be shown in a review or paper only when the attack
column and the warning below remain visible.

| Study | Dataset | Trigger family | Clean accuracy after defense (%) | Poisoned/corrected true-label accuracy (%) | ASR after defense (%) |
|---|---|---|---:|---:|---:|
| SampDetox | DTD | BadNets visible patch | 57.46 | 57.12 | **1.73** |
| SampDetox | DTD | WaNet invisible spatial warp | 58.33 | 58.12 | **2.21** |
| **Proposed method** | **DTD** | **FTrojan-style block-DCT frequency trigger, four positions** | **61.70** | **60.29** | **7.64** |

**Mandatory warning:** This is a contextual comparison, not a head-to-head
ranking. Although the dataset and broad metrics are similar, the attacks,
models, data splits, poison settings, target classes, and implementations are
different. Therefore, it is incorrect to say that one row universally
outperforms another.

### What can safely be said

> Published work confirms that DTD is a meaningful high-frequency backdoor-
> defense benchmark. Under our different and explicitly frequency-domain
> FTrojan protocol, the proposed generator reduced mean ASR from 99.48% to
> 7.64%, recovered 60.29% true-label accuracy, and retained 61.70% clean-path
> accuracy after the generator. These results are competitive in utility, but
> they are not a direct replication of SampDetox's BadNets or WaNet setting.

Do **not** write that our method is numerically better than SampDetox. SampDetox
has lower residual ASR in its own attack settings, while our method has a
different threat model and a frequency-selective, inspectable correction
mechanism.

---

## Fair same-protocol comparison for the proposed method

This is the strongest performance table for the paper because every row uses
the same DTD validation split, suspicious ResNet-18, four active FTrojan-style
positions, and evaluation code.

**Result status:** frozen-model validation comparison; no training occurred
during this evaluation, but it is not yet a separately locked final-test run.

| Method under our protocol | Mean ASR (%) (lower is better) | True-label accuracy (%) (higher is better) | Meaning |
|---|---:|---:|---|
| No defense | 99.48 | 0.50 | Verified active attack |
| **Full image-conditioned generator** | **7.64** | **60.29** | Best security-utility balance |
| Static average correction | 18.22 | 55.00 | Fixed template helps but is weaker |
| Broad frequency suppression | 66.60 | 24.61 | Indiscriminate filtering damages texture information |
| Strongest predicted 1% only | 9.29 | 59.44 | Small selected support carries most correction effect |
| Everything outside strongest 1% | 99.28 | 0.68 | Unselected region alone does not repair the attack |

This table supports a valid claim that the learned image-conditioned generator
outperforms the tested static and broad alternatives **under the same protocol**.

---

## Journal support for showing image and frequency similarity

**Journal article:** M. Yoshida and M. Okuda, *IFAP: Input-Frequency Adaptive
Adversarial Perturbation via Full-Spectrum Envelope Constraint for Spectral
Fidelity*, IEEE Access, vol. 13, 2025, DOI:
`10.1109/ACCESS.2025.3648201`.

IFAP is an adversarial-evasion attack rather than a backdoor defense. It cannot
serve as a direct ASR baseline. Its value for our paper is the evaluation style:

| Evidence | What it shows |
|---|---|
| Original, modified, and amplified-difference images | Whether the change is visually noticeable and where it appears |
| PSNR | Pixel-level distortion magnitude |
| SSIM | Preservation of image luminance, contrast, and structure |
| STSIM | Preservation of texture statistics across frequency sub-bands |
| Frequency cosine similarity (`Freq_Cossim`) | Alignment between flattened absolute DCT spectra |
| Averaged radial DCT-spectrum plot | Whether frequency energy follows a similar profile across frequency bands |

The paper defines frequency cosine similarity as:

\[
\operatorname{FreqCos}(x,z)=
\frac{v_x^\top v_z}{\lVert v_x\rVert_2\lVert v_z\rVert_2},
\quad
v_x=\operatorname{flatten}(|\operatorname{DCT}(x)|),
\quad
v_z=\operatorname{flatten}(|\operatorname{DCT}(z)|).
\]

A value closer to 1 indicates more similar global DCT-amplitude profiles.
However, it does not prove that two images have the same class or that a
backdoor has been removed.

### Recommended similarity evidence for our paper

For each representative DTD example, show:

1. Clean image and clean prediction.
2. Triggered image and target prediction.
3. Corrected image and recovered prediction.
4. Trigger-minus-clean difference with a fixed display multiplier.
5. Corrected-minus-trigger difference with the same multiplier.
6. Clean, triggered, and corrected log-amplitude spectra using one shared color
   scale.
7. Applied spectral correction map.
8. PSNR or pixel L1, SSIM, and optionally STSIM.
9. ASR and true-label recovery beside the visual evidence.

Our existing panels already provide most of the visual evidence and report
pixel L1. The project also reports support IoU and correction/evidence cosine
similarity for causal alignment. SSIM or STSIM would be an additional
perceptual-preservation measurement; it should not be assigned a value until it
is actually calculated on the final evaluation set.

---

## Slide-ready version

**Slide title:** Same-Dataset Literature Context and Fair Internal Comparison

**Left side: published DTD context**

| Method | Attack | PA (%) (higher is better) | ASR (%) (lower is better) |
|---|---|---:|---:|
| SampDetox | BadNets | 57.12 | 1.73 |
| SampDetox | WaNet | 58.12 | 2.21 |
| Proposed | FTrojan-DCT | 60.29 | 7.64 |

Place this sentence below the table:

> Same dataset and related metrics, but different attack protocols; results are
> contextual and not a direct ranking.

Add a small note under the right-side table: **Frozen validation protocol; a
separately locked final-test run remains pending.**

**Right side: fair internal comparison**

| Our DTD method | ASR (%) (lower is better) | Accuracy (%) (higher is better) |
|---|---:|---:|
| No defense | 99.48 | 0.50 |
| **Full generator** | **7.64** | **60.29** |
| Static correction | 18.22 | 55.00 |
| Broad suppression | 66.60 | 24.61 |

### What to say aloud

> I found published backdoor-defense results on the same DTD texture dataset,
> but not with the same FTrojan frequency trigger. SampDetox tests visible
> BadNets and invisible WaNet attacks, whereas my experiment tests a repeated
> block-DCT frequency trigger. I therefore use its table only as same-dataset
> context. My actual performance claim is based on the right-side comparison,
> where all methods use exactly the same model, attack, data, and evaluation.
> The learned generator substantially outperforms a fixed correction and broad
> filtering while recovering texture accuracy.

---

## Final recommendation

Use **SampDetox** as the same-DTD backdoor-defense reference even though it is a
conference paper. Use **IFAP** as the same-DTD frequency-analysis reference,
**WPDA** as the recent frequency-trigger reference, and **Freq-Pret** as the
recent frequency-defense reference. None should be presented as an exact
published head-to-head competitor for the current DTD experiment.

For a publishable direct comparison, either run Freq-Pret on the exact DTD and
FTrojan checkpoint, or extend the proposed method to CIFAR-100 and Tiny
ImageNet and evaluate it under the exact same attacks as Freq-Pret/WPDA. Until
one of those is completed, the internal static, broad-suppression, and support
ablation baselines remain the valid numerical head-to-head comparison.

## Sources

- [SampDetox paper and DTD Table 11](https://papers.nips.cc/paper_files/paper/2024/file/dbb5180957513805ebeea787b8c66ac9-Paper-Conference.pdf)
- [SampDetox publication page](https://proceedings.nips.cc/paper_files/paper/2024/hash/dbb5180957513805ebeea787b8c66ac9-Abstract-Conference.html)
- [WPDA, Neural Networks journal article](https://doi.org/10.1016/j.neunet.2025.108074)
- [Freq-Pret, IEEE Internet of Things Journal](https://doi.org/10.1109/JIOT.2026.3671197)
- [CLTBA, Journal on Information Security](https://doi.org/10.1186/s13635-026-00246-6)
- [TriBA, IEEE Transactions on Dependable and Secure Computing](https://doi.org/10.1109/TDSC.2025.3600427)
- [IFAP, IEEE Access journal article](https://doi.org/10.1109/ACCESS.2025.3648201)
- [FTrojan, ECCV 2022](https://doi.org/10.1007/978-3-031-19778-9_23)
- [Official DTD dataset page](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
- Supplied Freq-Pret PDF: `Paper/Backdoor_Defense_Strategy_for_Image_Classification_Tasks_Based_on_Frequency_Domain_Perturbation.pdf`
- Local final metrics: `Advanced_Outputs/dtd_position_generator_final_baselines_v1/final_baseline_summary.json`
- Local generator validation: `Advanced_Outputs/dtd_position_generalizing_generator_v1/validation_summary.json`
