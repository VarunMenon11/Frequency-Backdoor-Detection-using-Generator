# Literature Survey and Research Gap

## Project context

**Project:** Selective Frequency-Domain Backdoor Mitigation Using Adaptive Spectral Correction

This survey is organized around the exact project question:

> Can a backdoored image classifier be repaired by selectively correcting trigger-related frequency information while preserving useful image structure and clean classification performance?

The literature should not be presented as a list of unrelated papers. Explain how each paper contributes to one of four research directions:

1. Backdoor attack and threat-model foundations.
2. Classical detection and model-repair defenses.
3. Frequency-domain attacks and spectral analysis.
4. Image purification and adaptive frequency-aware defenses.

The central research gap is not that nobody has used Fourier analysis. Existing research already uses frequency-domain attacks, spectral detection, frequency purification, phase preservation, and model repair. The narrower potential contribution of this project is the combination of:

- clean/triggered amplitude-difference evidence;
- a learned image-conditioned correction gate;
- selective amplitude movement toward a clean reference;
- phase-preserving inverse-FFT reconstruction;
- classifier repair using clean, corrected-triggered, and raw-triggered samples;
- cross-resolution and trigger-family evaluation.

This combination must be compared against the recent Freq-Pret direction before a strong novelty claim is made.

---

## A. Papers that should be read first

### 1. Backdoor Defense Strategy for Image Classification Tasks Based on Frequency Domain Perturbation / Freq-Pret

**Authors:** Ping Zhang, Hongyuan Yue, Chau Yuen  
**Venue:** IEEE Internet of Things Journal, 2026  
**Link:** [Indexed paper record](https://www.researchgate.net/publication/401673151_Backdoor_Defense_Strategy_for_Image_Classification_Tasks_based_on_Frequency_Domain_Perturbation)

#### Method

The accessible abstract describes a frequency-domain defense called Freq-Pret. It preserves phase-related semantics, perturbs high-frequency amplitude features, and combines frequency perturbation with lightweight retraining to weaken the model's dependence on trigger information.

#### Limitation or comparison point

This is the closest recent paper to the broad direction of the current project. The full paper must be obtained and checked carefully for the following details:

- whether it uses a learned generator;
- whether the generator receives clean and triggered amplitude pairs;
- whether it predicts a per-frequency correction map;
- whether it uses the absolute amplitude difference as an input;
- whether it performs image reconstruction and classifier repair as separate stages;
- whether it evaluates unknown triggers and unknown images;
- which datasets, attacks, architectures, and baselines it uses.

Do not claim that the current project is the first phase-preserving amplitude defense until this comparison is complete.

#### Relevance to this project

Use it as the closest comparison in the research-gap slide. Explain that the present project investigates a more explicit adaptive correction-map formulation and evaluates the combination across resolutions, trigger strength, trigger functions, and a FIBA-style attack.

---

### 2. Rethinking the Backdoor Attacks' Triggers: A Frequency Perspective

**Authors:** Yi Zeng, Won Park, Z. Morley Mao, Ruoxi Jia  
**Venue:** ICCV 2021  
**Link:** [Official ICCV paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zeng_Rethinking_Backdoor_Attacks_Triggers_A_Frequency_Perspective_ICCV_2021_paper.pdf)

#### Method

This work analyzes common backdoor triggers in the frequency domain. It shows that many existing attacks leave high-frequency artifacts and demonstrates that frequency information can be used for detecting suspicious samples without prior knowledge of the exact attack details.

#### Limitation

Its main contribution is frequency-based analysis and detection, not a generator that selectively reconstructs each triggered image or repairs the classifier through a learned amplitude correction map. Detecting suspicious data does not automatically remove a learned backdoor from a deployed model.

#### Relevance

This paper supports the motivation for inspecting frequency spectra. It is useful when explaining why the trigger may be difficult to see in RGB space but still create measurable spectral evidence.

---

### 3. FIBA: Frequency-Injection Based Backdoor Attack in Medical Image Analysis

**Authors:** Yu Feng et al.  
**Venue:** CVPR 2022  
**Link:** [Official CVPR paper](https://openaccess.thecvf.com/content/CVPR2022/html/Feng_FIBA_Frequency-Injection_Based_Backdoor_Attack_in_Medical_Image_Analysis_CVPR_2022_paper.html)

#### Method

FIBA creates a frequency-domain backdoor by linearly blending the amplitude spectrum of a clean image with the amplitude spectrum of a trigger/reference image inside a frequency mask. It preserves the clean phase and reconstructs the poisoned image through inverse FFT. The method was designed for medical-image tasks and is evaluated in classification and dense-prediction settings.

#### Limitation

FIBA is an attack, not a defense. It shows how amplitude information can carry a backdoor, but it does not provide the proposed generator-based correction and classifier-repair pipeline. Its construction also depends on a reference trigger image, a mask, and attack-specific blending settings.

#### Relevance

FIBA is the most relevant attack benchmark for the current amplitude-side defense. The project should use it to test whether a correction method designed around amplitude differences can mitigate a trigger created directly in the Fourier domain.

---

### 4. Backdoor Attack through Frequency Domain / FTROJAN

**Authors:** Tong Wang et al.  
**Year:** 2021  
**Link:** [FTROJAN paper](https://arxiv.org/abs/2111.10991)

#### Method

FTROJAN embeds a backdoor directly through frequency-domain manipulation. The resulting spatial perturbation is distributed across the image and can be visually difficult to distinguish from clean data. The paper evaluates attack success, clean accuracy, and resistance to defenses.

#### Limitation

FTROJAN is primarily an attack method. It demonstrates the threat but does not solve selective frequency correction or model repair. Its trigger construction is also different from the paired clean/triggered correction setting used in the current project.

#### Relevance

It demonstrates why a defense should not rely only on visible pixel patches. It is a useful attack baseline for future evaluation.

---

### 5. UPure: Defending Against Repetitive Backdoor Attacks on Semi-Supervised Learning through Rate-Distortion-Perception Trade-Off

**Authors:** Cheng-Yi Lee et al.  
**Venue:** WACV 2025  
**Link:** [Official WACV paper](https://openaccess.thecvf.com/content/WACV2025/papers/Lee_Defending_Against_Repetitive_Backdoor_Attacks_on_Semi-Supervised_Learning_through_Lens_of_Rate-Distortion-Perception_Trade-Off_WACV_2025_paper.pdf)

#### Method

UPure purifies poisoned unlabeled data by introducing frequency-domain perturbations. It uses a rate-distortion-perception analysis to select an appropriate frequency region and applies purification strategies before model training.

#### Limitation

UPure is designed for semi-supervised learning and poisoned-data purification. It is not the same as an input-conditioned generator that receives a clean/triggered amplitude difference and predicts a soft correction map. Its frequency-region strategy is also different from the current learned amplitude-gating formulation.

#### Relevance

UPure is an important modern baseline because it directly challenges the statement that all frequency defenses are global filters. The project literature survey should acknowledge it and describe the narrower difference between frequency perturbation/purification and learned paired spectral correction.

---

### 6. Black-box Backdoor Defense via Zero-shot Image Purification

**Authors:** Yucheng Shi et al.  
**Venue:** NeurIPS 2023  
**Link:** [Official NeurIPS paper](https://papers.neurips.cc/paper_files/paper/2023/file/b36554b97da741b1c48c9de05c73993e-Paper-Conference.pdf)

#### Method

ZIP first applies a transformation such as blurring to damage the trigger, then uses a pre-trained diffusion model to recover semantic information and produce a purified image. It is designed for black-box or zero-shot settings where the defender may not know the trigger or have internal model information.

#### Limitation

Diffusion-based purification can be computationally expensive. It may introduce generative changes and is dependent on the availability and suitability of a pre-trained generative model. It does not explicitly learn the trigger-related amplitude difference or perform the current phase-preserving spectral reconstruction.

#### Relevance

ZIP is an important comparison for the future unknown-image version of the project. It shows that input purification can operate without a clean counterpart, which is a limitation of the current paired generator.

---

## B. Classical model-repair and backdoor-defense papers

### 7. Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks

**Authors:** Bolun Wang et al.  
**Venue:** IEEE Symposium on Security and Privacy, 2019  
**Link:** [Paper](https://people.cs.uchicago.edu/~ravenben/publications/pdf/backdoor-sp19.pdf)

#### Method

Neural Cleanse reverse-engineers a possible trigger for each target class by optimizing a mask and pattern that causes inputs to be classified as that target. It uses an anomaly score to identify unusually small trigger patterns and then applies mitigation such as filtering, pruning, or unlearning.

#### Limitation

The method relies on trigger reverse engineering and can be challenged by complex, dynamic, sample-specific, or frequency-domain triggers. It focuses on discovering a trigger through model behaviour rather than learning a direct amplitude correction map.

#### Relevance

It is a standard baseline for detection and mitigation. The current project should distinguish “reverse-engineer a trigger” from “correct a learned spectral difference.”

---

### 8. Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks

**Authors:** Kang Liu, Brendan Dolan-Gavitt, Siddharth Garg  
**Venue:** RAID 2018  
**Link:** [Paper](https://arxiv.org/abs/1805.12185)

#### Method

Fine-Pruning combines pruning and fine-tuning. It removes neurons that are considered less active on clean data and then fine-tunes the remaining model to reduce backdoor behaviour.

#### Limitation

Pruning can remove useful features along with backdoor-related features. Its performance depends on identifying suitable neurons and may be weaker against attacks that distribute backdoor behaviour broadly or are designed to resist pruning.

#### Relevance

It provides a classic model-level repair baseline. The current method differs because it first targets the suspected input spectrum and then repairs the classifier with trigger-containing samples.

---

### 9. Neural Attention Distillation: Erasing Backdoor Triggers from Deep Neural Networks

**Authors:** Yige Li et al.  
**Venue:** ICLR 2021  
**Link:** [OpenReview paper](https://openreview.net/forum?id=9l0K4OM-oXE)

#### Method

NAD fine-tunes a teacher network on a small clean subset and then distills the teacher's intermediate attention behaviour into the backdoored student. The goal is to redirect the model's attention away from the trigger.

#### Limitation

NAD requires a clean subset and a teacher/student training process. It works in feature/attention space and does not explicitly locate or correct a trigger-related frequency component. Its success can depend on teacher quality and the availability of clean data.

#### Relevance

NAD is a strong model-repair baseline. It helps explain why the current project includes a classifier-repair stage but uses spectral evidence rather than attention alignment.

---

### 10. Adversarial Neuron Pruning Purifies Backdoored Deep Models

**Authors:** Dongxian Wu and Yisen Wang  
**Venue:** NeurIPS 2021  
**Link:** [Official NeurIPS paper](https://proceedings.neurips.cc/paper/2021/hash/8cbe9ce23f42628c98f80fa0fac8b19a-Abstract.html)

#### Method

ANP observes that neurons associated with backdoor behaviour are unusually sensitive to adversarial perturbations of their weights. It searches for sensitive neurons and prunes them to repair the model.

#### Limitation

ANP is model-internal and architecture-dependent. It does not produce a corrected input image or provide a visual explanation of which spectral components were changed. Pruning can also affect clean features if the backdoor and semantic features are entangled.

#### Relevance

ANP is useful as a comparison against model-only repair. The current project adds an explicit image-level spectral correction before or alongside model repair.

---

### 11. Adversarial Unlearning of Backdoors via Implicit Hypergradient / I-BAU

**Authors:** Yi Zeng et al.  
**Venue:** ICLR 2022  
**Link:** [OpenReview paper](https://openreview.net/pdf?id=MeeQkFYVbzW)

#### Method

I-BAU formulates backdoor removal as a minimax problem using clean data. It uses an implicit hypergradient to account for the interaction between an adversarial inner problem and model-repair optimization.

#### Limitation

The optimization is more complex than ordinary fine-tuning and can be computationally expensive. It depends on clean data and model access, and it repairs model behaviour without providing the frequency-localized image correction used here.

#### Relevance

It is a strong unlearning baseline and a useful comparison for the model-repair part of the project.

---

### 12. Progressive Backdoor Erasing via Connecting Backdoor and Adversarial Attacks

**Authors:** Mu et al.  
**Venue:** CVPR 2023  
**Link:** [Official CVPR paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Mu_Progressive_Backdoor_Erasing_via_Connecting_Backdoor_and_Adversarial_Attacks_CVPR_2023_paper.pdf)

#### Method

Progressive Backdoor Erasing connects backdoor erasing with adversarial attack generation and progressively updates the model to remove backdoor behaviour. It evaluates against Fine-Pruning, Neural Cleanse, NAD, ANP, and other methods.

#### Limitation

The method is primarily a model-repair approach and may require clean extra data for its strongest setting. It does not provide a frequency-amplitude correction map or phase-preserving image reconstruction.

#### Relevance

This paper is useful because it provides a modern comparison table for several standard defenses and demonstrates that post-training repair should be evaluated against multiple attack families.

---

## C. Additional papers for broader context

### 13. Backdoor Cleansing With Unlabeled Data

**Venue:** CVPR 2023  
**Link:** [Official CVPR paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Pang_Backdoor_Cleansing_With_Unlabeled_Data_CVPR_2023_paper.pdf)

This paper studies cleansing a suspicious network using unlabeled data, with layer-wise weight re-initialization and knowledge distillation. Its importance for this project is the reduced dependence on labeled clean data. The limitation is that the method remains primarily model-space repair and does not provide the explicit spectral correction explanation used in the current project.

### 14. Neural Polarizer: A Lightweight and Effective Backdoor Defense via Purifying Poisoned Features

**Venue:** NeurIPS 2023  
**Link:** [Official NeurIPS paper](https://papers.neurips.cc/paper_files/paper/2023/hash/03df5246cc78af497940338dd3eacbaa-Abstract-Conference.html)

Neural Polarizer inserts a learnable intermediate module that filters trigger information while preserving benign information in feature space. It is relevant because it is also selective rather than simply deleting a broad part of the representation. Its limitation relative to this project is that it operates inside learned feature space and does not explicitly analyze Fourier amplitude or phase.

### 15. Towards Invisible Backdoor Attacks in the Frequency Domain Against Deep Neural Networks

**Authors:** Xinrui Liu et al.  
**Year:** 2023  
**Link:** [Paper](https://arxiv.org/abs/2305.10596)

This paper develops another frequency-domain backdoor attack and evaluates invisibility using image-similarity metrics. It is useful for the threat-model section and for future attack diversity. Its limitation is that it is an attack paper rather than a correction or model-repair defense.

### 16. Lite-BD: A Lightweight Black-box Backdoor Defense via Reviving Multi-Stage Image Transformations

**Authors:** Abdullah Arafat Miah and Yu Bi  
**Year:** 2026 preprint  
**Link:** [arXiv paper](https://arxiv.org/abs/2602.07197)

Lite-BD uses down-upscaling and query-based band-by-band frequency filtering for black-box backdoor defense. It is relevant to the future unknown-image setting and to input-time defenses. As a recent preprint, its peer-review status and exact experimental scope should be checked before treating it as a final state-of-the-art baseline.

---

## D. Comparison matrix for the PPT

| Paper or method | Domain | Main method | Needs clean data? | Changes image? | Repairs model? | Main limitation relative to this project |
|---|---|---|---|---|---|---|
| Neural Cleanse | Model/input analysis | Reverse-engineer minimal trigger | Not necessarily in the strict implementation | Sometimes | Yes | Trigger inversion can struggle with complex or frequency triggers |
| Fine-Pruning | Model repair | Prune low-activation neurons and fine-tune | Usually a clean subset | No explicit correction | Yes | May remove useful features and is model-dependent |
| NAD | Model repair | Teacher attention distillation | Small clean subset | No explicit correction | Yes | Feature-space method; depends on teacher and clean data |
| ANP | Model repair | Find sensitive neurons and prune them | Very small clean set | No explicit correction | Yes | Architecture/model-space dependent |
| I-BAU | Model repair | Minimax adversarial unlearning | Clean data | No explicit correction | Yes | More complex and computationally demanding optimization |
| Frequency Perspective | Spectral detection | Analyze frequency artifacts | Dataset samples | No | No | Detection does not itself remove the learned dependency |
| FTROJAN | Frequency attack | Inject trigger in frequency domain | Attack-side data | Yes | No | Attack method, not defense |
| FIBA | Frequency attack | Blend amplitude spectra and preserve phase | Attack-side reference | Yes | No | Attack method, not defense |
| ZIP | Black-box purification | Transform image and recover with diffusion | No clean pair required | Yes | No retraining required | Computational cost and generative distribution dependence |
| UPure | Frequency purification | Perturb selected frequency regions | Designed for SSL data purification | Yes | Trains model on purified data | Fixed purification setting and different threat/task scope |
| Freq-Pret | Frequency defense | Phase-preserving amplitude perturbation plus retraining | Check full paper | Likely | Yes | Closest comparison; exact overlap must be established from full paper |
| Proposed project | Spectral correction + repair | Learned correction map from paired amplitude evidence plus repair | Yes in current version | Yes | Yes | Current paired setting and limited repeated-seed/baseline analysis |

---

## E. Recommended literature-survey structure in the dissertation

### Paragraph 1: Backdoor threat

Introduce data-poisoning backdoors and explain the clean-accuracy/high-ASR behaviour.

### Paragraph 2: Classical defenses

Discuss Neural Cleanse, Fine-Pruning, NAD, ANP, and I-BAU. Group them into trigger inversion, pruning, distillation, and unlearning.

### Paragraph 3: Frequency-domain attacks

Discuss frequency perspectives, FTROJAN, FIBA, and other invisible frequency triggers. Explain why amplitude and phase matter.

### Paragraph 4: Frequency-aware purification

Discuss ZIP, UPure, Freq-Pret, and recent frequency-aware purification. Explain that these methods vary in whether they are black-box, input-time, data-time, model-time, fixed, or learned.

### Paragraph 5: Gap and positioning

State that the proposed project investigates an adaptive, image-conditioned amplitude correction map trained using spectral evidence and classifier feedback, followed by classifier repair. State the current known-trigger limitation and the planned input-only extension.

---

## F. Safe research-gap paragraph for the paper

> Existing backdoor defenses operate through trigger reverse engineering, activation or neuron analysis, model pruning, knowledge distillation, adversarial unlearning, or input purification. Frequency-domain research has also shown that backdoor triggers can be injected into or detected through spectral representations. However, broad frequency suppression or fixed purification may treat natural and trigger-related components similarly. Motivated by this limitation, this work investigates a selective spectral correction framework in which clean and triggered amplitude evidence is used to learn a frequency-dependent correction map. The corrected amplitude is combined with preserved phase information for image reconstruction, and the suspicious classifier is subsequently repaired using clean, corrected-triggered, and raw-triggered samples. The current study is a controlled known-trigger evaluation; blind unknown-image correction is left for future work.

This wording avoids claiming that no related frequency defense exists while still clearly defining the project contribution.

---

## G. What to read and what to show the guide

### Minimum reading set

Read these six first:

1. The Freq-Pret paper you already have.
2. Rethinking Backdoor Attacks: A Frequency Perspective.
3. FIBA.
4. Neural Cleanse.
5. NAD or ANP.
6. ZIP or UPure.

### What to show in the PPT

Use one comparison table with columns:

```text
Paper | Domain | Method | Clean-data requirement | Main limitation | Relation to our work
```

Do not put every equation from every paper on the slide. The panel mainly needs to see that you understand the categories, the trade-offs, and the precise position of your method.

### What to say if asked whether the method is completely new

> The underlying tools are established individually: FFT decomposition, phase preservation, frequency-domain attacks, image purification, and model repair. The research contribution I am investigating is their combination into a selective amplitude-correction framework with a learned correction map and joint image/model evaluation. I am treating Freq-Pret and other frequency-aware defenses as close comparisons rather than claiming that the overall direction has never been explored.

