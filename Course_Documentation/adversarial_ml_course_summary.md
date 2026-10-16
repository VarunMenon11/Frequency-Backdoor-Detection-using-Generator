Module 1 started with the introduction of adversarial machine learning using evasion attacks, data poisoning, model extraction, model inference, unsafe Pickle objects, and secure model deployment. Module 1 laboratory provided me with a reproducible environment consisting of Python, PyTorch, Git, and test environments. I applied the same concept in my project by ensuring that everything was separated.

The concepts of Module 2 include data validation, clean baseline, confusion matrix, hash, and configuration-based training. To implement the above, I recorded the information about the split of the DTD dataset, poison ratio, target class, trigger parameter, model setting, checkpoints, and JSON outputs. In addition to that, I trained a control model of ResNet-18 which would allow comparing the poisoned model with the normal target class.

Module 3 covered white box attacks and introduced FGSM as an attack function.
FGSM uses the gradient of a model to create small perturbations in pixels to
maximize classification error. While FGSM and my training time backdoor are
not the same thing, I learned through Module 3 that attacks should be
parameterized, bounded, tested against held out data, and tested for multiple
strengths. Transferability led me to consider training a generator on
trigger locations to then test it on unseen locations.

In Module 4, we discussed black-box attacks, suspicious query bursts, rate
limiting, output truncation, and watermarking. None of these techniques were
used in the offline test, but they will help when deploying the API, as
detection of repeated queries trying to trigger a response from the generator
will be necessary.

Module 5 had a direct relation to my research because it discussed
poisoning and backdoor attacks. I applied this method to Describable Textures
Dataset with an ImageNet-pretrained ResNet-18. The chosen non-target images
were attacked by an FTrojan-type of frequency trigger, which is related to the
target class `banded`. The initial models were dismissed because high
success rate in the poisoned training samples could not be generalized to
images outside the training set. I conducted multiple calibration attacks
sweeps and chose a validated attack model with acceptable clean performance.
I evaluated it based on clean accuracy, success rate of the triggered
attacks, rate of clean target, attack success lift and conditional attack
success in order to make sure that natural errors will not be counted as
attacks. After that, the suspicious classifier was frozen, and used for
training a reference-free generator. In the process of evaluation, the
generator receives only one possibly triggered image without its clean
version. It evaluates the amplitude, phase, local spectral residual and
frequency coordinates, and makes a selective prediction about amplitude
correction of the image before reconstruction and classification. I trained
the generator with two trigger position layouts and tested it on held-out
layouts. These tests provided evidence that correction was linked to
a small, meaningful spectral region rather than simple clean-trigger
subtraction or removal of the entire spectrum. MITRE ATLAS also helped place
this work within broader data-compromise and supply-chain threats.

Module 6 discussed the topic of randomized smoothing and calibration. These
two have not been used in the implemented generator, but they proved that
accuracy, robustness, and confidence are separate notions. Randomized
smoothing might be used as a generic baseline in future, while calibration
can show whether the initial confidence and confidence after
correction are reliable.

Module 7 discussed topics related to data engineering, evidence of governance,
the EU AI Act, and DORA. The project relevance of this module lies in the
procedure, according to which the claim should be proven using manifests,
configurations, hashes, checkpoints, metrics, figures, limitations and the link
between all of these.

Module 8 dealt with AI supply chain security, AIBOMs, provenance, and SLSA. This is
indirectly relevant since a backdoor can come in via a third-party data
set or model. The data set source, manifest, pretrained weights, software
versions, configuration, checkpoints, and output could be used to create an
artifact record, even though the project does not have any claim for SLSA
compliance. Module 9 covered application engineering with minimal impact on
the technique.

In summary, the class provided support for the whole project process:
creating and validating the backdoor attack, training a reference-free
spectral correction technique, testing on seen and unseen situations,
comparing other simpler techniques, examining the behavior of the noise, and
documenting both success and limitations.