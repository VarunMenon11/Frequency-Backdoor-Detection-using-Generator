# Experiment Record: Different Image Resolutions

## 1. Research question

The first baseline used CIFAR-100 at 32x32. A natural question was whether the approach was tied to tiny images. This experiment therefore repeated the complete pipeline on three resolutions:

- CIFAR-100: 32x32;
- Tiny ImageNet: 64x64;
- STL-10: 96x96.

The same principle, poison ratio, approximate trigger strength, model family, generator objective, repair procedure, and evaluation logic were retained. Dataset-specific class counts and target names changed as required.

## 2. Why resolution matters

When resolution increases, the image contains more pixels and the FFT contains a larger frequency grid. There are therefore more possible locations at which natural texture and trigger energy may appear. A method that only works on 32x32 images would have limited practical value.

Resolution scaling is not the same as simply enlarging a picture for display. The experiments actually trained and evaluated the models using the native dataset images at their stated sizes.

## 3. Fixed and scaled settings

| Dataset | Resolution | Classes | Target | Poison ratio | Alpha | Frequency |
|---|---:|---:|---|---:|---:|---:|
| CIFAR-100 | 32x32 | 100 | apple | 0.12 | 0.08 | (6,6) |
| Tiny ImageNet | 64x64 | 200 | goldfish | 0.12 | 0.08 | (12,12) |
| STL-10 | 96x96 | 10 | airplane | 0.12 | 0.08 | (18,18) |

The frequencies follow the proportional rule:

$$
f_{new}=f_{CIFAR}\times\frac{R_{new}}{32}.
$$

This keeps the trigger at a comparable relative frequency position. It is an experimental normalization, not proof that these are the best frequencies for each dataset.

## 4. Dataset details

### CIFAR-100

CIFAR-100 contains 50,000 training images, 10,000 test images, and 100 classes. The 32x32 input makes it a convenient baseline for verifying the code, FFT shapes, poisoning, generator training, and repair process.

### Tiny ImageNet

Tiny ImageNet contains 100,000 training images, 10,000 validation images, and 200 classes at 64x64 resolution. It is a harder classification task because the model must distinguish many more categories. The ASR set contains 9,950 non-target validation images after excluding the 50 target-class examples.

### STL-10

STL-10 contains 5,000 labelled training images and 8,000 test images at 96x96 resolution across 10 classes. Its larger image size makes it the clearest test of whether the correction map can operate on a richer frequency grid. The ASR set contains 7,200 non-airplane test images.

## 5. Results

| Dataset | Suspicious clean accuracy | Suspicious ASR | Generator corrected accuracy | Generator corrected ASR | Repaired clean accuracy | Repaired ASR |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-100 32x32 | 55.68% | 98.54% | 54.65% | 0.37% | 63.35% | 0.06% |
| Tiny ImageNet 64x64 | 40.96% | 99.99% | 40.51% | 0.37% | 46.11% | 0.09% |
| STL-10 96x96 | 65.75% | 100.00% | 63.51% | 3.15% | 76.19% | 0.71% |

## 6. Interpretation

The first column of defence evidence is the suspicious ASR. It is close to 100% on all three datasets, showing that the trigger was learned at every resolution. This makes the subsequent reductions meaningful.

The generator alone reduces ASR to 0.37% on CIFAR-100, 0.37% on Tiny ImageNet, and 3.15% on STL-10. The slightly larger residual on STL-10 does not invalidate the method; it indicates that the higher-resolution setting is more difficult and leaves more spectral variation to model.

After repair, raw-trigger ASR falls to 0.06%, 0.09%, and 0.71%, respectively. Clean accuracy also improves relative to the suspicious checkpoint in all three stored baseline runs. This improvement should be reported as an observed result, not as a guaranteed property of every training run.

## 7. What the result proves

The resolution experiment supports this carefully bounded statement:

> The proposed generator-based spectral correction and classifier-repair pipeline reduced a controlled sinusoidal frequency backdoor across 32x32, 64x64, and 96x96 RGB image inputs.

It does not prove that an arbitrary 128x128 image, a medical image, or a satellite image can be processed without adaptation. Such data may have different channels, dynamic ranges, class distributions, and frequency statistics.

## 8. Visual evidence

Use one panel from each resolution:

- CIFAR-100: [final correction panel](../outputs/final_cifar100_evaluation/sample_panels/final_panel_test_index_0.png)
- Tiny ImageNet: [validation correction panel](../outputs/tinyimagenet_64x64/sample_panels/tinyimagenet_64_panel_val_index_0.png)
- STL-10: [final correction panel](../outputs/STL_Outputs/stl10_96x96/sample_panels/stl10_96_panel_test_index_0.png)

The higher-resolution example can be embedded as follows:

![STL-10 96x96 correction panel](../outputs/STL_Outputs/stl10_96x96/sample_panels/stl10_96_panel_test_index_0.png)

Each panel should be explained as a sequence: original content, trigger-induced prediction, spectral correction, and final repaired prediction. The amplitude difference and correction map are more informative than independently normalized amplitude images because the natural image spectrum is much stronger than the small injected change.

## 9. Reproducibility files

- CIFAR-100 record: `../docs/final_cifar100_experiment_record.md`
- Tiny ImageNet record: `../docs/final_tinyimagenet_64x64_experiment_record.md`
- STL-10 record: `../docs/final_stl10_96x96_experiment_record.md`
- Aggregate record: `../docs/final_cross_resolution_results_record.md`
- Training and evaluation scripts: `../scripts/`

