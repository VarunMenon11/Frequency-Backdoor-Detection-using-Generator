# Datasets

This module loads CIFAR-100, defines tensor conversion, creates train/test
splits, and exposes clean or poisoned PyTorch datasets.

The first implementation target is dataset construction and inspection. We will
not train a model until the clean and poisoned data are verified.
