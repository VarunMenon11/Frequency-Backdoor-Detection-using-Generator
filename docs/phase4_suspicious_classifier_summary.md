# Phase 4 Summary: Suspicious Classifier Setup

This document summarizes the initial classifier implementation used to train a
suspicious model on the poisoned CIFAR-100 dataset.

## Purpose

The suspicious classifier is trained on a mixture of clean and poisoned training
samples.

Its purpose is to test whether the model learns two behaviors:

```text
clean image -> correct CIFAR-100 class
triggered image -> attacker target class
```

The second behavior is the backdoor. We do not assume it exists until it is
confirmed by Attack Success Rate.

## Model

Implemented model:

```text
SmallCIFARClassifier
```

Location:

```text
models/cifar_cnn.py
```

Input:

```text
image tensor with shape (B, 3, 32, 32)
```

Output:

```text
logits with shape (B, 100)
```

The model is a compact convolutional neural network with:

```text
convolution
batch normalization
ReLU
max pooling
adaptive average pooling
linear classifier
```

Trainable parameters:

```text
1,172,004
```

## Training Data

The model trains on:

```text
PoisonedCIFAR100Dataset
```

Current poisoning setup:

```text
total training samples: 50000
poisoned training samples: 6000
poison ratio: 0.12
target label: apple
trigger strength: 0.08
trigger frequency: (6, 6)
```

## Evaluation

The model is evaluated using two datasets.

### Clean Accuracy

Dataset:

```text
Clean CIFAR-100 test set
```

Purpose:

```text
Measures whether the model still performs normal classification.
```

### Attack Success Rate

Dataset:

```text
TriggeredCIFAR100TestDataset
```

Purpose:

```text
Measures how often triggered non-target test images are classified as apple.
```

Target-class test samples are excluded from ASR evaluation because predicting
apple on an actual apple image is not evidence of attack success.

## Guide Demonstration Script

Main runnable file:

```text
scripts/train_suspicious_classifier.py
```

This script:

```text
1. Loads poisoned CIFAR-100 training data.
2. Loads clean CIFAR-100 test data.
3. Loads triggered ASR test data.
4. Builds the classifier.
5. Trains the model.
6. Prints clean accuracy and ASR after each epoch.
7. Saves a checkpoint and JSON summary.
```

Quick smoke-test command:

```text
python -m scripts.train_suspicious_classifier --epochs 1 --batch-size 32 --max-train-batches 2 --max-eval-batches 1 --device cpu --output-dir experiments/suspicious_classifier_smoke_test
```

Real training should remove the batch limits:

```text
python -m scripts.train_suspicious_classifier --epochs 30 --batch-size 128 --device auto --output-dir experiments/suspicious_classifier_full
```

## Smoke Test Result

A smoke test was run to verify the full pipeline.

The result is not scientifically meaningful because it used only two training
batches and one evaluation batch.

Verified behavior:

```text
model builds successfully
training loop runs
clean accuracy evaluation runs
ASR evaluation runs
checkpoint is saved
summary JSON is saved
```

Smoke-test outputs:

```text
experiments/suspicious_classifier_smoke_test/training_summary.json
experiments/suspicious_classifier_smoke_test/suspicious_classifier.pt
```

## Next Requirement

We need to run real training until the model is confirmed to be backdoored.

A confirmed suspicious model should show:

```text
reasonable clean accuracy
high attack success rate
```

Only after this confirmation should we move to frequency analysis and generator
training.
