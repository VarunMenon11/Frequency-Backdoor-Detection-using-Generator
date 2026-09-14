"""Paired, source-aligned measurements for the advanced DTD classifier."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import ASBManifestDataset


SELECTION_POLICY = "maximum_validation_clean_accuracy_first_tie"


def checkpoint_selection_score(clean_accuracy):
    # Attack strength is a separate diagnostic, not a reward for target bias.
    return float(clean_accuracy)


def attack_checkpoint_score(validation_asr, clean_accuracy, minimum_clean_accuracy):
    """Rank attack checkpoints without rewarding ordinary target-class bias.

    The checkpoint must first satisfy the predeclared clean-accuracy floor.
    Among eligible epochs, conditional ASR is primary, same-model target-rate
    lift is the first tie-breaker, and clean accuracy is the final tie-breaker.
    """
    clean_accuracy = float(clean_accuracy)
    if clean_accuracy < float(minimum_clean_accuracy) or not validation_asr:
        return None
    conditional = [
        float(metrics["conditional_asr_clean_correct"])
        for metrics in validation_asr.values()
        if metrics["conditional_asr_clean_correct"] is not None
    ]
    if not conditional:
        return None
    mean_conditional = sum(conditional) / len(conditional)
    mean_lift = sum(
        float(metrics["same_model_target_rate_lift"])
        for metrics in validation_asr.values()
    ) / len(validation_asr)
    return mean_conditional, mean_lift, clean_accuracy


def evaluation_transform(image_size):
    return transforms.Compose([
        transforms.Resize(
            round(image_size * 256 / 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])


def apply_evaluation_configs(rows, attack_configs):
    """Override seen-trigger configs without changing the stored manifest."""
    return [
        {**row, "trigger_config": dict(attack_configs[str(row["variant_name"])])}
        if str(row["variant_name"]) in attack_configs else dict(row)
        for row in rows
    ]


def resolve_evaluation_config(rows, name, attack_configs):
    if name in attack_configs:
        return dict(attack_configs[name])
    configs = [row["trigger_config"] for row in rows
               if row["protocol_role"] == "final_test_triggered"
               and row["variant_name"] == name]
    if not configs or any(config != configs[0] for config in configs):
        raise ValueError(f"Missing or inconsistent manifest config for {name}")
    return dict(configs[0])


@torch.inference_mode()
def predict_rows(model, rows, images_root, transform, args, device, *, progress=None):
    """Return predictions in manifest order, retaining source IDs for pairing."""
    dataset = ASBManifestDataset(
        images_root, rows, transform=transform, label_field="original_label"
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    model.eval()
    predictions = []
    for batch_index, (images, labels) in enumerate(loader):
        if args.max_eval_batches is not None and batch_index >= args.max_eval_batches:
            break
        predicted = model(images.to(device, non_blocking=True)).argmax(1).cpu().tolist()
        start = len(predictions)
        for row, label, value in zip(rows[start:start + len(predicted)], labels.tolist(), predicted):
            predictions.append({
                "source_id": str(row["source_id"]),
                "original_label": int(label), "prediction": int(value),
            })
        if progress and (batch_index % 10 == 0 or len(predictions) == len(rows)):
            print(f"{progress}: {len(predictions)}/{len(rows)}", flush=True)
    if not predictions:
        raise ValueError("Evaluation produced no samples")
    if len({row["source_id"] for row in predictions}) != len(predictions):
        raise ValueError("Evaluation source IDs must be unique")
    return predictions


def clean_metrics(predictions):
    count = len(predictions)
    if not count:
        raise ValueError("No clean predictions")
    correct = sum(row["prediction"] == row["original_label"] for row in predictions)
    return {"accuracy": correct / count, "num_samples": count, "num_correct": correct}


def paired_metrics(clean, triggered, target_label):
    """ASR and transitions on identical non-target sources; rates are fractions."""
    clean = [row for row in clean if row["original_label"] != target_label]
    triggered = [row for row in triggered if row["original_label"] != target_label]
    before = {row["source_id"]: row for row in clean}
    after = {row["source_id"]: row for row in triggered}
    if len(before) != len(clean) or len(after) != len(triggered):
        raise ValueError("Duplicate source IDs in paired predictions")
    if not before or before.keys() != after.keys():
        raise ValueError("Paired evaluation needs identical nonempty source ID sets")
    pairs = []
    for source_id, row in before.items():
        other = after[source_id]
        if row["original_label"] != other["original_label"]:
            raise ValueError(f"Label mismatch for {source_id}")
        pairs.append((row["prediction"], other["prediction"], row["original_label"]))
    n = len(pairs)
    clean_target = sum(c == target_label for c, t, y in pairs)
    triggered_target = sum(t == target_label for c, t, y in pairs)
    new_target = sum(c != target_label and t == target_label for c, t, y in pairs)
    left_target = sum(c == target_label and t != target_label for c, t, y in pairs)
    clean_correct = sum(c == y for c, t, y in pairs)
    correct_to_target = sum(c == y and t == target_label for c, t, y in pairs)
    return {
        "num_non_target_samples": n,
        "clean_target_count": clean_target,
        "triggered_target_count": triggered_target,
        "clean_non_target_target_rate": clean_target / n,
        "asr": triggered_target / n,
        "same_model_target_rate_lift": (triggered_target - clean_target) / n,
        "new_target_flips": new_target,
        "left_target_flips": left_target,
        "unchanged_target_predictions": clean_target - left_target,
        "new_target_flip_rate": new_target / n,
        "num_clean_correct_non_target": clean_correct,
        "clean_correct_to_target_count": correct_to_target,
        "conditional_asr_clean_correct": correct_to_target / clean_correct if clean_correct else None,
        "triggered_clean_label_accuracy": sum(t == y for c, t, y in pairs) / n,
        "clean_non_target_accuracy": clean_correct / n,
    }


def evaluate_trigger(model, clean_rows, clean_predictions, *, name, config,
                     target_label, images_root, transform, args, device, progress=None):
    # Limit smoke tests to the exact clean sources already evaluated.
    ids = {row["source_id"] for row in clean_predictions
           if row["original_label"] != target_label}
    rows = [{**row, "variant_name": name, "trigger_config": dict(config)}
            for row in clean_rows if str(row["source_id"]) in ids]
    triggered = predict_rows(model, rows, images_root, transform, args, device, progress=progress)
    metrics = paired_metrics(clean_predictions, triggered, target_label)
    metrics["trigger_config"] = dict(config)
    return metrics, triggered
