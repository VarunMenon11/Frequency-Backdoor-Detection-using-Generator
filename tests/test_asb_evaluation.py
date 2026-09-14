"""Run with python -m unittest discover -s tests -v."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from argparse import Namespace
from unittest.mock import patch

import torch

from evaluation.asb_classifier import (
    apply_evaluation_configs, checkpoint_selection_score, evaluate_trigger,
    paired_metrics, resolve_evaluation_config,
)
from scripts.train_asb_dtd_pretrained_classifier import evaluate_final_test, save_prediction_panel


def prediction(source, label, value):
    return {"source_id": str(source), "original_label": label, "prediction": value}


class PairedMetricTests(unittest.TestCase):
    def test_bias_is_not_trigger_activation(self):
        rows = [prediction(i, 1, 0) for i in range(10)]
        result = paired_metrics(rows, rows, 0)
        self.assertEqual(result["asr"], 1)
        self.assertEqual(result["same_model_target_rate_lift"], 0)
        self.assertEqual(result["new_target_flips"], 0)
        self.assertIsNone(result["conditional_asr_clean_correct"])

    def test_both_flip_directions_and_conditional_denominator(self):
        clean = [prediction(0, 1, 1), prediction(1, 1, 0),
                 prediction(2, 2, 2), prediction(3, 2, 0), prediction(4, 0, 0)]
        triggered = [prediction(3, 2, 0), prediction(2, 2, 2),
                     prediction(1, 1, 2), prediction(0, 1, 0), prediction(4, 0, 0)]
        result = paired_metrics(clean, triggered, 0)
        self.assertEqual(result["num_non_target_samples"], 4)
        self.assertEqual(result["asr"], .5)
        self.assertEqual(result["new_target_flips"], 1)
        self.assertEqual(result["left_target_flips"], 1)
        self.assertEqual(result["same_model_target_rate_lift"], 0)
        self.assertEqual(result["conditional_asr_clean_correct"], .5)
        self.assertEqual(result["triggered_clean_label_accuracy"], .25)

    def test_misaligned_sources_and_labels_rejected(self):
        clean = [prediction(1, 1, 1)]
        for bad in ([prediction(2, 1, 0)], [prediction(1, 2, 0)], clean + clean):
            with self.assertRaises(ValueError):
                paired_metrics(clean, bad, 0)

    def test_empty_denominator_rejected(self):
        with self.assertRaises(ValueError):
            paired_metrics([prediction(1, 0, 0)], [prediction(1, 0, 0)], 0)

    def test_selection_cannot_reward_high_asr_target_collapse(self):
        self.assertGreater(checkpoint_selection_score(.61), checkpoint_selection_score(.46))
        self.assertEqual(checkpoint_selection_score(.65), .65)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"source_id": "a", "original_label": 1, "variant_name": "fourier_middle",
             "protocol_role": "final_test_triggered", "trigger_config": {"strength": .2}},
            {"source_id": "a", "original_label": 1, "variant_name": "haar_hh",
             "protocol_role": "final_test_triggered", "trigger_config": {"strength": .25}},
        ]
        self.configs = {"fourier_middle": {"strength": .6}}

    def test_override_used_for_seen_but_not_heldout_and_not_mutated(self):
        updated = apply_evaluation_configs(self.rows, self.configs)
        self.assertEqual(updated[0]["trigger_config"]["strength"], .6)
        self.assertEqual(updated[1]["trigger_config"]["strength"], .25)
        self.assertEqual(self.rows[0]["trigger_config"]["strength"], .2)
        self.assertEqual(resolve_evaluation_config(self.rows, "fourier_middle", self.configs)["strength"], .6)
        self.assertEqual(resolve_evaluation_config(self.rows, "haar_hh", self.configs)["strength"], .25)

    def test_missing_trigger_rejected(self):
        with self.assertRaises(ValueError):
            resolve_evaluation_config(self.rows, "missing", {})

    def test_partial_evaluation_uses_identical_clean_sources(self):
        clean_rows = [{"source_id": "a", "original_label": 1},
                      {"source_id": "b", "original_label": 1}]
        clean = [prediction("a", 1, 1)]
        with patch("evaluation.asb_classifier.predict_rows", return_value=[prediction("a", 1, 0)]) as predict:
            result, _ = evaluate_trigger(
                None, clean_rows, clean, name="fourier_middle", config={"strength": .6},
                target_label=0, images_root=None, transform=None, args=None, device=None,
            )
        evaluated_rows = predict.call_args.args[1]
        self.assertEqual(len(evaluated_rows), 1)
        self.assertEqual(evaluated_rows[0]["trigger_config"]["strength"], .6)
        self.assertEqual(result["new_target_flips"], 1)

    def test_final_evaluation_propagates_override_and_handles_clean_control(self):
        class Model:
            def metadata(self):
                return {}
        clean = [prediction("a", 1, 1)]
        for configs in (self.configs, {}):
            with patch("scripts.train_asb_dtd_pretrained_classifier.predict_rows", return_value=clean), patch(
                "scripts.train_asb_dtd_pretrained_classifier.evaluate_trigger",
                return_value=({"asr": .0, "triggered_clean_label_accuracy": 1}, []),
            ) as trigger:
                summary = evaluate_final_test(
                    model=Model(), rows=self.rows, images_root=None, transform=None,
                    args=Namespace(max_eval_batches=None), device=torch.device("cpu"),
                    target_label=0, target_class="banded", test_clean_rows=[],
                    test_trigger_names=["fourier_middle"], attack_trigger_names=list(configs),
                    attack_trigger_configs=configs, best_epoch=1, best_score=.6,
                )
            self.assertEqual(trigger.call_args.kwargs["config"]["strength"], .6 if configs else .2)
            self.assertEqual(summary["clean_test"]["accuracy"], 1)
            if not configs:
                self.assertIsNone(summary["mean_seen_trigger_asr"])

    def test_prediction_panel_uses_same_override(self):
        import matplotlib
        matplotlib.use("Agg")
        clean = {"source_id": "a", "original_label": 1, "class_name": "blotchy",
                 "variant_name": "clean", "protocol_role": "final_test_clean"}
        rows = [clean] + [
            {**clean, "variant_name": name, "protocol_role": "final_test_triggered",
             "trigger_config": {"strength": .2}}
            for name in ("fourier_middle", "fourier_low", "haar_hh")
        ]
        captured = []
        def dataset(root, selected, **kwargs):
            captured.append(selected[0])
            return [(torch.zeros(3, 8, 8), torch.tensor(1))]
        class Model:
            def eval(self):
                return self
            def __call__(self, images):
                return torch.tensor([[0., 1.]])
        with TemporaryDirectory() as directory, patch(
            "scripts.train_asb_dtd_pretrained_classifier.ASBManifestDataset", side_effect=dataset
        ):
            save_prediction_panel(
                Model(), rows, None, None, ["banded", "blotchy"], 0, 1,
                torch.device("cpu"), Path(directory), attack_trigger_configs=self.configs,
            )
        self.assertEqual(captured[1]["trigger_config"]["strength"], .6)
        self.assertEqual(captured[2]["trigger_config"]["strength"], .2)


if __name__ == "__main__":
    unittest.main()
