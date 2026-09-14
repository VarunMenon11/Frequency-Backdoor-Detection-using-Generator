import unittest

from evaluation.asb_classifier import attack_checkpoint_score


def metric(conditional, lift):
    return {
        "conditional_asr_clean_correct": conditional,
        "same_model_target_rate_lift": lift,
    }


class AttackCheckpointSelectionTests(unittest.TestCase):
    def test_requires_clean_floor(self):
        self.assertIsNone(
            attack_checkpoint_score({"t": metric(0.9, 0.8)}, 0.54, 0.55)
        )

    def test_uses_conditional_asr_before_lift(self):
        lower = attack_checkpoint_score({"t": metric(0.70, 0.60)}, 0.60, 0.55)
        higher = attack_checkpoint_score({"t": metric(0.80, 0.20)}, 0.58, 0.55)
        self.assertGreater(higher, lower)

    def test_ties_on_lift_then_clean_accuracy(self):
        first = attack_checkpoint_score({"t": metric(0.80, 0.20)}, 0.58, 0.55)
        second = attack_checkpoint_score({"t": metric(0.80, 0.30)}, 0.56, 0.55)
        third = attack_checkpoint_score({"t": metric(0.80, 0.30)}, 0.60, 0.55)
        self.assertGreater(second, first)
        self.assertGreater(third, second)


if __name__ == "__main__":
    unittest.main()
