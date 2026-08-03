"""
Unit Tests for Resource Scheduler
--------------------------------------
"""

import unittest

from omegaconf import OmegaConf

from src.master_corpus.scheduler import ResourceScheduler


class TestResourceScheduler(unittest.TestCase):
    """Test resource-weighted sampling logic."""

    def setUp(self) -> None:
        """Set up a scheduler with known weights for deterministic assertions."""
        cfg = OmegaConf.create(
            {
                "direction_weights": {"eng_to_eke": 2.0, "swa_to_eke": 1.0},
                "sampling_temperature": 1.0,
                "lexical_augmentation": {"enabled": False, "mix_ratio": 0.1},
            }
        )
        self.scheduler = ResourceScheduler(cfg, seed=42)

    def test_sampling_probs_sum_to_one(self) -> None:
        """Computed direction sampling probabilities must sum to 1.0."""
        probs = self.scheduler.compute_direction_sampling_probs({"eng_to_eke": 100, "swa_to_eke": 100})
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)

    def test_higher_weight_gets_higher_probability(self) -> None:
        """A direction with 2x the configured weight should get a higher share
        of the batch even with identical raw row counts."""
        probs = self.scheduler.compute_direction_sampling_probs({"eng_to_eke": 100, "swa_to_eke": 100})
        self.assertGreater(probs["eng_to_eke"], probs["swa_to_eke"])

    def test_batch_plan_sums_to_batch_size(self) -> None:
        """The integer per-direction quota must sum to the requested batch size."""
        plan = self.scheduler.build_mixed_batch_plan({"eng_to_eke": 100, "swa_to_eke": 100}, batch_size=32)
        self.assertEqual(sum(plan.values()), 32)

    def test_empty_direction_counts_raises(self) -> None:
        """An empty direction_counts dict must raise, not silently return nothing."""
        with self.assertRaises(ValueError):
            self.scheduler.compute_direction_sampling_probs({})


if __name__ == "__main__":
    unittest.main()
