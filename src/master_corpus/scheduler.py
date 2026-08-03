"""
Dynamic Resource Scheduler
-----------------------------
Combines multiple resources (bilingual pairs, trilingual triplets, the
lexical corpus) into a single weighted training mixture per
`configs/training/multilingual.yaml`'s `direction_weights` and
`sampling_temperature`, and optionally interleaves lexical-augmentation
tasks at a configured mix ratio. Backs notebook 06 (dataset_scheduler) and
experiments E5 (Full Resources) onward, where training draws from more than
one resource type simultaneously.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class ResourceScheduler:
    """Builds a single weighted training mixture from multiple task pools."""

    def __init__(self, multilingual_cfg: DictConfig, seed: int = 42) -> None:
        """Initialize the scheduler.

        Args:
            multilingual_cfg: Parsed `configs/training/multilingual.yaml`
                (must provide `direction_weights` and `sampling_temperature`).
            seed: Random seed for reproducible resampling.
        """
        self.cfg = multilingual_cfg
        self.rng = np.random.default_rng(seed)

    def compute_direction_sampling_probs(self, direction_counts: Dict[str, int]) -> Dict[str, float]:
        """Compute per-direction sampling probabilities from raw counts and config weights.

        Combines each direction's natural row count with its configured
        weight, then applies temperature smoothing: `sampling_temperature`
        > 1 flattens the distribution (up-sampling low-resource
        directions), while 1.0 samples proportionally to the raw
        weighted counts.

        Args:
            direction_counts: Mapping of task_type (e.g. "ENG_to_EKE") to
                the number of available rows for that direction.

        Returns:
            Dict[str, float]: Normalized sampling probability per direction,
                summing to 1.0.

        Raises:
            ValueError: If `direction_counts` is empty.
        """
        if not direction_counts:
            raise ValueError("direction_counts must not be empty.")

        weights = self.cfg.get("direction_weights", {})
        temperature = float(self.cfg.get("sampling_temperature", 1.0))

        weighted_counts = {}
        for direction, count in direction_counts.items():
            weight = weights.get(direction.lower(), 1.0)
            weighted_counts[direction] = max(count, 1) ** (1.0 / temperature) * weight

        total = sum(weighted_counts.values())
        probs = {direction: value / total for direction, value in weighted_counts.items()}
        logger.info(f"Computed sampling probabilities (temperature={temperature}): {probs}")
        return probs

    def build_mixed_batch_plan(self, direction_counts: Dict[str, int], batch_size: int) -> Dict[str, int]:
        """Convert sampling probabilities into an integer per-direction quota for one batch.

        Args:
            direction_counts: Mapping of task_type to available row count.
            batch_size: Total number of examples in one training batch.

        Returns:
            Dict[str, int]: Number of examples to draw from each direction
                in this batch, summing to `batch_size` (up to rounding).
        """
        probs = self.compute_direction_sampling_probs(direction_counts)
        directions = list(probs.keys())
        raw_counts = self.rng.multinomial(batch_size, list(probs.values()))
        return dict(zip(directions, (int(c) for c in raw_counts)))

    def mix_with_lexical(self, sentence_tasks: pd.DataFrame, lexical_tasks: pd.DataFrame) -> pd.DataFrame:
        """Interleave lexical-augmentation tasks into sentence-level tasks.

        Args:
            sentence_tasks: DataFrame of sentence-level instruction tasks
                (e.g. output of InstructionTaskGenerator).
            lexical_tasks: DataFrame of lexical instruction tasks (same
                schema: at minimum "prompt" and "response" columns).

        Returns:
            pd.DataFrame: Combined, shuffled DataFrame. If
                `lexical_augmentation.enabled` is False in the config, or
                `lexical_tasks` is empty, `sentence_tasks` is returned
                shuffled but otherwise unmodified.
        """
        lexical_cfg = self.cfg.get("lexical_augmentation", {})
        if not lexical_cfg.get("enabled", False) or len(lexical_tasks) == 0:
            return sentence_tasks.sample(frac=1.0, random_state=self.rng.integers(0, 2**31)).reset_index(drop=True)

        mix_ratio = float(lexical_cfg.get("mix_ratio", 0.1))
        n_lexical = int(len(sentence_tasks) * mix_ratio / (1 - mix_ratio))
        n_lexical = min(n_lexical, len(lexical_tasks))

        lexical_sample = lexical_tasks.sample(n=n_lexical, random_state=42)
        combined = pd.concat([sentence_tasks, lexical_sample], ignore_index=True)
        shuffled = combined.sample(frac=1.0, random_state=self.rng.integers(0, 2**31)).reset_index(drop=True)

        logger.info(
            f"Mixed {len(sentence_tasks):,} sentence tasks with {n_lexical:,} lexical tasks "
            f"(target ratio={mix_ratio:.2%})."
        )
        return shuffled
