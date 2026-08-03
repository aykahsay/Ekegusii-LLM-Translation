"""
Direction-Weighted Sampler
------------------------------
A `torch.utils.data.Sampler` that draws batch indices according to
per-row weights (e.g. one weight per translation direction, derived from
`configs/training/multilingual.yaml` via `ResourceScheduler`) rather than
uniform random sampling -- used when a single combined dataset (built by
`MultilingualPairMixer`) still needs its low-resource directions
up-sampled during actual DataLoader iteration.
"""

import logging
from typing import Iterator, List, Sequence

import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class WeightedDirectionSampler(Sampler[int]):
    """Samples dataset indices with replacement according to per-row weights."""

    def __init__(self, weights: Sequence[float], num_samples: int, seed: int = 42) -> None:
        """Initialize the sampler.

        Args:
            weights: Per-row sampling weight, same length and order as the
                underlying dataset (e.g. derived from each row's
                `task_type`/direction via a lookup table).
            num_samples: Total number of indices to draw per epoch
                (typically `len(dataset)`, but can be set higher to
                effectively over-sample low-resource directions per epoch).
            seed: Random seed for reproducible sampling.

        Raises:
            ValueError: If `weights` is empty or all weights are <= 0, or
                `num_samples` <= 0.
        """
        if len(weights) == 0:
            raise ValueError("weights must not be empty.")
        if sum(w for w in weights if w > 0) <= 0:
            raise ValueError("At least one weight must be positive.")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}.")

        self.weights = torch.as_tensor(list(weights), dtype=torch.double)
        self.num_samples = num_samples
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def __iter__(self) -> Iterator[int]:
        """Yield `num_samples` indices drawn with replacement per `self.weights`."""
        indices = torch.multinomial(self.weights, self.num_samples, replacement=True, generator=self.generator)
        yield from indices.tolist()

    def __len__(self) -> int:
        """Number of indices yielded per epoch."""
        return self.num_samples

    @staticmethod
    def weights_from_task_types(task_types: Sequence[str], direction_weights: dict) -> List[float]:
        """Build a per-row weight list from each row's task_type and a direction-weight map.

        Args:
            task_types: Per-row task_type strings (e.g. "ENG_to_EKE"),
                matching keys used in `InstructionTaskGenerator.TASK_PROMPTS`.
            direction_weights: Mapping of lowercase direction key (e.g.
                "eng_to_eke") to a weight, as in
                `configs/training/multilingual.yaml`'s `direction_weights`.

        Returns:
            List[float]: Per-row weight, defaulting to 1.0 for any
                task_type not found in `direction_weights`.
        """
        weights = []
        for task_type in task_types:
            key = task_type.lower()
            weights.append(float(direction_weights.get(key, 1.0)))
        return weights
