"""
Statistical Significance Testing
-------------------------------------
Paired bootstrap resampling significance test between two experiments'
per-sentence metric scores (e.g. COMET or chrF per test-set sentence).
Required before claiming in the paper that one resource configuration
(e.g. E4 Trilingual) "outperforms" another (e.g. E1 Bilingual) -- a
difference in mean scores alone does not establish significance,
especially with a ~4,928-sentence test set where per-sentence variance is
non-trivial.
"""

import logging
from typing import Any, Dict, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class PairedBootstrapTest:
    """Paired bootstrap significance test for comparing two systems' per-sentence scores."""

    def __init__(self, n_resamples: int = 10000, seed: int = 42) -> None:
        """Initialize the test.

        Args:
            n_resamples: Number of bootstrap resamples to draw.
            seed: Random seed for reproducible resampling.
        """
        self.n_resamples = n_resamples
        self.rng = np.random.default_rng(seed)

    def run(
        self, scores_a: Sequence[float], scores_b: Sequence[float], system_a_name: str = "A", system_b_name: str = "B"
    ) -> Dict[str, Any]:
        """Test whether system A's mean score is significantly higher than system B's.

        Both score sequences MUST be per-sentence scores computed on the
        SAME test set in the SAME sentence order (i.e. paired), e.g. one
        COMET score per test-set sentence for each system.

        Args:
            scores_a: Per-sentence scores for system A (e.g. the proposed/
                higher-resource experiment).
            scores_b: Per-sentence scores for system B (e.g. the baseline).
            system_a_name: Label for system A in the returned result.
            system_b_name: Label for system B in the returned result.

        Returns:
            Dict[str, object]: Keys "mean_diff" (A - B), "p_value" (one-sided,
                probability that a bootstrap resample shows B >= A when A's
                true mean is higher), "significant_at_0.05" (bool),
                "system_a", "system_b", "n_sentences".

        Raises:
            ValueError: If `scores_a` and `scores_b` have different lengths
                or are empty.
        """
        if len(scores_a) != len(scores_b):
            raise ValueError(f"len(scores_a)={len(scores_a)} must match len(scores_b)={len(scores_b)} (paired test).")
        if len(scores_a) == 0:
            raise ValueError("scores_a/scores_b must not be empty.")

        arr_a = np.asarray(scores_a, dtype=float)
        arr_b = np.asarray(scores_b, dtype=float)
        n = len(arr_a)

        observed_diff = float(arr_a.mean() - arr_b.mean())

        count_b_greater_equal = 0
        for _ in range(self.n_resamples):
            indices = self.rng.integers(0, n, size=n)
            resample_diff = arr_a[indices].mean() - arr_b[indices].mean()
            if resample_diff <= 0:
                count_b_greater_equal += 1

        p_value = count_b_greater_equal / self.n_resamples

        result = {
            "system_a": system_a_name,
            "system_b": system_b_name,
            "n_sentences": n,
            "mean_a": round(float(arr_a.mean()), 4),
            "mean_b": round(float(arr_b.mean()), 4),
            "mean_diff": round(observed_diff, 4),
            "p_value": round(p_value, 4),
            "significant_at_0.05": p_value < 0.05,
        }
        logger.info(
            f"Bootstrap test [{system_a_name} vs {system_b_name}]: "
            f"diff={observed_diff:.4f}, p={p_value:.4f}, significant={result['significant_at_0.05']}"
        )
        return result
