"""
Generic Numeric Aggregation Utilities
---------------------------------------
Statistics helpers shared by training-curve visualization and experiment
result aggregation: bootstrap confidence intervals, exponential moving
averages for noisy training-loss curves, and multi-run aggregation.

This module intentionally does NOT contain translation-quality metrics
(BLEU/chrF/COMET) -- those live in `src/evaluation/`. This module holds
only generic numeric utilities that operate on arbitrary float sequences.
"""

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    """Compute the mean and (sample) standard deviation of a sequence.

    Args:
        values: Non-empty sequence of numeric values.

    Returns:
        Tuple[float, float]: (mean, standard deviation). Standard deviation
            is 0.0 if fewer than 2 values are provided.

    Raises:
        ValueError: If `values` is empty.
    """
    if len(values) == 0:
        raise ValueError("Cannot compute mean/std of an empty sequence.")
    arr = np.asarray(values, dtype=float)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return float(arr.mean()), std


def bootstrap_confidence_interval(
    values: Sequence[float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute a bootstrap confidence interval for the mean of `values`.

    Used to attach uncertainty estimates to per-sentence metric scores
    (e.g. COMET or chrF scores across a test set) when comparing E0-E8
    experiments, since a single point estimate can hide overlapping
    confidence intervals between two resource configurations.

    Args:
        values: Per-sample metric scores (e.g. one COMET score per sentence).
        n_resamples: Number of bootstrap resamples to draw.
        confidence: Confidence level for the interval (e.g. 0.95 for 95%).
        seed: Random seed for reproducible resampling.

    Returns:
        Tuple[float, float, float]: (point_estimate, lower_bound, upper_bound).

    Raises:
        ValueError: If `values` is empty or `confidence` is not in (0, 1).
    """
    if len(values) == 0:
        raise ValueError("Cannot bootstrap an empty sequence.")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")

    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    point_estimate = float(arr.mean())

    resample_means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        resample_means[i] = sample.mean()

    alpha = 1.0 - confidence
    lower = float(np.percentile(resample_means, 100 * (alpha / 2)))
    upper = float(np.percentile(resample_means, 100 * (1 - alpha / 2)))
    return point_estimate, lower, upper


def exponential_moving_average(values: Sequence[float], alpha: float = 0.1) -> List[float]:
    """Smooth a noisy sequence (e.g. per-step training loss) via EMA.

    Args:
        values: Sequence of raw values in chronological order.
        alpha: Smoothing factor in (0, 1]; higher weights recent values more.

    Returns:
        List[float]: Smoothed sequence, same length as `values`.

    Raises:
        ValueError: If `alpha` is not in (0, 1].
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}.")
    if len(values) == 0:
        return []

    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return smoothed


def aggregate_runs(run_scores: Dict[str, Sequence[float]]) -> Dict[str, Dict[str, float]]:
    """Aggregate per-sample scores from multiple named runs into summary stats.

    Args:
        run_scores: Mapping of run/experiment name (e.g. "E1_English_Ekegusii")
            to a sequence of per-sample metric scores.

    Returns:
        Dict[str, Dict[str, float]]: For each run name, a dict with keys
            "mean", "std", "min", "max".
    """
    summary: Dict[str, Dict[str, float]] = {}
    for run_name, scores in run_scores.items():
        if len(scores) == 0:
            logger.warning(f"Run '{run_name}' has no scores; skipping.")
            continue
        mean, std = mean_std(scores)
        arr = np.asarray(scores, dtype=float)
        summary[run_name] = {
            "mean": mean,
            "std": std,
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return summary
