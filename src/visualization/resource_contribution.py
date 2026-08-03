"""
Resource Contribution Waterfall Chart
------------------------------------------
The headline ablation figure: shows the MARGINAL score change contributed
by each successively-added resource (E0 -> E1 -> ... -> E8), as a waterfall
of deltas rather than a bar chart of absolute scores -- directly answers
"how much did adding trilingual data / lexical augmentation / curriculum
learning actually help." Uses the diverging blue/red pair: blue for gains,
red for losses (a later resource occasionally hurting a metric is a
legitimate and interesting finding, not something to hide).
"""

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from src.visualization.palette import apply_publication_style

logger = logging.getLogger(__name__)

_GAIN_COLOR = "#2a78d6"
_LOSS_COLOR = "#e34948"


def plot_resource_waterfall(
    ordered_scores: pd.Series, metric_name: str = "SacreBLEU", output_path: Optional[Path] = None
) -> plt.Figure:
    """Plot a waterfall of marginal score deltas across successive experiments.

    Args:
        ordered_scores: Series indexed by experiment ID, in the intended
            resource-addition order (e.g. E0, E1, E3, E4, E5, E6, E7),
            values are that experiment's metric score.
        metric_name: Metric label for the y-axis/title.
        output_path: If given, saves the figure to this path (PNG, 150 dpi).

    Returns:
        plt.Figure: The created figure. The first bar shows E0's absolute
            score (baseline); each subsequent bar shows the delta from the
            previous experiment, colored blue (gain) or red (loss), floating
            at its correct cumulative height.

    Raises:
        ValueError: If `ordered_scores` has fewer than 2 entries.
    """
    if len(ordered_scores) < 2:
        raise ValueError("Need at least 2 experiments (a baseline plus one addition) to plot a waterfall.")

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(ordered_scores)), 5.5))

    labels: List[str] = [str(label) for label in ordered_scores.index.tolist()]
    values = ordered_scores.values.tolist()
    deltas = [values[0]] + [values[i] - values[i - 1] for i in range(1, len(values))]

    running_total = 0.0
    for i, (label, delta) in enumerate(zip(labels, deltas)):
        color = "#898781" if i == 0 else (_GAIN_COLOR if delta >= 0 else _LOSS_COLOR)
        bottom = running_total if delta >= 0 or i == 0 else running_total + delta
        ax.bar(i, abs(delta) if i > 0 else delta, bottom=bottom if i > 0 else 0, color=color, width=0.6)

        label_y = running_total + delta if i > 0 else delta
        sign = "+" if delta > 0 and i > 0 else ""
        ax.text(i, label_y, f"{sign}{delta:.1f}", ha="center", va="bottom" if delta >= 0 else "top", fontsize=9)

        running_total += delta

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Resource Contribution Waterfall: {metric_name}")
    ax.axhline(0, color="#c3c2b7", linewidth=0.8)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        logger.info(f"Saved resource contribution waterfall to {output_path}.")
    return fig


def compute_marginal_contributions(ordered_scores: pd.Series) -> pd.DataFrame:
    """Compute the marginal delta each successive experiment contributes.

    Args:
        ordered_scores: Series indexed by experiment ID, in resource-
            addition order.

    Returns:
        pd.DataFrame: Columns "score", "delta", "pct_of_total_gain" --
            useful for a companion results table alongside the waterfall figure.
    """
    values: List[float] = ordered_scores.tolist()
    deltas = [values[0]] + [values[i] - values[i - 1] for i in range(1, len(values))]
    total_gain = values[-1] - values[0]

    result = pd.DataFrame(
        {
            "score": values,
            "delta": deltas,
            "pct_of_total_gain": [
                round(100 * d / total_gain, 1) if total_gain != 0 and i > 0 else 0.0 for i, d in enumerate(deltas)
            ],
        },
        index=ordered_scores.index,
    )
    return result
