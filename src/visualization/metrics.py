"""
Automatic Metric Comparison Figures
----------------------------------------
Bar charts comparing SacreBLEU/chrF/COMET (or any single metric) across
experiments (E0-E8) or models, one fixed-order categorical color per
entity -- the standard "how does each experiment score" figure referenced
throughout notebooks 09 and 13.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from src.visualization.palette import get_categorical_color, apply_publication_style

logger = logging.getLogger(__name__)


def plot_metric_comparison(
    scores: pd.Series, metric_name: str = "SacreBLEU", output_path: Optional[Path] = None
) -> plt.Figure:
    """Plot a single metric's score across experiments/models as a bar chart.

    Args:
        scores: Series indexed by entity name (e.g. experiment ID), values
            are the metric score.
        metric_name: Metric label for the y-axis/title.
        output_path: If given, saves the figure to this path (PNG, 150 dpi).

    Returns:
        plt.Figure: The created figure, bars colored in fixed categorical order.
    """
    apply_publication_style()

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(scores)), 5))
    colors = [get_categorical_color(i) for i in range(len(scores))]
    bars = ax.bar(scores.index.astype(str), scores.values, color=colors)

    for bar, value in zip(bars, scores.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} by Experiment")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        logger.info(f"Saved {metric_name} comparison figure to {output_path}.")
    return fig


def plot_multi_metric_comparison(
    metrics_df: pd.DataFrame, output_path: Optional[Path] = None
) -> plt.Figure:
    """Plot several metrics side by side as grouped bars, one subplot per metric.

    Each metric gets its OWN axis (never a shared/dual y-scale), since
    SacreBLEU (0-100), chrF (0-100), and COMET (~0-1) live on different
    scales entirely.

    Args:
        metrics_df: DataFrame indexed by entity (e.g. experiment ID), one
            column per metric.
        output_path: If given, saves the figure to this path (PNG, 150 dpi).

    Returns:
        plt.Figure: The created figure with one subplot per metric column.
    """
    apply_publication_style()

    n_metrics = len(metrics_df.columns)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), squeeze=False)
    colors = [get_categorical_color(i) for i in range(len(metrics_df))]

    for ax, metric_name in zip(axes[0], metrics_df.columns):
        ax.bar(metrics_df.index.astype(str), metrics_df[metric_name].values, color=colors)
        ax.set_title(metric_name)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        logger.info(f"Saved multi-metric comparison figure to {output_path}.")
    return fig
