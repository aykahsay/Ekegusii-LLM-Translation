"""
Training Learning-Curve Figures
------------------------------------
Plots smoothed training/validation loss over steps for one or more
experiment runs, on a SINGLE axis (never dual-axis -- if loss and a metric
like COMET need to appear together, that's two separate figures, indexed
if needed, not one chart with two y-scales).
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt

from src.utils.metrics import exponential_moving_average
from src.visualization.palette import get_categorical_color, apply_publication_style

logger = logging.getLogger(__name__)


def plot_loss_curve(
    steps: Sequence[int], loss_values: Sequence[float], smoothing_alpha: float = 0.1, output_path: Optional[Path] = None
) -> plt.Figure:
    """Plot a single run's raw (faint) and smoothed (bold) loss curve.

    Args:
        steps: Global training step at each logged point.
        loss_values: Raw loss value at each step.
        smoothing_alpha: EMA smoothing factor (see
            `src.utils.metrics.exponential_moving_average`).
        output_path: If given, saves the figure to this path (PNG, 150 dpi).

    Returns:
        plt.Figure: The created figure.
    """
    apply_publication_style()
    smoothed = exponential_moving_average(list(loss_values), alpha=smoothing_alpha)
    color = get_categorical_color(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, loss_values, color=color, alpha=0.25, linewidth=1)
    ax.plot(steps, smoothed, color=color, linewidth=2, label="Training loss (smoothed)")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve")
    ax.legend(frameon=False)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        logger.info(f"Saved loss curve figure to {output_path}.")
    return fig


def plot_multi_run_comparison(
    runs: Dict[str, Dict[str, Sequence[float]]], smoothing_alpha: float = 0.1, output_path: Optional[Path] = None
) -> plt.Figure:
    """Compare smoothed loss curves across multiple experiment runs.

    Args:
        runs: Mapping of run/experiment label (e.g. "E1_English_Ekegusii")
            to {"steps": [...], "loss": [...]}.
        smoothing_alpha: EMA smoothing factor applied to each run independently.
        output_path: If given, saves the figure to this path (PNG, 150 dpi).

    Returns:
        plt.Figure: The created figure, one fixed-color line per run.
    """
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, (run_name, data) in enumerate(runs.items()):
        smoothed = exponential_moving_average(list(data["loss"]), alpha=smoothing_alpha)
        ax.plot(data["steps"], smoothed, color=get_categorical_color(i), linewidth=2, label=run_name)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss (EMA smoothed)")
    ax.set_title("Training Loss Across Experiments")
    ax.legend(frameon=False, ncol=2, fontsize=9)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        logger.info(f"Saved multi-run comparison figure to {output_path}.")
    return fig
