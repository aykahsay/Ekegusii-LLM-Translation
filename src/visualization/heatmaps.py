"""
Sequential Heatmap Figures
------------------------------
Single-hue sequential heatmaps (never a rainbow colormap) for magnitude
encoding across two categorical dimensions -- e.g. metric score per
(experiment, language-direction) cell, or per-term terminology accuracy.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from src.visualization.palette import SEQUENTIAL_BLUE_CMAP, apply_publication_style

logger = logging.getLogger(__name__)


def plot_metric_heatmap(
    matrix_df: pd.DataFrame,
    value_label: str = "Score",
    title: str = "Metric Heatmap",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot a 2D matrix (e.g. experiment x direction) as a sequential-hue heatmap.

    Args:
        matrix_df: DataFrame where the index is one categorical dimension
            (e.g. experiment ID) and columns are the other (e.g.
            translation direction), with numeric cell values.
        value_label: Label for the colorbar.
        title: Figure title.
        output_path: If given, saves the figure to this path (PNG, 150 dpi).

    Returns:
        plt.Figure: The created figure, with cell values annotated directly
            (selective direct labeling, since a full legend would be
            redundant with the colorbar).
    """
    apply_publication_style()

    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(matrix_df.columns)), max(4, 0.6 * len(matrix_df.index))))
    im = ax.imshow(matrix_df.values, cmap=SEQUENTIAL_BLUE_CMAP, aspect="auto")

    ax.set_xticks(range(len(matrix_df.columns)))
    ax.set_xticklabels(matrix_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix_df.index)))
    ax.set_yticklabels(matrix_df.index)

    vmax = matrix_df.values.max() if matrix_df.size else 1.0
    for i in range(len(matrix_df.index)):
        for j in range(len(matrix_df.columns)):
            value = matrix_df.values[i, j]
            text_color = "white" if value > 0.6 * vmax else "#0b0b0b"
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", color=text_color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_label)
    ax.set_title(title)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        logger.info(f"Saved heatmap figure to {output_path}.")
    return fig
