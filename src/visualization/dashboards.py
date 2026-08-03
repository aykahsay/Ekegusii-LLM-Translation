"""
Project Status Dashboard
-----------------------------
A single multi-panel figure summarizing overall project state -- corpus
domain/source distribution, split sizes, and completed-experiment
scorecard -- for quick visual review during development (e.g. re-run after
each new experiment finishes). Distinct from `publication.py`'s curated,
paper-ready 3-panel figure: this is a broader, less curated "everything at
a glance" view meant for the research team, not the manuscript.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from src.visualization.palette import apply_publication_style, get_categorical_color

logger = logging.getLogger(__name__)


def build_project_status_dashboard(
    source_distribution: pd.DataFrame,
    split_summary: pd.DataFrame,
    completed_experiment_scores: pd.Series,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Build a 3-panel project-status dashboard.

    Panel A: corpus source distribution (pie).
    Panel B: train/val/test split sizes (bar).
    Panel C: metric score for each experiment completed so far (bar; any
             not-yet-run experiment is simply absent, so this panel grows
             as the ablation study progresses).

    Args:
        source_distribution: Output of `CorpusStatistics.source_distribution`
            (index = source name, "count" column).
        split_summary: Output of `CorpusStatistics.split_summary` (index =
            split name, "rows" column).
        completed_experiment_scores: Series indexed by experiment ID for
            whichever experiments have a saved result so far.
        output_path: If given, saves the figure to this path (PNG, 150 dpi).

    Returns:
        plt.Figure: The assembled dashboard figure.
    """
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # Panel A: source distribution pie.
    ax_a = axes[0]
    colors_a = [get_categorical_color(i) for i in range(len(source_distribution))]
    ax_a.pie(
        source_distribution["count"],
        labels=source_distribution.index,
        colors=colors_a,
        autopct="%1.0f%%",
        textprops={"fontsize": 8},
    )
    ax_a.set_title("A. Corpus Source Distribution")

    # Panel B: split sizes.
    ax_b = axes[1]
    colors_b = [get_categorical_color(i) for i in range(len(split_summary))]
    ax_b.bar(split_summary.index.astype(str), split_summary["rows"], color=colors_b)
    ax_b.set_ylabel("Rows")
    ax_b.set_title("B. Train / Val / Test Split Sizes")

    # Panel C: experiments completed so far.
    ax_c = axes[2]
    if len(completed_experiment_scores) > 0:
        colors_c = [get_categorical_color(i) for i in range(len(completed_experiment_scores))]
        ax_c.bar(completed_experiment_scores.index.astype(str), completed_experiment_scores.values, color=colors_c)
        plt.setp(ax_c.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    else:
        ax_c.text(0.5, 0.5, "No experiments completed yet", ha="center", va="center", transform=ax_c.transAxes)
    ax_c.set_title(f"C. Experiments Completed ({len(completed_experiment_scores)}/9)")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        logger.info(f"Saved project status dashboard to {output_path}.")
    return fig
