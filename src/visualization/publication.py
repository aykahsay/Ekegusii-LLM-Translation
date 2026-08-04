"""
Publication-Ready Composite Figures
----------------------------------------
Assembles the individual chart functions (tokenizer, metrics,
resource_contribution) into a single multi-panel figure suitable for
direct inclusion in the paper (`paper/figures/`) -- notebook 13's core
deliverable. Each panel keeps its own single-hue/categorical color
encoding; panels are laid out as a grid, never combined onto one axis.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from src.visualization.palette import apply_publication_style, get_categorical_color
from src.visualization.resource_contribution import compute_marginal_contributions

logger = logging.getLogger(__name__)


def build_main_results_figure(
    fertility_df: pd.DataFrame,
    metric_scores: pd.Series,
    resource_scores: pd.Series,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Build the paper's main 3-panel results figure.

    Panel A: tokenizer fertility by language (Qwen vs Llama).
    Panel B: final metric score by experiment.
    Panel C: resource-contribution waterfall (marginal gains).

    Args:
        fertility_df: Output of `TokenizerComparator.compare` (MultiIndexed
            by (model, language), "mean_fertility" column).
        metric_scores: Series indexed by experiment ID -> final metric score.
        resource_scores: Series indexed by experiment ID (resource-addition
            order) -> metric score, for the waterfall panel.
        output_path: If given, saves the figure to this path (PNG, 300 dpi
            -- publication resolution, higher than the 150 dpi used for
            exploratory notebook figures).

    Returns:
        plt.Figure: The assembled 3-panel figure.
    """
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: tokenizer fertility.
    ax_a = axes[0]
    languages = fertility_df.index.get_level_values("language").unique().tolist()
    models = fertility_df.index.get_level_values("model").unique().tolist()
    x = range(len(languages))
    width = 0.8 / len(models)
    for i, model in enumerate(models):
        values = [fertility_df.loc[(model, lang), "mean_fertility"] for lang in languages]
        ax_a.bar([xi + i * width for xi in x], values, width, label=model.capitalize(), color=get_categorical_color(i))
    ax_a.set_xticks([xi + width * (len(models) - 1) / 2 for xi in x])
    ax_a.set_xticklabels(languages)
    ax_a.set_ylabel("Mean fertility")
    ax_a.set_title("A. Tokenizer Fertility")
    ax_a.legend(frameon=False, fontsize=8)

    # Panel B: metric score by experiment.
    ax_b = axes[1]
    colors_b = [get_categorical_color(i) for i in range(len(metric_scores))]
    ax_b.bar(metric_scores.index.astype(str), metric_scores.values, color=colors_b)
    ax_b.set_title("B. Score by Experiment")
    plt.setp(ax_b.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    # Panel C: resource contribution waterfall (delegates the math, draws inline
    # to keep all three panels on one consistent figure/axes grid).
    ax_c = axes[2]
    contributions = compute_marginal_contributions(resource_scores)
    running = 0.0
    for i, (exp_id, row) in enumerate(contributions.iterrows()):
        delta = row["delta"]
        color = "#898781" if i == 0 else ("#2a78d6" if delta >= 0 else "#e34948")
        bottom = 0 if i == 0 else (running if delta >= 0 else running + delta)
        ax_c.bar(i, abs(delta) if i > 0 else delta, bottom=bottom, color=color, width=0.6)
        running += delta
    ax_c.set_xticks(range(len(contributions)))
    ax_c.set_xticklabels(contributions.index.astype(str), rotation=45, ha="right", fontsize=8)
    ax_c.set_title("C. Resource Contribution")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300)
        logger.info(f"Saved main results figure to {output_path}.")
    return fig
