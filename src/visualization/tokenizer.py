"""
Tokenizer Comparison Figures
--------------------------------
Bar charts for `TokenizerComparator`'s Qwen-vs-Llama fragmentation and
vocabulary-coverage output -- the core figure for notebook 04
(tokenizer_analysis). One categorical color per model (fixed order: Qwen
first, Llama second), grouped by language on a single axis.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.palette import CATEGORICAL_PALETTE, apply_publication_style

logger = logging.getLogger(__name__)

_MODEL_COLORS = {"qwen": CATEGORICAL_PALETTE[0], "llama": CATEGORICAL_PALETTE[1]}


def plot_fertility_comparison(comparison_df: pd.DataFrame, output_path: Optional[Path] = None) -> plt.Figure:
    """Plot mean subword fertility per language, grouped by model.

    Args:
        comparison_df: Output of `TokenizerComparator.compare` (MultiIndexed
            by (model, language), with a "mean_fertility" column).
        output_path: If given, saves the figure to this path (PNG, 150 dpi).

    Returns:
        plt.Figure: The created figure.
    """
    apply_publication_style()

    languages = comparison_df.index.get_level_values("language").unique().tolist()
    models = comparison_df.index.get_level_values("model").unique().tolist()

    x = np.arange(len(languages))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model in enumerate(models):
        values = [comparison_df.loc[(model, lang), "mean_fertility"] for lang in languages]
        ax.bar(x + i * width, values, width, label=model.capitalize(), color=_MODEL_COLORS.get(model, CATEGORICAL_PALETTE[i]))

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(languages)
    ax.set_ylabel("Mean subword fertility (tokens/word)")
    ax.set_title("Tokenizer Fertility: Qwen2.5 vs. Llama-3.1")
    ax.legend(frameon=False)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        logger.info(f"Saved fertility comparison figure to {output_path}.")
    return fig


def plot_vocabulary_coverage(coverage_df: pd.DataFrame, output_path: Optional[Path] = None) -> plt.Figure:
    """Plot type-level vocabulary coverage percentage per language, grouped by model.

    Args:
        coverage_df: Output of `TokenizerComparator.compare_vocabulary_coverage`
            (MultiIndexed by (model, language), with a
            "type_level_coverage_pct" column).
        output_path: If given, saves the figure to this path (PNG, 150 dpi).

    Returns:
        plt.Figure: The created figure.
    """
    apply_publication_style()

    languages = coverage_df.index.get_level_values("language").unique().tolist()
    models = coverage_df.index.get_level_values("model").unique().tolist()

    x = np.arange(len(languages))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model in enumerate(models):
        values = [coverage_df.loc[(model, lang), "type_level_coverage_pct"] for lang in languages]
        ax.bar(x + i * width, values, width, label=model.capitalize(), color=_MODEL_COLORS.get(model, CATEGORICAL_PALETTE[i]))

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(languages)
    ax.set_ylabel("Single-token vocabulary coverage (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Type-Level Vocabulary Coverage: Qwen2.5 vs. Llama-3.1")
    ax.legend(frameon=False)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        logger.info(f"Saved vocabulary coverage figure to {output_path}.")
    return fig
