"""
Deterministic Corpus Sampling
--------------------------------
Reusable, seeded sampling utilities for constructing human-evaluation
subsets, few-shot example pools, and ablation-study subsamples from the
master corpus -- all without ever touching the frozen train/val/test split
assignment (samples should be drawn from within a single already-resolved
split, typically "test" for human eval or "train" for few-shot examples).
"""

import logging
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class CorpusSampler:
    """Deterministic sampling operations over a single corpus DataFrame."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize the sampler.

        Args:
            seed: Random seed applied to every sampling operation for
                reproducibility across runs.
        """
        self.seed = seed

    def sample_n(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Draw a deterministic random sample of `n` rows.

        Args:
            df: Source DataFrame to sample from.
            n: Number of rows to sample. Clamped to `len(df)` if larger.

        Returns:
            pd.DataFrame: Sampled rows with a reset index.
        """
        n = min(n, len(df))
        sample = df.sample(n=n, random_state=self.seed).reset_index(drop=True)
        logger.info(f"Sampled {n:,} of {len(df):,} rows.")
        return sample

    def stratified_sample(
        self, df: pd.DataFrame, stratify_column: str, n_per_stratum: int
    ) -> pd.DataFrame:
        """Draw a deterministic sample with a fixed count per category value.

        Useful for building a balanced human-evaluation set across
        `dataset_origin` or `source` values rather than an unweighted
        random sample that could be dominated by the largest source.

        Args:
            df: Source DataFrame to sample from.
            stratify_column: Categorical column to stratify by.
            n_per_stratum: Maximum rows to draw per distinct category value
                (fewer are drawn if a category has fewer available rows).

        Returns:
            pd.DataFrame: Concatenated stratified sample with a reset index.

        Raises:
            KeyError: If `stratify_column` is not present in `df`.
        """
        if stratify_column not in df.columns:
            raise KeyError(f"Column '{stratify_column}' not found in DataFrame.")

        frames: List[pd.DataFrame] = []
        for _, group in df.groupby(stratify_column):
            group_df: pd.DataFrame = group  # groupby over a DataFrame always yields DataFrame groups
            frames.append(self.sample_n(group_df, n_per_stratum))

        combined = pd.concat(frames, ignore_index=True) if frames else df.iloc[0:0].copy()
        logger.info(
            f"Stratified sample by '{stratify_column}': {len(combined):,} rows across "
            f"{df[stratify_column].nunique()} strata."
        )
        return combined

    def few_shot_examples(
        self, df: pd.DataFrame, k: int, exclude_ids: Optional[List[int]] = None, id_column: str = "concept_id"
    ) -> pd.DataFrame:
        """Select `k` few-shot in-context examples, excluding given IDs.

        Args:
            df: Source DataFrame to draw examples from (typically the
                training split).
            k: Number of examples to select.
            exclude_ids: Concept IDs to exclude (e.g. the current
                evaluation batch, to avoid the model seeing its own answer
                as a few-shot example).
            id_column: Name of the unique identifier column.

        Returns:
            pd.DataFrame: `k` selected rows with a reset index.
        """
        pool = df
        if exclude_ids:
            pool = df[~df[id_column].isin(exclude_ids)]
        return self.sample_n(pool, k)
