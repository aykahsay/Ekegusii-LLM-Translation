"""
Curriculum Learning Scheduler
--------------------------------
Orders training data into progressive stages for E7 (Curriculum Learning):
by default, sentence difficulty (short/common-vocabulary examples first,
long/rare-vocabulary examples later), with an alternate resource-based
staging mode (bilingual -> trilingual -> lexical) also provided since both
are common curriculum strategies for low-resource NMT.
"""

import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


class DifficultyCurriculum:
    """Stages examples from easiest to hardest based on sentence length and rarity."""

    def __init__(self, num_stages: int = 4) -> None:
        """Initialize the curriculum.

        Args:
            num_stages: Number of progressive difficulty buckets to create.

        Raises:
            ValueError: If `num_stages` < 1.
        """
        if num_stages < 1:
            raise ValueError(f"num_stages must be >= 1, got {num_stages}.")
        self.num_stages = num_stages

    def compute_difficulty_score(self, df: pd.DataFrame, text_column: str) -> pd.Series:
        """Score each row by a simple length + rare-word difficulty heuristic.

        Args:
            df: DataFrame containing the text to score.
            text_column: Column to compute difficulty from (typically the
                source language column for a given translation direction).

        Returns:
            pd.Series: Difficulty score per row (higher = harder). Combines
                normalized word count with the fraction of words that are
                rare within this DataFrame (appearing at or below the
                10th-percentile frequency).
        """
        texts = df[text_column].fillna("").astype(str)
        word_lists = texts.str.split()
        word_counts = word_lists.apply(len)

        all_words: List[str] = [w.lower() for words in word_lists for w in words]
        freq = pd.Series(all_words).value_counts()
        rare_threshold = freq.quantile(0.10) if len(freq) > 0 else 0

        def rare_word_fraction(words: List[str]) -> float:
            if not words:
                return 0.0
            rare = sum(1 for w in words if freq.get(w.lower(), 0) <= rare_threshold)
            return rare / len(words)

        rare_fraction = word_lists.apply(rare_word_fraction)

        normalized_length = (word_counts - word_counts.min()) / max(word_counts.max() - word_counts.min(), 1)
        score = 0.6 * normalized_length + 0.4 * rare_fraction
        return score

    def assign_stages(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Assign each row to a curriculum stage (0 = easiest) based on difficulty.

        Args:
            df: DataFrame to stage.
            text_column: Column used to compute difficulty (see
                `compute_difficulty_score`).

        Returns:
            pd.DataFrame: Copy of `df` with an added "curriculum_stage"
                column (int, 0-indexed) and "difficulty_score" column.
        """
        staged = df.copy()
        staged["difficulty_score"] = self.compute_difficulty_score(df, text_column)
        staged = staged.sort_values("difficulty_score").reset_index(drop=True)
        staged["curriculum_stage"] = pd.qcut(
            staged.index, q=self.num_stages, labels=False, duplicates="drop"
        )
        logger.info(f"Assigned {len(staged):,} rows to {staged['curriculum_stage'].nunique()} curriculum stages.")
        return staged


class ResourceCurriculum:
    """Stages training by resource type: bilingual -> trilingual -> lexical."""

    STAGE_ORDER = ("bilingual", "trilingual", "lexical")

    def build_schedule(
        self,
        bilingual_df: pd.DataFrame,
        trilingual_df: pd.DataFrame,
        lexical_df: pd.DataFrame,
    ) -> List[pd.DataFrame]:
        """Build an ordered list of DataFrames representing curriculum stages.

        Args:
            bilingual_df: Bilingual (two-language) training tasks.
            trilingual_df: Trilingual training tasks.
            lexical_df: Lexical-augmentation training tasks.

        Returns:
            List[pd.DataFrame]: Ordered [bilingual, bilingual+trilingual,
                bilingual+trilingual+lexical] -- each stage cumulatively
                includes all prior resources plus the new one, so later
                training epochs never "forget" earlier resources.
        """
        stage_1 = bilingual_df.reset_index(drop=True)
        stage_2 = pd.concat([bilingual_df, trilingual_df], ignore_index=True)
        stage_3 = pd.concat([bilingual_df, trilingual_df, lexical_df], ignore_index=True)

        logger.info(
            f"Built resource curriculum: stage sizes = "
            f"{len(stage_1):,} -> {len(stage_2):,} -> {len(stage_3):,}."
        )
        return [stage_1, stage_2, stage_3]
