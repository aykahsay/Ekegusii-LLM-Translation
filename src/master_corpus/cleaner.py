"""
Corpus-Level Cleaner
-----------------------
DataFrame-level hygiene operations: dropping fully-empty rows, removing
duplicate concept/text rows, and coercing text columns to a consistent
stripped-string dtype. This module operates at the row/DataFrame level;
per-cell text normalization (unicode normalization, punctuation handling,
code-switch marking) lives in `src/preprocessing/normalize.py` and is used
here for the per-cell strip step so the two modules stay complementary
rather than duplicating each other.
"""

import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


class CorpusCleaner:
    """Applies row-level cleaning operations to a corpus DataFrame."""

    def clean(
        self,
        df: pd.DataFrame,
        text_columns: List[str],
        id_column: str,
    ) -> pd.DataFrame:
        """Clean a corpus DataFrame: strip whitespace, drop empty/duplicate rows.

        Args:
            df: Raw corpus DataFrame.
            text_columns: Text columns to strip and check for emptiness
                (e.g. ["English", "Kiswahili", "Ekegusii"]).
            id_column: Unique identifier column used to drop exact
                duplicate rows (keeping the first occurrence).

        Returns:
            pd.DataFrame: Cleaned DataFrame with a reset index. Row order
                is otherwise preserved.
        """
        original_len = len(df)
        cleaned: pd.DataFrame = df.copy()

        for col in text_columns:
            if col in cleaned.columns:
                cleaned[col] = cleaned[col].apply(self._strip_or_none)

        present_columns = [c for c in text_columns if c in cleaned.columns]
        if present_columns:
            all_empty: pd.Series = cleaned[present_columns[0]].isna()
            for col in present_columns[1:]:
                all_empty = all_empty & cleaned[col].isna()
            cleaned = cleaned.loc[~all_empty]

        if id_column in cleaned.columns:
            cleaned = cleaned.drop_duplicates(subset=[id_column], keep="first")

        cleaned = cleaned.reset_index(drop=True)
        logger.info(
            f"Cleaned corpus: {original_len:,} -> {len(cleaned):,} rows "
            f"({original_len - len(cleaned):,} removed)."
        )
        return cleaned

    @staticmethod
    def _strip_or_none(value: object) -> object:
        """Strip whitespace from a string value, preserving NaN/None as-is."""
        if pd.isna(value):
            return value
        stripped = str(value).strip()
        return stripped if stripped else pd.NA
