"""
Quality & Relevance Filtering
--------------------------------
Row-level filters applied after cleaning/normalization: sentence-length
bounds (drop fragments and outlier walls-of-text), and language-consistency
filtering (drop rows whose English/Kiswahili column fails the
`language_detection` check). Filters are designed to be composed and each
returns a boolean mask rather than mutating the DataFrame directly, so
callers can inspect/log what would be dropped before committing to it.
"""

import logging
from typing import Optional

import pandas as pd

from src.preprocessing.language_detection import matches_expected_language

logger = logging.getLogger(__name__)


def length_filter_mask(
    series: pd.Series, min_words: int = 2, max_words: int = 200
) -> pd.Series:
    """Flag rows whose word count falls within an acceptable range.

    Args:
        series: Text column to check.
        min_words: Minimum acceptable word count (drops single-word
            fragments, stray headers, etc.).
        max_words: Maximum acceptable word count (drops mis-scraped
            multi-paragraph blocks that shouldn't be a single PSA sentence).

    Returns:
        pd.Series: Boolean mask, True where the row's word count is within
            [min_words, max_words].

    Raises:
        ValueError: If `min_words` > `max_words`.
    """
    if min_words > max_words:
        raise ValueError(f"min_words ({min_words}) must be <= max_words ({max_words}).")

    word_counts = series.fillna("").astype(str).apply(lambda s: len(s.split()))
    return (word_counts >= min_words) & (word_counts <= max_words)


def language_consistency_mask(series: pd.Series, expected_language: str) -> pd.Series:
    """Flag rows whose text plausibly matches its expected language.

    Rows where detection is inconclusive (returns None, e.g. very short
    text) are kept (marked True) rather than dropped, since inconclusive
    detection is not evidence of a labeling error.

    Args:
        series: Text column to check.
        expected_language: One of `src.utils.constants.LANG_ENGLISH` or
            `LANG_KISWAHILI` (Ekegusii is not supported here -- see
            `language_detection.ekegusii_lexicon_overlap` for that case).

    Returns:
        pd.Series: Boolean mask, True where the text matches the expected
            language or detection was inconclusive.
    """

    def _check(text: object) -> bool:
        if not isinstance(text, str) or not text.strip():
            return True
        result = matches_expected_language(text, expected_language)
        return True if result is None else result

    return series.apply(_check)


def apply_filters(
    df: pd.DataFrame,
    text_column: str,
    min_words: int = 2,
    max_words: int = 200,
    expected_language: Optional[str] = None,
) -> pd.DataFrame:
    """Apply length and (optionally) language-consistency filtering to a DataFrame.

    Args:
        df: DataFrame to filter.
        text_column: Column to filter on.
        min_words: Minimum acceptable word count.
        max_words: Maximum acceptable word count.
        expected_language: If provided, also apply
            `language_consistency_mask` for this language (English/Kiswahili
            only).

    Returns:
        pd.DataFrame: Filtered DataFrame with a reset index.
    """
    original_len = len(df)
    mask = length_filter_mask(df[text_column], min_words, max_words)

    if expected_language is not None:
        mask &= language_consistency_mask(df[text_column], expected_language)

    filtered_rows: pd.DataFrame = df.loc[mask]  # type: ignore[assignment]
    filtered = filtered_rows.reset_index(drop=True)
    logger.info(
        f"Filtered '{text_column}': {original_len:,} -> {len(filtered):,} rows "
        f"({original_len - len(filtered):,} removed)."
    )
    return filtered
