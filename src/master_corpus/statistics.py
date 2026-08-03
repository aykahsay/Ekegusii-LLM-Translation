"""
Master Corpus Statistics
---------------------------
Computes descriptive statistics over the master sentence/lexical corpora:
per-language sentence-length and vocabulary statistics, source/dataset-origin
distribution, and split-size summaries. Backs notebook 01
(master_corpus_analysis) and notebook 03 (resource_statistics).
"""

import logging
import re
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


class CorpusStatistics:
    """Computes descriptive statistics for a corpus DataFrame."""

    def language_statistics(self, df: pd.DataFrame, languages: List[str]) -> pd.DataFrame:
        """Compute per-language sentence-length and vocabulary statistics.

        Args:
            df: Corpus DataFrame containing one column per language.
            languages: Column names to compute statistics for (e.g.
                ["English", "Kiswahili", "Ekegusii"]).

        Returns:
            pd.DataFrame: Indexed by language, with columns "non_null_rows",
                "avg_word_count", "median_word_count", "max_word_count",
                "total_tokens", "vocabulary_size", "type_token_ratio".
        """
        rows = []
        for lang in languages:
            if lang not in df.columns:
                logger.warning(f"Language column '{lang}' not found; skipping.")
                continue

            series = df[lang].dropna().astype(str)
            word_counts = series.apply(lambda s: len(s.split()))
            tokens: List[str] = []
            for s in series:
                tokens.extend(re.findall(r"\w+", str(s).lower()))

            total_tokens = len(tokens)
            vocab_size = len(set(tokens))
            ttr = vocab_size / total_tokens if total_tokens > 0 else 0.0

            rows.append(
                {
                    "language": lang,
                    "non_null_rows": len(series),
                    "avg_word_count": round(float(word_counts.mean()), 2) if len(word_counts) else 0.0,
                    "median_word_count": word_counts.median() if len(word_counts) else 0.0,
                    "max_word_count": int(word_counts.max()) if len(word_counts) else 0,
                    "total_tokens": total_tokens,
                    "vocabulary_size": vocab_size,
                    "type_token_ratio": round(ttr, 4),
                }
            )

        return pd.DataFrame(rows).set_index("language")

    def source_distribution(self, df: pd.DataFrame, column: str = "source") -> pd.DataFrame:
        """Compute the row-count and percentage distribution over a categorical column.

        Args:
            df: Corpus DataFrame.
            column: Categorical column to summarize (e.g. "source" or
                "dataset_origin").

        Returns:
            pd.DataFrame: Columns "count" and "percentage", indexed by
                category value, sorted by count descending.

        Raises:
            KeyError: If `column` is not present in `df`.
        """
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")

        counts = df[column].value_counts()
        pct = (counts / len(df) * 100).round(2)
        return pd.DataFrame({"count": counts, "percentage": pct})

    def split_summary(self, splits: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Summarize row counts and proportions across dataset splits.

        Args:
            splits: Mapping of split name (e.g. "train", "val", "test") to
                its DataFrame.

        Returns:
            pd.DataFrame: Columns "rows" and "percentage", indexed by split name.
        """
        total = sum(len(df) for df in splits.values())
        rows = {
            name: {"rows": len(df), "percentage": round(100 * len(df) / total, 2) if total else 0.0}
            for name, df in splits.items()
        }
        return pd.DataFrame(rows).T
