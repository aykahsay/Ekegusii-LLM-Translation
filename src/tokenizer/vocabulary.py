"""
Word-Type Vocabulary Coverage Analysis
------------------------------------------
`TokenizerMetrics.compute_corpus_fertility` measures single-token coverage
per word *instance* (so a common word appearing 1000 times counts 1000
times). This module instead analyzes coverage at the word *type* level
(each unique word counted once), which better answers "how much of this
language's vocabulary does the tokenizer actually know" independent of
corpus word-frequency skew, and additionally surfaces the most-fragmented
word types for qualitative inspection.
"""

import logging
import re
from typing import List

import pandas as pd
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class VocabularyCoverageAnalyzer:
    """Analyzes tokenizer coverage over unique word types in a corpus."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Initialize with a HuggingFace tokenizer to analyze.

        Args:
            tokenizer: Tokenizer under analysis.
        """
        self.tokenizer = tokenizer

    def extract_word_types(self, sentences: List[str]) -> List[str]:
        """Extract the set of unique lowercased word types from a sentence list.

        Args:
            sentences: Sentences to extract vocabulary from.

        Returns:
            List[str]: Sorted list of unique word types.
        """
        vocabulary: set = set()
        for sentence in sentences:
            if isinstance(sentence, str):
                vocabulary.update(re.findall(r"\w+", sentence.lower()))
        return sorted(vocabulary)

    def coverage_table(self, sentences: List[str]) -> pd.DataFrame:
        """Build a per-word-type table of subword token counts.

        Args:
            sentences: Sentences to extract vocabulary from and analyze.

        Returns:
            pd.DataFrame: Columns "word", "token_count", "is_single_token",
                sorted by "token_count" descending (most-fragmented first).
        """
        word_types = self.extract_word_types(sentences)
        rows = []
        for word in word_types:
            token_count = len(self.tokenizer.encode(word, add_special_tokens=False))
            rows.append({"word": word, "token_count": token_count, "is_single_token": token_count == 1})

        table = pd.DataFrame(rows).sort_values("token_count", ascending=False).reset_index(drop=True)
        return table

    def summarize(self, sentences: List[str]) -> dict:
        """Summarize type-level vocabulary coverage for a sentence list.

        Args:
            sentences: Sentences to extract vocabulary from and analyze.

        Returns:
            dict: Keys "vocabulary_size" (unique word types), "single_token_types",
                "type_level_coverage_pct", "mean_tokens_per_type".
        """
        table = self.coverage_table(sentences)
        if len(table) == 0:
            return {
                "vocabulary_size": 0,
                "single_token_types": 0,
                "type_level_coverage_pct": 0.0,
                "mean_tokens_per_type": 0.0,
            }

        single_token_types = int(table["is_single_token"].sum())
        coverage_pct = 100.0 * single_token_types / len(table)

        summary = {
            "vocabulary_size": len(table),
            "single_token_types": single_token_types,
            "type_level_coverage_pct": round(coverage_pct, 2),
            "mean_tokens_per_type": round(table["token_count"].mean(), 3),
        }
        logger.info(f"Type-level coverage: {summary}")
        return summary

    def most_fragmented_words(self, sentences: List[str], top_n: int = 20) -> pd.DataFrame:
        """Return the `top_n` word types requiring the most subword tokens.

        Args:
            sentences: Sentences to extract vocabulary from and analyze.
            top_n: Number of most-fragmented word types to return.

        Returns:
            pd.DataFrame: Top `top_n` rows of `coverage_table`, i.e. the
                words the tokenizer fragments most heavily -- useful
                qualitative evidence for why lexical augmentation (E6)
                should help.
        """
        return self.coverage_table(sentences).head(top_n)
