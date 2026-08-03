"""
Rare-Word Identification
----------------------------
Identifies low-frequency word types within a corpus (candidates for
rare-word translation-accuracy evaluation, notebook 10) and reports how
severely the tokenizer fragments them relative to high-frequency words --
the expected finding being that rare Ekegusii words are fragmented far more
than common ones, motivating targeted lexical augmentation (E6).
"""

import logging
import re
from collections import Counter
from typing import List, Optional

import pandas as pd
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class RareWordIdentifier:
    """Identifies rare word types and analyzes their tokenizer fragmentation."""

    def __init__(self, tokenizer: Optional[PreTrainedTokenizerBase] = None) -> None:
        """Initialize the identifier.

        Args:
            tokenizer: Optional tokenizer used to compute per-word token
                counts. If None, frequency analysis still works but
                `fragmentation_table` cannot be used.
        """
        self.tokenizer = tokenizer

    @staticmethod
    def word_frequencies(sentences: List[str]) -> Counter:
        """Compute word-type frequencies across a sentence list.

        Args:
            sentences: Sentences to tokenize into words (whitespace + regex
                word-boundary split, lowercased).

        Returns:
            Counter: Maps lowercased word -> occurrence count.
        """
        counter: Counter = Counter()
        for sentence in sentences:
            if isinstance(sentence, str):
                counter.update(re.findall(r"\w+", sentence.lower()))
        return counter

    def identify_rare_words(
        self, sentences: List[str], max_frequency: int = 2, min_length: int = 3
    ) -> List[str]:
        """Identify word types occurring at or below a frequency threshold.

        Args:
            sentences: Sentences to analyze.
            max_frequency: Maximum occurrence count for a word to be
                considered "rare".
            min_length: Minimum character length -- filters out rare
                one/two-letter tokens that are usually noise rather than
                meaningful rare vocabulary.

        Returns:
            List[str]: Rare word types, sorted alphabetically.
        """
        freqs = self.word_frequencies(sentences)
        rare = [word for word, count in freqs.items() if count <= max_frequency and len(word) >= min_length]
        logger.info(f"Identified {len(rare):,} rare word type(s) (freq <= {max_frequency}).")
        return sorted(rare)

    def fragmentation_table(self, sentences: List[str], max_frequency: int = 2) -> pd.DataFrame:
        """Compare tokenizer fragmentation between rare and common word types.

        Args:
            sentences: Sentences to analyze.
            max_frequency: Frequency threshold defining "rare" (see
                `identify_rare_words`).

        Returns:
            pd.DataFrame: Columns "word", "corpus_frequency", "token_count",
                "is_rare", sorted by "corpus_frequency" ascending.

        Raises:
            ValueError: If no tokenizer was provided at construction time.
        """
        if self.tokenizer is None:
            raise ValueError("A tokenizer must be provided to compute fragmentation_table.")

        freqs = self.word_frequencies(sentences)
        rows = []
        for word, freq in freqs.items():
            token_count = len(self.tokenizer.encode(word, add_special_tokens=False))
            rows.append(
                {
                    "word": word,
                    "corpus_frequency": freq,
                    "token_count": token_count,
                    "is_rare": freq <= max_frequency,
                }
            )

        return pd.DataFrame(rows).sort_values("corpus_frequency").reset_index(drop=True)

    def compare_rare_vs_common_fertility(self, sentences: List[str], max_frequency: int = 2) -> dict:
        """Summarize mean token count for rare vs. common word types.

        Args:
            sentences: Sentences to analyze.
            max_frequency: Frequency threshold defining "rare".

        Returns:
            dict: Keys "rare_mean_tokens", "common_mean_tokens",
                "rare_to_common_ratio" (>1 means rare words fragment more).
        """
        table = self.fragmentation_table(sentences, max_frequency)
        rare_mean = table.loc[table["is_rare"], "token_count"].mean() if table["is_rare"].any() else 0.0
        common_mean = table.loc[~table["is_rare"], "token_count"].mean() if (~table["is_rare"]).any() else 0.0

        ratio = rare_mean / common_mean if common_mean > 0 else float("inf")
        result = {
            "rare_mean_tokens": round(float(rare_mean), 3),
            "common_mean_tokens": round(float(common_mean), 3),
            "rare_to_common_ratio": round(ratio, 3) if ratio != float("inf") else ratio,
        }
        logger.info(f"Rare vs. common fertility: {result}")
        return result
