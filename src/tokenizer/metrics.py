"""
Tokenizer Fertility & Subword Fragmentation Analyzer
----------------------------------------------------
Computes scientific tokenizer metrics across languages (English, Kiswahili, Ekegusii):
1. Subword Fertility (tokens per word)
2. Character Compression Ratio (characters per token)
3. Vocabulary Coverage (% of words mapped to single token)
4. Out-of-Vocabulary / Fragmentation Rate
"""

import logging
from typing import Dict, List, Union
import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class TokenizerMetrics:
    """Computes tokenizer subword fertility and fragmentation statistics."""

    def __init__(self, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> None:
        """Initialize with a HuggingFace PreTrainedTokenizer instance."""
        self.tokenizer = tokenizer

    def compute_sentence_fertility(self, text: str) -> Dict[str, float]:
        """Compute fertility metrics for a single text sentence.

        Args:
            text: Input sentence text string.

        Returns:
            Dict containing fertility, compression_ratio, and word_count.
        """
        words = text.strip().split()
        if not words:
            return {"fertility": 0.0, "compression_ratio": 0.0, "word_count": 0, "token_count": 0}

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        num_words = len(words)
        num_tokens = len(tokens)
        num_chars = len(text)

        fertility = num_tokens / num_words if num_words > 0 else 0.0
        compression_ratio = num_chars / num_tokens if num_tokens > 0 else 0.0

        return {
            "fertility": float(fertility),
            "compression_ratio": float(compression_ratio),
            "word_count": num_words,
            "token_count": num_tokens,
        }

    def compute_corpus_fertility(self, sentences: List[str]) -> Dict[str, float]:
        """Compute aggregated fertility statistics across a list of corpus sentences.

        Args:
            sentences: List of text sentences.

        Returns:
            Dict containing mean_fertility, std_fertility, mean_compression_ratio, vocabulary_coverage.
        """
        fertilities: List[float] = []
        compressions: List[float] = []
        single_token_words = 0
        total_words = 0

        for sentence in sentences:
            if not sentence or not isinstance(sentence, str):
                continue
            m = self.compute_sentence_fertility(sentence)
            if m["word_count"] > 0:
                fertilities.append(m["fertility"])
                compressions.append(m["compression_ratio"])
                
                # Check word-level single-token coverage
                words = sentence.strip().split()
                for word in words:
                    subwords = self.tokenizer.encode(word, add_special_tokens=False)
                    if len(subwords) == 1:
                        single_token_words += 1
                    total_words += 1

        vocab_coverage = (single_token_words / total_words * 100.0) if total_words > 0 else 0.0

        return {
            "mean_fertility": float(np.mean(fertilities)) if fertilities else 0.0,
            "std_fertility": float(np.std(fertilities)) if fertilities else 0.0,
            "mean_compression_ratio": float(np.mean(compressions)) if compressions else 0.0,
            "vocabulary_coverage_pct": float(vocab_coverage),
            "total_words_analyzed": total_words,
        }
