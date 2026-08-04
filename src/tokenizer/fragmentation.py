"""
Per-Language Fragmentation Analysis
--------------------------------------
Applies `TokenizerMetrics` (src.tokenizer.metrics) across each of English,
Kiswahili, and Ekegusii independently to produce a per-language comparison
table -- the core deliverable of notebook 04 (tokenizer_analysis). High
fragmentation for Ekegusii relative to English/Kiswahili is the expected
finding motivating lexical augmentation (E6): it demonstrates the tokenizer
was not trained on Ekegusii and splits its words into many more subword
pieces than a language it was trained on.
"""

import logging
from typing import Dict, List, Union

import pandas as pd
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from src.tokenizer.metrics import TokenizerMetrics

logger = logging.getLogger(__name__)


class FragmentationAnalyzer:
    """Compares tokenizer fragmentation across multiple languages."""

    def __init__(self, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> None:
        """Initialize with a HuggingFace tokenizer to analyze.

        Args:
            tokenizer: Tokenizer under analysis (e.g. Qwen2.5 or Mistral-7B).
        """
        self.metrics = TokenizerMetrics(tokenizer)

    def analyze_languages(self, language_sentences: Dict[str, List[str]]) -> pd.DataFrame:
        """Compute fragmentation statistics for each language's sentence sample.

        Args:
            language_sentences: Mapping of language name (e.g. "English")
                to a list of sentences from that language.

        Returns:
            pd.DataFrame: Indexed by language, with columns
                "mean_fertility", "std_fertility", "mean_compression_ratio",
                "vocabulary_coverage_pct", "total_words_analyzed", and
                "fragmentation_rate_pct" (100 - vocabulary_coverage_pct,
                i.e. the fraction of words split into >1 subword token).
        """
        rows = []
        for language, sentences in language_sentences.items():
            stats = self.metrics.compute_corpus_fertility(sentences)
            stats["language"] = language
            stats["fragmentation_rate_pct"] = 100.0 - stats["vocabulary_coverage_pct"]
            rows.append(stats)
            logger.info(
                f"[{language}] mean_fertility={stats['mean_fertility']:.3f}, "
                f"fragmentation_rate={stats['fragmentation_rate_pct']:.2f}%"
            )

        return pd.DataFrame(rows).set_index("language")

    def relative_fragmentation(self, comparison_df: pd.DataFrame, baseline_language: str) -> pd.DataFrame:
        """Express each language's fertility relative to a baseline language.

        Args:
            comparison_df: Output of `analyze_languages`.
            baseline_language: Language to normalize against (typically
                "English", since it is the language the base tokenizer was
                most heavily trained on).

        Returns:
            pd.DataFrame: `comparison_df` with an added
                "fertility_ratio_vs_baseline" column (e.g. 2.5 means that
                language needs 2.5x as many tokens per word as the baseline).

        Raises:
            KeyError: If `baseline_language` is not a row in `comparison_df`.
        """
        if baseline_language not in comparison_df.index:
            raise KeyError(f"Baseline language '{baseline_language}' not found in comparison_df.")

        baseline_fertility = comparison_df.loc[baseline_language, "mean_fertility"]
        result = comparison_df.copy()
        result["fertility_ratio_vs_baseline"] = (result["mean_fertility"] / baseline_fertility).round(3)
        return result
