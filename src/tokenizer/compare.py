"""
Aya vs. Llama Tokenizer Comparison
-------------------------------------
Ties together the Aya/Llama tokenizer loaders and the fragmentation/
vocabulary analyzers into a single side-by-side comparison table across
English, Kiswahili, and Ekegusii -- the core deliverable of notebook 04
(tokenizer_analysis), directly informing which of the two base models is
the better starting point for QLoRA fine-tuning on Ekegusii.
"""

import logging
from typing import Dict, List

import pandas as pd

from src.tokenizer.aya import load_aya_tokenizer
from src.tokenizer.fragmentation import FragmentationAnalyzer
from src.tokenizer.llama import load_llama_tokenizer
from src.tokenizer.vocabulary import VocabularyCoverageAnalyzer

logger = logging.getLogger(__name__)


class TokenizerComparator:
    """Runs the full Aya vs. Llama tokenizer comparison across languages."""

    def compare(self, language_sentences: Dict[str, List[str]]) -> pd.DataFrame:
        """Compare Aya-23 and Llama-3.1 tokenizer fragmentation across languages.

        Args:
            language_sentences: Mapping of language name to a list of
                sentences from that language (e.g. from
                `configs/datasets/monolingual.yaml`'s per-language
                projection of the master sentence corpus).

        Returns:
            pd.DataFrame: MultiIndexed by (model, language) with the same
                columns as `FragmentationAnalyzer.analyze_languages`
                ("mean_fertility", "fragmentation_rate_pct", etc.), so rows
                for "aya"/"English" and "llama"/"English" sit adjacent for
                direct comparison.
        """
        aya_tokenizer = load_aya_tokenizer()
        llama_tokenizer = load_llama_tokenizer()

        aya_df = FragmentationAnalyzer(aya_tokenizer).analyze_languages(language_sentences)
        llama_df = FragmentationAnalyzer(llama_tokenizer).analyze_languages(language_sentences)

        aya_df["model"] = "aya"
        llama_df["model"] = "llama"

        combined: pd.DataFrame = pd.concat([aya_df, llama_df])
        combined = combined.reset_index().set_index(["model", "language"]).sort_index()
        logger.info(f"Compared Aya vs. Llama across {len(language_sentences)} language(s).")
        return combined

    def compare_vocabulary_coverage(self, language_sentences: Dict[str, List[str]]) -> pd.DataFrame:
        """Compare Aya vs. Llama type-level vocabulary coverage across languages.

        Args:
            language_sentences: Mapping of language name to sentence list.

        Returns:
            pd.DataFrame: Rows are (model, language) pairs; columns are the
                keys returned by
                `VocabularyCoverageAnalyzer.summarize` ("vocabulary_size",
                "type_level_coverage_pct", "mean_tokens_per_type", etc.).
        """
        aya_analyzer = VocabularyCoverageAnalyzer(load_aya_tokenizer())
        llama_analyzer = VocabularyCoverageAnalyzer(load_llama_tokenizer())

        rows = []
        for model_name, analyzer in (("aya", aya_analyzer), ("llama", llama_analyzer)):
            for language, sentences in language_sentences.items():
                summary = analyzer.summarize(sentences)
                summary["model"] = model_name
                summary["language"] = language
                rows.append(summary)

        return pd.DataFrame(rows).set_index(["model", "language"]).sort_index()

    def recommend_base_model(self, comparison_df: pd.DataFrame, target_language: str = "Ekegusii") -> str:
        """Recommend which base model tokenizes `target_language` more efficiently.

        Args:
            comparison_df: Output of `compare` (must contain `target_language`
                rows for both "aya" and "llama").
            target_language: Language to base the recommendation on
                (defaults to "Ekegusii", the project's core low-resource
                target).

        Returns:
            str: "aya" or "llama" -- whichever has the lower mean_fertility
                (fewer tokens per word) for `target_language`.

        Raises:
            KeyError: If `target_language` rows are missing for either model.
        """
        aya_fertility = comparison_df.loc[("aya", target_language), "mean_fertility"]
        llama_fertility = comparison_df.loc[("llama", target_language), "mean_fertility"]

        recommendation = "aya" if aya_fertility <= llama_fertility else "llama"
        logger.info(
            f"[{target_language}] aya_fertility={aya_fertility:.3f}, llama_fertility={llama_fertility:.3f} "
            f"-> recommend '{recommendation}'."
        )
        return recommendation
