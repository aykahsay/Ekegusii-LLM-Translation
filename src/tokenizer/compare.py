"""
Qwen vs. Mistral Tokenizer Comparison
---------------------------------------
Ties together the Qwen/Mistral tokenizer loaders and the fragmentation/
vocabulary analyzers into a single side-by-side comparison table across
English, Kiswahili, and Ekegusii -- the core deliverable of notebook 04
(tokenizer_analysis), directly informing which of the two base models is
the better starting point for QLoRA fine-tuning on Ekegusii.
"""

import logging
from typing import Dict, List

import pandas as pd

from src.tokenizer.fragmentation import FragmentationAnalyzer
from src.tokenizer.mistral import load_mistral_tokenizer
from src.tokenizer.qwen import load_qwen_tokenizer
from src.tokenizer.vocabulary import VocabularyCoverageAnalyzer

logger = logging.getLogger(__name__)


class TokenizerComparator:
    """Runs the full Qwen vs. Mistral tokenizer comparison across languages."""

    def compare(self, language_sentences: Dict[str, List[str]]) -> pd.DataFrame:
        """Compare Qwen2.5 and Mistral-7B tokenizer fragmentation across languages.

        Args:
            language_sentences: Mapping of language name to a list of
                sentences from that language (e.g. from
                `configs/datasets/monolingual.yaml`'s per-language
                projection of the master sentence corpus).

        Returns:
            pd.DataFrame: MultiIndexed by (model, language) with the same
                columns as `FragmentationAnalyzer.analyze_languages`
                ("mean_fertility", "fragmentation_rate_pct", etc.), so rows
                for "qwen"/"English" and "mistral"/"English" sit adjacent for
                direct comparison.
        """
        qwen_tokenizer = load_qwen_tokenizer()
        mistral_tokenizer = load_mistral_tokenizer()

        qwen_df = FragmentationAnalyzer(qwen_tokenizer).analyze_languages(language_sentences)
        mistral_df = FragmentationAnalyzer(mistral_tokenizer).analyze_languages(language_sentences)

        qwen_df["model"] = "qwen"
        mistral_df["model"] = "mistral"

        combined: pd.DataFrame = pd.concat([qwen_df, mistral_df])
        combined = combined.reset_index().set_index(["model", "language"]).sort_index()
        logger.info(f"Compared Qwen vs. Mistral across {len(language_sentences)} language(s).")
        return combined

    def compare_vocabulary_coverage(self, language_sentences: Dict[str, List[str]]) -> pd.DataFrame:
        """Compare Qwen vs. Mistral type-level vocabulary coverage across languages.

        Args:
            language_sentences: Mapping of language name to sentence list.

        Returns:
            pd.DataFrame: Rows are (model, language) pairs; columns are the
                keys returned by
                `VocabularyCoverageAnalyzer.summarize` ("vocabulary_size",
                "type_level_coverage_pct", "mean_tokens_per_type", etc.).
        """
        qwen_analyzer = VocabularyCoverageAnalyzer(load_qwen_tokenizer())
        mistral_analyzer = VocabularyCoverageAnalyzer(load_mistral_tokenizer())

        rows = []
        for model_name, analyzer in (("qwen", qwen_analyzer), ("mistral", mistral_analyzer)):
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
                rows for both "qwen" and "mistral").
            target_language: Language to base the recommendation on
                (defaults to "Ekegusii", the project's core low-resource
                target).

        Returns:
            str: "qwen" or "mistral" -- whichever has the lower mean_fertility
                (fewer tokens per word) for `target_language`.

        Raises:
            KeyError: If `target_language` rows are missing for either model.
        """
        qwen_fertility = comparison_df.loc[("qwen", target_language), "mean_fertility"]
        mistral_fertility = comparison_df.loc[("mistral", target_language), "mean_fertility"]

        recommendation = "qwen" if qwen_fertility <= mistral_fertility else "mistral"
        logger.info(
            f"[{target_language}] qwen_fertility={qwen_fertility:.3f}, mistral_fertility={mistral_fertility:.3f} "
            f"-> recommend '{recommendation}'."
        )
        return recommendation
