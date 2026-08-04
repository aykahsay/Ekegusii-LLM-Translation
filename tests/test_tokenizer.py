"""
Unit Tests for Tokenizer Fertility & Fragmentation Analyzers
-----------------------------------------------------------------
Uses `bert-base-multilingual-cased` (small, ungated, always downloadable)
as a stand-in tokenizer rather than Qwen2.5/Mistral-7B -- those require
gated Hub access and 8B-parameter downloads, which unit tests must not
depend on. The metrics logic under test is tokenizer-agnostic.
"""

import unittest

from transformers import AutoTokenizer

from src.tokenizer.fragmentation import FragmentationAnalyzer
from src.tokenizer.metrics import TokenizerMetrics
from src.tokenizer.vocabulary import VocabularyCoverageAnalyzer


class TestTokenizerMetrics(unittest.TestCase):
    """Test subword fertility and fragmentation computations."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the stand-in tokenizer once for the whole test class."""
        cls.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    def test_sentence_fertility_nonzero_for_nonempty_text(self) -> None:
        """A non-empty sentence must have positive fertility and word count."""
        metrics = TokenizerMetrics(self.tokenizer)
        result = metrics.compute_sentence_fertility("Hello world, this is a test.")
        self.assertGreater(result["fertility"], 0.0)
        self.assertGreater(result["word_count"], 0)

    def test_sentence_fertility_empty_text(self) -> None:
        """An empty/whitespace-only sentence must return all-zero metrics, not raise."""
        metrics = TokenizerMetrics(self.tokenizer)
        result = metrics.compute_sentence_fertility("   ")
        self.assertEqual(result["word_count"], 0)
        self.assertEqual(result["fertility"], 0.0)

    def test_corpus_fertility_vocabulary_coverage_bounds(self) -> None:
        """Vocabulary coverage percentage must be within [0, 100]."""
        metrics = TokenizerMetrics(self.tokenizer)
        result = metrics.compute_corpus_fertility(["Hello world.", "This is a test sentence."])
        self.assertGreaterEqual(result["vocabulary_coverage_pct"], 0.0)
        self.assertLessEqual(result["vocabulary_coverage_pct"], 100.0)

    def test_fragmentation_analyzer_flags_higher_fertility_language(self) -> None:
        """A language built from rare, unseen word forms should fragment more
        than plain English, since the tokenizer wasn't trained on it."""
        analyzer = FragmentationAnalyzer(self.tokenizer)
        comparison = analyzer.analyze_languages(
            {
                "English": ["The cat sat on the mat."] * 3,
                "Nonsense": ["Zqxvlorp fribbnaxxo tremquilibus."] * 3,
            }
        )
        relative = analyzer.relative_fragmentation(comparison, baseline_language="English")
        self.assertGreaterEqual(relative.loc["Nonsense", "fertility_ratio_vs_baseline"], 1.0)

    def test_vocabulary_coverage_analyzer_type_level_counts(self) -> None:
        """Vocabulary size must equal the number of unique word types, not word instances."""
        analyzer = VocabularyCoverageAnalyzer(self.tokenizer)
        summary = analyzer.summarize(["the the the cat sat", "the dog ran"])
        # Unique word types: {the, cat, sat, dog, ran} = 5
        self.assertEqual(summary["vocabulary_size"], 5)


if __name__ == "__main__":
    unittest.main()
