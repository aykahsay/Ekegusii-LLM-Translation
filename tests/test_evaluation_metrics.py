"""
Unit Tests for SacreBLEU, chrF++, and Lexical Evaluators using Standard unittest
---------------------------------------------------------------------------------
"""

import unittest

from src.evaluation.chrf import ChrFEvaluator
from src.evaluation.lexical_evaluator import LexicalEvaluator
from src.evaluation.sacrebleu import SacreBLEUEvaluator


class TestEvaluationMetrics(unittest.TestCase):
    """Test suite for SacreBLEU, chrF++, and Lexical Evaluators."""

    def test_sacrebleu_evaluator(self) -> None:
        """Test SacreBLEU evaluation calculation."""
        evaluator = SacreBLEUEvaluator()
        hyps = ["The government helps citizens and families every single day in the country."]
        refs = ["The government helps citizens and families every single day in the country."]
        res = evaluator.compute_corpus_bleu(hyps, refs)

        self.assertIn("sacrebleu_score", res)
        self.assertAlmostEqual(float(res["sacrebleu_score"]), 100.0, places=1)

    def test_chrf_evaluator(self) -> None:
        """Test chrF++ evaluation calculation."""
        evaluator = ChrFEvaluator()
        hyps = ["eserikari yakonyere abanto"]
        refs = ["eserikari yakonyere abanto"]
        res = evaluator.compute_chrf(hyps, refs)

        self.assertIn("chrf_plus_plus_score", res)
        self.assertAlmostEqual(float(res["chrf_plus_plus_score"]), 100.0, places=1)

    def test_lexical_evaluator(self) -> None:
        """Test exact and stem match lexical evaluation."""
        evaluator = LexicalEvaluator()
        preds = [
            ("eserikari yakonyere abanto", "eserikari yakonyere abanto"),
            ("eserikari yakonyera abanto", "eserikari yakonyere abanto"),
        ]
        res = evaluator.evaluate_predictions(preds)

        self.assertEqual(res["exact_term_accuracy"], 50.0)
        self.assertEqual(res["morphological_coverage"], 100.0)


if __name__ == "__main__":
    unittest.main()
