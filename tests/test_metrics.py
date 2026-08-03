"""
Unit Test for Metrics Evaluation
--------------------------------
"""

import unittest
from src.evaluation.sacrebleu import SacreBLEUEvaluator


class TestMetrics(unittest.TestCase):
    """Test translation evaluation metric calculations."""

    def test_sacrebleu(self) -> None:
        evaluator = SacreBLEUEvaluator()
        hyps = ["The government helps citizens and families every single day in the country."]
        refs = ["The government helps citizens and families every single day in the country."]
        res = evaluator.compute_corpus_bleu(hyps, refs)
        self.assertAlmostEqual(float(res["sacrebleu_score"]), 100.0, places=1)


if __name__ == "__main__":
    unittest.main()
