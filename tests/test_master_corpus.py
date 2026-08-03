"""
Unit Tests for MasterCorpusManager & DataLeakageChecker using Standard unittest
---------------------------------------------------------------------------------
"""

import unittest
import pandas as pd

from src.master_corpus.manager import MasterCorpusManager
from src.master_corpus.integrity import DataLeakageChecker


class TestMasterCorpus(unittest.TestCase):
    """Test suite for MasterCorpusManager and DataLeakageChecker."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.manager = MasterCorpusManager()

    def test_sentence_corpus_loading(self) -> None:
        """Test loading and column structure of Master Sentence Corpus."""
        df = self.manager.load_sentence_corpus()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        for col in ["concept_id", "English", "Kiswahili", "Ekegusii", "source"]:
            self.assertIn(col, df.columns)

    def test_lexical_corpus_loading(self) -> None:
        """Test loading of Master Lexical Corpus."""
        df = self.manager.load_lexical_corpus()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 268)
        self.assertIn("lexicon_id", df.columns)

    def test_splits_loading(self) -> None:
        """Test train, val, and test split loading."""
        train_df = self.manager.load_train_split()
        val_df = self.manager.load_val_split()
        test_df = self.manager.load_test_split()

        self.assertGreater(len(train_df), 0)
        self.assertGreater(len(val_df), 0)
        self.assertGreater(len(test_df), 0)

    def test_zero_leakage_audit(self) -> None:
        """Test that DataLeakageChecker verifies 0% overlap."""
        checker = DataLeakageChecker(self.manager)
        self.assertTrue(checker.verify_all())


if __name__ == "__main__":
    unittest.main()
