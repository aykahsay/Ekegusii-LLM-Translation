"""
Unit Test for Data Loader
-------------------------
"""

import unittest
from src.master_corpus.manager import MasterCorpusManager


class TestDataLoader(unittest.TestCase):
    """Test data loader functionality."""

    def test_corpus_load(self) -> None:
        manager = MasterCorpusManager()
        df = manager.load_sentence_corpus()
        self.assertGreater(len(df), 0)


if __name__ == "__main__":
    unittest.main()
