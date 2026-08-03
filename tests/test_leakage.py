"""
Unit Test for Data Leakage Prevention
-------------------------------------
"""

import unittest
from src.master_corpus.manager import MasterCorpusManager
from src.master_corpus.integrity import DataLeakageChecker


class TestDataLeakage(unittest.TestCase):
    """Test data leakage verification across splits."""

    def test_zero_leakage(self) -> None:
        """Verify zero leakage across train, val, and test concept IDs."""
        manager = MasterCorpusManager()
        checker = DataLeakageChecker(manager)
        self.assertTrue(checker.verify_all())


if __name__ == "__main__":
    unittest.main()
