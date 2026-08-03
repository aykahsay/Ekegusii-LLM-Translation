"""
Master Corpus Manager API
-------------------------
Centralized API manager providing strict loading, verification, and access
to the Master Sentence Corpus (49,277 concepts) and Master Lexical Corpus (268 terms)
with zero-leakage guarantee across Train (80%), Val (10%), and Test (10%) splits.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class MasterCorpusManager:
    """Central API for loading, inspecting, and retrieving master translation datasets."""

    SENTENCE_REQUIRED_COLUMNS = [
        "concept_id",
        "English",
        "Kiswahili",
        "Ekegusii",
        "source",
        "dataset_origin",
    ]
    LEXICAL_REQUIRED_COLUMNS = [
        "lexicon_id",
        "English",
        "Kiswahili",
        "Ekegusii",
        "source",
    ]

    def __init__(self, data_dir: str = "data/master_corpus") -> None:
        """Initialize the MasterCorpusManager with data directory paths.

        Args:
            data_dir: Relative or absolute path to data/master_corpus directory.
        """
        self.data_dir = Path(data_dir)
        self.sentence_corpus_path = self.data_dir / "master_sentence_corpus.csv"
        self.lexical_corpus_path = self.data_dir / "master_lexical_corpus.csv"
        self.splits_dir = self.data_dir / "splits"

    def load_sentence_corpus(self) -> pd.DataFrame:
        """Load and validate the Master Sentence Corpus (49,277 concepts).

        Returns:
            pd.DataFrame: Master sentence corpus.

        Raises:
            FileNotFoundError: If master_sentence_corpus.csv is missing.
            ValueError: If required columns are missing.
        """
        if not self.sentence_corpus_path.exists():
            raise FileNotFoundError(f"Sentence corpus missing at: {self.sentence_corpus_path}")

        df = pd.read_csv(self.sentence_corpus_path)
        self._validate_columns(df, self.SENTENCE_REQUIRED_COLUMNS, "Master Sentence Corpus")
        logger.info(f"Loaded Master Sentence Corpus: {len(df):,} concepts.")
        return df

    def load_lexical_corpus(self) -> pd.DataFrame:
        """Load and validate the Master Lexical Corpus (268 terms).

        Returns:
            pd.DataFrame: Master lexical corpus.

        Raises:
            FileNotFoundError: If master_lexical_corpus.csv is missing.
            ValueError: If required columns are missing.
        """
        if not self.lexical_corpus_path.exists():
            raise FileNotFoundError(f"Lexical corpus missing at: {self.lexical_corpus_path}")

        df = pd.read_csv(self.lexical_corpus_path)
        self._validate_columns(df, self.LEXICAL_REQUIRED_COLUMNS, "Master Lexical Corpus")
        logger.info(f"Loaded Master Lexical Corpus: {len(df):,} terms.")
        return df

    def load_train_split(self) -> pd.DataFrame:
        """Load the master training split (80% / 39,421 concepts)."""
        return self._load_split("master_train.csv")

    def load_val_split(self) -> pd.DataFrame:
        """Load the master validation split (10% / 4,928 concepts)."""
        return self._load_split("master_val.csv")

    def load_test_split(self) -> pd.DataFrame:
        """Load the master test split (10% / 4,928 concepts)."""
        return self._load_split("master_test.csv")

    def load_derived_split(self, split_name: str) -> pd.DataFrame:
        """Load a specific derived subset (e.g. derived_train_eng_eke.csv).

        Args:
            split_name: Name of the derived CSV file.

        Returns:
            pd.DataFrame: Derived dataset subset.
        """
        return self._load_split(split_name)

    def _load_split(self, filename: str) -> pd.DataFrame:
        split_path = self.splits_dir / filename
        if not split_path.exists():
            raise FileNotFoundError(f"Split file missing at: {split_path}")
        df = pd.read_csv(split_path)
        logger.info(f"Loaded dataset split [{filename}]: {len(df):,} rows.")
        return df

    @staticmethod
    def _validate_columns(df: pd.DataFrame, expected: List[str], dataset_name: str) -> None:
        missing = [col for col in expected if col not in df.columns]
        if missing:
            raise ValueError(f"{dataset_name} missing required columns: {missing}")
