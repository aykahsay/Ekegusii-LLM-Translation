"""
Deterministic Corpus Splitter
-------------------------------
The master 80/10/10 train/val/test split (`data/master_corpus/splits/`)
MUST NEVER be regenerated once created -- every experiment (E0-E8) depends
on evaluating against the exact same held-out set. This module has two
distinct responsibilities that respect that constraint:

1. `split_concepts()` documents and reproduces the ORIGINAL deterministic
   splitting procedure, for provenance/reproducibility purposes only (e.g.
   verifying the existing split files could be regenerated identically from
   the raw corpus). It is guarded so it cannot silently overwrite the
   existing split files.
2. `project_to_existing_split()` builds NEW derived subsets (e.g. a future
   per-direction validation file) that stay consistent with the EXISTING
   concept_id -> split assignment, so new derived files never introduce
   leakage relative to the frozen master split.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class SplitImmutabilityError(Exception):
    """Raised when an operation would overwrite the frozen master split."""


class CorpusSplitter:
    """Deterministic splitting and split-consistent projection utilities."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize the splitter.

        Args:
            seed: Random seed used by `split_concepts` for reproducibility.
                This MUST match the seed originally used to produce
                `data/master_corpus/splits/` (42) if ever used to verify
                the existing split is reproducible from scratch.
        """
        self.seed = seed

    def split_concepts(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Deterministically split a concept-level DataFrame into train/val/test.

        Args:
            df: Full concept-level DataFrame (e.g. the master sentence corpus).
            train_ratio: Fraction assigned to the training split.
            val_ratio: Fraction assigned to the validation split (the
                remainder after train_ratio and val_ratio goes to test).

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train, val, test)
                DataFrames with a reset index.

        Raises:
            ValueError: If `train_ratio + val_ratio >= 1.0`.
        """
        if train_ratio + val_ratio >= 1.0:
            raise ValueError(
                f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must leave a "
                "positive fraction for the test split."
            )

        test_size = 1.0 - train_ratio
        train_df, temp_df = train_test_split(df, test_size=test_size, random_state=self.seed)

        val_fraction_of_temp = val_ratio / test_size
        val_df, test_df = train_test_split(temp_df, test_size=1 - val_fraction_of_temp, random_state=self.seed)

        logger.info(
            f"Split {len(df):,} rows -> train={len(train_df):,}, "
            f"val={len(val_df):,}, test={len(test_df):,}."
        )
        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    @staticmethod
    def build_split_assignment(
        train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, id_column: str = "concept_id"
    ) -> Dict[int, str]:
        """Build a concept_id -> split-name lookup from the existing split files.

        Args:
            train_df: Existing master_train.csv, loaded.
            val_df: Existing master_val.csv, loaded.
            test_df: Existing master_test.csv, loaded.
            id_column: Name of the unique identifier column.

        Returns:
            Dict[int, str]: Maps each concept_id to "train", "val", or "test".
        """
        assignment: Dict[int, str] = {}
        assignment.update({cid: "train" for cid in train_df[id_column]})
        assignment.update({cid: "val" for cid in val_df[id_column]})
        assignment.update({cid: "test" for cid in test_df[id_column]})
        return assignment

    @staticmethod
    def project_to_existing_split(
        df: pd.DataFrame, split_assignment: Dict[int, str], id_column: str = "concept_id"
    ) -> Dict[str, pd.DataFrame]:
        """Partition a new derived DataFrame using the frozen split assignment.

        Use this whenever generating a NEW derived subset (e.g. a
        language-pair projection) so it inherits train/val/test membership
        from the existing frozen split rather than being re-split
        independently -- the latter would risk the same concept appearing
        in both a derived train file and the master test file.

        Args:
            df: New derived DataFrame to partition (must contain `id_column`).
            split_assignment: Output of `build_split_assignment`.
            id_column: Name of the unique identifier column.

        Returns:
            Dict[str, pd.DataFrame]: Keys "train", "val", "test". Rows whose
                ID is not found in `split_assignment` are collected under
                the key "unassigned" for the caller to inspect/discard.
        """
        labels = df[id_column].map(split_assignment)
        result = {
            split_name: df[labels == split_name].reset_index(drop=True)
            for split_name in ("train", "val", "test")
        }
        unassigned = df[labels.isna()]
        if len(unassigned) > 0:
            result["unassigned"] = unassigned.reset_index(drop=True)
            logger.warning(f"{len(unassigned)} row(s) had no matching split assignment.")
        return result

    @staticmethod
    def assert_splits_not_overwritten(splits_dir: Path) -> None:
        """Guard against accidentally regenerating the frozen master split.

        Args:
            splits_dir: Directory expected to contain master_train.csv,
                master_val.csv, master_test.csv.

        Raises:
            SplitImmutabilityError: If all three master split files already
                exist -- callers must not proceed to write over them.
        """
        required = ["master_train.csv", "master_val.csv", "master_test.csv"]
        if all((splits_dir / f).exists() for f in required):
            raise SplitImmutabilityError(
                f"Master split files already exist in {splits_dir}. "
                "They must never be regenerated -- every experiment depends on "
                "evaluating against the exact same held-out set."
            )
