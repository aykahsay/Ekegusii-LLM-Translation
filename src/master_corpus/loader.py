"""
Config-Driven Dataset Loader
------------------------------
Bridges `configs/datasets/*.yaml` (dataset selection/config) to
`MasterCorpusManager` (fixed-schema master corpus access). While
`MasterCorpusManager` only knows the fixed master corpus filenames, this
loader resolves a *named* dataset config (e.g. "bilingual", "trilingual",
"lexical") to the correct derived-split DataFrame(s) -- so callers can write
`UnifiedDatasetLoader().load("bilingual")` instead of hardcoding which CSV
backs each experiment's resource configuration.
"""

import logging
from typing import Dict, Optional

import pandas as pd

from src.master_corpus.manager import MasterCorpusManager
from src.utils.config import load_dataset_config

logger = logging.getLogger(__name__)


class UnifiedDatasetLoader:
    """Resolves a named dataset config to its underlying DataFrame(s)."""

    def __init__(self, manager: Optional[MasterCorpusManager] = None) -> None:
        """Initialize with an optional MasterCorpusManager instance.

        Args:
            manager: Existing MasterCorpusManager to reuse. If None, a
                default-configured instance is created.
        """
        self.manager = manager or MasterCorpusManager()

    def load_train(self, dataset_name: str) -> pd.DataFrame:
        """Load the training data for a named dataset config.

        Args:
            dataset_name: One of "master", "bilingual", "trilingual",
                "lexical" (filename stem under `configs/datasets/`).

        Returns:
            pd.DataFrame: Training data. For "bilingual", the two configured
                pairs are concatenated with a `pair` column identifying
                which pair each row came from.

        Raises:
            ValueError: If `dataset_name` is unrecognized or its config is
                missing a required key for this operation.
        """
        cfg = load_dataset_config(dataset_name)

        if dataset_name == "master":
            return self.manager.load_train_split()

        if dataset_name == "trilingual":
            return self.manager.load_derived_split("derived_train_trilingual.csv")

        if dataset_name == "bilingual":
            frames: list[pd.DataFrame] = []
            for pair_name, pair_cfg in cfg["pairs"].items():
                filename = pair_cfg["train_path"].split("/")[-1]
                df = self.manager.load_derived_split(filename)
                df = df.copy()
                df["pair"] = pair_name
                frames.append(df)
            combined: pd.DataFrame = pd.concat(frames, ignore_index=True)
            logger.info(f"Loaded bilingual training data: {len(combined):,} rows across {len(frames)} pairs.")
            return combined

        if dataset_name == "lexical":
            return self.manager.load_lexical_corpus()

        raise ValueError(f"No training-data resolution rule for dataset '{dataset_name}'.")

    def load_eval_splits(self, dataset_name: str) -> Dict[str, pd.DataFrame]:
        """Load the fixed val/test evaluation splits for a named dataset config.

        Every dataset config's `eval_source` (or "master" for the master
        config itself) resolves to the SAME master val/test files -- this is
        intentional: every experiment must be scored against the identical
        held-out set regardless of which resources it trained on.

        Args:
            dataset_name: One of "master", "bilingual", "trilingual",
                "lexical".

        Returns:
            Dict[str, pd.DataFrame]: Keys "val" and "test".
        """
        cfg = load_dataset_config(dataset_name)
        eval_source = cfg.get("eval_source", "master") if dataset_name != "master" else "master"

        if eval_source != "master":
            raise ValueError(
                f"Dataset '{dataset_name}' declares eval_source='{eval_source}', "
                "but only 'master' is currently supported to guarantee a single "
                "shared held-out set across experiments."
            )

        return {
            "val": self.manager.load_val_split(),
            "test": self.manager.load_test_split(),
        }
