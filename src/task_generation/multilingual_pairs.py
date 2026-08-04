"""
Weighted Multilingual Pair Mixing
-------------------------------------
Bridges `translation_pairs.py` (raw pair extraction) and
`src.master_corpus.scheduler.ResourceScheduler` (direction-weighted
sampling) to build a single resource-mixed training pair dataset per
`configs/training/multilingual.yaml` -- used by experiments that train on
more than one direction simultaneously (E3 Combined Bilingual onward).
"""

import logging
from typing import Dict

import pandas as pd

from src.utils.bootstrap import ensure_package

ensure_package("omegaconf", "omegaconf==2.3.0")
ensure_package("hydra", "hydra-core==1.3.2")
from omegaconf import DictConfig

from src.master_corpus.scheduler import ResourceScheduler
from src.task_generation.translation_pairs import extract_pairs
from src.utils.constants import LANGUAGE_CODES, TRANSLATION_DIRECTIONS

logger = logging.getLogger(__name__)


def _direction_config_key(source_lang: str, target_lang: str) -> str:
    """Build the lowercase config key for a direction (e.g. "eng_to_eke")."""
    return f"{LANGUAGE_CODES[source_lang]}_to_{LANGUAGE_CODES[target_lang]}"


class MultilingualPairMixer:
    """Builds a single weighted-mixture pair dataset across all translation directions."""

    def __init__(self, multilingual_cfg: DictConfig, seed: int = 42) -> None:
        """Initialize the mixer.

        Args:
            multilingual_cfg: Parsed `configs/training/multilingual.yaml`.
            seed: Random seed for reproducible mixing.
        """
        self.scheduler = ResourceScheduler(multilingual_cfg, seed=seed)

    def build_direction_pairs(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Extract pairs for every direction, keyed by config-style direction name.

        Args:
            df: Concept-level DataFrame with English, Kiswahili, Ekegusii columns.

        Returns:
            Dict[str, pd.DataFrame]: Maps direction config key (e.g.
                "eng_to_eke") to its extracted pair DataFrame.
        """
        direction_pairs = {}
        for source_lang, target_lang in TRANSLATION_DIRECTIONS:
            key = _direction_config_key(source_lang, target_lang)
            direction_pairs[key] = extract_pairs(df, source_lang, target_lang)
        return direction_pairs

    def build_weighted_mixture(self, df: pd.DataFrame, target_total: int) -> pd.DataFrame:
        """Build a single weighted-mixture DataFrame of a target total size.

        Args:
            df: Concept-level DataFrame with English, Kiswahili, Ekegusii columns.
            target_total: Desired total number of rows in the mixture.

        Returns:
            pd.DataFrame: Concatenation of per-direction samples drawn
                according to `configs/training/multilingual.yaml`'s
                `direction_weights`/`sampling_temperature`, shuffled.
                Columns: "concept_id" (if available), "source", "target",
                "source_lang", "target_lang".
        """
        direction_pairs = self.build_direction_pairs(df)
        direction_counts = {key: len(pairs) for key, pairs in direction_pairs.items() if len(pairs) > 0}

        if not direction_counts:
            logger.warning("No non-empty translation directions found; returning empty mixture.")
            return pd.DataFrame(columns=["source", "target", "source_lang", "target_lang"])

        quotas = self.scheduler.build_mixed_batch_plan(direction_counts, target_total)

        frames = []
        for direction_key, quota in quotas.items():
            pool = direction_pairs[direction_key]
            n = min(quota, len(pool))
            if n > 0:
                frames.append(pool.sample(n=n, random_state=42))

        mixture = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        mixture = mixture.sample(frac=1.0, random_state=42).reset_index(drop=True)
        logger.info(f"Built weighted mixture: {len(mixture):,} rows (target was {target_total:,}).")
        return mixture
