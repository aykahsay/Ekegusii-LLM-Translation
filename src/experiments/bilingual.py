"""
E1/E2/E3: Bilingual Experiments
------------------------------------
Covers the three bilingual resource configurations:
  E1_English_Ekegusii: trained only on English<->Ekegusii pairs.
  E2_Swahili_Ekegusii: trained only on Kiswahili<->Ekegusii pairs.
  E3_Bilingual: trained on BOTH pairs combined (tests whether stacking two
                bilingual resources beats either alone).
All three share the same shape (bidirectional pairs, no trilingual/lexical
supervision), differing only in which pair(s) feed training -- hence one
parametrized class rather than three near-identical files.
"""

import logging
from typing import Literal

import pandas as pd

from src.experiments.base import SentenceLevelExperiment
from src.task_generation.translation_pairs import extract_pairs

logger = logging.getLogger(__name__)

BilingualMode = Literal["eng_eke", "swa_eke", "combined"]

_EXPERIMENT_IDS = {
    "eng_eke": "E1_English_Ekegusii",
    "swa_eke": "E2_Swahili_Ekegusii",
    "combined": "E3_Bilingual",
    "eng_swa": "E10_Model_A_English_Swahili",
}


class BilingualExperiment(SentenceLevelExperiment):
    """E1/E2/E3/E10: bidirectional training on one or both bilingual pairs."""

    def __init__(self, mode: str, **kwargs) -> None:
        if mode not in _EXPERIMENT_IDS:
            raise ValueError(f"mode must be one of {list(_EXPERIMENT_IDS)}, got '{mode}'.")
        self.mode = mode
        self.experiment_id = _EXPERIMENT_IDS[mode]
        super().__init__(**kwargs)

    def build_training_tasks(self) -> pd.DataFrame:
        """Build bidirectional instruction tasks for this experiment's pair(s).

        Returns:
            pd.DataFrame: Columns "prompt", "response".
        """
        train_df = self.manager.load_train_split()
        frames = []

        if self.mode in ("eng_eke", "combined"):
            frames.append(extract_pairs(train_df, "English", "Ekegusii"))
            frames.append(extract_pairs(train_df, "Ekegusii", "English"))
        if self.mode in ("swa_eke", "combined"):
            frames.append(extract_pairs(train_df, "Kiswahili", "Ekegusii"))
            frames.append(extract_pairs(train_df, "Ekegusii", "Kiswahili"))
        if self.mode == "eng_swa":
            frames.append(extract_pairs(train_df, "English", "Kiswahili"))
            frames.append(extract_pairs(train_df, "Kiswahili", "English"))

        pairs_df = pd.concat(frames, ignore_index=True)
        tasks_df = self.build_instruction_tasks_from_pairs(pairs_df)
        logger.info(f"[{self.experiment_id}] Built {len(tasks_df):,} training tasks (mode={self.mode}).")
        return tasks_df
