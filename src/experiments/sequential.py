"""
E9: Sequential Transfer Experiment (Two-Stage Transfer Fine-Tuning)
---------------------------------------------------------------------
Stage 1: Pre-tunes Qwen/Mistral on Swahili-Ekegusii (E2) or English-Swahili pivot data to establish Bantu syntax & structural alignment.
Stage 2: Continues QLoRA fine-tuning on English-Ekegusii target data (E1).

Tests whether sequential 2-stage transfer fine-tuning beats 1-stage joint trilingual tuning (E4) or direct single-pair tuning (E1).
"""

import logging
from typing import Optional
import pandas as pd

from src.experiments.base import BaseExperiment
from src.experiments.bilingual import BilingualExperiment
from src.master_corpus.manager import MasterCorpusManager

logger = logging.getLogger(__name__)


class SequentialTransferExperiment(BaseExperiment):
    """E9: Two-Stage Sequential Pivot Transfer Fine-Tuning."""

    experiment_id = "E9_Sequential_Transfer"

    def __init__(self, manager: Optional[MasterCorpusManager] = None) -> None:
        """Initialize the 2-stage sequential transfer experiment."""
        super().__init__(manager)
        self.stage1_exp_id = "E2_Swahili_Ekegusii"
        self.target_bilingual = BilingualExperiment("eng_eke", manager=self.manager)

    def build_training_tasks(self) -> pd.DataFrame:
        """Build Stage 2 target adaptation tasks (English ↔ Ekegusii).

        Returns:
            pd.DataFrame: Instruction tasks for Stage 2 target adaptation.
        """
        tasks_df = self.target_bilingual.build_training_tasks()
        logger.info(
            f"[{self.experiment_id}] Stage 2 target tasks built: {len(tasks_df):,} tasks. "
            f"(Stage 1 pivot adapter loaded from '{self.stage1_exp_id}')."
        )
        return tasks_df
