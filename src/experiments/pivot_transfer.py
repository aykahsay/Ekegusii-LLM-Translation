"""
E10: Three-Model Pivot Transfer Experiment (Model A, Model B, Model C)
-----------------------------------------------------------------------
Model A: Pre-trained on (English ↔ Kiswahili) pivot data.
Model B: Adapts Model A weights to (English ↔ Ekegusii) target data.
Model C: Adapts Model A weights to (Kiswahili ↔ Ekegusii) target data.

All three models (A, B, C) and their training/validation loss CSVs are saved,
plotted, uploaded to Hugging Face, and pushed to GitHub.
"""

import logging
from typing import Optional, Dict, Any
import pandas as pd

from src.experiments.base import BaseExperiment
from src.experiments.bilingual import BilingualExperiment
from src.master_corpus.manager import MasterCorpusManager

logger = logging.getLogger(__name__)


class PivotTransferExperimentABC(BaseExperiment):
    """E10: Three-Model Pivot Transfer Experiment (Model A, B, C)."""

    experiment_id = "E10_Pivot_Transfer_ABC"

    def __init__(self, manager: Optional[MasterCorpusManager] = None) -> None:
        """Initialize the 3-model pivot transfer experiment."""
        super().__init__(manager)
        self.exp_model_a = BilingualExperiment("eng_swa", manager=self.manager)
        self.exp_model_b = BilingualExperiment("eng_eke", manager=self.manager)
        self.exp_model_c = BilingualExperiment("swa_eke", manager=self.manager)

    def build_training_tasks_for_model_a(self) -> pd.DataFrame:
        """Build Model A training tasks (English ↔ Kiswahili)."""
        tasks_df = self.exp_model_a.build_training_tasks()
        logger.info(f"[{self.experiment_id}] Model A (English ↔ Kiswahili) tasks: {len(tasks_df):,} tasks.")
        return tasks_df

    def build_training_tasks_for_model_b(self) -> pd.DataFrame:
        """Build Model B training tasks (English ↔ Ekegusii)."""
        tasks_df = self.exp_model_b.build_training_tasks()
        logger.info(f"[{self.experiment_id}] Model B (English ↔ Ekegusii) tasks: {len(tasks_df):,} tasks.")
        return tasks_df

    def build_training_tasks_for_model_c(self) -> pd.DataFrame:
        """Build Model C training tasks (Kiswahili ↔ Ekegusii)."""
        tasks_df = self.exp_model_c.build_training_tasks()
        logger.info(f"[{self.experiment_id}] Model C (Kiswahili ↔ Ekegusii) tasks: {len(tasks_df):,} tasks.")
        return tasks_df

    def build_training_tasks(self) -> pd.DataFrame:
        """Default fallback returns Model B target adaptation tasks."""
        return self.build_training_tasks_for_model_b()
