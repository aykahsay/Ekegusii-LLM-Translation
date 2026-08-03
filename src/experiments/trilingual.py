"""
E4: Trilingual Multi-Task Experiment
------------------------------------------
Trains on complete English-Kiswahili-Ekegusii triplets using the full
6-way bidirectional instruction task set (`InstructionTaskGenerator`),
testing whether joint multilingual supervision outperforms the E3
combined-bilingual configuration -- the trilingual triplets provide
cross-lingual signal (e.g. English<->Kiswahili alignment) that pure
bilingual pairs cannot.
"""

import logging
from typing import Optional

import pandas as pd

from src.experiments.base import BaseExperiment
from src.master_corpus.manager import MasterCorpusManager
from src.task_generation.instruction_generator import InstructionTaskGenerator

logger = logging.getLogger(__name__)


class TrilingualExperiment(BaseExperiment):
    """E4: 6-way bidirectional training on complete trilingual triplets."""

    experiment_id = "E4_Trilingual"

    def __init__(self, manager: Optional[MasterCorpusManager] = None) -> None:
        """Initialize the experiment.

        Args:
            manager: Existing MasterCorpusManager to reuse. If None, a
                default-configured instance is created.
        """
        super().__init__(manager)
        self.task_generator = InstructionTaskGenerator(self.manager)

    def build_training_tasks(self) -> pd.DataFrame:
        """Build 6-way bidirectional instruction tasks from the trilingual split.

        Returns:
            pd.DataFrame: Output of
                `InstructionTaskGenerator.generate_tasks_from_dataframe`
                (columns "concept_id", "task_type", "prompt", "response"),
                restricted to rows with complete English/Kiswahili/Ekegusii
                triplets.
        """
        train_df = self.manager.load_train_split()
        complete_triplets = train_df.dropna(subset=["English", "Kiswahili", "Ekegusii"])

        tasks_df = self.task_generator.generate_tasks_from_dataframe(complete_triplets)
        logger.info(
            f"[{self.experiment_id}] Built {len(tasks_df):,} tasks from "
            f"{len(complete_triplets):,} complete triplets."
        )
        return tasks_df
