"""
E5: Full Resources Experiment
------------------------------------
Named `mono.py` in the scaffold because it additionally exploits rows the
OTHER experiments discard: `TrilingualExperiment` (E4) restricts to
complete English-Kiswahili-Ekegusii triplets, while `BilingualExperiment`
(E1-E3) restricts to exactly one or two specific pairs. This experiment
instead uses `InstructionTaskGenerator` over the ENTIRE training split
unfiltered -- including rows where only one language pair is present
(effectively monolingual-sourced additions that never made it into a
complete triplet) -- maximizing total training signal. Tests whether more
data, even partially-aligned, beats the cleaner-but-smaller E4 triplet-only
set.
"""

import logging
from typing import Optional

import pandas as pd

from src.experiments.base import BaseExperiment
from src.master_corpus.manager import MasterCorpusManager
from src.task_generation.instruction_generator import InstructionTaskGenerator

logger = logging.getLogger(__name__)


class FullResourcesExperiment(BaseExperiment):
    """E5: 6-way bidirectional training on the full training split, no triplet filter."""

    experiment_id = "E5_Full_Resources"

    def __init__(self, manager: Optional[MasterCorpusManager] = None) -> None:
        """Initialize the experiment.

        Args:
            manager: Existing MasterCorpusManager to reuse. If None, a
                default-configured instance is created.
        """
        super().__init__(manager)
        self.task_generator = InstructionTaskGenerator(self.manager)

    def build_training_tasks(self) -> pd.DataFrame:
        """Build 6-way bidirectional instruction tasks from the entire training split.

        Returns:
            pd.DataFrame: Output of
                `InstructionTaskGenerator.generate_tasks_from_dataframe`
                over the full (unfiltered) training split -- rows missing
                one or two languages still contribute whichever
                direction(s) they have complete data for.
        """
        train_df = self.manager.load_train_split()
        tasks_df = self.task_generator.generate_tasks_from_dataframe(train_df)
        logger.info(f"[{self.experiment_id}] Built {len(tasks_df):,} tasks from full {len(train_df):,}-row split.")
        return tasks_df
