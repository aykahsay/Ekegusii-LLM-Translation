"""
E7: Curriculum Learning Experiment
----------------------------------------
Uses the SAME data as E6 (Lexical Augmentation), but orders it via
`DifficultyCurriculum` (easy-to-hard, by response length + rare-word
fraction) instead of E6's random shuffle. Tests whether curriculum
ordering improves convergence/final quality over an identical resource
pool -- isolating the ordering variable from the resource-composition
variable that E0-E6 vary.
"""

import logging
from typing import Optional

import pandas as pd

from src.experiments.lexical import LexicalAugmentationExperiment
from src.master_corpus.curriculum import DifficultyCurriculum
from src.master_corpus.manager import MasterCorpusManager

logger = logging.getLogger(__name__)


class CurriculumLearningExperiment(LexicalAugmentationExperiment):
    """E7: E6's data reordered easy-to-hard via DifficultyCurriculum."""

    experiment_id = "E7_Curriculum_Learning"

    def __init__(self, manager: Optional[MasterCorpusManager] = None, num_stages: int = 4) -> None:
        """Initialize the experiment.

        Args:
            manager: Existing MasterCorpusManager to reuse. If None, a
                default-configured instance is created.
            num_stages: Number of progressive difficulty buckets (see
                `DifficultyCurriculum`).
        """
        super().__init__(manager)
        self.curriculum = DifficultyCurriculum(num_stages=num_stages)

    def build_training_tasks(self) -> pd.DataFrame:
        """Build E6's mixed tasks, reordered from easiest to hardest.

        Returns:
            pd.DataFrame: E6's "prompt"/"response" tasks with added
                "difficulty_score" and "curriculum_stage" columns, SORTED
                by increasing difficulty (a Trainer using this dataset
                without further shuffling will see easy examples first).
        """
        mixed_tasks = super().build_training_tasks()
        staged = self.curriculum.assign_stages(mixed_tasks, text_column="response")
        logger.info(
            f"[{self.experiment_id}] Staged {len(staged):,} tasks into "
            f"{staged['curriculum_stage'].nunique()} curriculum stages."
        )
        return staged
