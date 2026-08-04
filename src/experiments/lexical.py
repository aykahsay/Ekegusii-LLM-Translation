"""
E6: Lexical Augmentation Experiment
------------------------------------------
Builds on E5 (Full Resources) by additionally mixing in dictionary-level
instruction tasks from `LexicalTaskGenerator` (268 entries from
`master_lexical_corpus.csv`), per `configs/training/multilingual.yaml`'s
`lexical_augmentation.mix_ratio`. Tests whether explicit term-level
supervision improves rare-word/terminology translation accuracy beyond
what sentence-level data alone provides (H3).
"""

import logging
from typing import Optional

import pandas as pd

from src.experiments.mono import FullResourcesExperiment
from src.master_corpus.manager import MasterCorpusManager
from src.master_corpus.scheduler import ResourceScheduler
from src.task_generation.lexical_tasks import LexicalTaskGenerator
from src.utils.config_dict import load_yaml
from src.utils.constants import CONFIGS_DIR

logger = logging.getLogger(__name__)


class LexicalAugmentationExperiment(FullResourcesExperiment):
    """E6: E5's full-resource sentence tasks mixed with lexical-corpus tasks."""

    experiment_id = "E6_Lexical_Augmentation"

    def __init__(self, manager: Optional[MasterCorpusManager] = None) -> None:
        """Initialize the experiment.

        Args:
            manager: Existing MasterCorpusManager to reuse. If None, a
                default-configured instance is created.
        """
        super().__init__(manager)
        self.lexical_generator = LexicalTaskGenerator(self.manager)

    def build_training_tasks(self) -> pd.DataFrame:
        """Build E5's sentence tasks mixed with lexical-augmentation tasks.

        Returns:
            pd.DataFrame: Columns "prompt", "response" (task_type/lexicon_id/
                concept_id columns dropped for a uniform schema across the
                mixed sources before tokenization).
        """
        sentence_tasks = super().build_training_tasks()[["prompt", "response"]]
        lexical_tasks = self.lexical_generator.generate_all_tasks()[["prompt", "response"]]

        multilingual_cfg = load_yaml(CONFIGS_DIR / "training" / "multilingual.yaml")
        # This experiment IS the lexical-augmentation arm, so force it on
        # regardless of the shared config's default (which stays False for
        # E0-E5, which must NOT see lexical data).
        multilingual_cfg.lexical_augmentation.enabled = True
        scheduler = ResourceScheduler(multilingual_cfg)
        mixed = scheduler.mix_with_lexical(sentence_tasks, lexical_tasks)

        logger.info(
            f"[{self.experiment_id}] Mixed {len(sentence_tasks):,} sentence tasks with "
            f"{len(lexical_tasks):,} lexical tasks -> {len(mixed):,} total."
        )
        return mixed
