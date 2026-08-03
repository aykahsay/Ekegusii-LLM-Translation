"""
Shared Experiment Runner
----------------------------
Every experiment (E0-E8) follows the same shape -- build a training
dataset from some resource combination, tokenize it, fine-tune a model,
evaluate against the FIXED master test split, and persist results under
`experiments/{EXPERIMENT_ID}/`. What differs between experiments is only
WHICH resources feed the training set. This base class holds that common
shape once; `baseline.py`, `bilingual.py`, `trilingual.py`, etc. each
implement just `build_training_tasks()`.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.datasets.builder import InstructionDatasetBuilder
from src.evaluation.chrf import ChrFEvaluator
from src.evaluation.lexical_accuracy import compute_lexical_accuracy
from src.evaluation.sacrebleu import SacreBLEUEvaluator
from src.master_corpus.manager import MasterCorpusManager
from src.task_generation.translation_pairs import extract_pairs
from src.utils.constants import EXPERIMENTS_DIR
from src.utils.helpers import write_json

logger = logging.getLogger(__name__)


class BaseExperiment(ABC):
    """Common scaffold for a single E0-E8 experiment run.

    Attributes:
        experiment_id: One of `src.utils.constants.EXPERIMENT_IDS`.
        output_dir: `experiments/{experiment_id}/`, created on init.
    """

    experiment_id: str = "E_UNSET"

    def __init__(self, manager: Optional[MasterCorpusManager] = None) -> None:
        """Initialize the experiment.

        Args:
            manager: Existing MasterCorpusManager to reuse. If None, a
                default-configured instance is created.
        """
        self.manager = manager or MasterCorpusManager()
        self.output_dir = EXPERIMENTS_DIR / self.experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def build_training_tasks(self) -> pd.DataFrame:
        """Build this experiment's training instruction tasks.

        Returns:
            pd.DataFrame: Columns "prompt", "response" (see
                `InstructionTaskGenerator`/`LexicalTaskGenerator` output
                schema), ready for `InstructionDatasetBuilder`.
        """
        raise NotImplementedError

    def build_test_pairs(self, source_lang: str = "English", target_lang: str = "Ekegusii") -> pd.DataFrame:
        """Extract evaluation pairs from the FIXED master test split.

        Every experiment MUST evaluate against this same test split
        (never a derived/experiment-specific test set) for results to be
        comparable across E0-E8.

        Args:
            source_lang: Source language column.
            target_lang: Target language column.

        Returns:
            pd.DataFrame: Columns "source", "target" (see `extract_pairs`).
        """
        test_df = self.manager.load_test_split()
        return extract_pairs(test_df, source_lang, target_lang)

    def build_tokenized_dataset(self, tasks_df: pd.DataFrame, tokenizer: Any, max_length: int = 512):
        """Tokenize this experiment's training tasks for a given model's tokenizer.

        Args:
            tasks_df: Output of `build_training_tasks`.
            tokenizer: Tokenizer to encode with (Aya or Llama).
            max_length: Maximum sequence length (prompt + response).

        Returns:
            datasets.Dataset: Tokenized dataset ready for `run_qlora_training`.
        """
        builder = InstructionDatasetBuilder(tokenizer, max_length=max_length)
        return builder.build(tasks_df)

    def evaluate_predictions(
        self, predictions: list, references: list
    ) -> Dict[str, Any]:
        """Compute SacreBLEU + chrF + lexical accuracy for a set of predictions.

        Args:
            predictions: Model translations, aligned with `references`.
            references: Reference translations from `build_test_pairs`.

        Returns:
            Dict[str, Any]: Merged SacreBLEU/chrF/lexical-accuracy results.
        """
        bleu = SacreBLEUEvaluator().compute_corpus_bleu(predictions, references)
        chrf = ChrFEvaluator().compute_chrf(predictions, references)
        lexical = compute_lexical_accuracy(list(zip(predictions, references)), self.manager)

        return {**bleu, **chrf, **lexical}

    def save_results(self, results: Dict[str, Any]) -> Path:
        """Persist this experiment's evaluation results.

        Args:
            results: Metrics dict (e.g. output of `evaluate_predictions`),
                per model (aya/llama) if both were run.

        Returns:
            Path: `experiments/{experiment_id}/results.json`.
        """
        output_path = self.output_dir / "results.json"
        write_json({"experiment_id": self.experiment_id, **results}, output_path)
        logger.info(f"[{self.experiment_id}] Saved results to {output_path}.")
        return output_path


class SentenceLevelExperiment(BaseExperiment):
    """Common base for experiments whose training data is sentence-level pairs."""

    def build_instruction_tasks_from_pairs(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """Wrap raw (source, target) pairs into an instruction-task DataFrame.

        Args:
            pairs_df: Output of `extract_pairs`/`extract_all_directions`
                (columns "source", "target", "source_lang", "target_lang").

        Returns:
            pd.DataFrame: Columns "prompt", "response", built via
                `src.task_generation.prompt_templates.format_completion_prompt`.
        """
        from src.task_generation.prompt_templates import format_completion_prompt

        prompts = [
            format_completion_prompt(row["source_lang"], row["target_lang"], row["source"])
            for _, row in pairs_df.iterrows()
        ]
        return pd.DataFrame({"prompt": prompts, "response": pairs_df["target"].tolist()})
