"""
Mistral-7B-Instruct-v0.3 Training Pipeline
-----------------------------------------------
Wires together `MistralQLoRATrainer.load_model_and_tokenizer` with
`src.models.common.run_qlora_training`, mirroring `src.models.qwen.trainer`.
"""

import logging
from pathlib import Path

from datasets import Dataset
from transformers import Trainer

from src.models.common import run_qlora_training
from src.models.mistral.qlora import MistralQLoRATrainer
from src.utils.config import load_qlora_config
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


class MistralTrainingPipeline:
    """End-to-end QLoRA training pipeline for Mistral-7B-Instruct-v0.3."""

    def __init__(self, output_dir: str = "checkpoints/mistral", seed: int = 42) -> None:
        """Initialize the pipeline.

        Args:
            output_dir: Directory for this run's checkpoints/logs (e.g.
                `checkpoints/mistral/E1_English_Ekegusii`).
            seed: Random seed applied before model/data construction.
        """
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.qlora_trainer = MistralQLoRATrainer(output_dir=str(self.output_dir))

    def run(
        self, train_dataset: Dataset, eval_dataset: Dataset, early_stopping_patience: int = 3
    ) -> Trainer:
        """Run QLoRA fine-tuning for Mistral-7B-Instruct-v0.3 on the given tokenized datasets.

        Args:
            train_dataset: Tokenized training dataset, tokenized with THIS
                model's tokenizer (see `qlora.py`).
            eval_dataset: Tokenized validation dataset.
            early_stopping_patience: Evaluations tolerated with no improvement.

        Returns:
            Trainer: Trained HuggingFace Trainer instance.
        """
        set_seed(self.seed)
        model, tokenizer = self.qlora_trainer.load_model_and_tokenizer()
        qlora_cfg = load_qlora_config("mistral")

        logger.info(f"[Mistral] Starting training run -> {self.output_dir}")
        return run_qlora_training(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            qlora_cfg=qlora_cfg,
            output_dir=self.output_dir,
            early_stopping_patience=early_stopping_patience,
        )
