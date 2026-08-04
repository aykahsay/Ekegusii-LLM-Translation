"""
Qwen2.5-7B-Instruct Training Pipeline
-------------------------------------------
Wires together `QwenQLoRATrainer.load_model_and_tokenizer` (fresh QLoRA
init) with `src.models.common.run_qlora_training` (the shared training
loop) and `src.utils.config.load_qlora_config` (hyperparameters), giving a
single entry point to fine-tune Qwen2.5-7B-Instruct for one experiment.
"""

import logging
from pathlib import Path

from datasets import Dataset
from transformers import Trainer

from src.models.common import run_qlora_training
from src.models.qwen.qlora import QwenQLoRATrainer
from src.utils.config import load_qlora_config
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


class QwenTrainingPipeline:
    """End-to-end QLoRA training pipeline for Qwen2.5-7B-Instruct."""

    def __init__(self, output_dir: str = "checkpoints/qwen", seed: int = 42) -> None:
        """Initialize the pipeline.

        Args:
            output_dir: Directory for this run's checkpoints/logs (e.g.
                `checkpoints/qwen/E1_English_Ekegusii`).
            seed: Random seed applied before model/data construction.
        """
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.qlora_trainer = QwenQLoRATrainer(output_dir=str(self.output_dir))

    def run(
        self, train_dataset: Dataset, eval_dataset: Dataset, early_stopping_patience: int = 3
    ) -> Trainer:
        """Run QLoRA fine-tuning for Qwen2.5-7B-Instruct on the given tokenized datasets.

        Args:
            train_dataset: Tokenized training dataset (see `InstructionDatasetBuilder`,
                tokenized with THIS model's tokenizer -- see `qlora.py`).
            eval_dataset: Tokenized validation dataset.
            early_stopping_patience: Evaluations tolerated with no improvement.

        Returns:
            Trainer: Trained HuggingFace Trainer instance.
        """
        set_seed(self.seed)
        model, tokenizer = self.qlora_trainer.load_model_and_tokenizer()
        qlora_cfg = load_qlora_config("qwen")

        logger.info(f"[Qwen] Starting training run -> {self.output_dir}")
        return run_qlora_training(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            qlora_cfg=qlora_cfg,
            output_dir=self.output_dir,
            early_stopping_patience=early_stopping_patience,
        )
