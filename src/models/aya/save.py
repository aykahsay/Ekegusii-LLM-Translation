"""
Aya-23 Checkpoint & Deployment Saving
-----------------------------------------
Saves training checkpoints (via `CheckpointManager`) and, for the final
deployed model, merges the trained LoRA adapter into the base model
weights (via `src.models.common.merge_and_save_adapter`) so serving does
not require PEFT at inference time.
"""

import logging
from pathlib import Path
from typing import Any, Dict

from transformers import PreTrainedTokenizerBase

from src.models.common import merge_and_save_adapter
from src.utils.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


def save_aya_checkpoint(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    checkpoint_dir: str,
    step: int,
    epoch: float,
    metrics: Dict[str, float],
    is_best: bool = False,
) -> Path:
    """Save a training checkpoint for Aya-23-8B.

    Args:
        model: PEFT-wrapped model to save (only the adapter is persisted).
        tokenizer: Tokenizer to save alongside the adapter.
        checkpoint_dir: Root checkpoint directory for this run (e.g.
            `checkpoints/aya/E1_English_Ekegusii`).
        step: Global training step.
        epoch: Fractional epoch.
        metrics: Validation metrics recorded at this step.
        is_best: Whether this is the best checkpoint so far.

    Returns:
        Path: Directory the checkpoint was written to.
    """
    manager = CheckpointManager(Path(checkpoint_dir))
    return manager.save(model, tokenizer, step, epoch, metrics, is_best)


def save_aya_final_model(model: Any, tokenizer: PreTrainedTokenizerBase, output_dir: str = "models/aya_final") -> Path:
    """Merge the trained adapter into the base model and save for deployment.

    Args:
        model: A `PeftModel` with a trained adapter attached.
        tokenizer: Tokenizer to save alongside the merged model.
        output_dir: Destination directory for the merged, deployable model.

    Returns:
        Path: `output_dir`, containing the standalone merged model.
    """
    result = merge_and_save_adapter(model, tokenizer, Path(output_dir))
    logger.info(f"[Aya] Final deployable model saved to {result}.")
    return result
