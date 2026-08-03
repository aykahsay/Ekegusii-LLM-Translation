"""
Training DataLoader Factory
--------------------------------
Wires together `InstructionDatasetBuilder` (tokenization),
`CausalLMDataCollator` (dynamic padding), and optionally
`WeightedDirectionSampler` (resource-weighted sampling) into a single
`torch.utils.data.DataLoader` ready to pass to a `Seq2SeqTrainer`/custom
training loop.
"""

import logging
from typing import Optional, Sequence

from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from src.datasets.collator import CausalLMDataCollator
from src.datasets.sampler import WeightedDirectionSampler

logger = logging.getLogger(__name__)


def build_train_dataloader(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 8,
    weights: Optional[Sequence[float]] = None,
    num_samples_per_epoch: Optional[int] = None,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    """Build a training DataLoader with dynamic padding and optional resource weighting.

    Args:
        dataset: Tokenized dataset (output of `InstructionDatasetBuilder.build`).
        tokenizer: Tokenizer used to build `dataset` (for pad token id/side).
        batch_size: Per-step batch size.
        weights: Optional per-row sampling weights (see
            `WeightedDirectionSampler.weights_from_task_types`). If None,
            standard shuffled sampling is used instead.
        num_samples_per_epoch: Number of examples to draw per epoch when
            `weights` is provided. Defaults to `len(dataset)` if None.
        seed: Random seed for the weighted sampler (ignored if `weights`
            is None, since `DataLoader(shuffle=True)` uses global torch RNG).
        num_workers: Number of DataLoader worker processes.

    Returns:
        DataLoader: Yields batches with keys "input_ids", "attention_mask",
            "labels" (see `CausalLMDataCollator`).

    Raises:
        ValueError: If `weights` is provided but its length doesn't match `len(dataset)`.
    """
    collator = CausalLMDataCollator(tokenizer=tokenizer)

    if weights is not None:
        if len(weights) != len(dataset):
            raise ValueError(f"len(weights)={len(weights)} must match len(dataset)={len(dataset)}.")
        sampler = WeightedDirectionSampler(
            weights=weights, num_samples=num_samples_per_epoch or len(dataset), seed=seed
        )
        loader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, collate_fn=collator, num_workers=num_workers  # type: ignore[arg-type]
        )
        logger.info(f"Built weighted-sampling DataLoader: {len(sampler):,} samples/epoch, batch_size={batch_size}.")
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, num_workers=num_workers  # type: ignore[arg-type]
        )
        logger.info(f"Built shuffled DataLoader: {len(dataset):,} examples, batch_size={batch_size}.")

    return loader


def build_eval_dataloader(
    dataset: Dataset, tokenizer: PreTrainedTokenizerBase, batch_size: int = 8, num_workers: int = 0
) -> DataLoader:
    """Build an evaluation DataLoader (no shuffling, no resource weighting).

    Args:
        dataset: Tokenized dataset (output of `InstructionDatasetBuilder.build`).
        tokenizer: Tokenizer used to build `dataset`.
        batch_size: Per-step batch size.
        num_workers: Number of DataLoader worker processes.

    Returns:
        DataLoader: Sequential (unshuffled) batches for deterministic evaluation.
    """
    collator = CausalLMDataCollator(tokenizer=tokenizer)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=num_workers  # type: ignore[arg-type]
    )
    logger.info(f"Built eval DataLoader: {len(dataset):,} examples, batch_size={batch_size}.")
    return loader
