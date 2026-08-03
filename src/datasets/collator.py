"""
Causal-LM Data Collator
---------------------------
Dynamically pads a batch of variable-length tokenized examples (from
`InstructionDatasetBuilder`) to the longest sequence in that batch --
left-padding `input_ids`/`attention_mask` (matching the tokenizer's
configured `padding_side`) and padding `labels` with -100 so padded
positions never contribute to the loss.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List

import torch
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


@dataclass
class CausalLMDataCollator:
    """Pads a batch of tokenized instruction-tuning examples for causal-LM training.

    Attributes:
        tokenizer: Tokenizer providing `pad_token_id` and `padding_side`.
    """

    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        """Collate a list of tokenized examples into a padded batch.

        Args:
            features: List of dicts with keys "input_ids", "attention_mask",
                "labels" (variable-length lists of ints), as produced by
                `InstructionDatasetBuilder`.

        Returns:
            Dict[str, torch.Tensor]: "input_ids", "attention_mask",
                "labels", each of shape (batch_size, max_seq_len_in_batch).

        Raises:
            ValueError: If `tokenizer.pad_token_id` is None.
        """
        if self.tokenizer.pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id must be set for batch collation.")

        max_len = max(len(f["input_ids"]) for f in features)
        pad_left = self.tokenizer.padding_side == "left"

        input_ids_batch, attention_mask_batch, labels_batch = [], [], []

        for feature in features:
            input_ids = feature["input_ids"]
            attention_mask = feature["attention_mask"]
            labels = feature["labels"]

            pad_len = max_len - len(input_ids)
            id_padding = [self.tokenizer.pad_token_id] * pad_len
            mask_padding = [0] * pad_len
            label_padding = [IGNORE_INDEX] * pad_len

            if pad_left:
                input_ids_batch.append(id_padding + input_ids)
                attention_mask_batch.append(mask_padding + attention_mask)
                labels_batch.append(label_padding + labels)
            else:
                input_ids_batch.append(input_ids + id_padding)
                attention_mask_batch.append(attention_mask + mask_padding)
                labels_batch.append(labels + label_padding)

        return {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
        }
