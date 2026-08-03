"""
HuggingFace Dataset Builder
------------------------------
Converts a prompt/response instruction-task DataFrame (output of
`InstructionTaskGenerator`, `LexicalTaskGenerator`, or
`MultilingualPairMixer`) into a tokenized HuggingFace `datasets.Dataset`
ready for causal-LM QLoRA training: concatenates prompt + response +
EOS, and masks the prompt portion's labels with -100 so the loss is only
computed on the response tokens (standard instruction-tuning practice --
otherwise the model would be partially rewarded for memorizing prompts).
"""

import logging
from typing import Dict, List

import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


class InstructionDatasetBuilder:
    """Builds a tokenized causal-LM training dataset from prompt/response pairs."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int = 512) -> None:
        """Initialize the builder.

        Args:
            tokenizer: Tokenizer to encode prompts/responses with. Must
                have a `pad_token` set (see `src.tokenizer.aya`/`llama`
                loaders, which set this automatically).
            max_length: Maximum total sequence length (prompt + response);
                longer examples are truncated from the left of the prompt
                to preserve the response and the most recent prompt context.

        Raises:
            ValueError: If `tokenizer.pad_token` is not set.
        """
        if tokenizer.pad_token is None:
            raise ValueError("tokenizer.pad_token must be set before building a dataset.")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _tokenize_example(self, prompt: str, response: str) -> Dict[str, List[int]]:
        """Tokenize a single (prompt, response) pair into input_ids/attention_mask/labels."""
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(
            response + (self.tokenizer.eos_token or ""), add_special_tokens=False
        )

        if len(response_ids) >= self.max_length:
            # Response alone doesn't fit: drop the prompt entirely and keep
            # only the START of the response (truncating the response harms
            # supervision, but a hard max_length must still be respected).
            logger.warning(
                f"Response alone ({len(response_ids)} tokens) >= max_length={self.max_length}; "
                "dropping prompt and truncating response."
            )
            prompt_ids = []
            response_ids = response_ids[: self.max_length]
        else:
            total_len = len(prompt_ids) + len(response_ids)
            if total_len > self.max_length:
                overflow = total_len - self.max_length
                prompt_ids = prompt_ids[overflow:]  # truncate from the LEFT of the prompt
                logger.debug(f"Truncated {overflow} prompt token(s) to fit max_length={self.max_length}.")

        input_ids = prompt_ids + response_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + list(response_ids)
        attention_mask = [1] * len(input_ids)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def build(self, df: pd.DataFrame, prompt_col: str = "prompt", response_col: str = "response") -> Dataset:
        """Build a tokenized `datasets.Dataset` from a prompt/response DataFrame.

        Args:
            df: DataFrame with at least `prompt_col` and `response_col`
                text columns.
            prompt_col: Column name containing the instruction prompt.
            response_col: Column name containing the target response.

        Returns:
            Dataset: HuggingFace Dataset with columns "input_ids",
                "attention_mask", "labels" (variable-length; pad via
                `src.datasets.collator.CausalLMDataCollator` at batch time).

        Raises:
            KeyError: If `prompt_col` or `response_col` is missing from `df`.
        """
        for col in (prompt_col, response_col):
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")

        records = [
            self._tokenize_example(str(row[prompt_col]), str(row[response_col]))
            for _, row in df.iterrows()
        ]
        dataset = Dataset.from_list(records)
        logger.info(f"Built tokenized dataset: {len(dataset):,} examples, max_length={self.max_length}.")
        return dataset
