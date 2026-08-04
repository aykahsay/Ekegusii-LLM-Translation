"""
Llama-3.1 Tokenizer Loader
-----------------------------
Loads the Meta Llama-3.1-8B-Instruct tokenizer configured per
`configs/models/llama31_8b.yaml`, mirroring `src.tokenizer.aya` so both
models' tokenizers are loaded through an identical, config-driven code path.
"""

import logging

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.utils.config import load_model_config
from src.utils.hub_auth import raise_with_access_guidance

logger = logging.getLogger(__name__)


def load_llama_tokenizer() -> PreTrainedTokenizerBase:
    """Load the Llama-3.1-8B-Instruct tokenizer with project-standard configuration.

    Returns:
        PreTrainedTokenizerBase: Configured tokenizer with `padding_side`,
            `truncation_side`, and `model_max_length` set from
            `configs/models/llama31_8b.yaml`.

    Raises:
        OSError: If the model repository cannot be reached/downloaded, or
            gated-repo access has not been granted for the calling
            HuggingFace account (Llama models require accepting Meta's
            license on the Hub).
    """
    cfg = load_model_config("llama")

    try:
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            cfg.model.hf_path,
            use_fast=bool(cfg.model.use_fast_tokenizer),
            trust_remote_code=bool(cfg.model.trust_remote_code),
        )
    except OSError as exc:
        raise_with_access_guidance(exc, cfg.model.hf_path)
    tokenizer.padding_side = cfg.tokenizer.padding_side
    tokenizer.truncation_side = cfg.tokenizer.truncation_side
    tokenizer.model_max_length = int(cfg.tokenizer.max_length)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Llama tokenizer had no pad_token; defaulted to eos_token.")

    logger.info(f"Loaded Llama-3.1 tokenizer from '{cfg.model.hf_path}' (vocab_size={len(tokenizer):,}).")
    return tokenizer
