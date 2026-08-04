"""
Mistral-7B-Instruct-v0.3 Tokenizer Loader
-----------------------------------------------
Loads the Mistral-7B-Instruct-v0.3 tokenizer configured per
`configs/models/mistral_7b.yaml`, mirroring `src.tokenizer.qwen` so both
models' tokenizers are loaded through an identical, config-driven code path.
"""

import logging

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.utils.config import load_model_config
from src.utils.hub_auth import raise_with_access_guidance

logger = logging.getLogger(__name__)


def load_mistral_tokenizer() -> PreTrainedTokenizerBase:
    """Load the Mistral-7B-Instruct-v0.3 tokenizer with project-standard configuration.

    Returns:
        PreTrainedTokenizerBase: Configured tokenizer with `padding_side`,
            `truncation_side`, and `model_max_length` set from
            `configs/models/mistral_7b.yaml`.

    Raises:
        OSError: If the model repository cannot be reached/downloaded (the
            model itself is fully open on HuggingFace Hub, so this should
            only happen for connectivity issues).
    """
    cfg = load_model_config("mistral")

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
        logger.info("Mistral tokenizer had no pad_token; defaulted to eos_token.")

    logger.info(f"Loaded Mistral-7B-Instruct-v0.3 tokenizer from '{cfg.model.hf_path}' (vocab_size={len(tokenizer):,}).")
    return tokenizer
