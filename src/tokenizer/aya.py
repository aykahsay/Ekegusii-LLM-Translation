"""
Aya-23 Tokenizer Loader
--------------------------
Loads the Cohere Aya-23-8B tokenizer configured per `configs/models/aya_8b.yaml`
(padding side, truncation side, max length), for use by tokenizer-analysis
notebooks and the training/inference pipeline alike so both always load the
tokenizer identically configured.
"""

import logging

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.utils.config import load_model_config
from src.utils.hub_auth import raise_with_access_guidance

logger = logging.getLogger(__name__)


def load_aya_tokenizer() -> PreTrainedTokenizerBase:
    """Load the Aya-23-8B tokenizer with project-standard configuration.

    Returns:
        PreTrainedTokenizerBase: Configured tokenizer with `padding_side`,
            `truncation_side`, and `model_max_length` set from
            `configs/models/aya_8b.yaml`.

    Raises:
        OSError: If the model repository cannot be reached/downloaded
            (e.g. no internet access or gated-repo access not granted).
    """
    cfg = load_model_config("aya")

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
        logger.info("Aya tokenizer had no pad_token; defaulted to eos_token.")

    logger.info(f"Loaded Aya-23 tokenizer from '{cfg.model.hf_path}' (vocab_size={len(tokenizer):,}).")
    return tokenizer
