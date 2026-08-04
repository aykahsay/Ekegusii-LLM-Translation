"""
Qwen2.5-7B-Instruct Translation Inference
------------------------------------------------
High-level "translate a batch of sentences" entry point for
Qwen2.5-7B-Instruct, combining `load_qwen_for_inference`, prompt
formatting, and `src.models.common.generate_translations`.
"""

import logging
from typing import List, Optional

from src.models.common import generate_translations
from src.models.qwen.load import load_qwen_for_inference
from src.task_generation.prompt_templates import format_completion_prompt
from src.utils.config import load_generation_config

logger = logging.getLogger(__name__)


def translate_with_qwen(
    sentences: List[str],
    source_lang: str,
    target_lang: str,
    adapter_path: Optional[str] = None,
    generation_profile: str = "default",
    batch_size: int = 8,
) -> List[str]:
    """Translate a list of sentences with Qwen2.5-7B-Instruct.

    Args:
        sentences: Source-language sentences to translate.
        source_lang: Source language (e.g. "English").
        target_lang: Target language (e.g. "Ekegusii").
        adapter_path: Optional trained LoRA adapter checkpoint path. If
            None, uses the zero-shot base model.
        generation_profile: Decoding profile name (see
            `configs/generation/`), e.g. "default", "beam_search", "greedy".
        batch_size: Number of sentences translated per forward pass.

    Returns:
        List[str]: Translated sentences, same order and length as `sentences`.
    """
    model, tokenizer = load_qwen_for_inference(adapter_path)
    generation_cfg = load_generation_config(generation_profile)

    prompts = [format_completion_prompt(source_lang, target_lang, s) for s in sentences]
    translations = generate_translations(model, tokenizer, prompts, generation_cfg, batch_size=batch_size)

    logger.info(f"[Qwen] Translated {len(sentences):,} sentence(s): {source_lang} -> {target_lang}.")
    return translations
