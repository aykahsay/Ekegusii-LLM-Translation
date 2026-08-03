"""
CLI: translate
------------------
Interactive single/batch sentence translation entry point, dispatching to
Aya or Llama inference.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def run_translate(
    sentences: List[str],
    source_lang: str,
    target_lang: str,
    model_name: str = "aya",
    adapter_path: Optional[str] = None,
    generation_profile: str = "default",
) -> List[str]:
    """Translate one or more sentences with the requested model.

    Args:
        sentences: Source-language sentences to translate.
        source_lang: Source language (e.g. "English").
        target_lang: Target language (e.g. "Ekegusii").
        model_name: "aya" or "llama".
        adapter_path: Optional trained LoRA adapter checkpoint path. If
            None, uses the zero-shot base model.
        generation_profile: Decoding profile name (see `configs/generation/`).

    Returns:
        List[str]: Translations, same order as `sentences`.

    Raises:
        ValueError: If `model_name` is not "aya" or "llama".
    """
    if model_name == "aya":
        from src.models.aya.inference import translate_with_aya

        return translate_with_aya(sentences, source_lang, target_lang, adapter_path, generation_profile)
    if model_name == "llama":
        from src.models.llama.inference import translate_with_llama

        return translate_with_llama(sentences, source_lang, target_lang, adapter_path, generation_profile)

    raise ValueError(f"model_name must be 'aya' or 'llama', got '{model_name}'.")
