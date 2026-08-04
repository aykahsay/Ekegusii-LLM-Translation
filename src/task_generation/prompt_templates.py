"""
Inference-Time Prompt Formatting
-------------------------------------
Thin accessor over `configs/prompts/templates.yaml` (completion-style) and
`configs/prompts/translation.yaml` (chat-style) for formatting a SINGLE
translation prompt at inference time -- as opposed to
`InstructionTaskGenerator`, which bulk-generates prompt/response pairs for
an entire training split. Used by `src/models/*/inference.py` and the
evaluation pipeline.
"""

import logging
from typing import List, Tuple

from src.utils.config import load_prompt_templates as _load_templates
from src.utils.config_dict import ConfigDict, load_yaml
from src.utils.constants import CONFIGS_DIR, LANGUAGE_CODES

logger = logging.getLogger(__name__)


def _direction_key(source_lang: str, target_lang: str) -> str:
    """Build the templates.yaml key for a direction (e.g. "eng_to_eke")."""
    return f"{LANGUAGE_CODES[source_lang]}_to_{LANGUAGE_CODES[target_lang]}"


def format_completion_prompt(source_lang: str, target_lang: str, source_text: str) -> str:
    """Format a completion-style translation prompt for a base/causal-LM model.

    Args:
        source_lang: Source language (e.g. "English").
        target_lang: Target language (e.g. "Ekegusii").
        source_text: Sentence to translate.

    Returns:
        str: Formatted prompt from `configs/prompts/templates.yaml`.

    Raises:
        KeyError: If no template exists for the requested direction, or
            either language is not in `src.utils.constants.LANGUAGE_CODES`.
    """
    templates = _load_templates()["translation"]
    key = _direction_key(source_lang, target_lang)
    if key not in templates:
        raise KeyError(f"No prompt template found for direction '{key}'.")
    return templates[key].format(src=source_text)


def format_chat_messages(source_lang: str, target_lang: str, source_text: str) -> List[dict]:
    """Format a chat-style message list for an instruction-tuned model.

    Suitable for passing directly to `tokenizer.apply_chat_template()`.

    Args:
        source_lang: Source language (e.g. "English").
        target_lang: Target language (e.g. "Ekegusii").
        source_text: Sentence to translate.

    Returns:
        List[dict]: [{"role": "system", "content": ...}, {"role": "user",
            "content": ...}], using `configs/prompts/translation.yaml`.
    """
    cfg: ConfigDict = load_yaml(CONFIGS_DIR / "prompts" / "translation.yaml")
    user_content = cfg["user_turn_template"].format(src_lang=source_lang, tgt_lang=target_lang, src=source_text)
    return [
        {"role": "system", "content": cfg["system_prompt"]},
        {"role": "user", "content": user_content},
    ]


def available_directions() -> List[Tuple[str, str]]:
    """List the translation directions with a defined completion-style template.

    Returns:
        List[Tuple[str, str]]: (source_lang_code, target_lang_code) pairs
            (e.g. ("eng", "eke")) parsed from the template keys in
            `configs/prompts/templates.yaml`.
    """
    templates = _load_templates()["translation"]
    directions = []
    for key in templates:
        if key == "system_prompt" or "_to_" not in key:
            continue
        source_code, target_code = key.split("_to_")
        directions.append((source_code, target_code))
    return directions
