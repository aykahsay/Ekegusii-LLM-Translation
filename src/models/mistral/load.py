"""
Mistral-7B-Instruct-v0.3 Inference Model Loader
----------------------------------------------------
Thin wrapper over `src.models.common.load_adapter_for_inference` bound to
Mistral-7B-Instruct-v0.3, mirroring `src.models.qwen.load`.
"""

import logging
from typing import Any, Optional, Tuple

from transformers import PreTrainedTokenizerBase

from src.models.common import load_adapter_for_inference
from src.models.mistral.qlora import MistralQLoRATrainer

logger = logging.getLogger(__name__)


def load_mistral_for_inference(adapter_path: Optional[str] = None) -> Tuple[Any, PreTrainedTokenizerBase]:
    """Load Mistral-7B-Instruct-v0.3 for inference, optionally with a trained LoRA adapter.

    Args:
        adapter_path: Path to a saved PEFT adapter checkpoint. If None,
            loads the raw base model (zero-shot baseline, E0).

    Returns:
        Tuple[Any, PreTrainedTokenizerBase]: (model, tokenizer).
    """
    return load_adapter_for_inference(MistralQLoRATrainer.MODEL_ID, adapter_path)
