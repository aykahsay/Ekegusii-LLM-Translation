"""
Llama-3.1 Inference Model Loader
------------------------------------
Thin wrapper over `src.models.common.load_adapter_for_inference` bound to
Llama-3.1-8B-Instruct, mirroring `src.models.qwen.load`.
"""

import logging
from typing import Any, Optional, Tuple

from transformers import PreTrainedTokenizerBase

from src.models.common import load_adapter_for_inference
from src.models.llama.qlora import LlamaQLoRATrainer

logger = logging.getLogger(__name__)


def load_llama_for_inference(adapter_path: Optional[str] = None) -> Tuple[Any, PreTrainedTokenizerBase]:
    """Load Llama-3.1-8B-Instruct for inference, optionally with a trained LoRA adapter.

    Args:
        adapter_path: Path to a saved PEFT adapter checkpoint. If None,
            loads the raw base model (zero-shot baseline, E0).

    Returns:
        Tuple[Any, PreTrainedTokenizerBase]: (model, tokenizer).
    """
    return load_adapter_for_inference(LlamaQLoRATrainer.MODEL_ID, adapter_path)
