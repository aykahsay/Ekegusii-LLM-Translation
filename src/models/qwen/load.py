"""
Qwen2.5-7B-Instruct Inference Model Loader
------------------------------------------------
Thin wrapper over `src.models.common.load_adapter_for_inference` bound to
Qwen2.5-7B-Instruct, for loading a trained (or zero-shot baseline)
checkpoint for evaluation/inference -- as opposed to `qlora.py`'s
`QwenQLoRATrainer`, which always initializes a fresh adapter for training.
"""

import logging
from typing import Any, Optional, Tuple

from transformers import PreTrainedTokenizerBase

from src.models.common import load_adapter_for_inference
from src.models.qwen.qlora import QwenQLoRATrainer

logger = logging.getLogger(__name__)


def load_qwen_for_inference(adapter_path: Optional[str] = None) -> Tuple[Any, PreTrainedTokenizerBase]:
    """Load Qwen2.5-7B-Instruct for inference, optionally with a trained LoRA adapter.

    Args:
        adapter_path: Path to a saved PEFT adapter checkpoint (e.g.
            `checkpoints/qwen/E1_English_Ekegusii/checkpoint-1500`). If
            None, loads the raw base model (zero-shot baseline, E0).

    Returns:
        Tuple[Any, PreTrainedTokenizerBase]: (model, tokenizer).
    """
    return load_adapter_for_inference(QwenQLoRATrainer.MODEL_ID, adapter_path)
