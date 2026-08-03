"""
Aya-23 Inference Model Loader
--------------------------------
Thin wrapper over `src.models.common.load_adapter_for_inference` bound to
Aya-23-8B, for loading a trained (or zero-shot baseline) checkpoint for
evaluation/inference -- as opposed to `qlora.py`'s `AyaQLoRATrainer`, which
always initializes a fresh adapter for training.
"""

import logging
from typing import Any, Optional, Tuple

from transformers import PreTrainedTokenizerBase

from src.models.aya.qlora import AyaQLoRATrainer
from src.models.common import load_adapter_for_inference

logger = logging.getLogger(__name__)


def load_aya_for_inference(adapter_path: Optional[str] = None) -> Tuple[Any, PreTrainedTokenizerBase]:
    """Load Aya-23-8B for inference, optionally with a trained LoRA adapter.

    Args:
        adapter_path: Path to a saved PEFT adapter checkpoint (e.g.
            `checkpoints/aya/E1_English_Ekegusii/checkpoint-1500`). If
            None, loads the raw base model (zero-shot baseline, E0).

    Returns:
        Tuple[Any, PreTrainedTokenizerBase]: (model, tokenizer).
    """
    return load_adapter_for_inference(AyaQLoRATrainer.MODEL_ID, adapter_path)
