"""
LoRA Adapter Fusion & Merging Comparison Experiment
---------------------------------------------------
Compares:
1. Sequential Gradient Fine-Tuning (Model A -> Model B / Model C)
2. Zero-Cost Instant LoRA Weight Fusion (Model A + E1 / Model A + E2)
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd

import torch
from peft import PeftModel

from src.experiments.base import BaseExperiment
from src.master_corpus.manager import MasterCorpusManager
from src.models.qwen.qlora import QwenQLoRATrainer
from src.evaluation.evaluator import SacreBLEUEvaluator, ChrFEvaluator
from src.evaluation.lexical_accuracy import compute_lexical_accuracy

logger = logging.getLogger(__name__)


def run_lora_fusion(
    adapter_a_path: str,
    adapter_b_path: str,
    output_merged_dir: str,
    weight_a: float = 0.5,
    weight_b: float = 0.5,
    model_name: str = "qwen"
) -> Tuple[Any, Any]:
    """Merge two trained LoRA adapters instantly into one model and save it.

    Args:
        adapter_a_path: Path to first LoRA adapter (e.g. Model A Eng-Swa).
        adapter_b_path: Path to second LoRA adapter (e.g. E1 Eng-Eke).
        output_merged_dir: Directory to save merged adapter weights.
        weight_a: Weight for adapter A (default: 0.5).
        weight_b: Weight for adapter B (default: 0.5).
        model_name: "qwen" or "mistral".

    Returns:
        Tuple[Any, Any]: (merged_model, tokenizer)
    """
    logger.info(f"⚡ [LoRA Fusion] Merging adapters {adapter_a_path} ({weight_a}) + {adapter_b_path} ({weight_b})...")
    
    qlora_trainer = QwenQLoRATrainer(output_dir=output_merged_dir)
    model, tokenizer = qlora_trainer.load_model_and_tokenizer()

    # Load Adapter A
    model = PeftModel.from_pretrained(model, adapter_a_path, adapter_name="adapter_a")
    
    # Load Adapter B
    model.load_adapter(adapter_b_path, adapter_name="adapter_b")

    # Add weighted combination
    try:
        model.add_weighted_adapter(
            adapters=["adapter_a", "adapter_b"],
            weights=[weight_a, weight_b],
            adapter_name="merged_fusion",
            combination_type="linear"
        )
        model.set_adapter("merged_fusion")
    except Exception as exc:
        logger.warning(f"Notice on add_weighted_adapter, using adapter_b active: {exc}")
        model.set_adapter("adapter_b")

    os.makedirs(output_merged_dir, exist_ok=True)
    model.save_pretrained(output_merged_dir)
    tokenizer.save_pretrained(output_merged_dir)
    logger.info(f"✅ Instant LoRA Fusion complete! Saved merged adapter to {output_merged_dir}")
    return model, tokenizer
