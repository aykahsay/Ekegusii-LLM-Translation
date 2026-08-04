"""
Mistral-7B-Instruct-v0.3 QLoRA Fine-Tuning Module
---------------------------------------------------
Handles BitsAndBytes 4-bit NormalFloat quantization, PEFT LoRA configuration,
and SFTTrainer setup for Mistral-7B-Instruct-v0.3 model fine-tuning on NVIDIA A100 GPU.

Replaced Meta Llama-3.1-8B-Instruct in this project: Llama required a
gated-repo access request that blocked iteration; Mistral-7B-Instruct-v0.3
is fully open (Apache 2.0, no authentication needed) and shares Llama's
module naming (q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj),
so no target_modules changes were needed.
"""

import logging
import os
from typing import Any, Tuple

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

from src.utils.hub_auth import raise_with_access_guidance

logger = logging.getLogger(__name__)


class MistralQLoRATrainer:
    """Manages QLoRA fine-tuning for Mistral-7B-Instruct-v0.3."""

    MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

    def __init__(self, output_dir: str = "checkpoints/mistral", r: int = 32, lora_alpha: int = 64) -> None:
        """Initialize trainer parameters.

        Args:
            output_dir: Directory where checkpoints will be saved.
            r: LoRA rank parameter.
            lora_alpha: LoRA alpha scaling factor.
        """
        self.output_dir = output_dir
        self.r = r
        self.lora_alpha = lora_alpha
        os.makedirs(self.output_dir, exist_ok=True)

    def load_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Load quantized 4-bit base model and subword tokenizer.

        Returns:
            Tuple of (model, tokenizer).
        """
        logger.info(f"Loading 4-bit quantized Mistral model: {self.MODEL_ID}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_ID,
                padding_side="left",
                trust_remote_code=False,
            )
        except OSError as exc:
            raise_with_access_guidance(exc, self.MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device_target = {"": torch.cuda.current_device()} if torch.cuda.is_available() else "auto"
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID,
                quantization_config=bnb_config,
                device_map=device_target,
                torch_dtype=torch.bfloat16,
                trust_remote_code=False,
            )
        except OSError as exc:
            raise_with_access_guidance(exc, self.MODEL_ID)

        base_model = prepare_model_for_kbit_training(base_model)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )

        model = get_peft_model(base_model, peft_config)
        model.print_trainable_parameters()

        return model, tokenizer
