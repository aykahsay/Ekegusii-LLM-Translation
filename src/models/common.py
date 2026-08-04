"""
Shared Model Handling Logic (Aya + Llama)
---------------------------------------------
Aya-23-8B and Llama-3.1-8B are fine-tuned, checkpointed, evaluated, and
deployed through an IDENTICAL QLoRA pipeline -- only the base model ID and
per-model hyperparameters differ (see `configs/models/*.yaml`,
`configs/training/*_qlora.yaml`). This module holds that shared logic once;
`src/models/aya/{load,trainer,inference,save}.py` and
`src/models/llama/{...}.py` are thin wrappers that call into it with their
own MODEL_ID, keeping the required per-model file layout without
duplicating the underlying implementation four times over.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset

try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    import subprocess as _subprocess
    import sys as _sys

    _subprocess.run(
        [_sys.executable, "-m", "pip", "install", "--quiet", "omegaconf==2.3.0", "hydra-core==1.3.2"],
        check=False,
    )
    from omegaconf import DictConfig, OmegaConf

from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
    Seq2SeqTrainingArguments,
    Trainer,
)

from src.datasets.collator import CausalLMDataCollator
from src.utils.helpers import resolve_device

logger = logging.getLogger(__name__)


def load_adapter_for_inference(
    base_model_id: str, adapter_path: Optional[str] = None
) -> Tuple[Any, PreTrainedTokenizerBase]:
    """Load a base model (4-bit quantized) with an optional trained LoRA adapter attached.

    Distinct from `AyaQLoRATrainer`/`LlamaQLoRATrainer.load_model_and_tokenizer`,
    which always initializes a FRESH LoRA adapter for training -- this
    loads a PREVIOUSLY TRAINED adapter checkpoint for inference/evaluation.

    Args:
        base_model_id: HuggingFace Hub ID of the base model (e.g.
            "CohereForAI/aya-23-8B").
        adapter_path: Path to a saved PEFT adapter checkpoint directory
            (e.g. from `CheckpointManager.save`). If None, returns the
            unmodified base model (useful for zero-shot baseline E0).

    Returns:
        Tuple[Any, PreTrainedTokenizerBase]: (model, tokenizer). The model
            is a `PeftModel` if `adapter_path` was given, else the raw base model.

    Raises:
        OSError: If the base model or adapter cannot be loaded (missing
            files, no Hub access, etc.).
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        adapter_path or base_model_id, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if adapter_path is None:
        logger.info(f"Loaded base model '{base_model_id}' with no adapter (zero-shot).")
        return base_model, tokenizer

    model = PeftModel.from_pretrained(base_model, adapter_path)
    logger.info(f"Loaded base model '{base_model_id}' with adapter from '{adapter_path}'.")
    return model, tokenizer


@torch.no_grad()
def generate_translations(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    generation_cfg: DictConfig,
    batch_size: int = 8,
) -> List[str]:
    """Generate translations for a list of prompts using a configured decoding profile.

    Args:
        model: A (optionally PEFT-wrapped) causal-LM model in eval mode.
        tokenizer: Tokenizer matching `model`.
        prompts: List of formatted prompts (see
            `src.task_generation.prompt_templates.format_completion_prompt`).
        generation_cfg: Decoding config (e.g. from
            `src.utils.config.load_generation_config`) -- passed directly
            as `model.generate(**generation_cfg)` keyword arguments.
        batch_size: Number of prompts to generate per forward pass.

    Returns:
        List[str]: Decoded generations, one per prompt, with the input
            prompt portion stripped off.
    """
    model.eval()
    device = next(model.parameters()).device
    raw_kwargs = OmegaConf.to_container(generation_cfg, resolve=True)
    generation_kwargs: Dict[str, Any] = raw_kwargs if isinstance(raw_kwargs, dict) else {}

    outputs: List[str] = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        encoded = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)

        generated = model.generate(
            **encoded,
            pad_token_id=tokenizer.pad_token_id,
            **generation_kwargs,
        )

        input_length = encoded["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(generated[:, input_length:], skip_special_tokens=True)
        outputs.extend(decoded)

    logger.info(f"Generated {len(outputs):,} translations (batch_size={batch_size}).")
    return outputs


def merge_and_save_adapter(model: Any, tokenizer: PreTrainedTokenizerBase, output_dir: Path) -> Path:
    """Merge a trained LoRA adapter into the base model and save a standalone checkpoint.

    Used for the final deployed model (E8_Final_Model), where inference
    should not depend on PEFT being installed/available at serve time.

    Args:
        model: A `PeftModel` with a trained adapter attached.
        tokenizer: Tokenizer to save alongside the merged model.
        output_dir: Destination directory for the merged model.

    Returns:
        Path: `output_dir`, containing the merged model and tokenizer files.

    Raises:
        AttributeError: If `model` is not a PEFT model (has no `merge_and_unload`).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Merged adapter into base model and saved to {output_dir}.")
    return output_dir


def build_training_arguments(qlora_cfg: DictConfig, output_dir: Path) -> Seq2SeqTrainingArguments:
    """Build `Seq2SeqTrainingArguments` from a merged QLoRA config.

    Args:
        qlora_cfg: Output of `src.utils.config.load_qlora_config` (must
            contain a `training_arguments` section, per
            `configs/training/qlora.yaml`).
        output_dir: Directory for Trainer checkpoints/logs.

    Returns:
        Seq2SeqTrainingArguments: Configured training arguments.
    """
    ta = qlora_cfg.training_arguments
    return Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(ta.per_device_train_batch_size),
        per_device_eval_batch_size=int(ta.per_device_eval_batch_size),
        gradient_accumulation_steps=int(ta.gradient_accumulation_steps),
        gradient_checkpointing=bool(ta.gradient_checkpointing),
        max_grad_norm=float(ta.max_grad_norm),
        num_train_epochs=float(ta.num_train_epochs),
        warmup_ratio=float(ta.warmup_ratio),
        learning_rate=float(ta.learning_rate),
        fp16=bool(ta.fp16),
        bf16=bool(ta.bf16),
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        report_to=["tensorboard"],
    )


def run_qlora_training(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    qlora_cfg: DictConfig,
    output_dir: Path,
    early_stopping_patience: int = 3,
) -> Trainer:
    """Run QLoRA fine-tuning to completion and return the trained Trainer.

    Args:
        model: A PEFT-wrapped model ready for training (see
            `AyaQLoRATrainer`/`LlamaQLoRATrainer.load_model_and_tokenizer`).
        tokenizer: Tokenizer matching `model`.
        train_dataset: Tokenized training dataset (see `InstructionDatasetBuilder`).
        eval_dataset: Tokenized validation dataset.
        qlora_cfg: Merged QLoRA config (see `build_training_arguments`).
        output_dir: Directory for checkpoints/logs.
        early_stopping_patience: Evaluations to tolerate with no COMET/loss
            improvement before stopping (see `src.utils.checkpoint.EarlyStopping`
            for the underlying policy -- Trainer's own callback is used
            here for the actual stop signal).

    Returns:
        Trainer: The HuggingFace Trainer after `.train()` has completed.
    """
    from transformers import EarlyStoppingCallback

    training_args = build_training_arguments(qlora_cfg, output_dir)
    collator = CausalLMDataCollator(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    logger.info(f"Starting QLoRA training -> {output_dir} (device={resolve_device()}).")
    trainer.train()
    return trainer
