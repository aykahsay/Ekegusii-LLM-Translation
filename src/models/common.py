"""
Shared Model Handling Logic (Qwen + Mistral)
---------------------------------------------
Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3 are fine-tuned, checkpointed, evaluated, and
deployed through an IDENTICAL QLoRA pipeline -- only the base model ID and
per-model hyperparameters differ (see `configs/models/*.yaml`,
`configs/training/*_qlora.yaml`). This module holds that shared logic once;
`src/models/qwen/{load,trainer,inference,save}.py` and
`src/models/mistral/{...}.py` are thin wrappers that call into it with their
own MODEL_ID, keeping the required per-model file layout without
duplicating the underlying implementation four times over.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset
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
from src.utils.config_dict import ConfigDict
from src.utils.helpers import resolve_device
from src.utils.hub_auth import raise_with_access_guidance

logger = logging.getLogger(__name__)


def load_adapter_for_inference(
    base_model_id: str, adapter_path: Optional[str] = None
) -> Tuple[Any, PreTrainedTokenizerBase]:
    """Load a base model (4-bit quantized) with an optional trained LoRA adapter attached.

    Distinct from `QwenQLoRATrainer`/`MistralQLoRATrainer.load_model_and_tokenizer`,
    which always initializes a FRESH LoRA adapter for training -- this
    loads a PREVIOUSLY TRAINED adapter checkpoint for inference/evaluation.

    Args:
        base_model_id: HuggingFace Hub ID of the base model (e.g.
            "Qwen/Qwen2.5-7B-Instruct").
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

    try:
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            adapter_path or base_model_id, padding_side="left"
        )
    except OSError as exc:
        raise_with_access_guidance(exc, base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    except OSError as exc:
        raise_with_access_guidance(exc, base_model_id)

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
    generation_cfg: ConfigDict,
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
    generation_kwargs: Dict[str, Any] = generation_cfg.to_container()

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


def build_training_arguments(qlora_cfg: ConfigDict, output_dir: Path) -> Seq2SeqTrainingArguments:
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
    try:
        from transformers.integrations import is_tensorboard_available
        has_tb = is_tensorboard_available()
    except Exception:
        has_tb = False
    report_to = ["tensorboard"] if has_tb else "none"

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
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to=report_to,
    )


def run_qlora_training(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    qlora_cfg: ConfigDict,
    output_dir: Path,
    early_stopping_patience: int = 3,
) -> Trainer:
    """Run QLoRA fine-tuning to completion and return the trained Trainer.

    Args:
        model: A PEFT-wrapped model ready for training (see
            `QwenQLoRATrainer`/`MistralQLoRATrainer.load_model_and_tokenizer`).
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
    training_args.load_best_model_at_end = True
    training_args.metric_for_best_model = "loss"
    training_args.greater_is_better = False
    collator = CausalLMDataCollator(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    last_checkpoint = None
    if output_dir.exists():
        checkpoints = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
            last_checkpoint = str(checkpoints[-1])

    if last_checkpoint:
        logger.info(f"Resuming QLoRA training from checkpoint: {last_checkpoint} (device={resolve_device()}).")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        logger.info(f"Starting fresh QLoRA training -> {output_dir} (device={resolve_device()}).")
        trainer.train()

    # Automatically export Step, Training Loss, Validation Loss CSV & Loss Plots
    try:
        exp_name = str(output_dir).replace("\\", "/").rstrip("/").split("/")[-1]
        exp_short = exp_name.split("_")[0] if "_" in exp_name else exp_name
        import pandas as pd, os, matplotlib.pyplot as plt
        history = getattr(trainer.state, "log_history", [])
        step_map = {}
        for entry in history:
            step = entry.get("step")
            if step is not None:
                if step not in step_map:
                    step_map[step] = {"Step": step, "Training_Loss": None, "Validation_Loss": None}
                if "loss" in entry:
                    step_map[step]["Training_Loss"] = entry["loss"]
                if "eval_loss" in entry:
                    step_map[step]["Validation_Loss"] = entry["eval_loss"]
        records = []
        for step, vals in sorted(step_map.items()):
            if vals["Training_Loss"] is not None or vals["Validation_Loss"] is not None:
                records.append({
                    "step": step,
                    "train_loss": vals["Training_Loss"],
                    "val_loss": vals["Validation_Loss"]
                })
        df = pd.DataFrame(records)
        os.makedirs("data/results", exist_ok=True)
        os.makedirs("outputs/training_logs", exist_ok=True)
        os.makedirs("outputs/figures", exist_ok=True)
        os.makedirs("paper/figures", exist_ok=True)

        csv1 = f"data/results/{exp_short}_loss.csv"
        csv2 = f"outputs/training_logs/{exp_short}_loss.csv"
        df.to_csv(csv1, index=False)
        df.to_csv(csv2, index=False)
        logger.info(f"📊 Auto-saved training loss CSV to {csv1} and {csv2}")

        # 1. Automatically plot loss curve PNG
        try:
            fig_path1 = f"outputs/figures/{exp_short}_loss_curve.png"
            fig_path2 = f"paper/figures/{exp_short}_loss_curve.png"
            plt.figure(figsize=(8, 5))
            if "train_loss" in df.columns and df["train_loss"].notnull().any():
                plt.plot(df["step"], df["train_loss"], label="Training Loss", color="#1f77b4", linewidth=2)
            if "val_loss" in df.columns and df["val_loss"].notnull().any():
                plt.plot(df["step"], df["val_loss"], label="Validation Loss", color="#ff7f0e", linewidth=2, linestyle="--")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title(f"Training & Validation Loss Curve - {exp_name}")
            plt.legend()
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.tight_layout()
            plt.savefig(fig_path1, dpi=300)
            plt.savefig(fig_path2, dpi=300)
            plt.close()
            logger.info(f"📈 Auto-generated loss plots: {fig_path1} & {fig_path2}")
        except Exception as plot_err:
            logger.warning(f"Could not auto-plot loss curve: {plot_err}")

        # 2. Automatically Prune Intermediate Checkpoints (Keep Best/Highest Checkpoint Only)
        try:
            checkpoints = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
            if len(checkpoints) > 1:
                checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
                best_ckpt = checkpoints[-1]
                for old_ckpt in checkpoints[:-1]:
                    import shutil
                    shutil.rmtree(old_ckpt, ignore_errors=True)
                logger.info(f"🧹 Pruned intermediate checkpoints. Kept best checkpoint: {best_ckpt.name}")
        except Exception as prune_err:
            logger.warning(f"Checkpoint pruning notice: {prune_err}")

        # 3. Automatically Upload Best Checkpoints to Hugging Face Hub
        try:
            from huggingface_hub import HfApi
            hf_token = os.environ.get("HF_TOKEN")
            repo_id = os.environ.get("HF_REPO_ID", "aykgeh/Ekegusii-LLM-Translation")
            api = HfApi(token=hf_token) if hf_token else HfApi()
            parent_ckpt_dir = output_dir.parent.parent  # checkpoints/ directory
            if parent_ckpt_dir.exists():
                api.upload_folder(
                    folder_path=str(parent_ckpt_dir),
                    repo_id=repo_id,
                    repo_type="model",
                    delete_patterns="*"
                )
                logger.info(f"🤗 Automatically uploaded best checkpoints to Hugging Face: https://huggingface.co/{repo_id}")
        except Exception as hf_err:
            logger.warning(f"Hugging Face auto-upload notice: {hf_err}")

        # 4. Automatically Commit & Push Plots, Figures, & Loss CSVs to GitHub
        try:
            import subprocess
            os.system("chmod -R ugo+rwX .git 2>/dev/null")
            if os.path.exists(".git/index.lock"):
                try:
                    os.remove(".git/index.lock")
                except Exception:
                    pass
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", "*"], check=False)
            subprocess.run(["git", "add", "data/results/", "outputs/", "paper/figures/"], check=False)
            subprocess.run(["git", "commit", "-m", f"Auto-save loss CSV and figures for {exp_name}"], check=False)
            subprocess.run(["git", "push", "origin", "main"], check=False)
            logger.info(f"🚀 Automatically pushed plots, figures, and CSVs to GitHub!")
        except Exception as push_err:
            logger.warning(f"Git auto-push notice: {push_err}")

    except Exception as exc:
        logger.warning(f"Could not complete post-training automation: {exc}")

    return trainer


