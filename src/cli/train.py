"""
CLI: train
--------------
Dispatches to the correct experiment (E1-E7; E0 is zero-shot-only, E8 is
selected by the ablation aggregator afterward, not trained directly) and
model (Qwen/Mistral) training pipeline. Splits each experiment's training
tasks into a training pool and a small in-training validation slice
(DISTINCT from the fixed master test split, which is reserved for final
scoring only).
"""

import logging
from pathlib import Path
from typing import Callable, Dict

from transformers import Trainer

from src.experiments.base import BaseExperiment
from src.experiments.bilingual import BilingualExperiment
from src.experiments.curriculum import CurriculumLearningExperiment
from src.experiments.lexical import LexicalAugmentationExperiment
from src.experiments.mono import FullResourcesExperiment
from src.experiments.sequential import SequentialTransferExperiment
from src.experiments.trilingual import TrilingualExperiment
from src.models.common import run_qlora_training
from src.models.mistral.qlora import MistralQLoRATrainer
from src.models.qwen.qlora import QwenQLoRATrainer
from src.utils.config import load_qlora_config
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

from src.experiments.pivot_transfer import PivotTransferExperimentABC

TRAINABLE_EXPERIMENTS: Dict[str, Callable[[], BaseExperiment]] = {
    "E1_English_Ekegusii": lambda: BilingualExperiment("eng_eke"),
    "E2_Swahili_Ekegusii": lambda: BilingualExperiment("swa_eke"),
    "E3_Bilingual": lambda: BilingualExperiment("combined"),
    "E5_Full_Resources": FullResourcesExperiment,
    "E6_Lexical_Augmentation": LexicalAugmentationExperiment,
    "E7_Curriculum_Learning": CurriculumLearningExperiment,
    "E9_Sequential_Transfer": SequentialTransferExperiment,
    "E10_Model_A_English_Swahili": lambda: BilingualExperiment("eng_swa"),
    "E10_Model_B_English_Ekegusii": lambda: BilingualExperiment("eng_eke"),
    "E10_Model_C_Swahili_Ekegusii": lambda: BilingualExperiment("swa_eke"),
}


_QLORA_TRAINERS = {"qwen": QwenQLoRATrainer, "mistral": MistralQLoRATrainer}


def run_train(experiment_id: str, model_name: str, val_fraction: float = 0.05) -> Trainer:
    """Train one model on one experiment's resource configuration.

    Args:
        experiment_id: One of `TRAINABLE_EXPERIMENTS`'s keys (E1-E7, E9).
        model_name: "qwen" or "mistral".
        val_fraction: Fraction of the experiment's training tasks held out
            for in-training validation/early-stopping (not the final test set).

    Returns:
        Trainer: The trained HuggingFace Trainer.

    Raises:
        ValueError: If `experiment_id` is not trainable, or `model_name`
            is not "qwen"/"mistral".
    """
    if experiment_id not in TRAINABLE_EXPERIMENTS:
        raise ValueError(
            f"'{experiment_id}' is not directly trainable. Valid options: "
            f"{list(TRAINABLE_EXPERIMENTS)}"
        )
    if model_name not in _QLORA_TRAINERS:
        raise ValueError(f"model_name must be 'qwen' or 'mistral', got '{model_name}'.")

    output_dir = Path(f"checkpoints/{model_name}/{experiment_id}")
    matched_files = list(output_dir.glob("**/adapter_model.safetensors"))
    if experiment_id == "E4_Trilingual" or len(matched_files) > 0:
        logger.info(f"⏩ Experiment {experiment_id} is already completed -- SKIPPING training!")
        print(f"⏩ Experiment {experiment_id} is already completed -- SKIPPING training!")
        return None


    set_seed(42)
    experiment = TRAINABLE_EXPERIMENTS[experiment_id]()
    tasks_df = experiment.build_training_tasks()

    split_idx = int(len(tasks_df) * (1 - val_fraction))
    shuffled = tasks_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    train_tasks, val_tasks = shuffled.iloc[:split_idx], shuffled.iloc[split_idx:]



    
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    qlora_trainer = _QLORA_TRAINERS[model_name](output_dir=str(output_dir))
    model, tokenizer = qlora_trainer.load_model_and_tokenizer()

    # Stage 1 Adapter Loading for E9 Sequential Transfer & E10 Pivot Transfer
    if experiment_id == "E9_Sequential_Transfer":
        stage1_dir = Path(f"checkpoints/{model_name}/E2_Swahili_Ekegusii")
        stage1_ckpts = list(stage1_dir.glob("**/adapter_model.safetensors"))
        if stage1_ckpts:
            from peft import PeftModel
            stage1_path = stage1_ckpts[0].parent
            logger.info(f"🔗 [E9_Sequential_Transfer] Loading Stage 1 Swahili-Bantu pivot adapter from {stage1_path}...")
            model = PeftModel.from_pretrained(model, str(stage1_path), is_trainable=True)

    elif experiment_id in ["E10_Model_B_English_Ekegusii", "E10_Model_C_Swahili_Ekegusii"]:
        stage1_dir = Path(f"checkpoints/{model_name}/E10_Model_A_English_Swahili")
        stage1_ckpts = list(stage1_dir.glob("**/adapter_model.safetensors"))
        if stage1_ckpts:
            from peft import PeftModel
            stage1_path = stage1_ckpts[0].parent
            logger.info(f"🔗 [{experiment_id}] Loading Model A (English-Swahili) pre-trained pivot adapter from {stage1_path}...")
            model = PeftModel.from_pretrained(model, str(stage1_path), is_trainable=True)

    train_dataset = experiment.build_tokenized_dataset(train_tasks, tokenizer)
    val_dataset = experiment.build_tokenized_dataset(val_tasks, tokenizer)

    qlora_cfg = load_qlora_config(model_name)
    logger.info(f"Training {model_name} on {experiment_id}: {len(train_tasks):,} train / {len(val_tasks):,} val.")
    return run_qlora_training(model, tokenizer, train_dataset, val_dataset, qlora_cfg, output_dir)
