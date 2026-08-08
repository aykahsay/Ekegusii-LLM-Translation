"""
CLI: evaluate
-----------------
Implements automatic-metric evaluation of a trained (or zero-shot)
checkpoint against the fixed master test split -- distinct from
`main.py`'s `evaluate` command, which aggregates ALREADY-SAVED per-
experiment results into the E0-E8 attribution matrix. This module actually
RUNS inference + scores it for one model/direction.
"""

import logging
from typing import Any, Dict, Optional

from src.experiments.base import BaseExperiment
from src.master_corpus.manager import MasterCorpusManager
from src.models.mistral.inference import translate_with_mistral
from src.models.qwen.inference import translate_with_qwen

logger = logging.getLogger(__name__)

_INFERENCE_FN = {"qwen": translate_with_qwen, "mistral": translate_with_mistral}


def run_evaluate(
    model_name: str,
    source_lang: str = "English",
    target_lang: str = "Ekegusii",
    adapter_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run inference + automatic-metric evaluation for one model/direction.

    Args:
        model_name: "qwen" or "mistral".
        source_lang: Source language for the test direction.
        target_lang: Target language for the test direction.
        adapter_path: Optional trained LoRA adapter checkpoint path. If
            None, evaluates the zero-shot base model.

    Returns:
        Dict[str, Any]: SacreBLEU/chrF/lexical-accuracy metrics (see
            `BaseExperiment.evaluate_predictions`).

    Raises:
        ValueError: If `model_name` is not "qwen" or "mistral".
    """
    if model_name not in _INFERENCE_FN:
        raise ValueError(f"model_name must be 'qwen' or 'mistral', got '{model_name}'.")

    manager = MasterCorpusManager()

    class _EvalHelper(BaseExperiment):
        experiment_id = f"cli_eval_{model_name}"

        def build_training_tasks(self):
            raise NotImplementedError

    helper = _EvalHelper(manager)
    test_pairs = helper.build_test_pairs(source_lang, target_lang)
    sources = test_pairs["source"].tolist()
    references = test_pairs["target"].tolist()

    logger.info(f"Evaluating {model_name} on {len(sources):,} {source_lang}->{target_lang} test pairs.")
    predictions = _INFERENCE_FN[model_name](sources, source_lang, target_lang, adapter_path=adapter_path)

    return helper.evaluate_predictions(predictions, references)


def evaluate_all_saved_checkpoints(model_name: str = "qwen") -> Any:
    """Scan checkpoints/{model_name}/ for all saved best model adapters,
    evaluate both directions (Eng->Eke and Eke->Eng), and export a master comparison CSV.

    Args:
        model_name: "qwen" or "mistral".

    Returns:
        pd.DataFrame: Summary evaluation table across all saved models.
    """
    import os, pandas as pd
    from pathlib import Path

    checkpoints_base = Path(f"checkpoints/{model_name}")
    results = []

    logger.info(f"🔍 Starting Master Evaluation Sweep for {model_name} across all saved checkpoints...")

    # 1. Evaluate Zero-Shot Baseline (E0)
    for src, tgt in [("English", "Ekegusii"), ("Ekegusii", "English")]:
        try:
            metrics = run_evaluate(model_name=model_name, source_lang=src, target_lang=tgt, adapter_path=None)
            results.append({
                "Experiment": "E0_Baseline",
                "Direction": f"{src}->{tgt}",
                "SacreBLEU": round(metrics.get("score", metrics.get("bleu", 0.0)), 2),
                "chrF++": round(metrics.get("chrf", 0.0), 2),
                "Lexical_Accuracy": round(metrics.get("lexical_accuracy", 0.0), 2)
            })
        except Exception as e:
            logger.warning(f"Notice on E0 evaluation ({src}->{tgt}): {e}")

    # 2. Evaluate all saved best checkpoints
    if checkpoints_base.exists():
        for exp_dir in sorted(checkpoints_base.iterdir()):
            if exp_dir.is_dir():
                ckpts = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
                if ckpts:
                    ckpts.sort(key=lambda x: int(x.name.split("-")[-1]))
                    best_ckpt = ckpts[-1]

                    for src, tgt in [("English", "Ekegusii"), ("Ekegusii", "English")]:
                        try:
                            logger.info(f"📊 Evaluating {exp_dir.name} ({best_ckpt.name}) for {src}->{tgt}...")
                            metrics = run_evaluate(model_name=model_name, source_lang=src, target_lang=tgt, adapter_path=str(best_ckpt))
                            results.append({
                                "Experiment": exp_dir.name,
                                "Direction": f"{src}->{tgt}",
                                "SacreBLEU": round(metrics.get("score", metrics.get("bleu", 0.0)), 2),
                                "chrF++": round(metrics.get("chrf", 0.0), 2),
                                "Lexical_Accuracy": round(metrics.get("lexical_accuracy", 0.0), 2)
                            })
                        except Exception as e:
                            logger.warning(f"Notice on {exp_dir.name} evaluation ({src}->{tgt}): {e}")

    df = pd.DataFrame(results)
    os.makedirs("outputs/evaluation_reports", exist_ok=True)
    os.makedirs("data/results", exist_ok=True)
    df.to_csv("outputs/evaluation_reports/master_evaluation_results.csv", index=False)
    df.to_csv("data/results/master_evaluation_results.csv", index=False)
    logger.info("🏆 Master evaluation sweep complete! Exported to outputs/evaluation_reports/master_evaluation_results.csv")
    return df

