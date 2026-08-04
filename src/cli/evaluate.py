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
from src.models.llama.inference import translate_with_llama
from src.models.qwen.inference import translate_with_qwen

logger = logging.getLogger(__name__)

_INFERENCE_FN = {"qwen": translate_with_qwen, "llama": translate_with_llama}


def run_evaluate(
    model_name: str,
    source_lang: str = "English",
    target_lang: str = "Ekegusii",
    adapter_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run inference + automatic-metric evaluation for one model/direction.

    Args:
        model_name: "qwen" or "llama".
        source_lang: Source language for the test direction.
        target_lang: Target language for the test direction.
        adapter_path: Optional trained LoRA adapter checkpoint path. If
            None, evaluates the zero-shot base model.

    Returns:
        Dict[str, Any]: SacreBLEU/chrF/lexical-accuracy metrics (see
            `BaseExperiment.evaluate_predictions`).

    Raises:
        ValueError: If `model_name` is not "qwen" or "llama".
    """
    if model_name not in _INFERENCE_FN:
        raise ValueError(f"model_name must be 'qwen' or 'llama', got '{model_name}'.")

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
