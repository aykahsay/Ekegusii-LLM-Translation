"""
COMET Neural Evaluation Metric
-----------------------------------
Computes COMET (source-aware, neural quality-estimation) scores via the
HuggingFace `evaluate` library's wrapper around Unbabel's COMET models.
Unlike SacreBLEU/chrF (surface n-gram overlap), COMET uses source, hypothesis,
AND reference to produce a learned quality estimate that correlates better
with human judgment -- particularly important for Ekegusii, where n-gram
metrics undercount valid paraphrases the tokenizer fragments unusually.
"""

import logging
from typing import Dict, List

import evaluate as hf_evaluate

logger = logging.getLogger(__name__)


class CometEvaluator:
    """Computes COMET scores for machine translation quality estimation."""

    def __init__(self, model_config: str = "Unbabel/wmt22-comet-da") -> None:
        """Initialize the COMET metric, downloading model weights on first use.

        Args:
            model_config: COMET checkpoint to use. "Unbabel/wmt22-comet-da"
                is the standard reference-based COMET model as of WMT22;
                requires `unbabel-comet` to be installed (see requirements.txt).

        Raises:
            ImportError: If `unbabel-comet` is not installed.
        """
        self._metric = hf_evaluate.load("comet", config_name=model_config)
        self.model_config = model_config

    def compute(
        self, predictions: List[str], references: List[str], sources: List[str]
    ) -> Dict[str, object]:
        """Compute per-sentence and mean COMET scores.

        Args:
            predictions: Model-generated translations.
            references: Reference (gold) translations, aligned with `predictions`.
            sources: Original source-language sentences, aligned with `predictions`.

        Returns:
            Dict[str, object]: Keys "mean_score" (float), "scores"
                (List[float], per-sentence), "model_config" (str).

        Raises:
            ValueError: If `predictions`, `references`, `sources` have
                mismatched lengths, or any is empty.
        """
        if not (len(predictions) == len(references) == len(sources)):
            raise ValueError(
                f"predictions ({len(predictions)}), references ({len(references)}), "
                f"and sources ({len(sources)}) must all be the same length."
            )
        if len(predictions) == 0:
            raise ValueError("predictions/references/sources must not be empty.")

        result = self._metric.compute(predictions=predictions, references=references, sources=sources)
        if result is None:
            raise RuntimeError("COMET metric computation returned no result.")
        scores = list(result["scores"])
        mean_score = sum(scores) / len(scores)

        logger.info(f"COMET ({self.model_config}) mean score: {mean_score:.4f} over {len(scores)} sentence(s).")
        return {"mean_score": mean_score, "scores": scores, "model_config": self.model_config}
