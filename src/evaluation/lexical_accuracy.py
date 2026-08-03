"""
Lexical Accuracy Metric Wrapper
------------------------------------
Thin, standardized-name wrapper around `LexicalEvaluator`
(lexical_evaluator.py). Exists so the metric name used in
`src.utils.constants.EVALUATION_METRICS` ("lexical_accuracy") and in
`ResourceAttributionAnalyzer`'s report columns has a matching module/function
to call, without reimplementing `LexicalEvaluator`'s exact/stem-match logic
a second time.
"""

import logging
from typing import Dict, List, Optional, Tuple

from src.evaluation.lexical_evaluator import LexicalEvaluator
from src.master_corpus.manager import MasterCorpusManager

logger = logging.getLogger(__name__)


def compute_lexical_accuracy(
    predictions: List[Tuple[str, str]], manager: Optional[MasterCorpusManager] = None
) -> Dict[str, float]:
    """Compute lexical accuracy (exact + morphological stem match) for a metric report.

    Args:
        predictions: List of (hypothesis_translation, reference_translation) tuples.
        manager: Optional MasterCorpusManager to reuse; a new one is
            created if not provided.

    Returns:
        Dict[str, float]: "exact_term_accuracy", "morphological_coverage",
            "total_terms_evaluated" -- see `LexicalEvaluator.evaluate_predictions`.
    """
    evaluator = LexicalEvaluator(manager)
    return evaluator.evaluate_predictions(predictions)
