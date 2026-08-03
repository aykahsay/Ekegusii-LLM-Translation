"""
chrF / chrF++ Machine Translation Metric Evaluator
--------------------------------------------------
Computes character n-gram F-score (chrF and chrF++) metric optimized
for agglutinative and morphologically rich languages like Ekegusii.
"""

import logging
from typing import Dict, List, Union
import sacrebleu

logger = logging.getLogger(__name__)


class ChrFEvaluator:
    """Computes chrF and chrF++ character n-gram evaluation scores."""

    def __init__(self, char_order: int = 6, word_order: int = 2, beta: float = 2.0) -> None:
        """Initialize with chrF parameters.

        Args:
            char_order: Maximum character n-gram length (default 6).
            word_order: Maximum word n-gram length for chrF++ (default 2).
            beta: Weight of recall relative to precision (default 2.0).
        """
        self.char_order = char_order
        self.word_order = word_order
        self.beta = beta

    def compute_chrf(
        self, hypotheses: List[str], references: List[Union[str, List[str]]]
    ) -> Dict[str, Union[float, str]]:
        """Compute corpus-level chrF++ score.

        Args:
            hypotheses: Model predicted translations list.
            references: Target reference translations list.

        Returns:
            Dict containing chrf_score, chrf_plus_plus_score, and signature string.
        """
        if isinstance(references[0], str):
            formatted_refs = [references]
        else:
            formatted_refs = references

        chrf = sacrebleu.corpus_chrf(
            hypotheses,
            formatted_refs,
            char_order=self.char_order,
            word_order=self.word_order,
            beta=self.beta,
        )

        sig = getattr(chrf, "signature", str(chrf))

        result = {
            "chrf_plus_plus_score": float(chrf.score),
            "char_order": self.char_order,
            "word_order": self.word_order,
            "signature": str(sig),
        }

        logger.info(f"chrF++ Score: {chrf.score:.2f}")
        return result
