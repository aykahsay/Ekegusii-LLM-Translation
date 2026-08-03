"""
SacreBLEU Machine Translation Metric Evaluator
----------------------------------------------
Provides standardized ACL/EMNLP translation evaluation using SacreBLEU
with official signature parsing, n-gram precisions, and brevity penalty calculations.
"""

import logging
from typing import Dict, List, Union
import sacrebleu

logger = logging.getLogger(__name__)


class SacreBLEUEvaluator:
    """Computes standardized SacreBLEU translation metrics."""

    def __init__(self, tokenize: str = "13a", lowercase: bool = False) -> None:
        """Initialize with SacreBLEU tokenizer configuration.

        Args:
            tokenize: Tokenizer type ('13a', 'intl', 'none').
            lowercase: Whether to compute case-insensitive BLEU.
        """
        self.tokenize = tokenize
        self.lowercase = lowercase

    def compute_corpus_bleu(
        self, hypotheses: List[str], references: List[Union[str, List[str]]]
    ) -> Dict[str, Union[float, str]]:
        """Compute corpus-level SacreBLEU score against reference translations.

        Args:
            hypotheses: Model predicted translations list.
            references: Target reference translations list (or list of reference lists).

        Returns:
            Dict containing score, precisions, bp (brevity penalty), and signature string.
        """
        if not hypotheses or not references:
            raise ValueError("Hypotheses and references lists cannot be empty.")

        if isinstance(references[0], str):
            formatted_refs = [references]
        else:
            formatted_refs = references

        bleu = sacrebleu.corpus_bleu(
            hypotheses,
            formatted_refs,
            tokenize=self.tokenize,
            lowercase=self.lowercase,
        )

        sig = getattr(bleu, "signature", str(bleu))

        result = {
            "sacrebleu_score": float(bleu.score),
            "precisions": [float(p) for p in bleu.precisions],
            "brevity_penalty": float(bleu.bp),
            "sys_len": int(bleu.sys_len),
            "ref_len": int(bleu.ref_len),
            "signature": str(sig),
        }

        logger.info(f"SacreBLEU Score: {bleu.score:.2f} (BP: {bleu.bp:.3f})")
        return result
