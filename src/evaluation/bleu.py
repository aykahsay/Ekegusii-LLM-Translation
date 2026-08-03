"""
Raw NLTK BLEU Evaluator
---------------------------
Computes "raw" BLEU via NLTK's tokenizer-dependent implementation, as a
SECONDARY reference point alongside `SacreBLEUEvaluator` -- NOT a
replacement for it. Raw BLEU scores are notoriously incomparable across
papers because they depend on ad-hoc tokenization choices; this module
exists specifically to demonstrate that discrepancy (report both, and show
they diverge) rather than to be the metric of record. `SacreBLEUEvaluator`
(sacrebleu.py) is the metric that should be reported in the paper's main
results tables.
"""

import logging
from typing import Dict, List, Union

import nltk
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

logger = logging.getLogger(__name__)


class NLTKBleuEvaluator:
    """Computes raw NLTK BLEU, for comparison against SacreBLEU only."""

    def __init__(self, tokenizer_name: str = "punkt_tab") -> None:
        """Initialize the evaluator, ensuring the NLTK tokenizer data is available.

        Args:
            tokenizer_name: NLTK tokenizer resource to download/use for
                whitespace+punctuation tokenization before scoring. NLTK
                >=3.9 renamed the resource `word_tokenize`/`sent_tokenize`
                actually load from "punkt" to "punkt_tab" -- both are
                checked/downloaded here so this works across NLTK versions.
        """
        for resource in ("punkt", "punkt_tab"):
            try:
                nltk.data.find(f"tokenizers/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)
        self.tokenizer_name = tokenizer_name

    def compute_corpus_bleu(
        self, hypotheses: List[str], references: List[Union[str, List[str]]]
    ) -> Dict[str, float]:
        """Compute raw corpus-level BLEU using NLTK's `word_tokenize` + smoothing.

        Args:
            hypotheses: Predicted translations.
            references: Reference translation(s) per hypothesis (a single
                string, or a list of acceptable reference strings, per row).

        Returns:
            Dict[str, float]: Key "nltk_bleu_score" (0-100 scale, matching
                SacreBLEU's scale for direct comparability).

        Raises:
            ValueError: If `hypotheses` or `references` is empty, or their
                lengths don't match.
        """
        if not hypotheses or not references:
            raise ValueError("hypotheses and references must not be empty.")
        if len(hypotheses) != len(references):
            raise ValueError(f"len(hypotheses)={len(hypotheses)} must match len(references)={len(references)}.")

        tokenized_hyps = [nltk.word_tokenize(h) for h in hypotheses]
        tokenized_refs = [
            [nltk.word_tokenize(r) for r in (refs if isinstance(refs, list) else [refs])] for refs in references
        ]

        smoothing = SmoothingFunction().method1
        score = corpus_bleu(tokenized_refs, tokenized_hyps, smoothing_function=smoothing)

        logger.info(f"Raw NLTK BLEU: {score * 100:.2f} (compare against SacreBLEUEvaluator for the same data).")
        return {"nltk_bleu_score": round(score * 100, 2)}
