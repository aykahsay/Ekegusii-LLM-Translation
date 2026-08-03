"""
Rare-Word Translation Accuracy
----------------------------------
Measures translation quality specifically on sentences containing RARE
source words (identified via `src.tokenizer.rare_words.RareWordIdentifier`),
as opposed to `LexicalEvaluator`/`lexical_accuracy.py`, which score
dictionary-term accuracy irrespective of corpus frequency. Rare-word
accuracy answers a different question: "does translation quality degrade
specifically on the long tail of infrequent vocabulary?" -- the notebook 10
("Rare-Word Precision Study") deliverable.
"""

import logging
from typing import Dict, List

from src.evaluation.chrf import ChrFEvaluator
from src.evaluation.sacrebleu import SacreBLEUEvaluator
from src.tokenizer.rare_words import RareWordIdentifier

logger = logging.getLogger(__name__)


class RareWordAccuracyEvaluator:
    """Compares translation quality on rare-word-containing vs. common sentences."""

    def __init__(self, max_frequency: int = 2) -> None:
        """Initialize the evaluator.

        Args:
            max_frequency: Corpus occurrence threshold defining "rare" (see
                `RareWordIdentifier.identify_rare_words`).
        """
        self.max_frequency = max_frequency
        self.identifier = RareWordIdentifier()
        self.bleu_eval = SacreBLEUEvaluator()
        self.chrf_eval = ChrFEvaluator()

    def split_by_rarity(
        self, sources: List[str], hypotheses: List[str], references: List[str]
    ) -> Dict[str, Dict[str, List[str]]]:
        """Partition parallel triples into rare-word-containing vs. common-only subsets.

        Args:
            sources: Source-language sentences.
            hypotheses: Model translations, aligned with `sources`.
            references: Reference translations, aligned with `sources`.

        Returns:
            Dict[str, Dict[str, List[str]]]: Keys "rare" and "common", each
                mapping to {"sources", "hypotheses", "references"} lists
                for that subset.

        Raises:
            ValueError: If `sources`, `hypotheses`, `references` lengths
                don't match.
        """
        if not (len(sources) == len(hypotheses) == len(references)):
            raise ValueError("sources, hypotheses, and references must be the same length.")

        rare_words = set(self.identifier.identify_rare_words(sources, max_frequency=self.max_frequency))

        split: Dict[str, Dict[str, List[str]]] = {
            "rare": {"sources": [], "hypotheses": [], "references": []},
            "common": {"sources": [], "hypotheses": [], "references": []},
        }

        for src, hyp, ref in zip(sources, hypotheses, references):
            words = set(src.lower().split())
            bucket = "rare" if words & rare_words else "common"
            split[bucket]["sources"].append(src)
            split[bucket]["hypotheses"].append(hyp)
            split[bucket]["references"].append(ref)

        return split

    def evaluate(self, sources: List[str], hypotheses: List[str], references: List[str]) -> Dict[str, Dict[str, float]]:
        """Compute SacreBLEU/chrF separately for rare-word vs. common sentences.

        Args:
            sources: Source-language sentences.
            hypotheses: Model translations, aligned with `sources`.
            references: Reference translations, aligned with `sources`.

        Returns:
            Dict[str, Dict[str, float]]: Keys "rare" and "common", each a
                dict with "sacrebleu_score", "chrf_plus_plus_score", and
                "n_sentences". A large gap (common >> rare) is the expected
                finding motivating lexical augmentation (E6).
        """
        split = self.split_by_rarity(sources, hypotheses, references)
        results: Dict[str, Dict[str, float]] = {}

        for bucket, data in split.items():
            if len(data["hypotheses"]) == 0:
                results[bucket] = {"sacrebleu_score": 0.0, "chrf_plus_plus_score": 0.0, "n_sentences": 0}
                continue

            bleu = self.bleu_eval.compute_corpus_bleu(data["hypotheses"], data["references"])
            chrf = self.chrf_eval.compute_chrf(data["hypotheses"], data["references"])

            results[bucket] = {
                "sacrebleu_score": bleu["sacrebleu_score"],
                "chrf_plus_plus_score": chrf["chrf_plus_plus_score"],
                "n_sentences": len(data["hypotheses"]),
            }

        logger.info(f"Rare-word accuracy split: {results}")
        return results
