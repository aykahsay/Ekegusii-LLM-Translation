"""
Lexical Terminology & Dictionary Evaluator
------------------------------------------
Evaluates translation model precision on specialized domain terminology and rare words
using exact string matching and morphological stem prefix coverage against the Master Lexical Corpus.
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd

from src.master_corpus.manager import MasterCorpusManager

logger = logging.getLogger(__name__)


class LexicalEvaluator:
    """Evaluates terminology precision and morphological stem match coverage."""

    def __init__(self, manager: Optional[MasterCorpusManager] = None) -> None:
        """Initialize with MasterCorpusManager instance."""
        self.manager = manager or MasterCorpusManager()
        try:
            self.lexical_df = self.manager.load_lexical_corpus()
        except Exception as e:
            logger.warning(f"Could not load lexical corpus automatically: {e}")
            self.lexical_df = pd.DataFrame()

    def evaluate_predictions(
        self, predictions: List[Tuple[str, str]]
    ) -> Dict[str, float]:
        """Evaluate translation hypothesis predictions against reference dictionary pairs.

        Args:
            predictions: List of tuples (hypothesis_translation, reference_translation).

        Returns:
            Dict containing exact_term_accuracy and morphological_coverage scores.
        """
        if not predictions:
            return {"exact_term_accuracy": 0.0, "morphological_coverage": 0.0, "total_eval": 0}

        exact_matches = 0
        stem_matches = 0
        total = len(predictions)

        for hypo, ref in predictions:
            hypo_clean = hypo.strip().lower()
            ref_clean = ref.strip().lower()

            # 1. Exact Term Match
            if hypo_clean == ref_clean:
                exact_matches += 1
                stem_matches += 1
            else:
                # 2. Morphological Stem Match (First 4 characters prefix overlap for Bantu stems)
                stem_ref = ref_clean[:4] if len(ref_clean) >= 4 else ref_clean
                if stem_ref in hypo_clean:
                    stem_matches += 1

        exact_acc = (exact_matches / total) * 100.0
        stem_cov = (stem_matches / total) * 100.0

        result = {
            "exact_term_accuracy": float(exact_acc),
            "morphological_coverage": float(stem_cov),
            "total_terms_evaluated": total,
        }

        logger.info(f"Lexical Accuracy: Exact={exact_acc:.2f}%, Stem={stem_cov:.2f}%")
        return result
