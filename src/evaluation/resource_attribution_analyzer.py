"""
Resource Attribution & Ablation Matrix Analyzer
-----------------------------------------------
Aggregates performance metrics (SacreBLEU, chrF++, Lexical Term Accuracy) across
Experiments E0 to E8 to evaluate research hypotheses H1, H2, H3, and H4.
"""

import logging
from typing import Dict, List, Optional
import pandas as pd

from src.evaluation.chrf import ChrFEvaluator
from src.evaluation.lexical_evaluator import LexicalEvaluator
from src.evaluation.sacrebleu import SacreBLEUEvaluator

logger = logging.getLogger(__name__)


class ResourceAttributionAnalyzer:
    """Aggregates metrics and generates publication-ready attribution report matrix."""

    EXPERIMENTS = [
        ("E0", "Base Model Baseline", False, False, False, False, False),
        ("E1", "Direct Bilingual (ENG-EKE)", True, False, False, False, False),
        ("E2", "Swahili Transfer (SWA-EKE)", False, True, False, False, False),
        ("E3", "Combined Dual Bilingual", True, True, False, False, False),
        ("E4", "Trilingual Supervision (H2)", True, True, True, True, False),
        ("E5", "Full Sentence Corpus (H1)", True, True, True, True, False),
        ("E6", "Lexical Augmentation (H3)", True, True, True, True, True),
        ("E7", "Curriculum Learning", True, True, True, True, True),
        ("E8", "Final Ensemble Model", True, True, True, True, True),
    ]

    def __init__(self) -> None:
        """Initialize evaluation engines."""
        self.bleu_eval = SacreBLEUEvaluator()
        self.chrf_eval = ChrFEvaluator()
        self.lex_eval = LexicalEvaluator()

    def generate_full_attribution_report(
        self, experiment_results: Optional[Dict[str, Dict[str, float]]] = None
    ) -> pd.DataFrame:
        """Generate attribution matrix DataFrame combining metrics across all experiments.

        Args:
            experiment_results: Optional dict of precomputed metrics per experiment.

        Returns:
            pd.DataFrame: Resource attribution matrix.
        """
        records: List[Dict[str, str]] = []

        for exp_id, name, eng_eke, swa_eke, eng_swa, tri, lex in self.EXPERIMENTS:
            res = (experiment_results or {}).get(exp_id, {})
            records.append(
                {
                    "Experiment": exp_id,
                    "Description": name,
                    "ENG-EKE": "✓" if eng_eke else "✗",
                    "SWA-EKE": "✓" if swa_eke else "✗",
                    "Trilingual": "✓" if tri else "✗",
                    "Lexical": "✓" if lex else "✗",
                    "SacreBLEU": f"{res.get('sacrebleu', 0.0):.2f}",
                    "chrF++": f"{res.get('chrf_pp', 0.0):.2f}",
                    "Lexical Acc (%)": f"{res.get('lexical_acc', 0.0):.2f}",
                }
            )

        report_df = pd.DataFrame(records)
        logger.info("Generated full resource attribution report matrix.")
        return report_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = ResourceAttributionAnalyzer()
    df = analyzer.generate_full_attribution_report()
    print(df.to_string(index=False))
