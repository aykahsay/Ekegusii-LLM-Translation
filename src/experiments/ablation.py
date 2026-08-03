"""
Resource Attribution Ablation Aggregator (+ E8 Final Model Selection)
--------------------------------------------------------------------------
Loads each completed experiment's `experiments/{ID}/results.json` (written
by `BaseExperiment.save_results`), feeds them into
`ResourceAttributionAnalyzer` for the publication matrix, runs pairwise
significance tests between adjacent experiments (E0 vs E1, E1 vs E3, etc.)
to support the "resource X significantly helps" claims, and determines
E8 (Final Model): whichever of E0-E7 scores highest on COMET/SacreBLEU
against the fixed test set is the one that gets merged and deployed.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.evaluation.resource_attribution_analyzer import ResourceAttributionAnalyzer
from src.evaluation.significance import PairedBootstrapTest
from src.utils.constants import EXPERIMENTS_DIR
from src.utils.helpers import read_json

logger = logging.getLogger(__name__)

# Adjacent-experiment comparisons meaningful for the paper's ablation
# narrative (each pair isolates ONE added resource/technique).
_ADJACENT_COMPARISONS: List[Tuple[str, str]] = [
    ("E0_Baseline", "E1_English_Ekegusii"),
    ("E1_English_Ekegusii", "E3_Bilingual"),
    ("E2_Swahili_Ekegusii", "E3_Bilingual"),
    ("E3_Bilingual", "E4_Trilingual"),
    ("E4_Trilingual", "E5_Full_Resources"),
    ("E5_Full_Resources", "E6_Lexical_Augmentation"),
    ("E6_Lexical_Augmentation", "E7_Curriculum_Learning"),
]


class AblationAggregator:
    """Aggregates completed experiment results into the final ablation study."""

    def __init__(self) -> None:
        """Initialize the aggregator."""
        self.attribution_analyzer = ResourceAttributionAnalyzer()

    def load_all_results(self) -> Dict[str, Dict[str, Any]]:
        """Load every experiment's saved `results.json`, skipping any not yet run.

        Returns:
            Dict[str, Dict[str, Any]]: Maps experiment_id -> its results
                dict (only experiments whose `results.json` exists).
        """
        results: Dict[str, Dict[str, Any]] = {}
        for exp_dir in sorted(EXPERIMENTS_DIR.glob("E*")):
            results_path = exp_dir / "results.json"
            if results_path.exists():
                results[exp_dir.name] = read_json(results_path)
            else:
                logger.warning(f"No results.json found for {exp_dir.name}; skipping.")
        return results

    def build_attribution_report(self, model_key: str = "aya"):
        """Build the E0-E8 attribution matrix from saved results for one model.

        Args:
            model_key: Which model's results to extract from each
                experiment's results dict ("aya" or "llama").

        Returns:
            pd.DataFrame: Output of `ResourceAttributionAnalyzer.generate_full_attribution_report`.
        """
        all_results = self.load_all_results()
        formatted = {
            # ResourceAttributionAnalyzer keys its lookup by the short code
            # ("E0", "E1", ...) prefixing each full experiment_id ("E0_Baseline",
            # "E1_English_Ekegusii", ...) -- split it off here.
            exp_id.split("_", 1)[0]: {
                "sacrebleu": res.get(model_key, res).get("sacrebleu_score", 0.0),
                "chrf_pp": res.get(model_key, res).get("chrf_plus_plus_score", 0.0),
                "lexical_acc": res.get(model_key, res).get("exact_term_accuracy", 0.0),
            }
            for exp_id, res in all_results.items()
        }
        return self.attribution_analyzer.generate_full_attribution_report(formatted)

    def run_significance_tests(
        self, per_sentence_scores: Dict[str, List[float]]
    ) -> List[Dict[str, Any]]:
        """Run paired bootstrap significance tests across adjacent experiment pairs.

        Args:
            per_sentence_scores: Maps experiment_id -> per-sentence metric
                scores (e.g. per-sentence COMET) on the fixed test set,
                same sentence order for every experiment.

        Returns:
            List[Dict[str, Any]]: One `PairedBootstrapTest.run` result per
                adjacent pair where BOTH experiments have scores available.
        """
        test = PairedBootstrapTest()
        results = []
        for exp_a, exp_b in _ADJACENT_COMPARISONS:
            if exp_a not in per_sentence_scores or exp_b not in per_sentence_scores:
                logger.warning(f"Skipping significance test {exp_a} vs {exp_b}: missing scores.")
                continue
            result = test.run(per_sentence_scores[exp_b], per_sentence_scores[exp_a], exp_b, exp_a)
            results.append(result)
        return results

    def select_final_model(self, model_key: str = "aya", metric: str = "sacrebleu_score") -> Optional[str]:
        """Determine which E0-E7 experiment should become E8 (Final Model).

        Args:
            model_key: Which model's results to compare ("aya" or "llama").
            metric: Metric key to rank experiments by (higher is better).

        Returns:
            Optional[str]: The experiment_id with the highest `metric`
                value, or None if no results have been saved yet.
        """
        all_results = self.load_all_results()
        candidates = {
            exp_id: res.get(model_key, res).get(metric, float("-inf"))
            for exp_id, res in all_results.items()
            if exp_id != "E8_Final_Model"
        }
        if not candidates:
            logger.warning("No experiment results available; cannot select a final model yet.")
            return None

        best_experiment = max(candidates, key=lambda k: candidates[k])
        logger.info(
            f"Selected '{best_experiment}' as E8 Final Model source "
            f"({model_key} {metric}={candidates[best_experiment]:.2f})."
        )
        return best_experiment
