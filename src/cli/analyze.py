"""
CLI: analyze
----------------
Wraps `AblationAggregator` for CLI use: builds the E0-E8 attribution matrix
from whichever experiments have saved results so far, and reports which
experiment should currently become E8 (Final Model).
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from src.experiments.ablation import AblationAggregator

logger = logging.getLogger(__name__)


def run_analyze(model_key: str = "aya") -> Dict[str, Any]:
    """Build the current attribution report and final-model recommendation.

    Args:
        model_key: Which model's results to analyze ("aya" or "llama").

    Returns:
        Dict[str, Any]: Keys "attribution_table" (pd.DataFrame) and
            "recommended_final_model" (Optional[str], the experiment_id
            with the best score so far, or None if no results exist yet).
    """
    aggregator = AblationAggregator()
    attribution_table: pd.DataFrame = aggregator.build_attribution_report(model_key)
    recommended: Optional[str] = aggregator.select_final_model(model_key)

    logger.info(f"Analysis complete. Recommended final model source: {recommended}.")
    return {"attribution_table": attribution_table, "recommended_final_model": recommended}
