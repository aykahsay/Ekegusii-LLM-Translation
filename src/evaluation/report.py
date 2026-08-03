"""
Publication Report Generator
--------------------------------
Combines automatic metrics (SacreBLEU/chrF/COMET/lexical/rare-word),
human-evaluation summaries, and significance-test results into a single
Markdown report -- the final aggregation step feeding notebook 13
(publication_figures) and the paper's results section. Builds on
`ResourceAttributionAnalyzer` (the E0-E8 matrix) rather than duplicating it.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.evaluation.resource_attribution_analyzer import ResourceAttributionAnalyzer

logger = logging.getLogger(__name__)


class PublicationReportGenerator:
    """Assembles a full Markdown evaluation report from all evaluation components."""

    def __init__(self) -> None:
        """Initialize with a ResourceAttributionAnalyzer for the E0-E8 matrix."""
        self.attribution_analyzer = ResourceAttributionAnalyzer()

    def build_report(
        self,
        experiment_results: Optional[Dict[str, Dict[str, float]]] = None,
        human_eval_summary: Optional[pd.DataFrame] = None,
        significance_results: Optional[list] = None,
    ) -> str:
        """Build the full Markdown evaluation report.

        Args:
            experiment_results: Per-experiment automatic metrics, passed
                through to `ResourceAttributionAnalyzer.generate_full_attribution_report`.
            human_eval_summary: Optional output of
                `HumanEvalAggregator.aggregate` (mean fluency/adequacy/
                cultural_accuracy per model).
            significance_results: Optional list of
                `PairedBootstrapTest.run` result dicts to include as a
                significance-testing section.

        Returns:
            str: Full Markdown report text.
        """
        sections = ["# Evaluation Report: Ekegusii-LLM-Translation", ""]

        sections.append("## Automatic Metrics: Resource Attribution Matrix (E0-E8)")
        attribution_df = self.attribution_analyzer.generate_full_attribution_report(experiment_results)
        sections.append(attribution_df.to_markdown(index=False))
        sections.append("")

        if human_eval_summary is not None and len(human_eval_summary) > 0:
            sections.append("## Human Evaluation Summary (Fluency / Adequacy / Cultural Accuracy)")
            sections.append(human_eval_summary.to_markdown())
            sections.append("")

        if significance_results:
            sections.append("## Statistical Significance Tests")
            sections.append("| System A | System B | Mean Diff | p-value | Significant (p<0.05) |")
            sections.append("|---|---|---|---|---|")
            for result in significance_results:
                sections.append(
                    f"| {result['system_a']} | {result['system_b']} | {result['mean_diff']} | "
                    f"{result['p_value']} | {result['significant_at_0.05']} |"
                )
            sections.append("")

        report_text = "\n".join(sections)
        logger.info(f"Built publication report ({len(report_text):,} characters).")
        return report_text

    def save_report(self, report_text: str, output_path: Path) -> None:
        """Save a generated report to disk.

        Args:
            report_text: Output of `build_report`.
            output_path: Destination `.md` file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text, encoding="utf-8")
        logger.info(f"Saved publication report to {output_path}.")
