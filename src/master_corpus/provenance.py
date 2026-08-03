"""
Dataset Provenance Reporting
-------------------------------
Summarizes where each row in the master corpus came from (`source`,
`dataset_origin`) into a structured lineage report, for reproducibility
documentation (docs/datasets.md) and the paper's data-statement appendix.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.utils.helpers import write_json

logger = logging.getLogger(__name__)


class ProvenanceReporter:
    """Builds and persists dataset lineage/provenance summaries."""

    def build_report(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Build a provenance summary for a corpus DataFrame.

        Args:
            df: Corpus DataFrame containing `source` and `dataset_origin` columns.
            dataset_name: Human-readable label for the report (e.g.
                "master_sentence_corpus").

        Returns:
            Dict[str, Any]: Report with keys "dataset_name", "total_rows",
                "source_distribution", and "dataset_origin_distribution"
                (each a mapping of category -> {"count", "percentage"}).

        Raises:
            KeyError: If neither `source` nor `dataset_origin` columns
                are present.
        """
        if "source" not in df.columns and "dataset_origin" not in df.columns:
            raise KeyError("DataFrame must contain at least one of 'source' or 'dataset_origin'.")

        report: Dict[str, Any] = {
            "dataset_name": dataset_name,
            "total_rows": len(df),
        }

        for column, key in (("source", "source_distribution"), ("dataset_origin", "dataset_origin_distribution")):
            if column in df.columns:
                counts = df[column].value_counts()
                pct = (counts / len(df) * 100).round(2)
                report[key] = {
                    category: {"count": int(count), "percentage": float(pct[category])}
                    for category, count in counts.items()
                }

        logger.info(f"Built provenance report for '{dataset_name}' ({len(df):,} rows).")
        return report

    def save_report(self, report: Dict[str, Any], output_path: Path) -> None:
        """Persist a provenance report to disk as JSON.

        Args:
            report: Output of `build_report`.
            output_path: Destination `.json` file path.
        """
        write_json(report, output_path)
        logger.info(f"Saved provenance report to {output_path}.")

    def to_markdown(self, report: Dict[str, Any]) -> str:
        """Render a provenance report as a Markdown table for documentation.

        Args:
            report: Output of `build_report`.

        Returns:
            str: Markdown-formatted lineage summary.
        """
        lines = [f"# Provenance: {report['dataset_name']}", "", f"Total rows: {report['total_rows']:,}", ""]

        for key, title in (("source_distribution", "Source"), ("dataset_origin_distribution", "Dataset Origin")):
            if key not in report:
                continue
            lines.append(f"## {title} Distribution")
            lines.append("")
            lines.append(f"| {title} | Count | Percentage |")
            lines.append("|---|---|---|")
            for category, stats in report[key].items():
                lines.append(f"| {category} | {stats['count']:,} | {stats['percentage']:.2f}% |")
            lines.append("")

        return "\n".join(lines)
