"""
Leakage Audit Reporting
--------------------------
`DataLeakageChecker` (integrity.py) answers a strict pass/fail question
about the master split. This module builds on top of it to produce a
structured, persistable audit report (for docs/reproducibility.md and CI-
style checks), and additionally verifies that NEW derived datasets (e.g. a
bilingual or trilingual projection) stay consistent with the frozen master
split assignment via `CorpusSplitter`.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.master_corpus.integrity import DataLeakageChecker
from src.master_corpus.manager import MasterCorpusManager
from src.master_corpus.splitter import CorpusSplitter
from src.utils.helpers import write_json

logger = logging.getLogger(__name__)


class LeakageAuditReporter:
    """Produces structured leakage audit reports for the master corpus and derived datasets."""

    def __init__(self, manager: MasterCorpusManager) -> None:
        """Initialize the reporter.

        Args:
            manager: MasterCorpusManager used to load the master splits.
        """
        self.manager = manager
        self.checker = DataLeakageChecker(manager)
        self.splitter = CorpusSplitter()

    def audit_master_split(self) -> Dict[str, Any]:
        """Run the master-split leakage audit and capture the result as a report.

        Returns:
            Dict[str, Any]: Report with keys "passed" (bool) and
                "failure_reason" (str, empty if passed).
        """
        try:
            self.checker.verify_all()
            return {"passed": True, "failure_reason": ""}
        except ValueError as exc:
            logger.error(f"Master split leakage audit failed: {exc}")
            return {"passed": False, "failure_reason": str(exc)}

    def audit_derived_dataset(
        self, derived_df: pd.DataFrame, dataset_name: str, id_column: str = "concept_id"
    ) -> Dict[str, Any]:
        """Verify a derived dataset's rows are consistent with the frozen master split.

        Args:
            derived_df: A derived dataset (e.g. an English-Ekegusii
                projection) that should have been built by filtering the
                master corpus, not by independently re-splitting it.
            dataset_name: Human-readable label for the report.
            id_column: Name of the unique identifier column shared with the
                master corpus.

        Returns:
            Dict[str, Any]: Report with keys "dataset_name", "total_rows",
                "split_breakdown" (rows per split), and "unassigned_rows"
                (rows whose ID doesn't exist in any master split -- a
                red flag suggesting the derived dataset wasn't built from
                the master corpus).
        """
        train_df = self.manager.load_train_split()
        val_df = self.manager.load_val_split()
        test_df = self.manager.load_test_split()

        assignment = self.splitter.build_split_assignment(train_df, val_df, test_df, id_column)
        partitioned = self.splitter.project_to_existing_split(derived_df, assignment, id_column)

        report = {
            "dataset_name": dataset_name,
            "total_rows": len(derived_df),
            "split_breakdown": {
                split_name: len(partitioned.get(split_name, derived_df.iloc[0:0]))
                for split_name in ("train", "val", "test")
            },
            "unassigned_rows": len(partitioned.get("unassigned", derived_df.iloc[0:0])),
        }

        if report["unassigned_rows"] > 0:
            logger.warning(
                f"[{dataset_name}] {report['unassigned_rows']} row(s) not found in any master split -- "
                "verify this dataset was derived from the master corpus."
            )
        else:
            logger.info(f"[{dataset_name}] All rows consistent with the frozen master split.")

        return report

    def save_report(self, report: Dict[str, Any], output_path: Path) -> None:
        """Persist an audit report to disk as JSON.

        Args:
            report: Output of `audit_master_split` or `audit_derived_dataset`.
            output_path: Destination `.json` file path.
        """
        write_json(report, output_path)
        logger.info(f"Saved leakage audit report to {output_path}.")
