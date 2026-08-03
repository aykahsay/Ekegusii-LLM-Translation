"""
Dataset Content Validator
---------------------------
Validates the *content* of a loaded corpus DataFrame -- null rates, empty
strings, duplicate concept IDs, and unexpected row counts -- as distinct
from `MasterCorpusManager._validate_columns` (which only checks that
required columns are present) and `DataLeakageChecker` (which checks
overlap *across* splits rather than within a single DataFrame).
"""

import logging
from dataclasses import dataclass, field
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Result of validating a single DataFrame.

    Attributes:
        dataset_name: Human-readable label for the validated DataFrame.
        row_count: Number of rows validated.
        issues: List of human-readable issue descriptions found. Empty if
            the dataset passed all checks.
    """

    dataset_name: str
    row_count: int
    issues: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Whether the dataset passed validation (no issues found)."""
        return len(self.issues) == 0


class CorpusValidator:
    """Runs content-quality checks against a corpus DataFrame."""

    def validate(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        id_column: str,
        text_columns: List[str],
        max_null_rate: float = 0.05,
    ) -> ValidationReport:
        """Validate a corpus DataFrame's content quality.

        Args:
            df: DataFrame to validate.
            dataset_name: Label used in the returned report and log messages.
            id_column: Name of the unique identifier column (e.g. "concept_id").
            text_columns: Columns expected to contain sentence/term text
                (e.g. ["English", "Kiswahili", "Ekegusii"]).
            max_null_rate: Maximum tolerated fraction of null/empty values
                per text column before it is flagged as an issue. Some text
                columns (e.g. lexical corpus "English") are legitimately
                sparse, so this is a warning threshold, not a hard failure.

        Returns:
            ValidationReport: Structured report of row count and any issues found.
        """
        report = ValidationReport(dataset_name=dataset_name, row_count=len(df))

        self._check_duplicate_ids(df, id_column, report)
        self._check_null_rates(df, text_columns, max_null_rate, report)
        self._check_all_columns_empty(df, text_columns, report)

        if report.is_valid:
            logger.info(f"[{dataset_name}] Validation passed ({report.row_count:,} rows).")
        else:
            logger.warning(f"[{dataset_name}] Validation found {len(report.issues)} issue(s): {report.issues}")

        return report

    @staticmethod
    def _check_duplicate_ids(df: pd.DataFrame, id_column: str, report: ValidationReport) -> None:
        if id_column not in df.columns:
            report.issues.append(f"ID column '{id_column}' not present.")
            return
        num_duplicates = df[id_column].duplicated().sum()
        if num_duplicates > 0:
            report.issues.append(f"{num_duplicates} duplicate '{id_column}' values found.")

    @staticmethod
    def _check_null_rates(
        df: pd.DataFrame, text_columns: List[str], max_null_rate: float, report: ValidationReport
    ) -> None:
        for col in text_columns:
            if col not in df.columns:
                report.issues.append(f"Expected text column '{col}' is missing.")
                continue
            null_rate = df[col].isna().mean()
            if null_rate > max_null_rate:
                report.issues.append(
                    f"Column '{col}' null rate {null_rate:.1%} exceeds threshold {max_null_rate:.1%}."
                )

    @staticmethod
    def _check_all_columns_empty(df: pd.DataFrame, text_columns: List[str], report: ValidationReport) -> None:
        present_columns = [c for c in text_columns if c in df.columns]
        if not present_columns:
            return
        all_empty_mask = df[present_columns[0]].isna()
        for col in present_columns[1:]:
            all_empty_mask = all_empty_mask & df[col].isna()
        num_fully_empty = all_empty_mask.sum()
        if num_fully_empty > 0:
            report.issues.append(f"{num_fully_empty} row(s) have all text columns empty.")
