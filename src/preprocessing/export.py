"""
Pipeline Output Export
--------------------------
Writes a cleaned/filtered DataFrame to disk (CSV or JSON) as the final step
of a preprocessing pipeline stage, logging the output schema and row count
alongside the write for reproducibility/debugging -- so a pipeline run's
logs always show exactly what was written where, without needing to
re-open the output file to check.
"""

import json
import logging
from pathlib import Path
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)


def export_dataframe(
    df: pd.DataFrame,
    output_path: Path,
    file_format: Literal["csv", "json", "jsonl"] = "csv",
) -> Path:
    """Write a DataFrame to disk in the requested format.

    Args:
        df: DataFrame to export.
        output_path: Destination file path. Parent directories are created
            if missing.
        file_format: One of "csv", "json" (a single JSON array), or "jsonl"
            (newline-delimited JSON, one record per line -- convenient for
            streaming instruction-tuning datasets into HuggingFace
            `datasets.load_dataset("json", ...)`).

    Returns:
        Path: The path written to (same as `output_path`).

    Raises:
        ValueError: If `file_format` is not one of the supported values.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if file_format == "csv":
        df.to_csv(output_path, index=False, encoding="utf-8")
    elif file_format == "json":
        df.to_json(output_path, orient="records", force_ascii=False, indent=2)
    elif file_format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for record in df.to_dict(orient="records"):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        raise ValueError(f"Unsupported file_format '{file_format}'. Use 'csv', 'json', or 'jsonl'.")

    logger.info(
        f"Exported {len(df):,} rows x {len(df.columns)} columns to {output_path} "
        f"(format={file_format}). Columns: {list(df.columns)}."
    )
    return output_path
