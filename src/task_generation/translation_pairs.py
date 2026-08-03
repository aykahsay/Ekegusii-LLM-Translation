"""
Raw Translation Pair Extraction
----------------------------------
Extracts plain (source_text, target_text, direction) parallel pairs from a
concept-level DataFrame -- WITHOUT the instruction-prompt wrapping that
`InstructionTaskGenerator` applies. Useful anywhere a plain parallel corpus
is needed rather than a causal-LM instruction format: SacreBLEU/chrF/COMET
evaluation (which need reference translations, not prompts), or seq2seq-
style baselines evaluated for comparison against the QLoRA instruction-
tuned models.
"""

import logging
from typing import List, Tuple

import pandas as pd

from src.utils.constants import TRANSLATION_DIRECTIONS

logger = logging.getLogger(__name__)


def extract_pairs(df: pd.DataFrame, source_lang: str, target_lang: str) -> pd.DataFrame:
    """Extract non-empty (source, target) pairs for one translation direction.

    Args:
        df: Concept-level DataFrame containing `source_lang` and
            `target_lang` columns.
        source_lang: Source language column name (e.g. "English").
        target_lang: Target language column name (e.g. "Ekegusii").

    Returns:
        pd.DataFrame: Columns "concept_id" (if present in `df`), "source",
            "target", "source_lang", "target_lang" -- one row per valid
            pair (both source and target non-empty).

    Raises:
        KeyError: If either language column is missing from `df`.
    """
    for col in (source_lang, target_lang):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

    valid = df.dropna(subset=[source_lang, target_lang])
    valid = valid[
        (valid[source_lang].astype(str).str.strip() != "") & (valid[target_lang].astype(str).str.strip() != "")
    ]

    result = pd.DataFrame(
        {
            "source": valid[source_lang].astype(str).str.strip().values,
            "target": valid[target_lang].astype(str).str.strip().values,
            "source_lang": source_lang,
            "target_lang": target_lang,
        }
    )
    if "concept_id" in df.columns:
        result.insert(0, "concept_id", valid["concept_id"].values)

    result = result.reset_index(drop=True)
    logger.info(f"Extracted {len(result):,} '{source_lang}'->'{target_lang}' pairs from {len(df):,} rows.")
    return result


def extract_all_directions(df: pd.DataFrame) -> pd.DataFrame:
    """Extract parallel pairs for all six standard translation directions.

    Args:
        df: Concept-level DataFrame with English, Kiswahili, Ekegusii columns.

    Returns:
        pd.DataFrame: Concatenation of `extract_pairs` results for every
            direction in `src.utils.constants.TRANSLATION_DIRECTIONS`, with
            a reset index.
    """
    frames: List[pd.DataFrame] = []
    for source_lang, target_lang in TRANSLATION_DIRECTIONS:
        frames.append(extract_pairs(df, source_lang, target_lang))

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Extracted {len(combined):,} total pairs across {len(TRANSLATION_DIRECTIONS)} directions.")
    return combined


def direction_counts(df: pd.DataFrame) -> "List[Tuple[str, int]]":
    """Count available pairs per translation direction, without materializing them.

    Cheaper than `extract_all_directions` when only counts are needed
    (e.g. for the resource scheduler's `direction_counts` input).

    Args:
        df: Concept-level DataFrame with English, Kiswahili, Ekegusii columns.

    Returns:
        List[Tuple[str, int]]: (direction_label, count) pairs, e.g.
            ("English->Ekegusii", 37721).
    """
    counts = []
    for source_lang, target_lang in TRANSLATION_DIRECTIONS:
        pair_df = extract_pairs(df, source_lang, target_lang)
        counts.append((f"{source_lang}->{target_lang}", len(pair_df)))
    return counts
