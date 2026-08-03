"""
Lexical-Substitution Augmentation Wrapper
-----------------------------------------------
Unlike bilingual/trilingual/lexical/curriculum.py, augmentation is NOT one
of the nine named E0-E8 experiments in
`ResourceAttributionAnalyzer.EXPERIMENTS` -- it's a cross-cutting technique
any experiment can optionally apply to its own training pairs. This module
wraps `LexicalSubstitutionAugmenter` (task_generation) as a reusable
decorator over any experiment's pair-building step, rather than being tied
to one specific experiment ID.
"""

import logging

import pandas as pd

from src.master_corpus.manager import MasterCorpusManager
from src.task_generation.augmentation import LexicalSubstitutionAugmenter

logger = logging.getLogger(__name__)


def augment_pairs_with_lexical_substitution(
    pairs_df: pd.DataFrame,
    source_lang: str,
    target_lang: str,
    manager: MasterCorpusManager,
    max_new_rows: int = 5000,
) -> pd.DataFrame:
    """Augment a (source, target) pair DataFrame with lexical-substitution variants.

    Args:
        pairs_df: DataFrame with "source"/"target" columns (see
            `src.task_generation.translation_pairs.extract_pairs`).
        source_lang: Source language name (must match a column in the
            lexical corpus, e.g. "English").
        target_lang: Target language name (e.g. "Ekegusii").
        manager: MasterCorpusManager used to load the lexical corpus.
        max_new_rows: Cap on the number of augmented rows generated.

    Returns:
        pd.DataFrame: `pairs_df` concatenated with augmented rows (columns
            "source", "target"; augmented rows' extra metadata columns are
            dropped for schema consistency), with a reset index.
    """
    lexical_df = manager.load_lexical_corpus()
    augmenter = LexicalSubstitutionAugmenter(lexical_df, source_lang, target_lang)

    augmented = augmenter.augment_dataframe(pairs_df, "source", "target", max_new_rows=max_new_rows)
    if len(augmented) == 0:
        logger.info("No lexical-substitution augmentations found; returning original pairs unchanged.")
        return pairs_df.reset_index(drop=True)

    combined = pd.concat([pairs_df[["source", "target"]], augmented[["source", "target"]]], ignore_index=True)
    logger.info(f"Augmented {len(pairs_df):,} pairs with {len(augmented):,} lexical-substitution variants.")
    return combined
