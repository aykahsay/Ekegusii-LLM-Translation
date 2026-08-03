"""
Lexical Substitution Augmentation
-------------------------------------
Generates synthetic parallel-sentence variants by substituting a known
dictionary term (from `master_lexical_corpus.csv`) for its translation
inside an existing parallel sentence pair, when that term appears in both
the source and target sentence. This is a lightweight augmentation
technique that -- unlike back-translation -- requires no trained model,
making it usable from the very first experiment rather than only after a
baseline model exists.
"""

import logging
import re
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class LexicalSubstitutionAugmenter:
    """Creates augmented sentence pairs via dictionary-term substitution."""

    def __init__(self, lexical_df: pd.DataFrame, source_lang: str, target_lang: str) -> None:
        """Initialize the augmenter with a lexicon for one language pair.

        Args:
            lexical_df: Master lexical corpus DataFrame (or a subset),
                containing at least `source_lang` and `target_lang` columns.
            source_lang: Source language column name (e.g. "English").
            target_lang: Target language column name (e.g. "Ekegusii").

        Raises:
            KeyError: If either language column is missing from `lexical_df`.
        """
        for col in (source_lang, target_lang):
            if col not in lexical_df.columns:
                raise KeyError(f"Column '{col}' not found in lexical_df.")

        self.source_lang = source_lang
        self.target_lang = target_lang
        pairs = lexical_df.dropna(subset=[source_lang, target_lang])
        self.lexicon = [
            (str(row[source_lang]).strip().lower(), str(row[target_lang]).strip())
            for _, row in pairs.iterrows()
            if str(row[source_lang]).strip() and str(row[target_lang]).strip()
        ]

    def augment_pair(self, source_text: str, target_text: str, max_substitutions: int = 1) -> List[dict]:
        """Generate augmented variants of one (source, target) sentence pair.

        For each lexicon entry whose source term appears (as a whole word,
        case-insensitive) in `source_text`, produce a variant with that
        term's word-boundary-safe first occurrence replaced -- but only if
        the corresponding target term also appears in `target_text`, so the
        substitution stays translation-consistent on both sides.

        Args:
            source_text: Original source-language sentence.
            target_text: Original target-language sentence (aligned translation).
            max_substitutions: Maximum number of augmented variants to return.

        Returns:
            List[dict]: Each dict has keys "source", "target",
                "substituted_term" (the source-language dictionary term
                used). Empty list if no applicable substitution was found.
        """
        variants: List[dict] = []
        source_lower = source_text.lower()

        for src_term, tgt_term in self.lexicon:
            if len(variants) >= max_substitutions:
                break

            src_pattern = rf"\b{re.escape(src_term)}\b"
            if not re.search(src_pattern, source_lower):
                continue
            if tgt_term.lower() not in target_text.lower():
                continue

            new_source = re.sub(src_pattern, src_term, source_text, count=1, flags=re.IGNORECASE)
            variants.append({"source": new_source, "target": target_text, "substituted_term": src_term})

        return variants

    def augment_dataframe(
        self, df: pd.DataFrame, source_col: str, target_col: str, max_new_rows: Optional[int] = None
    ) -> pd.DataFrame:
        """Augment an entire parallel DataFrame via lexical substitution.

        Args:
            df: DataFrame with `source_col` and `target_col` sentence columns.
            source_col: Source-language column name in `df`.
            target_col: Target-language column name in `df`.
            max_new_rows: Optional cap on the total number of augmented rows
                produced across the whole DataFrame.

        Returns:
            pd.DataFrame: New augmented rows only (does NOT include the
                original rows) with columns "source", "target",
                "substituted_term", "augmentation_method" (constant
                "lexical_substitution").
        """
        augmented_rows: List[dict] = []
        for _, row in df.iterrows():
            if max_new_rows is not None and len(augmented_rows) >= max_new_rows:
                break

            src_text = row.get(source_col)
            tgt_text = row.get(target_col)
            if not isinstance(src_text, str) or not isinstance(tgt_text, str):
                continue

            variants = self.augment_pair(src_text, tgt_text, max_substitutions=1)
            for variant in variants:
                variant["augmentation_method"] = "lexical_substitution"
                augmented_rows.append(variant)

        result = pd.DataFrame(augmented_rows)
        logger.info(f"Generated {len(result):,} augmented rows via lexical substitution.")
        return result
