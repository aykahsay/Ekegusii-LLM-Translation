"""
Near-Duplicate Detection
---------------------------
Detects exact and near-duplicate text (e.g. two scraped PSAs differing only
by punctuation or a trailing date) via normalized-text hashing and n-gram
Jaccard similarity. Distinct from `src.master_corpus.cleaner.CorpusCleaner`,
which only drops rows with duplicate IDs or fully-empty text -- this module
catches duplicate *content* that arrived under different concept_ids
(e.g. the same advisory scraped from two different source websites).
"""

import hashlib
import logging
from typing import List, Set, Tuple

import pandas as pd

from src.preprocessing.normalize import normalize_text

logger = logging.getLogger(__name__)


def exact_duplicate_mask(series: pd.Series) -> pd.Series:
    """Flag rows whose normalized text is an exact duplicate of an earlier row.

    Args:
        series: Text column to check (raw, un-normalized values are fine --
            normalization is applied internally before comparison).

    Returns:
        pd.Series: Boolean mask, True for every occurrence after the first
            of a given normalized text (i.e. rows that would be removed by
            `series[~mask]` to keep only first occurrences).
    """
    normalized = series.apply(lambda x: normalize_text(x) or "")
    hashes = normalized.apply(lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest())
    return hashes.duplicated(keep="first")


def _char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Return the set of character n-grams for a string."""
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard_similarity(text_a: str, text_b: str, n: int = 3) -> float:
    """Compute character n-gram Jaccard similarity between two strings.

    Args:
        text_a: First string.
        text_b: Second string.
        n: Character n-gram size.

    Returns:
        float: Jaccard similarity in [0, 1]. 1.0 if both strings are empty.
    """
    grams_a = _char_ngrams(text_a.lower(), n)
    grams_b = _char_ngrams(text_b.lower(), n)
    if not grams_a and not grams_b:
        return 1.0
    intersection = len(grams_a & grams_b)
    union = len(grams_a | grams_b)
    return intersection / union if union > 0 else 0.0


def find_near_duplicates(
    series: pd.Series, threshold: float = 0.9, ngram_size: int = 3
) -> List[Tuple[int, int, float]]:
    """Find pairs of rows whose text is near-duplicate (above a similarity threshold).

    Uses an O(n^2) pairwise comparison, so this is intended for use on
    already-small candidate pools (e.g. rows sharing the same `source` or
    `dataset_origin`), not the full 49k-row corpus at once.

    Args:
        series: Text column to compare (index preserved in output).
        threshold: Minimum Jaccard similarity to report a pair as near-duplicate.
        ngram_size: Character n-gram size used for similarity.

    Returns:
        List[Tuple[int, int, float]]: (index_a, index_b, similarity) triples
            for every pair exceeding `threshold`, sorted by similarity
            descending.

    Raises:
        ValueError: If `threshold` is not in (0, 1].
    """
    if not (0.0 < threshold <= 1.0):
        raise ValueError(f"threshold must be in (0, 1], got {threshold}.")

    texts = series.apply(lambda x: normalize_text(x) or "")
    indices = texts.index.tolist()
    values = texts.tolist()

    matches: List[Tuple[int, int, float]] = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if not values[i] or not values[j]:
                continue
            sim = jaccard_similarity(values[i], values[j], n=ngram_size)
            if sim >= threshold:
                matches.append((indices[i], indices[j], sim))

    matches.sort(key=lambda m: m[2], reverse=True)
    logger.info(f"Found {len(matches)} near-duplicate pair(s) at threshold >= {threshold}.")
    return matches
