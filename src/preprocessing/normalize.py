"""
Per-Cell Text Normalization
------------------------------
String-level normalization functions applied to individual sentence/term
cells: Unicode NFC normalization, whitespace collapsing, control-character
stripping, and quote/punctuation standardization. Complements
`src/master_corpus/cleaner.py` (which operates at the row/DataFrame level,
e.g. dropping empty or duplicate rows) -- this module supplies the actual
per-cell text transform that cleaner and other pipeline stages call.
"""

import logging
import re
import unicodedata
from typing import Literal, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Curly/typographic quotes and dashes standardized to their ASCII equivalents
# so downstream tokenizers see a consistent character set across sources
# scraped from different websites/PDF extractors.
_QUOTE_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-",
}


def normalize_unicode(text: str, form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFC") -> str:
    """Apply Unicode normalization to a string.

    Args:
        text: Input text.
        form: Unicode normalization form ("NFC", "NFD", "NFKC", "NFKD").
            "NFC" is the default and generally correct choice for
            comparing/tokenizing text that may come from different sources
            with different composed/decomposed accent representations.

    Returns:
        str: Unicode-normalized text.
    """
    return unicodedata.normalize(form, text)


def collapse_whitespace(text: str) -> str:
    """Collapse consecutive whitespace (including newlines/tabs) to single spaces.

    Args:
        text: Input text.

    Returns:
        str: Text with internal whitespace collapsed and leading/trailing
            whitespace stripped.
    """
    return _WHITESPACE_RE.sub(" ", text).strip()


def strip_control_characters(text: str) -> str:
    """Remove non-printable control characters left over from PDF/HTML extraction.

    Args:
        text: Input text.

    Returns:
        str: Text with control characters (except standard whitespace) removed.
    """
    return _CONTROL_CHAR_RE.sub("", text)


def standardize_punctuation(text: str) -> str:
    """Replace typographic quotes/dashes with their plain ASCII equivalents.

    Args:
        text: Input text.

    Returns:
        str: Text with curly quotes and em/en dashes standardized.
    """
    for original, replacement in _QUOTE_MAP.items():
        text = text.replace(original, replacement)
    return text


def normalize_text(text: Optional[str]) -> Optional[str]:
    """Apply the full normalization pipeline to a single text value.

    Order: Unicode NFC -> control-character strip -> punctuation
    standardization -> whitespace collapse.

    Args:
        text: Input text, or None/NaN (passed through unchanged).

    Returns:
        Optional[str]: Normalized text, or None if input was None/NaN, or
            None if the result is an empty string after normalization.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None

    result = normalize_unicode(text if isinstance(text, str) else str(text))
    result = strip_control_characters(result)
    result = standardize_punctuation(result)
    result = collapse_whitespace(result)
    return result if result else None


def normalize_column(series: pd.Series) -> pd.Series:
    """Apply `normalize_text` to every value in a pandas Series.

    Args:
        series: Column of raw text values (may contain NaN).

    Returns:
        pd.Series: Normalized text column, with empty results as None.
    """
    return series.apply(normalize_text)
