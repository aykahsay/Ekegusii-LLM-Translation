"""
Language Identification / Consistency Checking
--------------------------------------------------
Verifies that text in a given column actually matches its expected
language, catching scraping/alignment errors (e.g. an "Ekegusii" cell that
is actually untranslated English).

`langdetect` (Google's port of the Naive-Bayes-based language-detection
library) supports English and Swahili, so those two are checked with it
directly. It does NOT have a model for Ekegusii (no major statistical LID
library does -- it is not in CLD3/langdetect/fastText's language-code
tables). For Ekegusii, this module instead uses a lexicon-overlap heuristic
against `data/master_corpus/master_lexical_corpus.csv`'s Ekegusii vocabulary,
which is a much weaker signal than a trained LID model but is the only
practical option without training a bespoke classifier.
"""

import logging
import re
from typing import Optional, Set

from langdetect import DetectorFactory, LangDetectException, detect

from src.utils.constants import LANG_ENGLISH, LANG_KISWAHILI

logger = logging.getLogger(__name__)

# langdetect is non-deterministic by default (uses a random seed internally);
# fix it so repeated calls on the same text return the same result.
DetectorFactory.seed = 0

_LANGDETECT_CODE_MAP = {
    LANG_ENGLISH: "en",
    LANG_KISWAHILI: "sw",
}


def detect_language_code(text: str) -> Optional[str]:
    """Detect the ISO 639-1 language code of a text snippet via langdetect.

    Args:
        text: Text to identify.

    Returns:
        Optional[str]: Two-letter ISO code (e.g. "en", "sw"), or None if
            detection fails (e.g. text too short or non-alphabetic).
    """
    try:
        return detect(text)
    except LangDetectException:
        return None


def matches_expected_language(text: str, expected_language: str) -> Optional[bool]:
    """Check whether `text` appears to be written in `expected_language`.

    Only supports English and Kiswahili (the two languages langdetect
    covers here). For Ekegusii, use `ekegusii_lexicon_overlap` instead.

    Args:
        text: Text to check.
        expected_language: One of `src.utils.constants.LANG_ENGLISH` or
            `LANG_KISWAHILI`.

    Returns:
        Optional[bool]: True/False if a detection could be made, or None
            if detection failed (e.g. text too short) or the language
            isn't supported by this function.

    Raises:
        ValueError: If `expected_language` is not English or Kiswahili.
    """
    if expected_language not in _LANGDETECT_CODE_MAP:
        raise ValueError(
            f"matches_expected_language only supports {list(_LANGDETECT_CODE_MAP)}, "
            f"got '{expected_language}'. Use ekegusii_lexicon_overlap for Ekegusii."
        )

    detected = detect_language_code(text)
    if detected is None:
        return None
    return detected == _LANGDETECT_CODE_MAP[expected_language]


def ekegusii_lexicon_overlap(text: str, ekegusii_vocabulary: Set[str]) -> float:
    """Estimate whether `text` is plausibly Ekegusii via lexicon word overlap.

    This is a weak heuristic (not a trained classifier): it computes the
    fraction of `text`'s words that appear in a known Ekegusii vocabulary
    set (e.g. built from `master_lexical_corpus.csv` and/or the Ekegusii
    column of the sentence corpus). A low overlap for a cell labeled
    "Ekegusii" is a signal worth manual review, not proof of an error --
    short function words often coincidentally overlap with Kiswahili too.

    Args:
        text: Text to check.
        ekegusii_vocabulary: Set of known lowercased Ekegusii word forms.

    Returns:
        float: Fraction of words in `text` found in `ekegusii_vocabulary`,
            in [0, 1]. Returns 0.0 for empty text.
    """
    words = re.findall(r"\w+", text.lower())
    if not words:
        return 0.0
    overlap = sum(1 for w in words if w in ekegusii_vocabulary)
    return overlap / len(words)


def build_vocabulary(texts: "list[str]") -> Set[str]:
    """Build a lowercased word-form vocabulary set from a collection of texts.

    Args:
        texts: Sentences/terms to extract vocabulary from (e.g. the
            Ekegusii column of the master sentence corpus).

    Returns:
        Set[str]: Set of unique lowercased word forms across all texts.
    """
    vocabulary: Set[str] = set()
    for text in texts:
        vocabulary.update(re.findall(r"\w+", text.lower()))
    return vocabulary
