"""
Domain Terminology Consistency Checker
-------------------------------------------
Checks whether specific curated institutional/domain terms (e.g. proper
nouns, government body names, technical health/agriculture terms) are
translated CONSISTENTLY across a set of hypotheses -- distinct from
`LexicalEvaluator`, which checks general dictionary-term accuracy against
individual reference pairs. Terminology consistency matters even when no
single "correct" translation exists (e.g. an acronym might be legitimately
left untranslated OR localized), so this checks self-consistency and
presence rather than exact-match accuracy.
"""

import logging
from collections import Counter
from typing import Dict, List

logger = logging.getLogger(__name__)


class TerminologyConsistencyChecker:
    """Checks consistent handling of curated domain terms across translations."""

    def __init__(self, terminology_map: Dict[str, List[str]]) -> None:
        """Initialize with a curated terminology map.

        Args:
            terminology_map: Mapping of source term -> list of acceptable
                target-language renderings (e.g.
                {"Ministry of Health": ["Wizara ya Afya", "Ministry of Health"]}).
                Multiple acceptable renderings are allowed since some terms
                are legitimately transliterated OR left in English.
        """
        self.terminology_map = terminology_map

    def find_source_terms(self, source_text: str) -> List[str]:
        """Find which curated terms appear in a source sentence.

        Args:
            source_text: Source-language sentence to scan.

        Returns:
            List[str]: Curated terms (keys of `terminology_map`) found in
                `source_text` (case-insensitive substring match).
        """
        source_lower = source_text.lower()
        return [term for term in self.terminology_map if term.lower() in source_lower]

    def check_translation(self, source_text: str, translation: str) -> Dict[str, bool]:
        """Check whether each curated term found in `source_text` is rendered acceptably.

        Args:
            source_text: Source-language sentence.
            translation: Corresponding model translation.

        Returns:
            Dict[str, bool]: Maps each found term to whether at least one
                of its acceptable renderings appears in `translation`.
        """
        found_terms = self.find_source_terms(source_text)
        results = {}
        for term in found_terms:
            acceptable = self.terminology_map[term]
            results[term] = any(rendering.lower() in translation.lower() for rendering in acceptable)
        return results

    def evaluate_corpus(self, sources: List[str], translations: List[str]) -> Dict[str, object]:
        """Evaluate terminology consistency across an entire corpus.

        Args:
            sources: Source-language sentences.
            translations: Corresponding model translations, aligned with `sources`.

        Returns:
            Dict[str, object]: Keys "overall_accuracy" (float, fraction of
                term occurrences rendered acceptably), "per_term_accuracy"
                (Dict[str, float]), "term_occurrence_counts" (Dict[str, int]).

        Raises:
            ValueError: If `sources` and `translations` lengths don't match.
        """
        if len(sources) != len(translations):
            raise ValueError(f"len(sources)={len(sources)} must match len(translations)={len(translations)}.")

        term_hits: Counter = Counter()
        term_total: Counter = Counter()

        for source_text, translation in zip(sources, translations):
            checks = self.check_translation(source_text, translation)
            for term, is_correct in checks.items():
                term_total[term] += 1
                if is_correct:
                    term_hits[term] += 1

        per_term_accuracy = {
            term: round(100 * term_hits[term] / term_total[term], 2) for term in term_total
        }
        overall_total = sum(term_total.values())
        overall_accuracy = round(100 * sum(term_hits.values()) / overall_total, 2) if overall_total > 0 else 0.0

        logger.info(f"Terminology consistency: overall={overall_accuracy}% across {overall_total} occurrence(s).")
        return {
            "overall_accuracy": overall_accuracy,
            "per_term_accuracy": per_term_accuracy,
            "term_occurrence_counts": dict(term_total),
        }

    @staticmethod
    def build_from_lexical_corpus(lexical_df, source_lang: str, target_lang: str, min_term_length: int = 4) -> "TerminologyConsistencyChecker":
        """Build a terminology map from the master lexical corpus's longer entries.

        Args:
            lexical_df: Master lexical corpus DataFrame.
            source_lang: Source language column name.
            target_lang: Target language column name.
            min_term_length: Minimum character length for a lexicon entry
                to count as a "term" worth tracking (filters out short
                function words that aren't meaningful terminology).

        Returns:
            TerminologyConsistencyChecker: Initialized with terms drawn
                from `lexical_df`.
        """
        terminology_map: Dict[str, List[str]] = {}
        for _, row in lexical_df.iterrows():
            src_term = row.get(source_lang)
            tgt_term = row.get(target_lang)
            if not isinstance(src_term, str) or not isinstance(tgt_term, str):
                continue
            src_term, tgt_term = src_term.strip(), tgt_term.strip()
            if len(src_term) >= min_term_length and len(tgt_term) >= 1:
                terminology_map.setdefault(src_term, []).append(tgt_term)

        return TerminologyConsistencyChecker(terminology_map)
