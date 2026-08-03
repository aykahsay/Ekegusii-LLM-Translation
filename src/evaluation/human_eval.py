"""
Human Evaluation Template & Aggregation
--------------------------------------------
Generates a native-speaker evaluation template (100+ sentences, per the
project's evaluation requirement) sampled across domains/experiments, and
aggregates completed scores (fluency, adequacy, cultural accuracy -- each
1-5) into summary statistics per experiment/model. Distinct from the
automatic metrics (SacreBLEU/chrF/COMET): this handles the human-scored
side of evaluation, which automatic metrics cannot substitute for.
"""

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.master_corpus.sampling import CorpusSampler

logger = logging.getLogger(__name__)

SCORE_COLUMNS = ("fluency", "adequacy", "cultural_accuracy")


class HumanEvalTemplateBuilder:
    """Builds a native-speaker evaluation template CSV from model translations."""

    def build_template(
        self,
        source_df: pd.DataFrame,
        translations_by_model: Dict[str, List[str]],
        source_col: str = "English",
        n_sentences: int = 100,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Build a human-evaluation template with blank score columns per model.

        Args:
            source_df: DataFrame containing at least `source_col` (and
                ideally "Domain"/"concept_id" for stratified sampling and
                traceability).
            translations_by_model: Mapping of model/experiment label (e.g.
                "aya_E4_Trilingual") to a list of translations, ALIGNED
                ROW-FOR-ROW with `source_df` (same order, same length).
            source_col: Column in `source_df` holding the source sentence.
            n_sentences: Number of sentences to sample for evaluation (the
                project requires 100+).
            seed: Random seed for reproducible sampling.

        Returns:
            pd.DataFrame: One row per (sampled sentence, model) pair, with
                columns "eval_id", "concept_id" (if available), "source",
                "model", "translation", "fluency", "adequacy",
                "cultural_accuracy", "evaluator_notes" (the score/notes
                columns left blank for the evaluator to fill in).

        Raises:
            ValueError: If any model's translation list length doesn't
                match `len(source_df)`, or `n_sentences` < 1.
        """
        if n_sentences < 1:
            raise ValueError(f"n_sentences must be >= 1, got {n_sentences}.")
        for model_name, translations in translations_by_model.items():
            if len(translations) != len(source_df):
                raise ValueError(
                    f"Model '{model_name}' has {len(translations)} translations, "
                    f"expected {len(source_df)} (must align with source_df)."
                )

        sampler = CorpusSampler(seed=seed)
        sampled = sampler.sample_n(source_df.reset_index(drop=True), n_sentences)

        rows: List[dict] = []
        eval_id = 1
        for idx in sampled.index:
            source_text = sampled.loc[idx, source_col]
            concept_id = sampled.loc[idx, "concept_id"] if "concept_id" in sampled.columns else None

            for model_name, translations in translations_by_model.items():
                rows.append(
                    {
                        "eval_id": f"HUMAN-EVAL-{eval_id:04d}",
                        "concept_id": concept_id,
                        "source": source_text,
                        "model": model_name,
                        "translation": translations[idx],
                        "fluency": None,
                        "adequacy": None,
                        "cultural_accuracy": None,
                        "evaluator_notes": "",
                    }
                )
            eval_id += 1

        template_df = pd.DataFrame(rows)
        logger.info(
            f"Built human-eval template: {n_sentences} sentence(s) x "
            f"{len(translations_by_model)} model(s) = {len(template_df):,} rows."
        )
        return template_df


class HumanEvalAggregator:
    """Aggregates completed human evaluation scores into summary statistics."""

    def aggregate(self, completed_df: pd.DataFrame, group_by: str = "model") -> pd.DataFrame:
        """Summarize fluency/adequacy/cultural_accuracy scores per group.

        Args:
            completed_df: A filled-in human-eval template (see
                `HumanEvalTemplateBuilder.build_template`) with numeric
                scores in `SCORE_COLUMNS`.
            group_by: Column to group by (typically "model", but could be
                "concept_id" or a domain column if present).

        Returns:
            pd.DataFrame: Indexed by `group_by`, with mean/std columns for
                each of fluency/adequacy/cultural_accuracy, plus
                "n_scored" (count of non-null rows contributing).

        Raises:
            KeyError: If `group_by` or any score column is missing.
        """
        missing = [c for c in (group_by, *SCORE_COLUMNS) if c not in completed_df.columns]
        if missing:
            raise KeyError(f"completed_df is missing required column(s): {missing}")

        scored = completed_df.dropna(subset=list(SCORE_COLUMNS), how="all")
        grouped = scored.groupby(group_by)[list(SCORE_COLUMNS)]

        summary = grouped.agg(["mean", "std", "count"])
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary = summary.round(3)

        logger.info(f"Aggregated human eval scores across {summary.shape[0]} group(s) by '{group_by}'.")
        return summary

    def completion_rate(self, template_df: pd.DataFrame) -> float:
        """Compute the fraction of rows with at least one score filled in.

        Args:
            template_df: A (possibly partially completed) human-eval template.

        Returns:
            float: Fraction in [0, 1] of rows where any of `SCORE_COLUMNS`
                is non-null -- useful for tracking evaluator progress
                toward the 100+-sentence requirement.
        """
        if len(template_df) == 0:
            return 0.0
        has_any_score = template_df[SCORE_COLUMNS[0]].notna()
        for col in SCORE_COLUMNS[1:]:
            has_any_score = has_any_score | template_df[col].notna()
        return float(has_any_score.mean())

    def save_template(self, template_df: pd.DataFrame, output_path: Path) -> None:
        """Save a human-eval template/results DataFrame to CSV.

        Args:
            template_df: Template or completed results DataFrame.
            output_path: Destination `.csv` file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        template_df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"Saved human-eval template ({len(template_df):,} rows) to {output_path}.")
