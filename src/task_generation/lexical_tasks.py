"""
Lexical-Augmentation Task Generator
---------------------------------------
Generates instruction-tuning tasks from `master_lexical_corpus.csv` (268
dictionary entries) using `configs/prompts/lexical.yaml`'s templates.
Distinct from `InstructionTaskGenerator` (sentence-level, 6-way
translation): this produces shorter, term-level "define" and
"rare-word-probe" tasks used for E6 (Lexical Augmentation).
"""

import logging
from typing import List, Optional

import pandas as pd

from src.master_corpus.manager import MasterCorpusManager
from src.utils.config_dict import ConfigDict, load_yaml
from src.utils.constants import CONFIGS_DIR, LANGUAGE_CODES, SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)


class LexicalTaskGenerator:
    """Generates term-level instruction tasks from the master lexical corpus."""

    def __init__(self, manager: Optional[MasterCorpusManager] = None) -> None:
        """Initialize the generator.

        Args:
            manager: Existing MasterCorpusManager to reuse. If None, a
                default-configured instance is created.
        """
        self.manager = manager or MasterCorpusManager()
        self.templates: ConfigDict = load_yaml(CONFIGS_DIR / "prompts" / "lexical.yaml")

    def generate_tasks_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Expand lexical corpus rows into "define_term" instruction tasks.

        For each row, generates one task per language pair where both the
        source and target term are present (up to 6 directions, mirroring
        `InstructionTaskGenerator`'s sentence-level coverage).

        Args:
            df: DataFrame containing lexicon_id, English, Kiswahili,
                Ekegusii columns.

        Returns:
            pd.DataFrame: Columns "lexicon_id", "task_type", "prompt", "response".
        """
        tasks: List[dict] = []
        template = self.templates["define_term"]

        for _, row in df.iterrows():
            lex_id = row.get("lexicon_id")
            terms = {
                lang: (str(row[lang]).strip() if pd.notna(row.get(lang)) else None)
                for lang in SUPPORTED_LANGUAGES
            }

            for src_lang in SUPPORTED_LANGUAGES:
                for tgt_lang in SUPPORTED_LANGUAGES:
                    if src_lang == tgt_lang:
                        continue
                    src_term, tgt_term = terms[src_lang], terms[tgt_lang]
                    if not src_term or not tgt_term:
                        continue

                    task_type = f"{LANGUAGE_CODES[src_lang].upper()}_to_{LANGUAGE_CODES[tgt_lang].upper()}_lexical"
                    prompt = template.format(src_lang=src_lang, tgt_lang=tgt_lang, term=src_term)
                    tasks.append(
                        {"lexicon_id": lex_id, "task_type": task_type, "prompt": prompt, "response": tgt_term}
                    )

        task_df = pd.DataFrame(tasks)
        logger.info(f"Generated {len(task_df):,} lexical instruction tasks from {len(df):,} lexicon entries.")
        return task_df

    def generate_all_tasks(self) -> pd.DataFrame:
        """Generate lexical instruction tasks from the full master lexical corpus.

        Returns:
            pd.DataFrame: Output of `generate_tasks_from_dataframe` applied
                to the entire lexical corpus.
        """
        lexical_df = self.manager.load_lexical_corpus()
        return self.generate_tasks_from_dataframe(lexical_df)
