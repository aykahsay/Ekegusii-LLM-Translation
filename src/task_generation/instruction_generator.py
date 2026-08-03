"""
6-Way Multilingual Instruction Task Generator
----------------------------------------------
Transforms multilingual parallel concepts into 6 bidirectional translation tasks:
1. English -> Ekegusii
2. Ekegusii -> English
3. Kiswahili -> Ekegusii
4. Ekegusii -> Kiswahili
5. English -> Kiswahili
6. Kiswahili -> English
"""

import logging
from typing import Dict, List, Optional
import pandas as pd

from src.master_corpus.manager import MasterCorpusManager
from src.utils.config import load_prompt_templates

logger = logging.getLogger(__name__)

# Maps each task_type to (config key in configs/prompts/templates.yaml's
# `translation` section, target language name). Templates are loaded from
# config rather than hardcoded here so a single source of truth
# (configs/prompts/templates.yaml) governs prompt wording for both this
# generator and any inference-time prompt construction.
_TASK_CONFIG_KEYS = {
    "ENG_to_EKE": ("eng_to_eke", "Ekegusii"),
    "EKE_to_ENG": ("eke_to_eng", "English"),
    "SWA_to_EKE": ("swa_to_eke", "Ekegusii"),
    "EKE_to_SWA": ("eke_to_swa", "Kiswahili"),
    "ENG_to_SWA": ("eng_to_swa", "Kiswahili"),
    "SWA_to_ENG": ("swa_to_eng", "English"),
}


class InstructionTaskGenerator:
    """Generates 6-way bidirectional instruction-tuning tasks from multilingual concepts."""

    def __init__(self, manager: Optional[MasterCorpusManager] = None) -> None:
        """Initialize with an optional MasterCorpusManager instance.

        Loads prompt templates from `configs/prompts/templates.yaml` at
        construction time.

        Raises:
            src.utils.config.ConfigError: If the templates config is
                missing or malformed.
        """
        self.manager = manager or MasterCorpusManager()
        templates = load_prompt_templates()["translation"]
        self.TASK_PROMPTS = {
            task_type: (templates[config_key], target_lang)
            for task_type, (config_key, target_lang) in _TASK_CONFIG_KEYS.items()
        }

    def generate_tasks_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Expand a DataFrame of multilingual concepts into 6 bidirectional instruction tasks.

        Args:
            df: DataFrame containing concept_id, English, Kiswahili, Ekegusii columns.

        Returns:
            pd.DataFrame: Expanded task dataset with prompt, instruction, and response.
        """
        tasks: List[Dict[str, str]] = []

        for idx, row in df.iterrows():
            cid = row.get("concept_id", idx)
            eng = str(row["English"]).strip() if pd.notna(row.get("English")) else None
            swa = str(row["Kiswahili"]).strip() if pd.notna(row.get("Kiswahili")) else None
            eke = str(row["Ekegusii"]).strip() if pd.notna(row.get("Ekegusii")) else None

            # 1. ENG ↔ EKE Tasks
            if eng and eke:
                p1, _ = self.TASK_PROMPTS["ENG_to_EKE"]
                tasks.append({"concept_id": cid, "task_type": "ENG_to_EKE", "prompt": p1.format(src=eng), "response": eke})
                p2, _ = self.TASK_PROMPTS["EKE_to_ENG"]
                tasks.append({"concept_id": cid, "task_type": "EKE_to_ENG", "prompt": p2.format(src=eke), "response": eng})

            # 2. SWA ↔ EKE Tasks
            if swa and eke:
                p3, _ = self.TASK_PROMPTS["SWA_to_EKE"]
                tasks.append({"concept_id": cid, "task_type": "SWA_to_EKE", "prompt": p3.format(src=swa), "response": eke})
                p4, _ = self.TASK_PROMPTS["EKE_to_SWA"]
                tasks.append({"concept_id": cid, "task_type": "EKE_to_SWA", "prompt": p4.format(src=eke), "response": swa})

            # 3. ENG ↔ SWA Tasks
            if eng and swa:
                p5, _ = self.TASK_PROMPTS["ENG_to_SWA"]
                tasks.append({"concept_id": cid, "task_type": "ENG_to_SWA", "prompt": p5.format(src=eng), "response": swa})
                p6, _ = self.TASK_PROMPTS["SWA_to_ENG"]
                tasks.append({"concept_id": cid, "task_type": "SWA_to_ENG", "prompt": p6.format(src=swa), "response": eng})

        task_df = pd.DataFrame(tasks)
        logger.info(f"Expanded {len(df):,} concepts into {len(task_df):,} instruction tasks.")
        return task_df

    def generate_all_splits(self) -> Dict[str, pd.DataFrame]:
        """Generate instruction task DataFrames for Train, Val, and Test splits."""
        train_tasks = self.generate_tasks_from_dataframe(self.manager.load_train_split())
        val_tasks = self.generate_tasks_from_dataframe(self.manager.load_val_split())
        test_tasks = self.generate_tasks_from_dataframe(self.manager.load_test_split())

        return {
            "train": train_tasks,
            "val": val_tasks,
            "test": test_tasks,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gen = InstructionTaskGenerator()
    splits = gen.generate_all_splits()
    print(f"Generated Train Tasks: {len(splits['train']):,}")
