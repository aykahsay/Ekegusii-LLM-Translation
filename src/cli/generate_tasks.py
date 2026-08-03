"""
CLI: generate-tasks
------------------------
Implements the `ekegusii-nmt generate-tasks` command's logic. Kept as a
plain function here (not a Typer app) so it's independently testable and
importable from `main.py` without needing a live Typer/Rich console.
"""

import logging
from typing import Dict

import pandas as pd

from src.task_generation.instruction_generator import InstructionTaskGenerator

logger = logging.getLogger(__name__)


def run_generate_tasks() -> Dict[str, pd.DataFrame]:
    """Generate 6-way multilingual instruction-tuning tasks for all splits.

    Returns:
        Dict[str, pd.DataFrame]: Keys "train", "val", "test", each the
            generated instruction-task DataFrame for that split (see
            `InstructionTaskGenerator.generate_all_splits`).
    """
    generator = InstructionTaskGenerator()
    splits = generator.generate_all_splits()
    for split_name, split_df in splits.items():
        logger.info(f"Generated {len(split_df):,} instruction tasks for '{split_name}' split.")
    return splits
