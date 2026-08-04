"""
CLI: schedule-preview
--------------------------
Shows the resource-weighted sampling plan `ResourceScheduler` would
produce for a given batch size, WITHOUT actually training -- useful for
sanity-checking `configs/training/multilingual.yaml`'s direction weights
before committing to a multi-hour training run.
"""

import logging
from typing import Dict

try:
    from omegaconf import OmegaConf
except ImportError:
    import subprocess as _subprocess
    import sys as _sys

    _subprocess.run(
        [_sys.executable, "-m", "pip", "install", "--quiet", "omegaconf==2.3.0", "hydra-core==1.3.2"],
        check=False,
    )
    from omegaconf import OmegaConf

from src.master_corpus.manager import MasterCorpusManager
from src.master_corpus.scheduler import ResourceScheduler
from src.task_generation.translation_pairs import direction_counts
from src.utils.constants import CONFIGS_DIR

logger = logging.getLogger(__name__)


def run_schedule_preview(batch_size: int = 32) -> Dict[str, int]:
    """Preview the per-direction sample quota for one training batch.

    Args:
        batch_size: Batch size to simulate a quota for.

    Returns:
        Dict[str, int]: Per-direction quota (e.g. {"eng_to_eke": 12, ...})
            summing to `batch_size`, computed from the current training
            split's available row counts and
            `configs/training/multilingual.yaml`'s weights/temperature.
    """
    manager = MasterCorpusManager()
    train_df = manager.load_train_split()

    counts = direction_counts(train_df)
    from src.utils.constants import LANGUAGE_CODES

    counts_by_key = {
        f"{LANGUAGE_CODES[src]}_to_{LANGUAGE_CODES[tgt]}": count
        for direction_label, count in counts
        for src, tgt in [direction_label.split("->")]
    }

    multilingual_cfg = OmegaConf.load(CONFIGS_DIR / "training" / "multilingual.yaml")
    scheduler = ResourceScheduler(multilingual_cfg)
    quota = scheduler.build_mixed_batch_plan(counts_by_key, batch_size)

    logger.info(f"Schedule preview (batch_size={batch_size}): {quota}")
    return quota
