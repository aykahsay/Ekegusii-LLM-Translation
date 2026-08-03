"""
Reproducibility Seeding
-----------------------
Sets a single deterministic seed across Python's `random`, NumPy, and PyTorch
(CPU + all CUDA devices), and enables PyTorch's deterministic algorithms.
Every training/evaluation entry point in this project must call
`set_seed()` before constructing datasets or models -- without it, QLoRA
runs on the same config would not be reproducible, and E0-E8 result
comparisons would be confounded by run-to-run noise rather than the
resource being tested.
"""

import logging
import os
import random

import numpy as np
import torch

from src.utils.constants import DEFAULT_SEED

logger = logging.getLogger(__name__)


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> None:
    """Seed all relevant RNGs for reproducible experiments.

    Args:
        seed: Integer seed applied to Python's `random`, NumPy, and PyTorch.
        deterministic: If True, additionally requests deterministic (but
            slower) CUDA kernels via `torch.use_deterministic_algorithms`.
            Set to False when raw throughput matters more than bit-exact
            reproducibility (e.g. large-scale QLoRA training runs where a
            handful of nondeterministic ops are an acceptable trade-off).

    Raises:
        ValueError: If `seed` is negative.
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, RuntimeError) as exc:
            logger.warning(f"Could not enable fully deterministic algorithms: {exc}")

    logger.info(f"Global seed set to {seed} (deterministic={deterministic}).")
