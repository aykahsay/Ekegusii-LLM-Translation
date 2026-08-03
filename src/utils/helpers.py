"""
General-Purpose Helper Utilities
----------------------------------
Small, dependency-light utilities reused across master_corpus, tokenizer,
model, evaluation, and experiment modules: device resolution, JSON I/O,
timestamped run identifiers, and directory bootstrapping.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


def resolve_device(prefer_cuda: bool = True) -> torch.device:
    """Resolve the compute device to use for model loading/inference.

    Args:
        prefer_cuda: If True, returns a CUDA device when available.

    Returns:
        torch.device: "cuda" if available and requested, else "cpu".
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Resolved device: cuda ({torch.cuda.get_device_name(0)}).")
    else:
        device = torch.device("cpu")
        logger.info("Resolved device: cpu.")
    return device


def generate_run_id(prefix: str = "run") -> str:
    """Generate a sortable, unique run identifier for experiment tracking.

    Args:
        prefix: Short label prepended to the timestamp (e.g. an experiment
            ID like "E1_English_Ekegusii").

    Returns:
        str: Identifier of the form "{prefix}_{YYYYMMDD}_{HHMMSS}".
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def ensure_dir(path: Path) -> Path:
    """Create a directory (including parents) if it does not already exist.

    Args:
        path: Directory path to create.

    Returns:
        Path: The same path, guaranteed to exist as a directory.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON file into a dictionary.

    Args:
        path: Path to a `.json` file.

    Returns:
        Dict[str, Any]: Parsed JSON content.

    Raises:
        FileNotFoundError: If `path` does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    return data


def write_json(data: Dict[str, Any], path: Path, indent: int = 2) -> None:
    """Write a dictionary to a JSON file, creating parent directories as needed.

    Args:
        data: Serializable dictionary to write.
        path: Destination `.json` file path.
        indent: JSON indentation level for human-readable output.
    """
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    logger.debug(f"Wrote JSON to {path}.")


def count_trainable_parameters(model: torch.nn.Module) -> Dict[str, Any]:
    """Count trainable vs. total parameters of a (PEFT-wrapped) model.

    Useful for confirming QLoRA is only training adapter weights (typically
    <1% of total parameters for an 8B base model).

    Args:
        model: A PyTorch (optionally PEFT-wrapped) module.

    Returns:
        Dict[str, Any]: Keys "trainable" (int), "total" (int), and
            "trainable_pct" (float, percentage rounded to 4 decimals).
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = round(100 * trainable / total, 4) if total > 0 else 0.0
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({pct}%).")
    return {"trainable": trainable, "total": total, "trainable_pct": pct}
