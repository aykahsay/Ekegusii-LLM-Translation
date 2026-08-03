"""
Centralized Logging Configuration
----------------------------------
Configures Python's standard `logging` module with a Rich-formatted console
handler (and an optional rotating file handler) so every module across the
project -- master_corpus, tokenizer, models, evaluation, experiments, cli --
produces consistent, readable, timestamped log output instead of ad-hoc
`print()` calls.
"""

import logging
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler

from src.utils.constants import OUTPUTS_DIR

_CONFIGURED = False


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    quiet_libraries: bool = True,
) -> None:
    """Configure the root logger once for the whole process.

    Safe to call multiple times -- subsequent calls are no-ops so that
    importing this module from several entry points (CLI, notebooks,
    training scripts) doesn't register duplicate handlers.

    Args:
        level: Logging level for the project's own loggers (e.g. logging.INFO).
        log_file: Optional path to also write logs to a file. If None and
            `quiet_libraries` allows it, defaults to
            `outputs/logs/run.log` relative to the project root.
        quiet_libraries: If True, raises third-party library loggers
            (transformers, datasets, urllib3, etc.) to WARNING to reduce
            noise from their default INFO-level chatter.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    handlers: list[logging.Handler] = [
        RichHandler(rich_tracebacks=True, show_path=False, markup=True)
    ]

    resolved_log_file = log_file
    if resolved_log_file is None:
        log_dir = OUTPUTS_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        resolved_log_file = log_dir / "run.log"

    file_handler = logging.FileHandler(resolved_log_file, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True,
    )

    if quiet_libraries:
        for noisy_logger in ("transformers", "datasets", "urllib3", "filelock", "httpx"):
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    _CONFIGURED = True
    logging.getLogger(__name__).info(f"Logging configured (level={logging.getLevelName(level)}).")


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger, initializing the root logger on first use.

    Args:
        name: Usually `__name__` of the calling module.

    Returns:
        logging.Logger: Logger instance ready for use.
    """
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
