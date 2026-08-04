"""
Checkpoint Management for PEFT/QLoRA Training
-----------------------------------------------
Saves and restores PEFT adapter checkpoints together with JSON training-state
metadata (global step, epoch, best metric so far, optimizer/scheduler state
paths), enabling resume-from-checkpoint and early-stopping decisions during
long-running QLoRA fine-tuning of Qwen2.5-7B / Llama-3.1-8B on the A100.

The base model weights are never re-saved per checkpoint (only the small
LoRA adapter + tokenizer + metadata are), keeping checkpoints on the order
of tens of MB rather than duplicating the 8B-parameter base model each time.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.helpers import ensure_dir, read_json, write_json

logger = logging.getLogger(__name__)

_STATE_FILENAME = "training_state.json"


class CheckpointManager:
    """Manages PEFT adapter checkpoints for a single training run.

    Attributes:
        checkpoint_dir: Root directory under which numbered checkpoint
            subdirectories (e.g. `checkpoint-500/`) are created.
    """

    def __init__(self, checkpoint_dir: Path) -> None:
        """Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Root directory for this run's checkpoints, e.g.
                `checkpoints/qwen/E1_English_Ekegusii/`.
        """
        self.checkpoint_dir = ensure_dir(Path(checkpoint_dir))

    def save(
        self,
        model: Any,
        tokenizer: Any,
        step: int,
        epoch: float,
        metrics: Dict[str, float],
        is_best: bool = False,
    ) -> Path:
        """Save a PEFT adapter checkpoint with training-state metadata.

        Args:
            model: A PEFT-wrapped model exposing `save_pretrained`.
            tokenizer: The tokenizer paired with `model`, exposing
                `save_pretrained`.
            step: Global training step at which this checkpoint was taken.
            epoch: Fractional epoch at which this checkpoint was taken.
            metrics: Validation metrics recorded at this step (e.g.
                {"sacrebleu": 24.1, "chrf": 51.3, "comet": 0.71}).
            is_best: If True, also updates the `best/` checkpoint symlink
                directory (implemented as a plain copy for filesystem
                portability across platforms without symlink privileges).

        Returns:
            Path: The directory the checkpoint was written to.

        Raises:
            AttributeError: If `model` or `tokenizer` do not implement
                `save_pretrained`.
        """
        checkpoint_path = ensure_dir(self.checkpoint_dir / f"checkpoint-{step}")

        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)

        state = {
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
            "is_best": is_best,
        }
        write_json(state, checkpoint_path / _STATE_FILENAME)
        self._update_latest_pointer(checkpoint_path)

        if is_best:
            self._update_best_pointer(checkpoint_path)

        logger.info(f"Saved checkpoint at step {step} -> {checkpoint_path}")
        return checkpoint_path

    def load_latest_step(self) -> Optional[int]:
        """Return the global step of the most recent checkpoint, if any.

        Used to decide whether to resume training and from which step.

        Returns:
            Optional[int]: The latest checkpoint's step, or None if no
                checkpoint exists yet in `checkpoint_dir`.
        """
        pointer_path = self.checkpoint_dir / "latest.json"
        if not pointer_path.exists():
            return None
        pointer = read_json(pointer_path)
        return int(pointer["step"])

    def get_checkpoint_path(self, step: int) -> Path:
        """Return the directory for a specific checkpoint step.

        Args:
            step: Global training step of the desired checkpoint.

        Returns:
            Path: Directory expected to contain that checkpoint's files.

        Raises:
            FileNotFoundError: If no checkpoint exists at that step.
        """
        path = self.checkpoint_dir / f"checkpoint-{step}"
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint found at step {step} in {self.checkpoint_dir}.")
        return path

    def load_state(self, step: int) -> Dict[str, Any]:
        """Load the training-state metadata for a specific checkpoint.

        Args:
            step: Global training step of the checkpoint to inspect.

        Returns:
            Dict[str, Any]: The saved state dict (step, epoch, metrics, is_best).
        """
        checkpoint_path = self.get_checkpoint_path(step)
        return read_json(checkpoint_path / _STATE_FILENAME)

    def _update_latest_pointer(self, checkpoint_path: Path) -> None:
        state = read_json(checkpoint_path / _STATE_FILENAME)
        write_json({"step": state["step"], "path": str(checkpoint_path)}, self.checkpoint_dir / "latest.json")

    def _update_best_pointer(self, checkpoint_path: Path) -> None:
        state = read_json(checkpoint_path / _STATE_FILENAME)
        write_json({"step": state["step"], "path": str(checkpoint_path)}, self.checkpoint_dir / "best.json")
        logger.info(f"New best checkpoint recorded at step {state['step']}.")


class EarlyStopping:
    """Tracks a monitored metric across evaluations and signals when to stop.

    Attributes:
        patience: Number of consecutive non-improving evaluations tolerated.
        mode: "max" if higher metric values are better (e.g. COMET, BLEU),
            "min" if lower is better (e.g. eval loss).
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.0, mode: str = "max") -> None:
        """Initialize the early-stopping tracker.

        Args:
            patience: Number of evaluations with no improvement to tolerate
                before signaling that training should stop.
            min_delta: Minimum change in the monitored metric to qualify as
                an improvement.
            mode: "max" or "min", indicating the direction of improvement.

        Raises:
            ValueError: If `mode` is not "max" or "min", or `patience` < 1.
        """
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got '{mode}'.")
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}.")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: Optional[float] = None
        self.num_bad_evaluations = 0

    def step(self, score: float) -> bool:
        """Register a new evaluation score and check whether training should stop.

        Args:
            score: The monitored metric's value at this evaluation.

        Returns:
            bool: True if training should stop now, False otherwise.
        """
        if self.best_score is None or self._is_improvement(score):
            self.best_score = score
            self.num_bad_evaluations = 0
            return False

        self.num_bad_evaluations += 1
        logger.info(
            f"No improvement for {self.num_bad_evaluations}/{self.patience} evaluations "
            f"(best={self.best_score}, current={score})."
        )
        return self.num_bad_evaluations >= self.patience

    def _is_improvement(self, score: float) -> bool:
        assert self.best_score is not None
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta
