"""
Configuration Composition (Hydra/OmegaConf)
---------------------------------------------
Loads and merges the layered YAML config groups under `configs/` into a
single resolved configuration per model and experiment:

    configs/models/common.yaml           (shared hardware/seed/caching)
      -> configs/models/{model}_8b.yaml  (per-model spec, generation, tokenizer)
      -> configs/training/qlora.yaml     (generic PEFT/quantization defaults)
      -> configs/training/{model}_8b_qlora.yaml  (per-model QLoRA overrides)
      -> configs/generation/{decoding}.yaml       (optional decoding override)
      -> configs/datasets/{dataset}.yaml          (optional dataset override)

Later layers override earlier ones key-for-key via `OmegaConf.merge`, so a
per-model file only needs to specify what differs from the shared default.
This keeps configuration additive rather than duplicated across files.
"""

import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.utils.constants import CONFIGS_DIR, SUPPORTED_MODELS

logger = logging.getLogger(__name__)

_MODEL_CONFIG_FILES = {"aya": "aya_8b.yaml", "llama": "llama31_8b.yaml"}
_QLORA_CONFIG_FILES = {"aya": "aya_8b_qlora.yaml", "llama": "llama31_8b_qlora.yaml"}


class ConfigError(Exception):
    """Raised when a requested configuration file is missing or fails to parse."""


def _load_yaml(path: Path) -> DictConfig:
    """Load a single YAML file as an OmegaConf DictConfig.

    Args:
        path: Absolute path to the YAML file.

    Returns:
        DictConfig: Parsed configuration.

    Raises:
        ConfigError: If the file does not exist or is not a valid mapping.
    """
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        cfg = OmegaConf.load(path)
    except Exception as exc:  # noqa: BLE001 - surface any YAML parse error uniformly
        raise ConfigError(f"Failed to parse config file {path}: {exc}") from exc
    if not isinstance(cfg, DictConfig):
        raise ConfigError(f"Config file {path} did not resolve to a mapping.")
    return cfg


def _validate_model_name(model_name: str) -> None:
    if model_name not in SUPPORTED_MODELS:
        raise ConfigError(f"Unknown model '{model_name}'. Supported models: {SUPPORTED_MODELS}.")


def load_model_config(model_name: str) -> DictConfig:
    """Load and merge the model configuration for the given model.

    Merges `configs/models/common.yaml` (shared hardware/seed/caching) with
    `configs/models/{model}_8b.yaml` (per-model architecture/generation spec).

    Args:
        model_name: One of `src.utils.constants.SUPPORTED_MODELS` ("aya", "llama").

    Returns:
        DictConfig: Merged model configuration.

    Raises:
        ConfigError: If `model_name` is unsupported or a config file is missing.
    """
    _validate_model_name(model_name)
    common = _load_yaml(CONFIGS_DIR / "models" / "common.yaml")
    specific = _load_yaml(CONFIGS_DIR / "models" / _MODEL_CONFIG_FILES[model_name])
    merged = OmegaConf.merge(common, specific)
    logger.info(f"Loaded model config for '{model_name}'.")
    return merged  # type: ignore[return-value]


def load_qlora_config(model_name: str) -> DictConfig:
    """Load and merge QLoRA training configuration for the given model.

    Merges `configs/training/qlora.yaml` (generic PEFT/quantization defaults)
    with `configs/training/{model}_8b_qlora.yaml` (per-model overrides such as
    learning rate, epochs, and target modules).

    Args:
        model_name: One of `src.utils.constants.SUPPORTED_MODELS` ("aya", "llama").

    Returns:
        DictConfig: Merged QLoRA training configuration.

    Raises:
        ConfigError: If `model_name` is unsupported or a config file is missing.
    """
    _validate_model_name(model_name)
    generic = _load_yaml(CONFIGS_DIR / "training" / "qlora.yaml")
    specific = _load_yaml(CONFIGS_DIR / "training" / _QLORA_CONFIG_FILES[model_name])
    merged = OmegaConf.merge(generic, specific)
    logger.info(f"Loaded QLoRA config for '{model_name}'.")
    return merged  # type: ignore[return-value]


def load_dataset_config(dataset_name: str) -> DictConfig:
    """Load a dataset configuration by name.

    Args:
        dataset_name: Filename stem under `configs/datasets/` (e.g. "master",
            "bilingual", "trilingual", "monolingual", "lexical").

    Returns:
        DictConfig: Parsed dataset configuration.

    Raises:
        ConfigError: If the config file does not exist.
    """
    return _load_yaml(CONFIGS_DIR / "datasets" / f"{dataset_name}.yaml")


def load_generation_config(profile_name: str = "default") -> DictConfig:
    """Load a decoding/generation configuration profile.

    Args:
        profile_name: Filename stem under `configs/generation/` (e.g.
            "default", "beam_search", "greedy").

    Returns:
        DictConfig: Parsed generation configuration.

    Raises:
        ConfigError: If the config file does not exist.
    """
    return _load_yaml(CONFIGS_DIR / "generation" / f"{profile_name}.yaml")


def load_prompt_templates() -> DictConfig:
    """Load the translation prompt templates.

    Returns:
        DictConfig: Prompt templates keyed by translation direction.

    Raises:
        ConfigError: If `configs/prompts/templates.yaml` is missing.
    """
    return _load_yaml(CONFIGS_DIR / "prompts" / "templates.yaml")


def compose_experiment_config(
    model_name: str,
    dataset_name: str = "master",
    generation_profile: str = "default",
) -> DictConfig:
    """Compose the full configuration needed to run one experiment.

    Merges model, QLoRA training, dataset, generation, and prompt configs
    into a single namespaced configuration object.

    Args:
        model_name: One of `src.utils.constants.SUPPORTED_MODELS` ("aya", "llama").
        dataset_name: Dataset config to attach (see `load_dataset_config`).
        generation_profile: Generation config profile to attach (see
            `load_generation_config`).

    Returns:
        DictConfig: Namespaced config with `model`, `qlora`, `dataset`,
            `generation`, and `prompts` top-level keys.

    Raises:
        ConfigError: If any underlying config file is missing or invalid.
    """
    composed = OmegaConf.create(
        {
            "model": load_model_config(model_name),
            "qlora": load_qlora_config(model_name),
            "dataset": load_dataset_config(dataset_name),
            "generation": load_generation_config(generation_profile),
            "prompts": load_prompt_templates(),
        }
    )
    logger.info(
        f"Composed experiment config: model={model_name}, dataset={dataset_name}, "
        f"generation_profile={generation_profile}."
    )
    return composed


def save_resolved_config(cfg: DictConfig, output_path: Path) -> None:
    """Write a fully-resolved configuration to disk for run provenance.

    Args:
        cfg: Configuration to persist (typically the output of
            `compose_experiment_config`).
        output_path: Destination YAML file path. Parent directories are
            created if missing.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=output_path, resolve=True)
    logger.info(f"Saved resolved config to {output_path}.")
