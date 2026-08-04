"""
Minimal YAML Config Wrapper (PyYAML-based, no OmegaConf/Hydra dependency)
------------------------------------------------------------------------------
`omegaconf` cannot be installed at all on some Jupyter hosts (Kineses Cloud):
its `antlr4-python3-runtime` dependency has no wheel there, so pip falls back
to a source (setup.py) build, and that host's pip cannot obtain a working
`setuptools` for the build even with `--no-build-isolation`, `--user`, or an
explicit `antlr4-python3-runtime` pin -- all tried and all failed identically.

Rather than keep chasing environment-specific pip workarounds, this module
replaces the small subset of OmegaConf's API this project actually used
(load a YAML file, deep-merge two configs, attribute + dict access, dump
back to YAML) with a minimal wrapper over PyYAML + stdlib `dict`. PyYAML
ships wheels for every platform and is already a transitive dependency of
`transformers`/`accelerate`, so it is present wherever this project's other
required packages already verified as installed.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

logger = logging.getLogger(__name__)


class ConfigDict(dict):
    """A dict that also supports attribute-style get/set, recursively.

    Behaves like OmegaConf's `DictConfig` for the operations this project
    needs: `cfg.model.hf_path` attribute chains, `cfg.get(key, default)`,
    `cfg["key"]` subscripting, `for key in cfg` iteration, and in-place
    attribute mutation (`cfg.section.flag = True` mutates the same nested
    object the parent holds, not a copy).
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = value

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, _wrap(value))

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def to_container(self) -> Dict[str, Any]:
        """Recursively convert back to plain `dict`/`list`/scalar values."""
        return _unwrap(self)


def _wrap(value: Any) -> Any:
    if isinstance(value, ConfigDict):
        return value
    if isinstance(value, dict):
        return ConfigDict(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _unwrap(value: Any) -> Any:
    if isinstance(value, ConfigDict):
        return {k: _unwrap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    return value


def load_yaml(path: Union[str, Path]) -> ConfigDict:
    """Load a YAML file into a `ConfigDict`.

    Args:
        path: Path to the YAML file.

    Returns:
        ConfigDict: Parsed configuration. An empty file yields an empty ConfigDict.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If the YAML content is not a mapping at the top level.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} did not resolve to a mapping (got {type(data).__name__}).")
    return ConfigDict(data)


def merge(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    """Deep-merge two ConfigDicts; keys in `override` win, recursively for nested mappings.

    Args:
        base: Base configuration.
        override: Configuration whose keys take precedence over `base`.

    Returns:
        ConfigDict: A new merged configuration (neither input is mutated).
    """
    result = ConfigDict(base.to_container())
    for key, value in override.items():
        if key in result and isinstance(result[key], ConfigDict) and isinstance(value, ConfigDict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def save_yaml(cfg: ConfigDict, path: Union[str, Path]) -> None:
    """Write a ConfigDict to disk as YAML, creating parent directories as needed.

    Args:
        cfg: Configuration to persist.
        path: Destination YAML file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_container(), f, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved config to {path}.")
