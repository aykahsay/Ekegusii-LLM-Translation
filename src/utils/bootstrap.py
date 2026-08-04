"""
Environment Bootstrap Helpers
---------------------------------
Pure-stdlib helper for defensively ensuring an optional dependency is
installed before importing it. Exists because some Jupyter hosts (e.g.
Kineses Cloud conda envs) ship an incomplete package set relative to
requirements.txt: `pip install omegaconf` there failed with "setuptools is
not available in the build environment" -- omegaconf pulls in
antlr4-python3-runtime, which has no wheel for that host and falls back to
a source (setup.py) build, and pip's isolated build sandbox couldn't fetch
setuptools into itself to run it. `--no-build-isolation` (reuse the main
env's own setuptools, which conda almost always already has) plus
`--prefer-binary` (avoid the source build entirely when a wheel exists)
fixes it.

This module has NO third-party or `src`-internal imports, so it can always
be imported first, from anywhere, without risk of the same bootstrapping
problem it's meant to solve.
"""

import importlib
import subprocess
import sys


def ensure_package(module_name: str, pip_spec: str) -> None:
    """Import `module_name`, installing `pip_spec` first if that import fails.

    Args:
        module_name: The name used in `import module_name` (e.g. "omegaconf").
        pip_spec: The pip install argument (e.g. "omegaconf==2.3.0").

    Raises:
        ModuleNotFoundError: If the module still cannot be imported after
            the install attempt -- surfaces pip's own failure to the caller
            rather than silently swallowing it.
    """
    try:
        importlib.import_module(module_name)
        return
    except ImportError:
        pass

    import site
    user_site = site.getusersitepackages()
    if user_site not in sys.path:
        sys.path.insert(0, user_site)

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--user", "setuptools", "wheel", "antlr4-python3-runtime==4.9.3"],
        check=False
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--user", "--prefer-binary", "--no-build-isolation", pip_spec],
        check=False
    )
    importlib.import_module(module_name)
