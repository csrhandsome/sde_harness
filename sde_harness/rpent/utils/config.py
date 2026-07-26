"""Path resolution and environment-variable configuration."""
from __future__ import annotations

import os
from pathlib import Path

_LOCAL_ENV_LOADED = False


# ============================================================================
# Repository / package roots
# ============================================================================

def get_repo_root() -> Path:
    """Return the ``sde_harness`` repository root directory.

    Resolution: ``SDE_HARNESS_ROOT`` / ``RPENT_REPO_ROOT`` env var, then the
    parent of the ``rpent/`` package directory.
    """
    for key in ("SDE_HARNESS_ROOT", "RPENT_REPO_ROOT"):
        env = os.environ.get(key)
        if env:
            return Path(env).expanduser().resolve()
    # config.py lives at <repo>/rpent/utils/config.py
    return Path(__file__).resolve().parents[2]


def load_local_env(*, override: bool = False) -> Path | None:
    """Load ``sde_harness/.env.local`` into ``os.environ`` (once).

    Existing environment variables win unless ``override=True``.
    Returns the path that was loaded, or ``None`` if missing.
    """
    global _LOCAL_ENV_LOADED
    if _LOCAL_ENV_LOADED and not override:
        return None
    path = get_repo_root() / ".env.local"
    if not path.is_file():
        _LOCAL_ENV_LOADED = True
        return None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value
    _LOCAL_ENV_LOADED = True
    return path


def get_vla_adapter_root() -> Path:
    """Return the parent VLA-Adapter repository root.

    Resolution: ``VLA_ADAPTER_ROOT`` env var, then the parent of
    ``sde_harness/``.
    """
    env = os.environ.get("VLA_ADAPTER_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return get_repo_root().parent


# ============================================================================
# Paths derived from the repo root  (callable so tests can override)
# ============================================================================

def get_resources_dir(env_name: str) -> Path:
    """Return the per-env resources directory (memory + reference corpora)."""
    return get_repo_root() / "resources" / env_name


def get_memory_dir(env_name: str) -> Path:
    """Return the persistent, cross-run memory directory for an env."""
    return get_resources_dir(env_name) / "memory"


def get_adapter_checkpoint_path() -> str:
    """VLA-Adapter checkpoint directory (replaces RLinf Pi0.5 weights)."""
    return os.environ.get("ADAPTER_CHECKPOINT_PATH", "") or os.environ.get(
        "VLA_ADAPTER_CHECKPOINT", ""
    )


def get_libero_type() -> str:
    return os.environ.get("LIBERO_TYPE", "pro")


def get_libero_root(libero_type: str | None = None) -> Path:
    """Return the third_party LIBERO / LIBERO-PRO / LIBERO-plus root."""
    kind = (libero_type or get_libero_type()).lower()
    adapter = get_vla_adapter_root()
    mapping = {
        "standard": adapter / "LIBERO",
        "pro": adapter / "third_party" / "LIBERO-PRO",
        "plus": adapter / "third_party" / "LIBERO-plus",
    }
    if kind not in mapping:
        raise ValueError(f"unknown LIBERO_TYPE={kind!r}; expected standard|pro|plus")
    return mapping[kind]


def get_libero_config_dir(libero_type: str | None = None) -> Path:
    """Return the per-variant ``.libero_*_config`` directory used by eval scripts."""
    kind = (libero_type or get_libero_type()).lower()
    adapter = get_vla_adapter_root()
    mapping = {
        "standard": adapter / "experiments" / "robot" / "libero" / ".libero_config",
        "pro": adapter / "experiments" / "robot" / "libero" / ".libero_pro_config",
        "plus": adapter / "experiments" / "robot" / "libero" / ".libero_plus_config",
    }
    if kind not in mapping:
        raise ValueError(f"unknown LIBERO_TYPE={kind!r}; expected standard|pro|plus")
    return mapping[kind]
