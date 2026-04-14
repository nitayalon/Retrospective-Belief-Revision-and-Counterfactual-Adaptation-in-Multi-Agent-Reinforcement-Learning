"""JAX device configuration utility.

Call ``setup_device(config)`` at the very top of ``main()``, before any JAX
operations, so that ``JAX_PLATFORM_NAME`` is set before the runtime selects
its back-end.  JAX reads this environment variable on the *first* call to
``jax.devices()`` (not at import time), so setting it here is always safe even
though JAX modules are already imported at the top of ``train.py``.
"""

import os
import warnings
from typing import Any


def setup_device(config: Any) -> str:
    """Configure the JAX backend and print a one-line device summary.

    Reads ``config.compute.device`` (default ``'gpu'``) and sets
    ``JAX_PLATFORM_NAME`` when the environment variable is not already present.
    Falls back to CPU silently if the requested platform is unavailable.

    Args:
        config: ``SimpleConfig`` or any object with an optional
            ``compute.device`` attribute.

    Returns:
        The resolved platform string (``'gpu'`` or ``'cpu'``).
    """
    compute_cfg = getattr(config, "compute", None)
    desired = getattr(compute_cfg, "device", "gpu").lower()

    # Only override when the caller has not already set it explicitly.
    if "JAX_PLATFORM_NAME" not in os.environ:
        os.environ["JAX_PLATFORM_NAME"] = desired

    platform = os.environ["JAX_PLATFORM_NAME"]

    # Suppress spurious CUDA-not-found noise on CPU-only machines.
    if platform == "cpu":
        warnings.filterwarnings("ignore", message=".*No GPU.*")
        warnings.filterwarnings("ignore", message=".*CUDA.*")
        warnings.filterwarnings("ignore", message=".*cuda.*")

    try:
        import jax  # Imported after the env-var is set.
        devices = jax.devices(platform)
        print(f"[device] {platform.upper()} \u00d7{len(devices)}: {devices[0]}")
    except RuntimeError:
        # Requested device unavailable – fall back to CPU.
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        platform = "cpu"
        import jax
        devices = jax.devices()
        print(f"[device] '{desired}' unavailable \u2192 CPU: {devices[0]}")

    return platform
