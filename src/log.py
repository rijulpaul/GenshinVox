"""Minimal print-based logging helper for distributed runs.

Every message is prefixed with a timestamp and a severity level, e.g.::

    [14:03:22][INFO] Preprocessed 1423 samples
    [14:03:22][WARN][rank 3] Could not find reference audio, retrying

Behaviour in multi-process (accelerate / torchrun) runs:

- ``info`` is printed from the main process (rank 0) only, so progress
  reports do not spam the terminal once per worker.
- ``warning`` and ``error`` are printed from every process, so failures
  on any worker are never hidden.
- A ``[rank N]`` tag is added whenever more than one process is active.

Output goes straight to stdout via ``print(..., flush=True)`` and works
identically on every machine of a cluster (no logging config, no files).

Usage::

    from src.log import info, warning, error

    info("Model loaded")
    warning("Cache dir missing, downloading...")
    error("Failed to save checkpoint")
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

__all__ = ["info", "warning", "error", "log"]


def _world_size() -> int:
    """Number of distributed processes, 1 when not launched distributed."""
    world_size = os.environ.get("WORLD_SIZE")
    if world_size is not None:
        try:
            return int(world_size)
        except ValueError:
            pass
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    return 1


def _rank() -> int:
    """Global rank of the current process, 0 when not distributed."""
    rank = os.environ.get("RANK")
    if rank is not None:
        try:
            return int(rank)
        except ValueError:
            pass
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def log(level: str, message: str) -> None:
    """Low-level emitter; prefer :func:`info`, :func:`warning`, :func:`error`."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if _world_size() > 1:
        prefix = f"[{timestamp}][{level}][rank {_rank()}]"
    else:
        prefix = f"[{timestamp}][{level}]"
    print(f"{prefix} {message}", file=sys.stdout, flush=True)


def info(message: str) -> None:
    """Progress/informational message, printed from the main process only."""
    if _rank() != 0:
        return
    log("INFO", message)


def warning(message: str) -> None:
    """Warning message, printed from every process."""
    log("WARN", message)


def error(message: str) -> None:
    """Error message, printed from every process."""
    log("ERROR", message)