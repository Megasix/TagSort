"""ONNX Runtime sessions: which device runs the models, and on how many threads."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal, get_args

from tagsort.models import ModelError

if TYPE_CHECKING:
    import onnxruntime

__all__ = ["DEVICES", "Device", "create_session", "providers_for", "usable_cpus"]

logger = logging.getLogger("tagsort")

Device = Literal["cpu", "cuda", "auto"]
"""Where the models run: ``cpu``, ``cuda`` (an NVIDIA GPU), or ``auto`` (the GPU if one is
usable, else the CPU)."""

DEVICES: tuple[Device, ...] = get_args(Device)

_CPU = "CPUExecutionProvider"
_CUDA = "CUDAExecutionProvider"
_CGROUP = Path("/sys/fs/cgroup")


def _cgroup_quota(root: Path) -> float | None:
    """CPUs allowed by a cgroup v2 or v1 quota, or None when there is no quota."""
    try:
        quota, period = (root / "cpu.max").read_text().split()[:2]
        if quota != "max":
            return int(quota) / int(period)
        return None
    except (OSError, ValueError):
        pass
    try:
        quota_us = int((root / "cpu" / "cpu.cfs_quota_us").read_text())
        period_us = int((root / "cpu" / "cpu.cfs_period_us").read_text())
    except (OSError, ValueError):
        return None
    return quota_us / period_us if quota_us > 0 and period_us > 0 else None


def usable_cpus(cgroup_root: Path = _CGROUP) -> int:
    """How many CPUs this process may really use.

    Containers often see every core of the host while a quota lets them use only a few;
    ONNX Runtime then starts a thread per visible core and slows down. This counts the
    CPUs in the process's affinity mask, capped by the cgroup CPU quota.
    """
    count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 0
    count = count or os.cpu_count() or 1
    quota = _cgroup_quota(cgroup_root)
    if quota is not None:
        count = min(count, max(1, math.ceil(quota)))
    return max(1, count)


def providers_for(device: Device, available: list[str] | None = None) -> list[str]:
    """ONNX Runtime execution providers for ``device``, best first.

    Args:
        device: ``cpu``, ``cuda`` or ``auto``.
        available: The providers ONNX Runtime offers; read from it when None.

    Raises:
        ValueError: If ``device`` is not one of :data:`DEVICES`.
        ModelError: If ``device`` is ``cuda`` and ONNX Runtime has no CUDA provider.
    """
    if device not in DEVICES:
        raise ValueError(f"device must be one of {', '.join(DEVICES)}, not {device!r}")
    if device == "cpu":
        return [_CPU]
    if available is None:
        import onnxruntime

        available = onnxruntime.get_available_providers()
    if _CUDA in available:
        return [_CUDA, _CPU]
    if device == "cuda":
        raise ModelError(
            "device='cuda' needs onnxruntime-gpu with CUDA 13 and cuDNN 9; "
            "see docker/Dockerfile.gpu, or use device='auto' to fall back to the CPU"
        )
    return [_CPU]


def _preload_cuda() -> None:
    """Load CUDA and cuDNN from NVIDIA's pip wheels when they are installed."""
    import onnxruntime

    preload = getattr(onnxruntime, "preload_dlls", None)
    if preload is None:
        return
    try:
        preload()
    except Exception as error:  # Libraries may come from the system instead.
        logger.debug("onnxruntime.preload_dlls failed: %s", error)


def create_session(
    path: str | Path, *, device: Device = "cpu", threads: int | None = None
) -> onnxruntime.InferenceSession:
    """Load an ONNX model for inference.

    Args:
        path: The ONNX file.
        device: Where the model runs, see :data:`Device`.
        threads: Threads for one inference on the CPU; defaults to :func:`usable_cpus`.

    Raises:
        ValueError: If ``device`` is unknown or ``threads`` is below 1.
        ModelError: If ``device`` is ``cuda`` and no GPU can be used.
    """
    if threads is not None and threads < 1:
        raise ValueError("threads must be at least 1")
    providers = providers_for(device)
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = threads or usable_cpus()
    options.inter_op_num_threads = 1
    if providers[0] == _CUDA:
        _preload_cuda()
    session = onnxruntime.InferenceSession(str(path), sess_options=options, providers=providers)
    if providers[0] == _CUDA and session.get_providers()[0] != _CUDA:
        if device == "cuda":
            raise ModelError("the CUDA provider is installed but could not use a GPU")
        logger.warning("no usable GPU; reading on the CPU")
    return session
