"""Removes NVIDIA development headers from the GPU image (see NOTICE).

The NVIDIA pip wheels install C headers next to their libraries. They are not needed to
run TagSort, and NVIDIA's agreements let only some of them be redistributed. This keeps
the headers that Attachment A of the CUDA Toolkit EULA lists as distributable and deletes
every other file under an `include/` folder of the `nvidia` package, cuDNN's included.

    python prune_nvidia_headers.py /app/.venv/lib/python3.12/site-packages/nvidia
"""

import sys
from pathlib import Path

# Attachment A of the CUDA Toolkit EULA (v13.4): floating point type headers, headers for
# runtime compilation, the occupancy calculator and NVRTC's header.
DISTRIBUTABLE = {
    "cuda_fp16.h",
    "cuda_fp16.hpp",
    "cuda_bf16.h",
    "cuda_bf16.hpp",
    "cuda_fp8.h",
    "cuda_fp8.hpp",
    "cuda_fp6.h",
    "cuda_fp6.hpp",
    "cuda_fp4.h",
    "cuda_fp4.hpp",
    "crt/host_defines.h",
    "cuComplex.h",
    "cuda_awbarrier_helpers.h",
    "cuda_awbarrier_primitives.h",
    "cuda_awbarrier.h",
    "cuda_pipeline_helpers.h",
    "cuda_pipeline_primitives.h",
    "cuda_pipeline.h",
    "cuda_runtime_api.h",
    "cuda.h",
    "device_types.h",
    "vector_functions.h",
    "vector_types.h",
    "cuda/std/tuple",
    "cuda/std/type_traits",
    "cuda/std/utility",
    "cuda_occupancy.h",
    "nvrtc.h",
}


def prune(nvidia: Path) -> tuple[int, int]:
    """Deletes non-distributable headers under `nvidia`; returns (kept, removed)."""
    kept = removed = 0
    for include in nvidia.glob("*/include"):
        for path in sorted(include.rglob("*")):
            if not path.is_file():
                continue
            if path.relative_to(include).as_posix() in DISTRIBUTABLE:
                kept += 1
            else:
                path.unlink()
                removed += 1
    return kept, removed


if __name__ == "__main__":
    root = Path(sys.argv[1])
    if not root.is_dir():
        sys.exit(f"{root} is not a folder")
    kept, removed = prune(root)
    sys.stdout.write(f"NVIDIA headers: kept {kept} distributable, removed {removed}\n")
