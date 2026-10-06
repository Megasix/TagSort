"""Local models: manifests, cache folder and explicit download.

Each model has a manifest in ``manifests/`` recording where its files come from (pinned
to an upstream revision), their size and SHA-256, and everything needed to run them
identically in any language: preprocessing, decoder and character set.

TagSort never downloads anything on its own (CLAUDE.md principle 2): call
:func:`download_model` or run ``tagsort models download`` once.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from tagsort.errors import TagSortError

__all__ = [
    "DEFAULT_MODEL",
    "DetectionParams",
    "FileSpec",
    "Manifest",
    "ModelError",
    "RecognitionParams",
    "available_models",
    "download_model",
    "load_manifest",
    "model_files",
    "models_dir",
]

DEFAULT_MODEL = "ppocrv6-tiny"
"""Smallest model (about 6 MB), the one mobile applications embed."""

_KINDS = ("det", "rec")


class ModelError(TagSortError):
    """A model is unknown, not downloaded, or its files are corrupt."""


@dataclass(frozen=True)
class FileSpec:
    """One model file: where it comes from and how to check it."""

    url: str
    sha256: str
    size: int


@dataclass(frozen=True)
class DetectionParams:
    """Preprocessing and postprocessing of the text detector."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    channel_order: str
    limit_side: int
    multiple_of: int
    threshold: float
    box_threshold: float
    unclip_ratio: float


@dataclass(frozen=True)
class RecognitionParams:
    """Preprocessing and decoding of the text recognizer."""

    input_shape: tuple[int, int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    channel_order: str
    decoder: str
    blank_index: int
    space_appended: bool
    charset: tuple[str, ...]

    @property
    def classes(self) -> tuple[str, ...]:
        """Label of every output class; the blank is the empty string."""
        return ("", *self.charset, *((" ",) if self.space_appended else ()))


@dataclass(frozen=True)
class Manifest:
    """Everything known about one model."""

    name: str
    version: str
    license: str
    source: str
    files: dict[str, FileSpec]
    detection: DetectionParams
    recognition: RecognitionParams

    @property
    def size(self) -> int:
        """Total size of the model files, in bytes."""
        return sum(spec.size for spec in self.files.values())


def available_models() -> list[str]:
    """Names of the models TagSort knows, smallest first."""
    names = [
        entry.name.removesuffix(".json")
        for entry in resources.files("tagsort.models").joinpath("manifests").iterdir()
        if entry.name.endswith(".json")
    ]
    return sorted(names, key=lambda name: load_manifest(name).size)


def load_manifest(name: str) -> Manifest:
    """Return the manifest of model ``name``.

    Raises:
        ModelError: If no model has this name.
    """
    folder = resources.files("tagsort.models").joinpath("manifests")
    entry = folder.joinpath(f"{name}.json")
    if not entry.is_file():
        known = ", ".join(sorted(p.name.removesuffix(".json") for p in folder.iterdir()))
        raise ModelError(f"unknown model {name!r}; known models: {known}")
    data: dict[str, Any] = json.loads(entry.read_text(encoding="utf-8"))
    det, rec = data["detection"], data["recognition"]
    return Manifest(
        name=data["name"],
        version=data["version"],
        license=data["license"],
        source=data["source"],
        files={kind: FileSpec(**data["files"][kind]) for kind in _KINDS},
        detection=DetectionParams(
            mean=tuple(det["mean"]),
            std=tuple(det["std"]),
            channel_order=det["channel_order"],
            limit_side=det["limit_side"],
            multiple_of=det["multiple_of"],
            threshold=det["threshold"],
            box_threshold=det["box_threshold"],
            unclip_ratio=det["unclip_ratio"],
        ),
        recognition=RecognitionParams(
            input_shape=tuple(rec["input_shape"]),
            mean=tuple(rec["mean"]),
            std=tuple(rec["std"]),
            channel_order=rec["channel_order"],
            decoder=rec["decoder"],
            blank_index=rec["blank_index"],
            space_appended=rec["space_appended"],
            charset=tuple(rec["charset"]),
        ),
    )


def models_dir() -> Path:
    """Folder holding downloaded models.

    ``$TAGSORT_MODELS`` if set, else ``$XDG_CACHE_HOME/tagsort/models``, else
    ``~/.cache/tagsort/models``.
    """
    if os.environ.get("TAGSORT_MODELS"):
        return Path(os.environ["TAGSORT_MODELS"])
    cache = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache) / "tagsort" / "models"


def model_files(name: str, *, directory: str | Path | None = None) -> dict[str, Path]:
    """Return the paths of a downloaded model's files, checking their sizes.

    Raises:
        ModelError: If the model is unknown, not downloaded, or a file has the wrong size.
    """
    manifest = load_manifest(name)
    folder = Path(directory) if directory is not None else models_dir()
    paths = {}
    for kind, spec in manifest.files.items():
        path = folder / name / f"{kind}.onnx"
        if not path.is_file():
            raise ModelError(
                f"model {name!r} is not downloaded; run `tagsort models download {name}` "
                f"or call tagsort.download_model({name!r})"
            )
        if path.stat().st_size != spec.size:
            raise ModelError(f"{path} has the wrong size; download the model again")
        paths[kind] = path
    return paths


def download_model(
    name: str = DEFAULT_MODEL,
    *,
    directory: str | Path | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Path]:
    """Download a model's files, check their SHA-256, and return their paths.

    Files already present with the right checksum are kept. This is the only function
    in TagSort that downloads models; it needs the ``api`` extra (``httpx``).

    Args:
        name: Model to download, see :func:`available_models`.
        directory: Where to store models; defaults to :func:`models_dir`.
        progress: Called with ``(kind, bytes_done, bytes_total)`` while downloading.

    Raises:
        ModelError: If the model is unknown or a downloaded file fails its checksum.
        ImportError: If httpx is not installed.
    """
    manifest = load_manifest(name)
    try:
        import httpx
    except ModuleNotFoundError:
        raise ImportError(
            'downloading models needs the api extra: pip install "tagsort[api]"'
        ) from None
    folder = (Path(directory) if directory is not None else models_dir()) / name
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        for kind, spec in manifest.files.items():
            path = folder / f"{kind}.onnx"
            paths[kind] = path
            if path.is_file() and _sha256(path) == spec.sha256:
                continue
            partial = path.with_suffix(".part")
            digest = hashlib.sha256()
            done = 0
            with client.stream("GET", spec.url) as response, partial.open("wb") as out:
                if response.status_code != 200:
                    raise ModelError(f"cannot download {spec.url}: HTTP {response.status_code}")
                for chunk in response.iter_bytes(1 << 20):
                    out.write(chunk)
                    digest.update(chunk)
                    done += len(chunk)
                    if progress is not None:
                        progress(kind, done, spec.size)
            if digest.hexdigest() != spec.sha256:
                partial.unlink()
                raise ModelError(
                    f"{spec.url} does not match its manifest checksum; the file was not kept"
                )
            partial.replace(path)
    return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
