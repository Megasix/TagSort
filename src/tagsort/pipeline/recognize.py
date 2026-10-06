"""Text recognition: from a crop of one text line to per-step class probabilities."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from tagsort.models import RecognitionParams

__all__ = ["OnnxRecognizer", "Recognizer"]

MAX_WIDTH = 1600
"""Widest input sent to the recognizer, in pixels after resizing to its height."""


@runtime_checkable
class Recognizer(Protocol):
    """Reads one text line; any CTC recognizer can implement this."""

    @property
    def classes(self) -> tuple[str, ...]:
        """Label of every output class; ``classes[0]`` is the CTC blank."""
        ...

    def probabilities(self, line: Image.Image) -> NDArray[np.float32]:
        """Return the class probabilities of shape (T, len(classes)) for an upright line."""
        ...


class OnnxRecognizer:
    """A CTC recognizer exported to ONNX, such as PP-OCRv6's.

    Safe to share between threads.
    """

    def __init__(self, path: str | Path, params: RecognitionParams) -> None:
        """Load the model at ``path`` with the preprocessing described by ``params``."""
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3
        self._session = onnxruntime.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name
        self._params = params
        self._classes = params.classes
        _, self._height, self._min_width = params.input_shape
        self._mean = np.array(params.mean, dtype=np.float32)
        self._std = np.array(params.std, dtype=np.float32)

    @property
    def classes(self) -> tuple[str, ...]:
        """Label of every output class; ``classes[0]`` is the CTC blank."""
        return self._classes

    def probabilities(self, line: Image.Image) -> NDArray[np.float32]:
        """Return the class probabilities of shape (T, len(classes)) for an upright line."""
        image = line.convert("RGB")
        width = min(max(math.ceil(image.width * self._height / image.height), 1), MAX_WIDTH)
        resized = image.resize((width, self._height), Image.Resampling.BILINEAR)
        pixels = np.asarray(resized, dtype=np.float32) / 255.0
        if self._params.channel_order == "BGR":
            pixels = pixels[..., ::-1]
        pixels = (pixels - self._mean) / self._std
        # Pad on the right to the model's minimum width, as PaddleOCR does.
        batch = np.zeros((1, 3, self._height, max(width, self._min_width)), dtype=np.float32)
        batch[0, :, :, :width] = pixels.transpose(2, 0, 1)
        output: NDArray[np.float32] = self._session.run(None, {self._input: batch})[0][0]
        if output.shape[-1] != len(self._classes):
            raise ValueError(
                f"model outputs {output.shape[-1]} classes but the manifest lists "
                f"{len(self._classes)}"
            )
        return output
