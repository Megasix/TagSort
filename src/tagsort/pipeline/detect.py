"""Text detection: from a photo to oriented quadrilaterals around each text line.

The detector (a DB network such as PP-OCRv6's) outputs, for every pixel, the
probability that it belongs to text. Postprocessing turns that map into boxes:
threshold, connected components, minimum-area rotated rectangle of each component,
then an outward offset ("unclip") because DB predicts shrunken text regions. Everything
is done with numpy, without OpenCV, to keep the core light.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from tagsort.models import DetectionParams

__all__ = ["DetectedLine", "Detector", "OnnxDetector", "boxes_from_map", "min_area_rect"]

Point = tuple[float, float]
Quad = tuple[Point, Point, Point, Point]


@dataclass(frozen=True)
class DetectedLine:
    """A text line found in an image.

    Attributes:
        quad: Corners in pixels of the image given to the detector, clockwise from the
            top-left corner of the box as it lies in the image (not of the upright text).
        score: Mean text probability inside the region, in [0, 1].
    """

    quad: Quad
    score: float


@runtime_checkable
class Detector(Protocol):
    """Finds text lines; any detector can implement this."""

    def detect(self, image: Image.Image) -> list[DetectedLine]:
        """Return the text lines of an upright RGB image, in its pixel coordinates."""
        ...


class OnnxDetector:
    """A DB text detector exported to ONNX, such as PP-OCRv6's.

    Safe to share between threads.
    """

    def __init__(
        self, path: str | Path, params: DetectionParams, *, limit_side: int | None = None
    ) -> None:
        """Load the model.

        Args:
            path: The ONNX file.
            params: Preprocessing and postprocessing from the model manifest.
            limit_side: Longest side the image is resized to before detection;
                defaults to the manifest's. Larger finds smaller text but is slower.
        """
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3
        self._session = onnxruntime.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name
        self._params = params
        self._limit = limit_side or params.limit_side
        self._mean = np.array(params.mean, dtype=np.float32)
        self._std = np.array(params.std, dtype=np.float32)

    def detect(self, image: Image.Image) -> list[DetectedLine]:
        """Return the text lines of an upright RGB image, in its pixel coordinates."""
        rgb = image.convert("RGB")
        scale = min(1.0, self._limit / max(rgb.size))
        step = self._params.multiple_of
        width = max(step, round(rgb.width * scale / step) * step)
        height = max(step, round(rgb.height * scale / step) * step)
        resized = rgb.resize((width, height), Image.Resampling.BILINEAR)
        pixels = np.asarray(resized, dtype=np.float32) / 255.0
        if self._params.channel_order == "BGR":
            pixels = pixels[..., ::-1]
        pixels = (pixels - self._mean) / self._std
        batch = pixels.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        probability: NDArray[np.float32] = self._session.run(None, {self._input: batch})[0][0, 0]
        lines = boxes_from_map(
            probability,
            threshold=self._params.threshold,
            box_threshold=self._params.box_threshold,
            unclip_ratio=self._params.unclip_ratio,
        )
        sx, sy = rgb.width / width, rgb.height / height
        return [
            DetectedLine(
                quad=tuple((x * sx, y * sy) for x, y in line.quad),  # type: ignore[arg-type]
                score=line.score,
            )
            for line in lines
        ]


def boxes_from_map(
    probability: NDArray[np.floating],
    *,
    threshold: float,
    box_threshold: float,
    unclip_ratio: float,
    min_side: float = 3.0,
) -> list[DetectedLine]:
    """Turn a text probability map into rotated boxes, in map pixel coordinates."""
    mask = probability > threshold
    lines = []
    for runs in _components(mask):
        rows = np.array([r for r, _, _ in runs])
        starts = np.array([s for _, s, _ in runs])
        ends = np.array([e for _, _, e in runs])
        total = sum(float(probability[r, s : e + 1].sum()) for r, s, e in runs)
        score = total / float((ends - starts + 1).sum())
        if score < box_threshold:
            continue
        # Pixel corners of each run's end pixels bound the component exactly.
        xs = np.concatenate([starts, starts, ends + 1, ends + 1]).astype(np.float64)
        ys = np.concatenate([rows, rows + 1, rows, rows + 1]).astype(np.float64)
        center, (w, h), angle = min_area_rect(np.stack([xs, ys], axis=1))
        if min(w, h) < min_side:
            continue
        area, perimeter = w * h, 2 * (w + h)
        offset = area * unclip_ratio / perimeter
        quad = _rect_corners(center, (w + 2 * offset, h + 2 * offset), angle)
        lines.append(DetectedLine(quad=quad, score=score))
    return lines


def _components(mask: NDArray[np.bool_]) -> list[list[tuple[int, int, int]]]:
    """8-connected components as lists of (row, start, end) runs, via union-find on runs."""
    runs: list[tuple[int, int, int]] = []
    row_runs: list[list[int]] = []
    padded = np.zeros(mask.shape[1] + 2, dtype=np.int8)
    for row in range(mask.shape[0]):
        padded[1:-1] = mask[row]
        edges = np.flatnonzero(np.diff(padded))
        indices = []
        for start, stop in zip(edges[::2], edges[1::2], strict=True):
            indices.append(len(runs))
            runs.append((row, int(start), int(stop) - 1))
        row_runs.append(indices)

    parent = list(range(len(runs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for row in range(1, len(row_runs)):
        previous = row_runs[row - 1]
        j = 0
        for i in row_runs[row]:
            _, start, end = runs[i]
            # Skip runs of the previous row ending before this one starts (8-connectivity).
            while j < len(previous) and runs[previous[j]][2] < start - 1:
                j += 1
            k = j
            while k < len(previous) and runs[previous[k]][1] <= end + 1:
                a, b = find(i), find(previous[k])
                if a != b:
                    parent[a] = b
                k += 1

    groups: dict[int, list[tuple[int, int, int]]] = {}
    for i, run in enumerate(runs):
        groups.setdefault(find(i), []).append(run)
    return list(groups.values())


def min_area_rect(
    points: NDArray[np.float64],
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Minimum-area rectangle enclosing ``points``: (center, (width, height), angle in radians).

    Width lies along the angle, height across it; the angle is in (-pi/2, pi/2].
    """
    hull = _convex_hull(points)
    if len(hull) == 1:
        (x, y) = hull[0]
        return (float(x), float(y)), (0.0, 0.0), 0.0
    best: tuple[float, tuple[float, float], tuple[float, float], float] | None = None
    for i in range(len(hull)):
        edge = hull[(i + 1) % len(hull)] - hull[i]
        length = math.hypot(edge[0], edge[1])
        if length == 0:
            continue
        ux, uy = edge / length
        along = hull @ np.array([ux, uy])
        across = hull @ np.array([-uy, ux])
        width, height = along.max() - along.min(), across.max() - across.min()
        area = width * height
        if best is None or area < best[0] - 1e-9:
            mid_along = (along.max() + along.min()) / 2
            mid_across = (across.max() + across.min()) / 2
            center = (mid_along * ux - mid_across * uy, mid_along * uy + mid_across * ux)
            best = (area, center, (width, height), math.atan2(uy, ux))
    assert best is not None
    _, center, (width, height), angle = best
    # Normalize so the angle is in (-pi/2, pi/2].
    if angle <= -math.pi / 2:
        angle += math.pi
    elif angle > math.pi / 2:
        angle -= math.pi
    return (float(center[0]), float(center[1])), (float(width), float(height)), angle


def _convex_hull(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Andrew's monotone chain; returns hull vertices counterclockwise."""
    unique = np.unique(points, axis=0)
    if len(unique) <= 2:
        return unique

    def cross(o: NDArray[np.float64], a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[NDArray[np.float64]] = []
    for p in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[NDArray[np.float64]] = []
    for p in unique[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _rect_corners(center: tuple[float, float], size: tuple[float, float], angle: float) -> Quad:
    """Corners of a rotated rectangle, clockwise in image coordinates from the top-left."""
    cx, cy = center
    w, h = size
    ux, uy = math.cos(angle), math.sin(angle)
    vx, vy = -uy, ux
    corners = [
        (cx + sx * w / 2 * ux + sy * h / 2 * vx, cy + sx * w / 2 * uy + sy * h / 2 * vy)
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]
    # Start from the corner closest to the image's top-left, keeping clockwise order
    # (y grows downward, so increasing angle is clockwise on screen).
    start = min(range(4), key=lambda i: corners[i][0] + corners[i][1])
    ordered = corners[start:] + corners[:start]
    return (ordered[0], ordered[1], ordered[2], ordered[3])
