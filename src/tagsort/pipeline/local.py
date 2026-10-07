"""The local pipeline: detection, orientation, grammar-constrained recognition, scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from tagsort.grammar.decode import constrained, greedy
from tagsort.models import DEFAULT_MODEL, load_manifest, model_files
from tagsort.pipeline.detect import DetectedLine, Detector, OnnxDetector
from tagsort.pipeline.recognize import OnnxRecognizer, Recognizer
from tagsort.pipeline.runtime import Device
from tagsort.profile import Profile

__all__ = ["LineReading", "LocalPipeline", "score"]

DETECTION_SIDE = 1600
"""Working size of detection; on lot-02, 960 px missed small tags and 2,048 px was slower."""

MIN_LINE_PROBABILITY = 0.01
"""Lines whose best valid reading is less likely than this are not tags (rulers, names...).
On lot-02, every true tag scored at least 0.09 and every other line at most 0.0006."""

MIN_ACCEPT_PROBABILITY = 0.3
"""Below this probability a reading is never accepted, even when both decodings agree."""

CANDIDATES_PER_LINE = 3


@dataclass(frozen=True)
class LineReading:
    """What the local pipeline read on one text line that looks like a tag.

    Attributes:
        quad: Corners in pixels of the image read, clockwise from the top-left corner of
            the upright text.
        angle: Orientation of the text, clockwise from upright, in [0, 360).
        readings: Valid texts with their probability and profile tag id, most likely first.
        agrees: Whether the unconstrained reading spells the best valid text exactly.
        crop: The line, upright.
    """

    quad: tuple[tuple[float, float], ...]
    angle: float
    readings: tuple[tuple[str, float, str], ...]
    agrees: bool
    crop: Image.Image


def score(probability: float, agrees: bool) -> float:
    """Provisional confidence of a reading, until calibration on more labeled data.

    The model reading a valid text without the grammar's help (``agrees``) is strong
    independent evidence: on lot-02 it held for all 59 true tags and none of 138 other
    lines. Agreeing readings above :data:`MIN_ACCEPT_PROBABILITY` score at least 0.9;
    every other reading keeps its probability, below the acceptance threshold.
    """
    if agrees and probability >= MIN_ACCEPT_PROBABILITY:
        return 0.9 + 0.1 * probability
    return min(probability, 0.89)


class LocalPipeline:
    """Reads tags on the device with a downloaded model; no network access.

    Safe to share between threads.

    Args:
        model: Name of a downloaded model, see :func:`tagsort.available_models`.
        directory: Where models are stored; defaults to :func:`tagsort.models.models_dir`.
        detection_side: Longest image side for detection, in pixels.
        device: Where the models run: ``"cpu"`` (default), ``"cuda"`` for an NVIDIA GPU
            (needs ``onnxruntime-gpu``, see ``docker/Dockerfile.gpu``), or ``"auto"`` for
            the GPU when one is usable and the CPU otherwise.
        threads: CPU threads per inference. Defaults to the CPUs the process may really
            use, which in a container is its CPU quota rather than the host's cores.

    Raises:
        ModelError: If the model is unknown or not downloaded, or ``device="cuda"`` and
            no GPU can be used.
        ValueError: If ``device`` is unknown or ``threads`` is below 1.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        directory: str | Path | None = None,
        detection_side: int = DETECTION_SIDE,
        device: Device = "cpu",
        threads: int | None = None,
    ) -> None:
        """Load the model."""
        manifest = load_manifest(model)
        files = model_files(model, directory=directory)
        self.model = model
        self.version = manifest.version
        self._detector: Detector = OnnxDetector(
            files["det"],
            manifest.detection,
            limit_side=detection_side,
            device=device,
            threads=threads,
        )
        self._recognizer: Recognizer = OnnxRecognizer(
            files["rec"], manifest.recognition, device=device, threads=threads
        )

    def read(self, image: Image.Image, profile: Profile) -> list[LineReading]:
        """Return the lines of an upright image that read as a tag of ``profile``."""
        return [
            reading
            for line in self._detector.detect(image)
            if (reading := self._read_line(image, line, profile)) is not None
        ]

    def _read_line(
        self, image: Image.Image, line: DetectedLine, profile: Profile
    ) -> LineReading | None:
        tl, tr, br, bl = line.quad
        width = max(2, round(math.dist(tl, tr)))
        height = max(2, round(math.dist(tl, bl)))
        crop = image.transform(
            (width, height),
            Image.Transform.QUAD,
            data=(*tl, *bl, *br, *tr),
            resample=Image.Resampling.BILINEAR,
        )
        # A line lying across the box may be read in either direction; a tall box may
        # hold text turned a quarter either way.
        turns = (0, 180, 90, 270) if height > width * 1.2 else (0, 180)
        best: dict[str, tuple[float, str, int]] = {}
        agreeing: dict[int, str] = {}
        for turn in turns:
            upright = crop.rotate(turn, expand=True)
            probs = self._recognizer.probabilities(upright)
            agreeing[turn] = greedy(probs, self._recognizer.classes).text.replace(" ", "")
            for spec in profile.tags:
                for reading in constrained(probs, self._recognizer.classes, spec._automaton)[
                    :CANDIDATES_PER_LINE
                ]:
                    if reading.probability > best.get(reading.text, (0.0, "", 0))[0]:
                        best[reading.text] = (reading.probability, spec.id, turn)
        ranked = sorted(best.items(), key=lambda item: item[1][0], reverse=True)
        if not ranked or ranked[0][1][0] < MIN_LINE_PROBABILITY:
            return None
        text, (_, _, turn) = ranked[0]
        base = math.degrees(math.atan2(tr[1] - tl[1], tr[0] - tl[0]))
        corners = [tl, tr, br, bl]
        shift = turn // 90
        return LineReading(
            quad=tuple(corners[shift:] + corners[:shift]),
            angle=(base + turn) % 360,
            readings=tuple(
                (candidate, probability, tag_id)
                for candidate, (probability, tag_id, _) in ranked[:CANDIDATES_PER_LINE]
            ),
            agrees=agreeing[turn] == text,
            crop=crop.rotate(turn, expand=True),
        )
