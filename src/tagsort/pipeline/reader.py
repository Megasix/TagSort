"""The reader: from an image to a :class:`~tagsort.types.ReadResult`."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Generator, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace

from tagsort._version import __version__
from tagsort.fallback.base import ProviderTag, VisionProvider, VisionRequest
from tagsort.fallback.prompt import ANSWER_SCHEMA, build_instructions
from tagsort.pipeline.preprocess import ImageSource, PreparedImage, encode_jpeg, prepare
from tagsort.profile import Profile
from tagsort.types import Candidate, ImageInfo, Point, ReadResult, Tag, TagStatus

__all__ = ["Reader"]

# Provisional confidence scores, used until calibration lands (M3/M4). They only order
# readings by how much evidence supports them; they are not probabilities yet.
_CERTAIN_AND_VALID = 0.95
_UNCERTAIN_AND_VALID = 0.6
_ALTERNATIVE_VALID = 0.4
_NOT_VALID = 0.3
_UNREADABLE_FACTOR = 0.3


class Reader:
    """Finds and reads the tags in images, following a profile.

    A reader is built once and reused; it is safe to share between threads if its
    backend is.

    Args:
        profile: What the tags of the collection may say.
        backend: The vision API provider that finds and reads tags.
        accept_threshold: Minimum confidence for the ``accepted`` status.
        review_threshold: Minimum confidence for the ``review`` status; readings below it
            are ``unreadable`` and kept only as candidates.

    Raises:
        ValueError: If the thresholds are not ``0 <= review <= accept <= 1``.
    """

    def __init__(
        self,
        profile: Profile,
        *,
        backend: VisionProvider,
        accept_threshold: float = 0.9,
        review_threshold: float = 0.25,
    ) -> None:
        """Create the reader."""
        if not isinstance(profile, Profile):
            raise TypeError(f"profile must be a Profile, not {type(profile).__name__}")
        if not isinstance(backend, VisionProvider):
            raise TypeError(f"backend must be a VisionProvider, not {type(backend).__name__}")
        if not 0 <= review_threshold <= accept_threshold <= 1:
            raise ValueError("thresholds must satisfy 0 <= review <= accept <= 1")
        self._profile = profile
        self._backend = backend
        self._accept = accept_threshold
        self._review = review_threshold

    @property
    def profile(self) -> Profile:
        """The profile this reader follows."""
        return self._profile

    def read(self, image: ImageSource) -> ReadResult:
        """Find and read every tag in ``image``.

        Args:
            image: A file path, encoded image bytes, a PIL image or an array.

        Raises:
            ImageError: If the image cannot be decoded.
            ProviderError: If the vision API fails or returns an unusable answer.
        """
        start = time.perf_counter()
        prepared = prepare(image, max_side=self._backend.max_side)
        working = prepared.image
        request = VisionRequest(
            jpeg=encode_jpeg(working),
            width=working.width,
            height=working.height,
            instructions=build_instructions(
                self._profile,
                box_format=self._backend.box_format,
                width=working.width,
                height=working.height,
            ),
            schema=ANSWER_SCHEMA,
        )
        sent = time.perf_counter()
        answer = self._backend.read(request)
        done = time.perf_counter()
        return ReadResult(
            engine_version=__version__,
            model_version=f"{self._backend.name}:{answer.model}",
            image=ImageInfo(prepared.width, prepared.height, prepared.exif_rotation),
            tags=tuple(self._to_tag(raw, prepared) for raw in answer.tags),
            timings_ms={
                "preprocess": round((sent - start) * 1000, 1),
                "api": round((done - sent) * 1000, 1),
            },
        )

    def read_batch(
        self, images: Iterable[ImageSource], *, workers: int = 1
    ) -> Generator[ReadResult, None, None]:
        """Read images, yielding results in the order of ``images``.

        Args:
            images: Images to read; consumed lazily.
            workers: Images read at the same time. Vision APIs answer in seconds, so
                several workers cut the total time almost proportionally, within the
                provider's rate limits (rate-limited requests are retried).

        At most ``2 * workers`` images are in flight, so memory stays constant however
        long the batch. An error stops the batch; call :meth:`read` in a loop to
        handle errors image by image.

        Raises:
            ValueError: If ``workers`` is below 1.
        """
        if workers < 1:
            raise ValueError(f"workers must be at least 1, not {workers}")
        if workers == 1:
            for image in images:
                yield self.read(image)
            return
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tagsort") as pool:
            pending: deque[Future[ReadResult]] = deque()
            try:
                for image in images:
                    pending.append(pool.submit(self.read, image))
                    if len(pending) >= 2 * workers:
                        yield pending.popleft().result()
                while pending:
                    yield pending.popleft().result()
            finally:
                for future in pending:
                    future.cancel()

    def _to_tag(self, raw: ProviderTag, prepared: PreparedImage) -> Tag:
        raw = self._flag_upside_down_ambiguity(raw)
        readings = [raw.text, *raw.alternatives] if raw.text else list(raw.alternatives)
        scored = [(text, self._score(raw, index, text)) for index, text in enumerate(readings)]
        if raw.legibility == "unreadable":
            # Lower every guess but keep their order: a guess that fits the profile still
            # ranks first.
            scored = [(text, score * _UNREADABLE_FACTOR) for text, score in scored]
        # The first reading that fits the profile wins; ties keep the provider's order.
        scored.sort(key=lambda item: -item[1])
        best_text, confidence = scored[0] if scored else (None, 0.0)
        tag_id = self._profile.match(best_text) if best_text is not None else None
        status = self._status(confidence)
        if status == "accepted" and tag_id is None:
            # Whatever the thresholds, text outside the profile's grammar needs a person.
            status = "review"
        others = scored[1:]
        if status == "unreadable" and best_text is not None:
            others = scored
            best_text = None
        return Tag(
            tag_id=tag_id,
            text=best_text,
            confidence=confidence if best_text is not None else 0.0,
            status=status,
            polygon=_polygon(raw.box, raw.angle, prepared.scale),
            angle=raw.angle,
            candidates=tuple(
                Candidate(text, round(score * (1 - confidence) / (rank + 1), 3))
                for rank, (text, score) in enumerate(others)
            ),
            source="fallback",
        )

    def _flag_upside_down_ambiguity(self, raw: ProviderTag) -> ProviderTag:
        r"""Mark a reading uncertain when the tag read upside down also fits the profile.

        "0086" upside down reads "9800": both fit ``\d{4}``, so the orientation the
        provider chose decides the text, and a wrong guess would be accepted silently.
        """
        flipped = _upside_down(raw.text)
        if flipped is None or flipped == raw.text or self._profile.match(flipped) is None:
            return raw
        alternatives = raw.alternatives
        if flipped not in alternatives:
            alternatives = (*alternatives, flipped)
        legibility = "uncertain" if raw.legibility == "certain" else raw.legibility
        return replace(raw, legibility=legibility, alternatives=alternatives)

    def _score(self, raw: ProviderTag, index: int, text: str) -> float:
        if self._profile.match(text) is None:
            return _NOT_VALID * (0.9**index)
        if index > 0:
            return _ALTERNATIVE_VALID * (0.9 ** (index - 1))
        return _CERTAIN_AND_VALID if raw.legibility == "certain" else _UNCERTAIN_AND_VALID

    def _status(self, confidence: float) -> TagStatus:
        if confidence >= self._accept:
            return "accepted"
        if confidence >= self._review:
            return "review"
        return "unreadable"


# Characters that read as another valid character when turned upside down (180°).
_UPSIDE_DOWN = {
    **{c: c for c in "018HINOSXZlosxz-/ "},
    "6": "9",
    "9": "6",
    "M": "W",
    "W": "M",
    "b": "q",
    "q": "b",
    "d": "p",
    "p": "d",
    "n": "u",
    "u": "n",
}


def _upside_down(text: str) -> str | None:
    """Return ``text`` as it reads turned 180°, or ``None`` if some character cannot."""
    if not text or any(char not in _UPSIDE_DOWN for char in text):
        return None
    return "".join(_UPSIDE_DOWN[char] for char in reversed(text))


def _polygon(
    box: tuple[float, float, float, float], angle: int, scale: float
) -> tuple[Point, Point, Point, Point]:
    """Corners in original pixels, clockwise from the top-left corner of the upright text."""
    x0, y0, x1, y1 = (round(value * scale, 1) for value in box)
    corners: list[Point] = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    shift = angle // 90
    rotated = corners[shift:] + corners[:shift]
    return (rotated[0], rotated[1], rotated[2], rotated[3])
