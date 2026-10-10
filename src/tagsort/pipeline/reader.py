"""The reader: from an image to a :class:`~tagsort.types.ReadResult`."""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Generator, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from typing import Literal

from tagsort._version import __version__
from tagsort.errors import ProviderError
from tagsort.fallback.base import ProviderTag, Usage, VisionProvider, VisionRequest
from tagsort.fallback.prompt import ANSWER_SCHEMA, build_instructions
from tagsort.pipeline.local import LineReading, LocalPipeline, PlainLine, score
from tagsort.pipeline.openread import tag_likelihood
from tagsort.pipeline.preprocess import ImageSource, PreparedImage, encode_jpeg, prepare
from tagsort.profile import Profile
from tagsort.types import (
    ApiUsage,
    Candidate,
    ImageInfo,
    OtherText,
    Point,
    ReadResult,
    Tag,
    TagStatus,
)

__all__ = ["Reader"]

logger = logging.getLogger("tagsort")

LOCAL_MAX_SIDE = 4096
"""Longest side kept for the local pipeline; crops are cut from this image."""

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
        backend: ``"local"`` (the default) reads on the device with the default
            downloaded model; a :class:`LocalPipeline` chooses another model; a vision
            API provider sends each whole photo to that API instead.
        fallback: With a local backend, a vision API provider that reads again, from the
            crop of the tag only, every tag the local pipeline does not accept. Off by
            default; the photo itself never leaves the device.
        accept_threshold: Minimum confidence for the ``accepted`` status.
        review_threshold: Minimum confidence for the ``review`` status; readings below it
            are ``unreadable`` and kept only as candidates.
        open_reading: With a local backend, also return in ``other_texts`` the lines that
            fit no kind of tag, each with how likely it is a specimen tag
            (``docs/open-reading.md``). Off by default.

    Raises:
        ValueError: If the thresholds are not ``0 <= review <= accept <= 1``, or a
            fallback is given with an API backend.
        ModelError: If the local model is not downloaded.
    """

    def __init__(
        self,
        profile: Profile,
        *,
        backend: VisionProvider | LocalPipeline | Literal["local"] = "local",
        fallback: VisionProvider | None = None,
        accept_threshold: float = 0.9,
        review_threshold: float = 0.25,
        open_reading: bool = False,
    ) -> None:
        """Create the reader."""
        if not isinstance(profile, Profile):
            raise TypeError(f"profile must be a Profile, not {type(profile).__name__}")
        if not 0 <= review_threshold <= accept_threshold <= 1:
            raise ValueError("thresholds must satisfy 0 <= review <= accept <= 1")
        if fallback is not None and not isinstance(fallback, VisionProvider):
            raise TypeError(f"fallback must be a VisionProvider, not {type(fallback).__name__}")
        self._local: LocalPipeline | None = None
        self._backend: VisionProvider | None = None
        if backend == "local":
            self._local = LocalPipeline()
        elif isinstance(backend, LocalPipeline):
            self._local = backend
        elif isinstance(backend, VisionProvider):
            if fallback is not None:
                raise ValueError("a fallback is only used with the local backend")
            self._backend = backend
        else:
            raise TypeError(
                f"backend must be 'local', a LocalPipeline or a VisionProvider, not {backend!r}"
            )
        self._fallback = fallback
        self._profile = profile
        self._accept = accept_threshold
        self._review = review_threshold
        self._open_reading = open_reading

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
        if self._local is not None:
            return self._read_local(image, self._local)
        assert self._backend is not None
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
            api_usage=_total([answer.usage]),
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

    def _read_local(self, image: ImageSource, local: LocalPipeline) -> ReadResult:
        start = time.perf_counter()
        prepared = prepare(image, max_side=LOCAL_MAX_SIDE)
        prepared_at = time.perf_counter()
        if self._open_reading:
            lines, plain = local.read_all(prepared.image, self._profile)
        else:
            lines, plain = local.read(prepared.image, self._profile), []
        read_at = time.perf_counter()
        tags = []
        fallback_seconds = 0.0
        used_fallback = False
        billed: list[Usage] = []
        for line in lines:
            tag = self._local_tag(line, prepared)
            if tag.status != "accepted" and self._fallback is not None:
                asked = time.perf_counter()
                tag, usage = self._ask_fallback(self._fallback, line, prepared, tag)
                fallback_seconds += time.perf_counter() - asked
                used_fallback = True
                if usage is not None:
                    billed.append(usage)
            tags.append(tag)
        timings = {
            "preprocess": round((prepared_at - start) * 1000, 1),
            "local": round((read_at - prepared_at) * 1000, 1),
        }
        model_version = f"local:{local.model}-{local.version}"
        if used_fallback and self._fallback is not None:
            timings["fallback"] = round(fallback_seconds * 1000, 1)
            model_version += f"+{self._fallback.name}:{self._fallback.model}"
        return ReadResult(
            engine_version=__version__,
            model_version=model_version,
            image=ImageInfo(prepared.width, prepared.height, prepared.exif_rotation),
            tags=tuple(tags),
            timings_ms=timings,
            other_texts=_other_texts(plain, prepared),
            api_usage=_total(billed),
        )

    def _local_tag(self, line: LineReading, prepared: PreparedImage) -> Tag:
        text, probability, tag_id = line.readings[0]
        confidence = score(probability, line.agrees)
        others = [(t, p) for t, p, _ in line.readings[1:]]
        flipped = _upside_down(text)
        if flipped is not None and flipped != text and self._profile.match(flipped) is not None:
            # Same rule as for API readings: an orientation guess must not decide alone.
            confidence = min(confidence, _UNCERTAIN_AND_VALID)
            if flipped not in {t for t, _ in others}:
                others.append((flipped, 0.0))
        status = self._status(confidence)
        best: str | None = text
        if status == "unreadable":
            others.insert(0, (text, probability))
            best = None
        ceiling = confidence if best is not None else 1.0
        candidates = sorted(
            (Candidate(t, round(min(p, ceiling), 4)) for t, p in others),
            key=lambda candidate: -candidate.confidence,
        )
        return Tag(
            tag_id=tag_id if best is not None else self._profile.match(text),
            text=best,
            confidence=round(confidence, 4) if best is not None else 0.0,
            status=status,
            polygon=_scale(line.quad, prepared.scale),
            angle=round(line.angle, 1) % 360,
            candidates=tuple(candidates),
            source="local",
        )

    def _ask_fallback(
        self, provider: VisionProvider, line: LineReading, prepared: PreparedImage, local: Tag
    ) -> tuple[Tag, Usage | None]:
        """Read the tag again from its crop alone; keep the local tag if that fails.

        Also returns what the provider billed, or ``None`` when it did not answer.
        """
        crop = line.crop.copy()
        crop.thumbnail((provider.max_side, provider.max_side))
        request = VisionRequest(
            jpeg=encode_jpeg(crop),
            width=crop.width,
            height=crop.height,
            instructions=build_instructions(
                self._profile, box_format=provider.box_format, width=crop.width, height=crop.height
            ),
            schema=ANSWER_SCHEMA,
        )
        try:
            answer = provider.read(request)
        except ProviderError as error:
            logger.warning("fallback failed, keeping the local reading: %s", error)
            return local, None
        fitted = [self._without_stray_spaces(raw) for raw in answer.tags]
        valid = [raw for raw in fitted if self._profile.match(raw.text) is not None]
        if not valid:
            return local, answer.usage
        raw = max(valid, key=lambda r: r.legibility == "certain")
        tag = self._to_tag(raw, prepared)
        local_texts = [local.text, *(c.text for c in local.candidates)]
        candidates = [
            Candidate(t, round(min(0.05, tag.confidence), 4))
            for t in local_texts
            if t is not None and t != tag.text
        ]
        rescued = replace(
            tag,
            polygon=local.polygon,
            angle=local.angle,
            candidates=tuple(sorted({*tag.candidates, *candidates}, key=lambda c: -c.confidence)),
        )
        return rescued, answer.usage

    def _to_tag(self, raw: ProviderTag, prepared: PreparedImage) -> Tag:
        raw = self._flag_upside_down_ambiguity(self._without_stray_spaces(raw))
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

    def _without_stray_spaces(self, raw: ProviderTag) -> ProviderTag:
        """Drop the spaces of readings that fit the profile only without them.

        A gap in a written or printed tag (``GJ 07966``) is not a character: the local
        decoder ignores spaces a pattern does not allow, and so does this, for providers.
        """

        def fit(text: str) -> str:
            if self._profile.match(text) is not None:
                return text
            compact = "".join(text.split())
            return compact if self._profile.match(compact) is not None else text

        return replace(
            raw, text=fit(raw.text), alternatives=tuple(fit(text) for text in raw.alternatives)
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


def _scale(
    quad: tuple[tuple[float, float], ...], scale: float
) -> tuple[Point, Point, Point, Point]:
    """Corners in original pixels, rounded to a tenth of a pixel."""
    points = [(round(x * scale, 1), round(y * scale, 1)) for x, y in quad]
    return (points[0], points[1], points[2], points[3])


def _polygon(
    box: tuple[float, float, float, float], angle: int, scale: float
) -> tuple[Point, Point, Point, Point]:
    """Corners in original pixels, clockwise from the top-left corner of the upright text."""
    x0, y0, x1, y1 = (round(value * scale, 1) for value in box)
    corners: list[Point] = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    shift = angle // 90
    rotated = corners[shift:] + corners[:shift]
    return (rotated[0], rotated[1], rotated[2], rotated[3])


def _total(billed: list[Usage]) -> ApiUsage | None:
    """What the vision API billed for one image, or ``None`` when it never answered."""
    if not billed:
        return None
    return ApiUsage(
        calls=len(billed),
        input_tokens=sum(usage.input_tokens for usage in billed),
        output_tokens=sum(usage.output_tokens for usage in billed),
    )


def _other_texts(lines: list[PlainLine], prepared: PreparedImage) -> tuple[OtherText, ...]:
    """Lines no kind of tag fits, most likely specimen tags first (open reading)."""
    others = []
    for line in lines:
        text = " ".join(line.text.split())
        if not text:
            continue
        likelihood, kind = tag_likelihood(text)
        others.append(
            OtherText(
                text=text,
                confidence=round(line.confidence, 4),
                tag_likelihood=likelihood,
                kind_guess=kind,
                polygon=_scale(line.quad, prepared.scale),
                angle=round(line.angle, 1) % 360,
            )
        )
    return tuple(sorted(others, key=lambda other: other.tag_likelihood, reverse=True))
