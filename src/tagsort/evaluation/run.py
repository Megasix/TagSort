"""Reading a dataset with a reader, with a cache so no photo is paid for twice."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tagsort.errors import TagSortError
from tagsort.evaluation.dataset import LabeledImage
from tagsort.evaluation.metrics import ImageOutcome, match_tags
from tagsort.fallback.base import BoxFormat, Usage, VisionAnswer, VisionProvider, VisionRequest
from tagsort.pipeline.local import LocalPipeline
from tagsort.pipeline.reader import Reader
from tagsort.profile import Profile
from tagsort.types import Candidate, Tag

__all__ = ["CountingProvider", "RunResult", "run_dataset"]

logger = logging.getLogger("tagsort.evaluation")

USAGE_FILE = "_usage.json"


class CountingProvider:
    """Wraps a provider to record the tokens billed, per request and in total.

    Safe to share between threads: each thread sees the usage of its own last request.
    """

    def __init__(self, inner: VisionProvider) -> None:
        """Wrap ``inner``."""
        self.inner = inner
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = threading.Lock()
        self._local = threading.local()

    @property
    def name(self) -> str:
        """Name of the wrapped provider."""
        return self.inner.name

    @property
    def model(self) -> str:
        """Model of the wrapped provider."""
        return self.inner.model

    @property
    def max_side(self) -> int:
        """Image size of the wrapped provider."""
        return self.inner.max_side

    @property
    def box_format(self) -> BoxFormat:
        """Box format of the wrapped provider."""
        return self.inner.box_format

    def read(self, request: VisionRequest) -> VisionAnswer:
        """Read through the wrapped provider and add up the tokens."""
        answer = self.inner.read(request)
        spent = getattr(self._local, "usage", Usage())
        self._local.usage = Usage(
            spent.input_tokens + answer.usage.input_tokens,
            spent.output_tokens + answer.usage.output_tokens,
        )
        with self._lock:
            self.input_tokens += answer.usage.input_tokens
            self.output_tokens += answer.usage.output_tokens
        return answer

    def start_photo(self) -> None:
        """Start counting the calling thread's usage for a new photo."""
        self._local.usage = Usage()

    def photo_usage(self) -> Usage:
        """Usage of the calling thread's requests since :meth:`start_photo`."""
        usage: Usage = getattr(self._local, "usage", Usage())
        return usage


@dataclass(frozen=True)
class RunResult:
    """Outcomes of a run and the tokens billed for the photos it covers.

    Attributes:
        outcomes: One per photo, in the order given.
        input_tokens: Tokens billed for these photos, including earlier cached runs.
        output_tokens: Same, for output tokens.
        read_now: Photos sent to the provider during this run (not from the cache).
        wall_seconds: Wall-clock time of this run.
    """

    outcomes: list[ImageOutcome]
    input_tokens: int
    output_tokens: int
    read_now: int = 0
    wall_seconds: float = 0.0


def run_dataset(
    images: Sequence[LabeledImage],
    *,
    profile: Profile,
    provider: VisionProvider | None = None,
    local: LocalPipeline | None = None,
    fallback: VisionProvider | None = None,
    cache: Path,
    force: bool = False,
    workers: int = 1,
    progress: Callable[[int, int, ImageOutcome], None] | None = None,
) -> RunResult:
    """Read every photo, reusing results already in ``cache`` unless ``force`` is set.

    Give either an API ``provider``, or a ``local`` pipeline with an optional
    ``fallback`` provider; the tokens of whichever API is called are counted.

    ``workers`` photos are read at the same time. Each result is saved to
    ``cache/<photo>.json`` as soon as it is read, so an interrupted run resumes where it
    stopped. ``progress`` is called in the order of ``images``.
    """
    start = time.perf_counter()
    cache.mkdir(parents=True, exist_ok=True)
    usage_path = cache / USAGE_FILE
    usage: dict[str, list[int]] = (
        json.loads(usage_path.read_text(encoding="utf-8")) if usage_path.is_file() else {}
    )
    if (provider is None) == (local is None):
        raise ValueError("give either a provider or a local pipeline")
    api = provider if provider is not None else fallback
    counting = CountingProvider(api) if api is not None else None
    if local is not None:
        reader = Reader(profile, backend=local, fallback=counting)
    else:
        assert counting is not None
        reader = Reader(profile, backend=counting)

    def read_one(image: LabeledImage) -> tuple[ImageOutcome, Usage | None]:
        cached = cache / f"{image.path.stem}.json"
        if cached.is_file() and not force:
            return _outcome(image, json.loads(cached.read_text(encoding="utf-8"))), None
        if counting is not None:
            counting.start_photo()
        try:
            result = reader.read(image.path)
        except TagSortError as error:
            failed = ImageOutcome(image=image.path.name, session=image.session, error=str(error))
            return failed, None
        cached.write_text(result.to_json(indent=2), encoding="utf-8")
        spent = counting.photo_usage() if counting is not None else Usage()
        return _outcome(image, result.to_dict()), spent

    outcomes: list[ImageOutcome] = []
    read_now = 0
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        for index, (outcome, spent) in enumerate(pool.map(read_one, images), 1):
            if spent is not None:
                read_now += 1
                usage[outcome.image] = [spent.input_tokens, spent.output_tokens]
                usage_path.write_text(json.dumps(usage, indent=2), encoding="utf-8")
            outcomes.append(outcome)
            if progress is not None:
                progress(index, len(images), outcome)
    names = {image.path.name for image in images}
    return RunResult(
        outcomes=outcomes,
        input_tokens=sum(tokens[0] for name, tokens in usage.items() if name in names),
        output_tokens=sum(tokens[1] for name, tokens in usage.items() if name in names),
        read_now=read_now,
        wall_seconds=time.perf_counter() - start,
    )


def _outcome(image: LabeledImage, result: dict[str, Any]) -> ImageOutcome:
    tags = [_tag(data) for data in result["tags"]]
    pairs, invented = match_tags(tags, image.tags)
    seconds = sum(result.get("timings_ms", {}).values()) / 1000
    return ImageOutcome(
        image=image.path.name,
        session=image.session,
        pairs=pairs,
        invented=invented,
        seconds=seconds,
    )


def _tag(data: dict[str, Any]) -> Tag:
    return Tag(
        tag_id=data["tag_id"],
        text=data["text"],
        confidence=data["confidence"],
        status=data["status"],
        polygon=tuple(tuple(point) for point in data["polygon"]),  # type: ignore[arg-type]
        angle=data["angle"],
        candidates=tuple(Candidate(c["text"], c["confidence"]) for c in data["candidates"]),
        source=data["source"],
    )
