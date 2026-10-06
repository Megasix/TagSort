"""Reading a dataset with a reader, with a cache so no photo is paid for twice."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tagsort.errors import TagSortError
from tagsort.evaluation.dataset import LabeledImage
from tagsort.evaluation.metrics import ImageOutcome, match_tags
from tagsort.fallback.base import BoxFormat, VisionAnswer, VisionProvider, VisionRequest
from tagsort.pipeline.reader import Reader
from tagsort.profile import Profile
from tagsort.types import Candidate, Tag

__all__ = ["CountingProvider", "RunResult", "run_dataset"]

logger = logging.getLogger("tagsort.evaluation")

USAGE_FILE = "_usage.json"


@dataclass
class CountingProvider:
    """Wraps a provider to record the tokens billed for each photo."""

    inner: VisionProvider
    input_tokens: int = 0
    output_tokens: int = 0

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
        self.input_tokens += answer.usage.input_tokens
        self.output_tokens += answer.usage.output_tokens
        return answer


@dataclass(frozen=True)
class RunResult:
    """Outcomes of a run and the tokens billed for the photos it covers."""

    outcomes: list[ImageOutcome]
    input_tokens: int
    output_tokens: int


def run_dataset(
    images: Sequence[LabeledImage],
    *,
    profile: Profile,
    provider: VisionProvider,
    cache: Path,
    force: bool = False,
    progress: Callable[[int, int, ImageOutcome], None] | None = None,
) -> RunResult:
    """Read every photo, reusing results already in ``cache`` unless ``force`` is set.

    Each result is saved to ``cache/<photo>.json`` as soon as it is read, so an
    interrupted run resumes where it stopped.
    """
    cache.mkdir(parents=True, exist_ok=True)
    usage_path = cache / USAGE_FILE
    usage: dict[str, list[int]] = (
        json.loads(usage_path.read_text(encoding="utf-8")) if usage_path.is_file() else {}
    )
    counting = CountingProvider(provider)
    reader = Reader(profile, backend=counting)
    outcomes: list[ImageOutcome] = []
    for index, image in enumerate(images, 1):
        cached = cache / f"{image.path.stem}.json"
        if cached.is_file() and not force:
            outcome = _outcome(image, json.loads(cached.read_text(encoding="utf-8")))
        else:
            before = (counting.input_tokens, counting.output_tokens)
            try:
                result = reader.read(image.path)
            except TagSortError as error:
                outcome = ImageOutcome(
                    image=image.path.name, session=image.session, error=str(error)
                )
            else:
                cached.write_text(result.to_json(indent=2), encoding="utf-8")
                usage[image.path.name] = [
                    counting.input_tokens - before[0],
                    counting.output_tokens - before[1],
                ]
                usage_path.write_text(json.dumps(usage, indent=2), encoding="utf-8")
                outcome = _outcome(image, result.to_dict())
        outcomes.append(outcome)
        if progress is not None:
            progress(index, len(images), outcome)
    names = {image.path.name for image in images}
    return RunResult(
        outcomes=outcomes,
        input_tokens=sum(tokens[0] for name, tokens in usage.items() if name in names),
        output_tokens=sum(tokens[1] for name, tokens in usage.items() if name in names),
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
