"""Scoring read results against labels.

Definitions are in ``docs/evaluation.md``. In short, predicted tags are paired with
labeled tags by text similarity (labels have no polygons yet), then every metric is
computed from those pairs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from tagsort.evaluation.dataset import LabeledTag
from tagsort.types import Tag

__all__ = ["ImageOutcome", "Metrics", "levenshtein", "match_tags", "summarize"]

MAX_PAIRING_CER = 0.5
"""A prediction farther than this from a label is not a reading of that tag."""


def levenshtein(a: str, b: str) -> int:
    """Return the minimum number of single-character edits turning ``a`` into ``b``."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b))
            )
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class ImageOutcome:
    """How one photo was read.

    Attributes:
        image: Photo file name.
        session: Shooting session of the photo.
        pairs: Each labeled tag with the prediction paired to it, if any.
        invented: Predictions paired with no labeled tag.
        seconds: Wall-clock time spent reading the photo.
        error: Why the photo could not be read, or ``None``.
    """

    image: str
    session: str
    pairs: tuple[tuple[LabeledTag, Tag | None], ...] = ()
    invented: tuple[Tag, ...] = ()
    seconds: float = 0.0
    error: str | None = None


def match_tags(
    predicted: Sequence[Tag], labeled: Sequence[LabeledTag]
) -> tuple[tuple[tuple[LabeledTag, Tag | None], ...], tuple[Tag, ...]]:
    """Pair predictions with labels; return the pairs and the unpaired predictions.

    Readable labels are paired first, closest texts first, as long as the character
    error rate stays at or below :data:`MAX_PAIRING_CER`. Remaining predictions then
    pair with labels that no person can read, then predictions without text pair with
    any label left.
    """
    free_predictions = list(range(len(predicted)))
    free_labels = list(range(len(labeled)))
    pairs: dict[int, int] = {}

    candidates = []
    for li in free_labels:
        truth = labeled[li].text
        if truth is None:
            continue
        for pi in free_predictions:
            text = predicted[pi].text
            if text is None:
                continue
            cer = levenshtein(text, truth) / max(len(truth), 1)
            if cer <= MAX_PAIRING_CER:
                candidates.append((cer, li, pi))
    for _, li, pi in sorted(candidates):
        if li in free_labels and pi in free_predictions:
            pairs[li] = pi
            free_labels.remove(li)
            free_predictions.remove(pi)

    def pair_in_order(label_unreadable: bool, prediction_has_text: bool) -> None:
        for li in [i for i in free_labels if (labeled[i].text is None) == label_unreadable]:
            for pi in free_predictions:
                if (predicted[pi].text is not None) == prediction_has_text:
                    pairs[li] = pi
                    free_labels.remove(li)
                    free_predictions.remove(pi)
                    break

    pair_in_order(label_unreadable=True, prediction_has_text=True)  # "?" labels, read anyway
    pair_in_order(label_unreadable=False, prediction_has_text=False)  # readable, engine could not
    pair_in_order(label_unreadable=True, prediction_has_text=False)  # "?" labels, unreadable

    return (
        tuple(
            (label, predicted[pairs[i]] if i in pairs else None) for i, label in enumerate(labeled)
        ),
        tuple(predicted[i] for i in free_predictions),
    )


@dataclass(frozen=True)
class Metrics:
    """Aggregate scores; see ``docs/evaluation.md`` for every definition."""

    photos: int
    failed_photos: int
    readable_tags: int
    unreadable_tags: int
    predicted_tags: int
    exact_matches: int
    exact_match_rate: float
    character_error_rate: float
    accepted_tags: int
    accepted_correct: int
    automation_rate: float
    silent_errors: int
    silent_error_rate: float
    review_tags: int
    unreadable_predictions: int
    missed_tags: int
    invented_tags: int
    fallback_share: float
    seconds_per_photo: float
    sessions: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return the metrics as a JSON-ready dictionary."""
        return dict(vars(self))


def summarize(outcomes: Iterable[ImageOutcome], *, by_session: bool = True) -> Metrics:
    """Compute the metrics of a set of photos."""
    outcomes = list(outcomes)
    readable = unreadable = predicted = exact = accepted = accepted_ok = silent = 0
    review = unreadable_pred = missed = invented = fallback = 0
    edits = chars = 0
    seconds = 0.0
    read_photos = [o for o in outcomes if o.error is None]
    for outcome in read_photos:
        seconds += outcome.seconds
        tags: list[Tag] = [t for _, t in outcome.pairs if t is not None] + list(outcome.invented)
        predicted += len(tags)
        invented += len(outcome.invented)
        for tag in tags:
            accepted += tag.status == "accepted"
            review += tag.status == "review"
            unreadable_pred += tag.status == "unreadable"
            fallback += tag.source == "fallback"
        for label, paired in outcome.pairs:
            if label.text is None:
                unreadable += 1
                if paired is not None and paired.status == "accepted":
                    silent += 1  # accepted a text no person can check
                continue
            readable += 1
            chars += len(label.text)
            if paired is None:
                missed += 1
                edits += len(label.text)
                continue
            correct = paired.text == label.text
            exact += correct
            edits += levenshtein(paired.text or "", label.text)
            if paired.status == "accepted":
                accepted_ok += correct
                silent += not correct
        silent += sum(tag.status == "accepted" for tag in outcome.invented)

    sessions: dict[str, dict[str, float]] = {}
    if by_session:
        for session in sorted({o.session for o in outcomes}):
            part = summarize([o for o in outcomes if o.session == session], by_session=False)
            sessions[session] = {
                "photos": part.photos,
                "readable_tags": part.readable_tags,
                "exact_match_rate": part.exact_match_rate,
                "silent_errors": part.silent_errors,
            }

    return Metrics(
        photos=len(outcomes),
        failed_photos=len(outcomes) - len(read_photos),
        readable_tags=readable,
        unreadable_tags=unreadable,
        predicted_tags=predicted,
        exact_matches=exact,
        exact_match_rate=_ratio(exact, readable),
        character_error_rate=_ratio(edits, chars),
        accepted_tags=accepted,
        accepted_correct=accepted_ok,
        automation_rate=_ratio(accepted_ok, readable),
        silent_errors=silent,
        silent_error_rate=_ratio(silent, accepted),
        review_tags=review,
        unreadable_predictions=unreadable_pred,
        missed_tags=missed,
        invented_tags=invented,
        fallback_share=_ratio(fallback, predicted),
        seconds_per_photo=round(seconds / len(read_photos), 2) if read_photos else 0.0,
        sessions=sessions,
    )


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0
