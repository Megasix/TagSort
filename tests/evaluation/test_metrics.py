from typing import Any

import pytest

from tagsort import Tag
from tagsort.evaluation.dataset import LabeledTag
from tagsort.evaluation.metrics import ImageOutcome, levenshtein, match_tags, summarize

BOX = ((0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0))


def tag(text: str | None, status: Any = "accepted", tag_id: str | None = "primary") -> Tag:
    if text is None:
        status = "unreadable"
    return Tag(
        tag_id=tag_id,
        text=text,
        confidence=0.95 if status == "accepted" else 0.3,
        status=status,
        polygon=BOX,
        angle=0,
        source="fallback",
    )


def label(text: str | None) -> LabeledTag:
    return LabeledTag(text=text)


@pytest.mark.parametrize(
    ("a", "b", "distance"),
    [
        ("", "", 0),
        ("abc", "", 3),
        ("", "ab", 2),
        ("GJ07966", "GJ07966", 0),
        ("GJ07966", "GJ07986", 1),
        ("kitten", "sitting", 3),
        ("0086", "9800", 4),
    ],
)
def test_levenshtein(a: str, b: str, distance: int) -> None:
    assert levenshtein(a, b) == distance
    assert levenshtein(b, a) == distance


def test_closest_texts_pair_first() -> None:
    predicted = [tag("GJ07986"), tag("GJ07966")]
    labeled = [label("GJ07966"), label("GJ07986")]
    pairs, invented = match_tags(predicted, labeled)
    assert [(lab.text, t.text if t else None) for lab, t in pairs] == [
        ("GJ07966", "GJ07966"),
        ("GJ07986", "GJ07986"),
    ]
    assert invented == ()


def test_distant_reading_is_invented_and_label_missed() -> None:
    pairs, invented = match_tags([tag("XY1")], [label("GJ07966")])
    assert pairs == ((label("GJ07966"), None),)
    assert [t.text for t in invented] == ["XY1"]


def test_unreadable_labels_and_predictions_pair_last() -> None:
    predicted = [tag(None), tag("1234")]
    labeled = [label(None), label("GJ07966")]
    pairs, invented = match_tags(predicted, labeled)
    assert [(lab.text, t.text if t else "none") for lab, t in pairs] == [
        (None, "1234"),
        ("GJ07966", None),
    ]
    assert invented == ()


def test_textless_prediction_pairs_with_unreadable_label() -> None:
    pairs, invented = match_tags([tag(None)], [label(None)])
    assert pairs[0][1] is not None
    assert invented == ()


def outcome(
    pairs: Any = (), invented: Any = (), session: str = "s1", **kwargs: Any
) -> ImageOutcome:
    return ImageOutcome(
        image="x.jpg", session=session, pairs=tuple(pairs), invented=tuple(invented), **kwargs
    )


def test_summary_counts_every_case() -> None:
    outcomes = [
        outcome([(label("GJ07966"), tag("GJ07966"))], seconds=4.0),  # exact, accepted
        outcome([(label("GJ07967"), tag("GJ07961"))], seconds=2.0),  # wrong, accepted: silent
        outcome([(label("0086"), tag("9800", "review"))]),  # wrong, review
        outcome([(label("GJ07970"), None)]),  # missed
        outcome(invented=[tag("GJ00499", "review")], session="s2"),  # invented, review
        outcome(invented=[tag("GJ00500")], session="s2"),  # invented, accepted: silent
        outcome([(label(None), tag("1504"))], session="s2"),  # "?" accepted: silent
        outcome(error="quota", session="s2"),  # failed photo
    ]
    m = summarize(outcomes)
    assert m.photos == 8
    assert m.failed_photos == 1
    assert m.readable_tags == 4
    assert m.unreadable_tags == 1
    assert m.predicted_tags == 6
    assert m.exact_matches == 1
    assert m.exact_match_rate == 0.25
    # Edits: 0 + 1 + 4 + 7 (missed) over 7 + 7 + 4 + 7 characters.
    assert m.character_error_rate == round(12 / 25, 4)
    assert m.accepted_tags == 4
    assert m.accepted_correct == 1
    assert m.automation_rate == 0.25
    assert m.silent_errors == 3
    assert m.silent_error_rate == 0.75
    assert m.review_tags == 2
    assert m.missed_tags == 1
    assert m.invented_tags == 2
    assert m.fallback_share == 1.0
    assert m.seconds_per_photo == round(6.0 / 7, 2)
    assert set(m.sessions) == {"s1", "s2"}
    assert m.sessions["s1"]["exact_match_rate"] == 0.25
    assert m.sessions["s2"]["silent_errors"] == 2


def test_empty_summary() -> None:
    m = summarize([])
    assert m.photos == 0
    assert m.exact_match_rate == 0
    assert m.seconds_per_photo == 0
    assert m.to_dict()["photos"] == 0
