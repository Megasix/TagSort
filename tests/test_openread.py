"""Open reading: generic cues for how likely a line is a specimen tag."""

import pytest

from tagsort.pipeline.openread import tag_likelihood


@pytest.mark.parametrize(
    "text",
    ["15950-1", "GJ07966", "NMC16368", "MDX-4848", "12/345", "973213", "FWS 4019"],
)
def test_specimen_numbers_are_likely_tags(text: str) -> None:
    score, kind = tag_likelihood(text)
    assert kind == "id"
    assert score >= 0.65


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("NATIONAL MUSEUM OF CANADA", "header"),
        ("Herbarium", "header"),
        ("1 2 3 4 5 6", "noise"),
        ("x", "noise"),
        ("cm", "noise"),
        ("Collected by J. Smith on the shore 1921", "noise"),
    ],
)
def test_headers_rulers_and_sentences_are_not(text: str, kind: str) -> None:
    score, guessed = tag_likelihood(text)
    assert guessed == kind
    assert score < 0.5


def test_more_specimen_like_texts_score_higher() -> None:
    assert tag_likelihood("GJ07966")[0] > tag_likelihood("1921")[0] > tag_likelihood("A1")[0]
