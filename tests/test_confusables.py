from tagsort.pipeline.confusables import other_kind_readings
from tagsort.profile import Profile


def profile() -> Profile:
    return Profile.from_dict(
        {
            "schema_version": "1.0",
            "name": "P",
            "tags_per_individual": 1,
            "tags": [
                {"id": "FieldTag", "pattern": "[A-Z]{2}\\d{5}"},
                {"id": "FWS", "pattern": "FWS\\d{4}"},
                {"id": "GJ", "pattern": "GJ\\d{5}"},
            ],
        }
    )


def test_a_reading_one_swap_from_another_kind_is_ambiguous() -> None:
    assert other_kind_readings(profile(), "FW54019", "FieldTag") == ["FWS4019"]
    assert other_kind_readings(profile(), "FWS4019", "FWS") == ["FW54019"]


def test_swaps_within_the_same_kind_are_ordinary_alternatives() -> None:
    # GJ02948 → GJO2948 fits nothing; GJ0294B neither: no other kind is reachable.
    assert other_kind_readings(profile(), "GJ02948", "GJ") == []
