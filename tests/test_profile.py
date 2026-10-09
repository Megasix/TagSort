import copy
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from tagsort import PatternError, Profile, ProfileError, TagSpec
from tests.helpers import DATA


def museum_a() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((DATA / "museum_a.json").read_text(encoding="utf-8"))
    return data


def with_change(path: tuple[str | int, ...], value: Any) -> dict[str, Any]:
    """museum_a with the value at ``path`` replaced, or deleted when value is DELETE."""
    data = museum_a()
    node: Any = data
    for key in path[:-1]:
        node = node[key]
    if value is DELETE:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return data


DELETE = object()

# Documents that both the schema and the loader must reject, with the location the loader
# reports.
SCHEMA_INVALID: list[tuple[str, Any, str | None]] = [
    ("not an object", [], None),
    ("missing version", with_change(("schema_version",), DELETE), "schema_version"),
    ("missing name", with_change(("name",), DELETE), "name"),
    ("missing count", with_change(("tags_per_individual",), DELETE), "tags_per_individual"),
    ("missing tags", with_change(("tags",), DELETE), "tags"),
    ("unknown key", with_change(("colour",), "red"), None),
    ("future version", with_change(("schema_version",), "2.0"), "schema_version"),
    ("numeric version", with_change(("schema_version",), 1.0), "schema_version"),
    ("empty name", with_change(("name",), ""), "name"),
    ("numeric name", with_change(("name",), 3), "name"),
    ("zero count", with_change(("tags_per_individual",), 0), "tags_per_individual"),
    ("negative count", with_change(("tags_per_individual",), -1), "tags_per_individual"),
    ("float count", with_change(("tags_per_individual",), 2.5), "tags_per_individual"),
    ("boolean count", with_change(("tags_per_individual",), True), "tags_per_individual"),
    ("string count", with_change(("tags_per_individual",), "2"), "tags_per_individual"),
    ("tags not a list", with_change(("tags",), {"id": "a"}), "tags"),
    ("tag not an object", with_change(("tags", 1), "MD"), "tags[1]"),
    ("tag missing id", with_change(("tags", 0, "id"), DELETE), "tags[0].id"),
    ("tag missing pattern", with_change(("tags", 1, "pattern"), DELETE), "tags[1].pattern"),
    ("tag unknown key", with_change(("tags", 0, "label"), "x"), "tags[0]"),
    ("numeric id", with_change(("tags", 0, "id"), 1), "tags[0].id"),
    ("empty id", with_change(("tags", 0, "id"), ""), "tags[0].id"),
    ("id starts with digit", with_change(("tags", 0, "id"), "1st"), "tags[0].id"),
    ("id with space", with_change(("tags", 0, "id"), "first tag"), "tags[0].id"),
    ("id with accent", with_change(("tags", 0, "id"), "étiquette"), "tags[0].id"),
    ("id too long", with_change(("tags", 0, "id"), "a" * 65), "tags[0].id"),
    ("null pattern", with_change(("tags", 0, "pattern"), None), "tags[0].pattern"),
    ("empty pattern", with_change(("tags", 0, "pattern"), ""), "tags[0].pattern"),
]

# Documents the schema accepts.
SCHEMA_VALID: list[tuple[str, Any]] = [
    ("museum_a", museum_a()),
    ("one tag", with_change(("tags",), [{"id": "only", "pattern": "[A-Z]{3}"}])),
    # A collection whose kinds of tag are not known yet: read with open reading.
    ("no tags yet", with_change(("tags",), [])),
    ("id at max length", with_change(("tags", 0, "id"), "a" + "_-9" * 21)),
    ("unicode name", with_change(("name",), "Muséum d'histoire naturelle")),
    ("large count", with_change(("tags_per_individual",), 12)),
]

# Documents the schema accepts but the loader rejects: the rules JSON Schema cannot express.
SEMANTIC_INVALID: list[tuple[str, Any, str]] = [
    ("duplicate ids", with_change(("tags", 1, "id"), "primary"), "tags[1].id"),
    ("unbounded pattern", with_change(("tags", 0, "pattern"), "MD\\d+"), "tags[0].pattern"),
    ("anchored pattern", with_change(("tags", 1, "pattern"), "^AB$"), "tags[1].pattern"),
]


@pytest.mark.parametrize(("case", "document", "location"), SCHEMA_INVALID)
def test_schema_and_loader_both_reject(
    profile_validator: Draft202012Validator, case: str, document: Any, location: str | None
) -> None:
    assert not profile_validator.is_valid(document), case
    with pytest.raises(ProfileError) as info:
        Profile.from_dict(document)
    assert info.value.location == location


@pytest.mark.parametrize(("case", "document"), SCHEMA_VALID)
def test_schema_and_loader_both_accept(
    profile_validator: Draft202012Validator, case: str, document: Any
) -> None:
    profile_validator.validate(document)
    profile = Profile.from_dict(document)
    assert profile.to_dict() == document, case


@pytest.mark.parametrize(("case", "document", "location"), SEMANTIC_INVALID)
def test_loader_rejects_what_schema_cannot_express(
    profile_validator: Draft202012Validator, case: str, document: Any, location: str
) -> None:
    profile_validator.validate(document)
    with pytest.raises(ProfileError) as info:
        Profile.from_dict(document)
    assert info.value.location == location, case


def test_from_file() -> None:
    profile = Profile.from_file(DATA / "museum_a.json")
    assert profile.name == "Museum A"
    assert profile.tags_per_individual == 2
    assert profile.tags == (
        TagSpec(id="primary", pattern=r"MD\d{5}"),
        TagSpec(id="secondary", pattern=r"[A-Z]{2}-\d{3,4}"),
    )


def test_from_file_accepts_str_path() -> None:
    assert Profile.from_file(str(DATA / "museum_a.json")).name == "Museum A"


def test_from_file_missing() -> None:
    with pytest.raises(FileNotFoundError):
        Profile.from_file(DATA / "does_not_exist.json")


def test_from_file_not_utf8(tmp_path: Path) -> None:
    path = tmp_path / "latin1.json"
    path.write_bytes('{"name": "Muséum"}'.encode("latin-1"))
    with pytest.raises(ProfileError, match="not valid UTF-8"):
        Profile.from_file(path)


def test_json_round_trip() -> None:
    profile = Profile.from_dict(museum_a())
    assert Profile.from_json(profile.to_json()) == profile
    assert json.loads(profile.to_json(indent=None)) == museum_a()


def test_from_json_accepts_bytes() -> None:
    raw = (DATA / "museum_a.json").read_bytes()
    assert Profile.from_json(raw) == Profile.from_dict(museum_a())


def test_invalid_json_reports_line_and_column() -> None:
    # The exact position differs between Python versions; it must be reported.
    with pytest.raises(ProfileError, match=r"invalid JSON at line \d+, column \d+"):
        Profile.from_json('{"name": "A",\n}')


def test_duplicate_json_keys_are_rejected() -> None:
    """json.loads would silently keep the last value."""
    text = (DATA / "museum_a.json").read_text(encoding="utf-8")
    text = text.replace('"name": "Museum A",', '"name": "Museum A",\n  "name": "Museum B",')
    with pytest.raises(ProfileError, match="duplicate key 'name'"):
        Profile.from_json(text)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_standard_json_constants_are_rejected(constant: str) -> None:
    text = json.dumps(museum_a()).replace('"tags_per_individual": 2', f'"x": {constant}')
    with pytest.raises(ProfileError, match="not valid JSON"):
        Profile.from_json(text)


def test_pattern_error_is_located() -> None:
    document = with_change(("tags", 1, "pattern"), "[A-Z]{2}-\\d+")
    with pytest.raises(PatternError) as info:
        Profile.from_dict(document)
    error = info.value
    assert error.location == "tags[1].pattern"
    assert error.code == "unbounded_quantifier"
    assert error.position == 11
    assert str(error).startswith("tags[1].pattern: quantifier '+' is not supported")


def test_error_message_names_the_location() -> None:
    with pytest.raises(ProfileError) as info:
        Profile.from_dict(with_change(("tags", 0, "colour"), "red"))
    assert (
        str(info.value) == "tags[0]: unknown property 'colour'; expected only id, pattern, header"
    )


def test_match_and_tag_lookup() -> None:
    profile = Profile.from_dict(museum_a())
    assert profile.match("MD04127") == "primary"
    assert profile.match("AB-1234") == "secondary"
    assert profile.match("AB-12") is None
    assert profile.tag("secondary").pattern == r"[A-Z]{2}-\d{3,4}"
    with pytest.raises(KeyError):
        profile.tag("tertiary")


def test_match_returns_the_first_matching_tag() -> None:
    profile = Profile(
        name="Overlap",
        tags_per_individual=1,
        tags=(TagSpec("narrow", r"A\d"), TagSpec("wide", r"[A-Z]\d")),
    )
    assert profile.match("A1") == "narrow"
    assert profile.match("B1") == "wide"


def test_direct_construction_validates() -> None:
    with pytest.raises(ProfileError, match="duplicate tag id 'a'"):
        Profile(name="X", tags_per_individual=1, tags=(TagSpec("a", "x"), TagSpec("a", "y")))
    with pytest.raises(ProfileError) as info:
        Profile(name="X", tags_per_individual=1, tags=("a",))  # type: ignore[arg-type]
    assert info.value.location == "tags[0]"
    with pytest.raises(PatternError):
        TagSpec("a", "x*")
    with pytest.raises(ProfileError, match="must be a string"):
        TagSpec("a", 3)  # type: ignore[arg-type]


def test_tags_list_is_normalized_to_tuple() -> None:
    profile = Profile(name="X", tags_per_individual=1, tags=[TagSpec("a", "x")])  # type: ignore[arg-type]
    assert profile.tags == (TagSpec("a", "x"),)


def test_profile_is_immutable() -> None:
    profile = Profile.from_dict(museum_a())
    with pytest.raises(AttributeError):
        profile.name = "Other"  # type: ignore[misc]


def test_from_dict_does_not_modify_its_input() -> None:
    document = museum_a()
    snapshot = copy.deepcopy(document)
    Profile.from_dict(document)
    assert document == snapshot


def test_non_json_python_value_is_named() -> None:
    with pytest.raises(ProfileError, match="must be a string, not bytes"):
        Profile.from_dict(with_change(("name",), b"Museum A"))


def test_readme_example_is_museum_a() -> None:
    readme = (DATA.parents[1] / "README.md").read_text(encoding="utf-8")
    blocks = [b.split("```", 1)[0] for b in readme.split("```json\n")[1:]]
    profiles = [json.loads(b) for b in blocks if '"Museum A"' in b]
    assert profiles == [museum_a()]
