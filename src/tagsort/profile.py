"""Profiles: what the tags of a collection may say.

A profile follows ``schemas/profile.v1.json``. Loading checks everything the schema
checks, plus the rules it cannot express (unique tag ids, valid patterns), so an invalid
profile is rejected before any image is read.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from tagsort.errors import PatternError, ProfileError
from tagsort.grammar import Automaton, compile_pattern

__all__ = ["Profile", "TagSpec"]

_TAG_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}")
_PROFILE_KEYS = ("schema_version", "name", "tags_per_individual", "tags")
_TAG_KEYS = ("id", "pattern")


@dataclass(frozen=True)
class TagSpec:
    """One kind of tag in a profile.

    Attributes:
        id: Identifier of this kind of tag, reported as ``tag_id`` in results. Starts with
            a letter, then up to 63 letters, digits, ``_`` or ``-``.
        pattern: Every text this kind of tag may carry, in the regex subset described in
            ``docs/patterns.md``. Always matched against the whole text.

    Raises:
        PatternError: If ``pattern`` is outside the supported subset.
        ProfileError: If ``id`` is malformed.
    """

    id: str
    pattern: str
    _automaton: Automaton = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the id and compile the pattern."""
        if not isinstance(self.id, str) or not _TAG_ID.fullmatch(self.id):
            raise ProfileError(
                f"tag id {self.id!r} must start with a letter, then up to 63 letters, "
                "digits, '_' or '-'"
            )
        if not isinstance(self.pattern, str):
            raise ProfileError(f"pattern must be a string, not {type(self.pattern).__name__}")
        object.__setattr__(self, "_automaton", compile_pattern(self.pattern))

    def matches(self, text: str) -> bool:
        """Return whether ``text`` as a whole matches this tag's pattern."""
        return self._automaton.matches(text)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready form of this tag, as it appears in a profile."""
        return {"id": self.id, "pattern": self.pattern}


@dataclass(frozen=True)
class Profile:
    r"""The tags of one collection: how many each specimen carries and what they say.

    Build it from a JSON document that follows ``schemas/profile.v1.json``:

    >>> profile = Profile.from_dict(
    ...     {
    ...         "schema_version": "1.0",
    ...         "name": "Museum A",
    ...         "tags_per_individual": 2,
    ...         "tags": [
    ...             {"id": "primary", "pattern": "MD\\d{5}"},
    ...             {"id": "secondary", "pattern": "[A-Z]{2}-\\d{3,4}"},
    ...         ],
    ...     }
    ... )
    >>> profile.match("MD04127")
    'primary'
    >>> profile.match("MD4127") is None
    True

    Attributes:
        name: Human-readable name of the profile.
        tags_per_individual: Number of tags attached to one specimen.
        tags: The kinds of tag found in this collection, with unique ids.

    Raises:
        ProfileError: If a field is invalid. :class:`PatternError`, a subclass, is raised
            for an invalid pattern.
    """

    SCHEMA_VERSION: ClassVar[str] = "1.0"
    """Version of ``profile.v1.json`` this class reads and writes."""

    name: str
    tags_per_individual: int
    tags: tuple[TagSpec, ...]

    def __post_init__(self) -> None:
        """Validate the fields and normalize ``tags`` to a tuple."""
        if not isinstance(self.name, str) or not self.name:
            raise ProfileError("must be a non-empty string", location="name")
        count = self.tags_per_individual
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ProfileError("must be an integer of at least 1", location="tags_per_individual")
        tags = tuple(self.tags)
        if not tags:
            raise ProfileError("must list at least one tag", location="tags")
        seen: set[str] = set()
        for index, tag in enumerate(tags):
            if not isinstance(tag, TagSpec):
                raise ProfileError("must be a TagSpec", location=f"tags[{index}]")
            if tag.id in seen:
                raise ProfileError(f"duplicate tag id {tag.id!r}", location=f"tags[{index}].id")
            seen.add(tag.id)
        object.__setattr__(self, "tags", tags)

    def tag(self, tag_id: str) -> TagSpec:
        """Return the tag with id ``tag_id``.

        Raises:
            KeyError: If the profile has no such tag.
        """
        for tag in self.tags:
            if tag.id == tag_id:
                return tag
        raise KeyError(tag_id)

    def match(self, text: str) -> str | None:
        """Return the id of the first tag whose pattern matches ``text``, or ``None``."""
        for tag in self.tags:
            if tag.matches(text):
                return tag.id
        return None

    # Loading and saving

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Profile:
        """Build a profile from a parsed JSON document.

        Raises:
            ProfileError: If the document does not conform to ``profile.v1.json`` or a
                pattern is invalid. The error's ``location`` says where.
        """
        if not isinstance(data, Mapping):
            raise ProfileError(f"profile must be a JSON object, not {_json_type(data)}")
        _check_keys(data, _PROFILE_KEYS, location=None)
        version = data["schema_version"]
        if version != cls.SCHEMA_VERSION or not isinstance(version, str):
            raise ProfileError(
                f"unsupported version {version!r}; this version of TagSort reads "
                f"{cls.SCHEMA_VERSION!r}",
                location="schema_version",
            )
        for key, expected, description in (
            ("name", str, "a string"),
            ("tags_per_individual", int, "an integer"),
            ("tags", list | tuple, "an array"),
        ):
            value = data[key]
            if not isinstance(value, expected) or isinstance(value, bool):
                raise ProfileError(f"must be {description}, not {_json_type(value)}", location=key)
        return cls(
            name=data["name"],
            tags_per_individual=data["tags_per_individual"],
            tags=tuple(_tag_from_dict(tag, index) for index, tag in enumerate(data["tags"])),
        )

    @classmethod
    def from_json(cls, text: str | bytes) -> Profile:
        """Build a profile from a JSON document.

        Raises:
            ProfileError: If the text is not valid JSON, has duplicate keys, or does not
                describe a valid profile.
        """
        try:
            data = json.loads(
                text, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant
            )
        except json.JSONDecodeError as error:
            raise ProfileError(
                f"invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}"
            ) from None
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> Profile:
        """Read a profile from a UTF-8 JSON file.

        Raises:
            OSError: If the file cannot be read.
            ProfileError: If the file does not describe a valid profile.
        """
        try:
            text = Path(path).read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ProfileError(f"{os.fspath(path)} is not valid UTF-8: {error.reason}") from None
        return cls.from_json(text)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready form of this profile, conforming to ``profile.v1.json``."""
        return {
            "schema_version": self.SCHEMA_VERSION,
            "name": self.name,
            "tags_per_individual": self.tags_per_individual,
            "tags": [tag.to_dict() for tag in self.tags],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return this profile as a JSON document conforming to ``profile.v1.json``."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def _json_type(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, int):
        return "an integer"
    if isinstance(value, float):
        return "a number"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, Mapping):
        return "an object"
    if isinstance(value, list | tuple):
        return "an array"
    return type(value).__name__


def _check_keys(data: Mapping[str, Any], allowed: Iterable[str], location: str | None) -> None:
    allowed = tuple(allowed)
    prefix = f"{location}." if location else ""
    for key in data:
        if key not in allowed:
            raise ProfileError(
                f"unknown property {key!r}; expected only {', '.join(allowed)}", location=location
            )
    for key in allowed:
        if key not in data:
            raise ProfileError("required property is missing", location=f"{prefix}{key}")


def _tag_from_dict(data: object, index: int) -> TagSpec:
    location = f"tags[{index}]"
    if not isinstance(data, Mapping):
        raise ProfileError(f"must be an object, not {_json_type(data)}", location=location)
    _check_keys(data, _TAG_KEYS, location=location)
    for key in _TAG_KEYS:
        if not isinstance(data[key], str):
            raise ProfileError(
                f"must be a string, not {_json_type(data[key])}", location=f"{location}.{key}"
            )
    try:
        return TagSpec(id=data["id"], pattern=data["pattern"])
    except PatternError as error:
        raise error.at(f"{location}.pattern") from None
    except ProfileError as error:
        raise ProfileError(error.message, location=f"{location}.id") from None


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProfileError(f"duplicate key {key!r} in a JSON object")
        result[key] = value
    return result


def _reject_constant(name: str) -> None:
    raise ProfileError(f"{name} is not valid JSON")
