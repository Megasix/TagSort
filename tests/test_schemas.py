import json

import pytest
from jsonschema import Draft202012Validator

from tests.helpers import ROOT, load_schema

SCHEMAS = sorted(path.name for path in (ROOT / "schemas").glob("*.json"))


def test_both_schemas_exist() -> None:
    assert SCHEMAS == ["profile.v1.json", "result.v1.json"]


@pytest.mark.parametrize("name", SCHEMAS)
def test_schema_is_valid_draft_2020_12(name: str) -> None:
    Draft202012Validator.check_schema(load_schema(name))


@pytest.mark.parametrize("name", SCHEMAS)
def test_schema_metadata(name: str) -> None:
    schema = load_schema(name)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"].endswith(f"/schemas/{name}")
    assert schema["properties"]["schema_version"] == {
        "description": "Version of this schema the "
        + ("profile" if name.startswith("profile") else "result")
        + " follows.",
        "const": "1.0",
    }


@pytest.mark.parametrize("name", SCHEMAS)
def test_every_object_is_closed(name: str) -> None:
    """Unknown properties are rejected everywhere, so typos never pass silently."""

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                assert node.get("additionalProperties") is False, node
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(load_schema(name))


def test_claude_md_examples_conform() -> None:
    """The examples in CLAUDE.md §9 are valid documents."""
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    blocks = [block.split("```", 1)[0] for block in text.split("```json\n")[1:]]
    profile, result = (json.loads(block) for block in blocks[:2])
    Draft202012Validator(load_schema("profile.v1.json")).validate(profile)
    Draft202012Validator(load_schema("result.v1.json")).validate(result)
