import pytest
from jsonschema import Draft202012Validator

from tests.helpers import load_schema


@pytest.fixture(scope="session")
def profile_validator() -> Draft202012Validator:
    return Draft202012Validator(load_schema("profile.v1.json"))


@pytest.fixture(scope="session")
def result_validator() -> Draft202012Validator:
    return Draft202012Validator(load_schema("result.v1.json"))
