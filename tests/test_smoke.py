import re
from importlib.metadata import version

import tagsort


def test_version_is_semver() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", tagsort.__version__)


def test_version_matches_installed_metadata() -> None:
    assert tagsort.__version__ == version("tagsort")


def test_public_api_is_version_only() -> None:
    assert tagsort.__all__ == ["__version__"]
