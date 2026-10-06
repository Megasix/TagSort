import re
from importlib.metadata import version

import tagsort


def test_version_is_semver() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", tagsort.__version__)


def test_version_matches_installed_metadata() -> None:
    assert tagsort.__version__ == version("tagsort")


def test_public_api() -> None:
    """Changing this list changes the public API: it needs maintainer approval."""
    assert tagsort.__all__ == [
        "Candidate",
        "ImageInfo",
        "PatternError",
        "Point",
        "Profile",
        "ProfileError",
        "ReadResult",
        "Tag",
        "TagSortError",
        "TagSource",
        "TagSpec",
        "TagStatus",
        "__version__",
    ]
    for name in tagsort.__all__:
        assert hasattr(tagsort, name)


def test_core_has_no_runtime_dependencies() -> None:
    """The core must stay importable with the standard library only until M4."""
    from importlib.metadata import requires

    assert not [r for r in requires("tagsort") or [] if "extra ==" not in r]
