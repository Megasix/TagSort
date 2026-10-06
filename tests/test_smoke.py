import re
import subprocess
import sys
from importlib.metadata import version

import pytest

import tagsort


def test_version_is_semver() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", tagsort.__version__)


def test_version_matches_installed_metadata() -> None:
    assert tagsort.__version__ == version("tagsort")


def test_public_api() -> None:
    """Changing this list changes the public API: it needs maintainer approval."""
    assert tagsort.__all__ == [
        "AnthropicProvider",
        "Candidate",
        "DeepSeekProvider",
        "GeminiProvider",
        "ImageError",
        "ImageInfo",
        "LocalPipeline",
        "ModelError",
        "OpenAIProvider",
        "PatternError",
        "Point",
        "Profile",
        "ProfileError",
        "ProviderError",
        "ProviderTag",
        "ReadResult",
        "Reader",
        "Tag",
        "TagSortError",
        "TagSource",
        "TagSpec",
        "TagStatus",
        "Usage",
        "VisionAnswer",
        "VisionProvider",
        "VisionRequest",
        "__version__",
        "available_models",
        "download_model",
    ]
    for name in tagsort.__all__:
        assert hasattr(tagsort, name)


def test_runtime_dependencies() -> None:
    """The core needs Pillow, numpy and ONNX Runtime only; everything else is an extra."""
    from importlib.metadata import requires

    names = sorted(r.split(">")[0] for r in requires("tagsort") or [] if "extra ==" not in r)
    assert names == ["numpy", "onnxruntime", "pillow"]


def test_core_does_not_import_httpx() -> None:
    code = "import sys, tagsort; tagsort.Reader; assert 'httpx' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_providers_explain_the_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(sys.modules):
        if name.startswith("tagsort.fallback.") and name != "tagsort.fallback.base":
            monkeypatch.delitem(sys.modules, name)
    # Another test may have left the class cached on the module; drop it so the lazy
    # import runs again.
    monkeypatch.delattr(tagsort, "OpenAIProvider", raising=False)
    monkeypatch.setitem(sys.modules, "httpx", None)
    with pytest.raises(ImportError, match=r"tagsort\[api\]"):
        tagsort.OpenAIProvider  # noqa: B018


def test_unknown_attribute() -> None:
    with pytest.raises(AttributeError, match="no attribute 'Nope'"):
        tagsort.Nope  # noqa: B018
