import hashlib
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

import tagsort
from tagsort import ModelError, available_models, download_model
from tagsort.cli import main
from tagsort.models import load_manifest, model_files, models_dir


def test_known_models_smallest_first() -> None:
    assert available_models() == ["ppocrv6-tiny", "ppocrv6-small", "ppocrv6-medium"]
    assert tagsort.models.DEFAULT_MODEL == "ppocrv6-tiny"


@pytest.mark.parametrize("name", ["ppocrv6-tiny", "ppocrv6-small", "ppocrv6-medium"])
def test_manifests_are_complete_and_pinned(name: str) -> None:
    manifest = load_manifest(name)
    assert manifest.license == "Apache-2.0"
    for spec in manifest.files.values():
        assert re.fullmatch(
            r"https://huggingface\.co/PaddlePaddle/.+/resolve/[0-9a-f]{40}/inference\.onnx",
            spec.url,
        )
        assert re.fullmatch(r"[0-9a-f]{64}", spec.sha256)
        assert spec.size > 1_000_000
    rec = manifest.recognition
    assert rec.decoder == "ctc"
    assert rec.input_shape[1] == 48
    assert rec.classes[0] == ""
    assert rec.classes[-1] == " "
    assert len(rec.classes) == len(rec.charset) + 2
    for char in "GJ0123456789ABCDEFXYZ-":
        assert char in rec.charset, char
    assert 0 < manifest.detection.threshold < manifest.detection.box_threshold < 1


def test_unknown_model() -> None:
    with pytest.raises(ModelError, match="unknown model 'nope'; known models: ppocrv6-medium"):
        load_manifest("nope")


def test_models_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TAGSORT_MODELS", str(tmp_path / "m"))
    assert models_dir() == tmp_path / "m"
    monkeypatch.delenv("TAGSORT_MODELS")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "c"))
    assert models_dir() == tmp_path / "c" / "tagsort" / "models"
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert models_dir() == Path.home() / ".cache" / "tagsort" / "models"


class FakeServer:
    """Serves fake model files and lets the manifest checksums match them."""

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, *, corrupt: bool = False, status: int = 200
    ) -> None:
        self.requests: list[str] = []
        self.payload = {"det": b"d" * 3000, "rec": b"r" * 5000}
        manifest = load_manifest("ppocrv6-tiny")
        self.urls = {spec.url: kind for kind, spec in manifest.files.items()}
        files = {
            kind: tagsort.models.FileSpec(
                spec.url, hashlib.sha256(self.payload[kind]).hexdigest(), len(self.payload[kind])
            )
            for kind, spec in manifest.files.items()
        }
        patched = tagsort.models.Manifest(**{**vars(manifest), "files": files})
        monkeypatch.setattr(tagsort.models, "load_manifest", lambda name: patched)
        real_client = httpx.Client

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(str(request.url))
            kind = self.urls[str(request.url)]
            body = self.payload[kind] + (b"x" if corrupt else b"")
            return httpx.Response(status, content=body)

        def client(**kwargs: Any) -> httpx.Client:
            return real_client(transport=httpx.MockTransport(handler))

        monkeypatch.setattr(httpx, "Client", client)


def test_download_verifies_and_keeps_good_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server = FakeServer(monkeypatch)
    seen: list[tuple[str, int, int]] = []
    paths = download_model(directory=tmp_path, progress=lambda *a: seen.append(a))
    assert set(paths) == {"det", "rec"}
    assert paths["det"].read_bytes() == server.payload["det"]
    assert seen[-1] == ("rec", 5000, 5000)
    assert model_files("ppocrv6-tiny", directory=tmp_path) == paths
    download_model(directory=tmp_path)
    assert len(server.requests) == 2, "verified files are not downloaded again"


def test_download_rejects_a_wrong_checksum(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FakeServer(monkeypatch, corrupt=True)
    with pytest.raises(ModelError, match="does not match its manifest checksum"):
        download_model(directory=tmp_path)
    assert not list((tmp_path / "ppocrv6-tiny").iterdir())


def test_download_reports_http_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FakeServer(monkeypatch, status=404)
    with pytest.raises(ModelError, match="HTTP 404"):
        download_model(directory=tmp_path)


def test_download_needs_httpx(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "httpx", None)
    with pytest.raises(ImportError, match=r"tagsort\[api\]"):
        download_model(directory=tmp_path)


def test_model_files_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="tagsort models download ppocrv6-tiny"):
        model_files("ppocrv6-tiny", directory=tmp_path)
    FakeServer(monkeypatch)
    paths = download_model(directory=tmp_path)
    paths["rec"].write_bytes(b"short")
    with pytest.raises(ModelError, match="wrong size"):
        model_files("ppocrv6-tiny", directory=tmp_path)


def test_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TAGSORT_MODELS", str(tmp_path))
    assert main(["models", "path"]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path)
    assert main(["models", "list"]) == 0
    out = capsys.readouterr().out
    assert "ppocrv6-tiny" in out
    assert "Apache-2.0" in out
    assert "not downloaded" in out
    assert main(["models", "download", "nope"]) == 1
    assert "unknown model" in capsys.readouterr().err
    FakeServer(monkeypatch)
    assert main(["models", "download"]) == 0
    assert "checksum verified" in capsys.readouterr().out
    assert main(["models", "list"]) == 0
    assert "downloaded" in capsys.readouterr().out
