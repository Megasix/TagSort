import logging
import os
from pathlib import Path
from typing import Any

import pytest

from tagsort.models import ModelError
from tagsort.pipeline import runtime
from tagsort.pipeline.runtime import create_session, providers_for, usable_cpus

CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"


def test_cpu_never_asks_for_cuda() -> None:
    assert providers_for("cpu", [CUDA, CPU]) == [CPU]


def test_cuda_and_auto_prefer_the_gpu() -> None:
    assert providers_for("cuda", [CUDA, CPU]) == [CUDA, CPU]
    assert providers_for("auto", [CUDA, CPU]) == [CUDA, CPU]


def test_auto_falls_back_to_the_cpu_and_cuda_refuses() -> None:
    assert providers_for("auto", [CPU]) == [CPU]
    with pytest.raises(ModelError, match="onnxruntime-gpu"):
        providers_for("cuda", [CPU])


def test_unknown_device_is_refused() -> None:
    with pytest.raises(ValueError, match="device must be one of"):
        providers_for("tpu", [CPU])  # type: ignore[arg-type]


def test_installed_providers_are_read_when_not_given() -> None:
    assert providers_for("auto") in ([CPU], [CUDA, CPU])


def cgroup(tmp_path: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    return tmp_path


def test_cgroup_v2_quota_caps_the_cpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(16)), raising=False)
    assert usable_cpus(cgroup(tmp_path, {"cpu.max": "150000 100000\n"})) == 2
    assert usable_cpus(cgroup(tmp_path, {"cpu.max": "50000 100000\n"})) == 1


def test_no_quota_keeps_the_affinity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {0, 1, 2}, raising=False)
    assert usable_cpus(cgroup(tmp_path, {"cpu.max": "max 100000\n"})) == 3
    assert usable_cpus(tmp_path / "missing") == 3


def test_cgroup_v1_quota(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(8)), raising=False)
    root = cgroup(tmp_path, {"cpu/cpu.cfs_quota_us": "400000", "cpu/cpu.cfs_period_us": "100000"})
    assert usable_cpus(root) == 4
    unlimited = cgroup(tmp_path / "u", {"cpu/cpu.cfs_quota_us": "-1", "cpu/cpu.cfs_period_us": "1"})
    assert usable_cpus(unlimited) == 8


def test_without_affinity_the_cpu_count_is_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 6)
    assert usable_cpus(tmp_path) == 6


class FakeSession:
    def __init__(self, path: str, sess_options: Any, providers: list[str]) -> None:
        self.path = path
        self.options = sess_options
        self.asked = providers
        self.used = providers if FakeSession.gpu_works else [CPU]

    gpu_works = True

    def get_providers(self) -> list[str]:
        return self.used


@pytest.fixture
def fake_ort(monkeypatch: pytest.MonkeyPatch) -> Any:
    import onnxruntime

    FakeSession.gpu_works = True
    monkeypatch.setattr(onnxruntime, "InferenceSession", FakeSession)
    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: [CUDA, CPU])
    monkeypatch.setattr(onnxruntime, "preload_dlls", lambda: None, raising=False)
    monkeypatch.setattr(runtime, "usable_cpus", lambda: 2)
    return onnxruntime


def test_session_threads_default_to_the_usable_cpus(fake_ort: Any) -> None:
    session: Any = create_session("model.onnx")
    assert session.asked == [CPU]
    assert session.options.intra_op_num_threads == 2
    assert session.options.inter_op_num_threads == 1
    assert create_session("model.onnx", threads=5).options.intra_op_num_threads == 5


def test_session_on_the_gpu(fake_ort: Any) -> None:
    session: Any = create_session("model.onnx", device="cuda")
    assert session.asked == [CUDA, CPU]


def test_cuda_that_cannot_use_a_gpu_fails_and_auto_warns(
    fake_ort: Any, caplog: pytest.LogCaptureFixture
) -> None:
    FakeSession.gpu_works = False
    with pytest.raises(ModelError, match="could not use a GPU"):
        create_session("model.onnx", device="cuda")
    with caplog.at_level(logging.WARNING, logger="tagsort"):
        session: Any = create_session("model.onnx", device="auto")
    assert session.get_providers() == [CPU]
    assert "reading on the CPU" in caplog.text


def test_preload_errors_are_not_fatal(fake_ort: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> None:
        raise OSError("no wheels")

    monkeypatch.setattr(fake_ort, "preload_dlls", broken)
    assert create_session("model.onnx", device="cuda").get_providers()[0] == CUDA


def test_bad_threads_are_refused() -> None:
    with pytest.raises(ValueError, match="threads"):
        create_session("model.onnx", threads=0)


class Input:
    name = "x"


class Recorded:
    def __init__(self, calls: list[dict[str, Any]], path: Any, **options: Any) -> None:
        calls.append({"path": str(path), **options})

    def get_inputs(self) -> list[Input]:
        return [Input()]


def test_detector_and_recognizer_pass_device_and_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    from tagsort.models import DEFAULT_MODEL, load_manifest
    from tagsort.pipeline import detect, recognize

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(detect, "create_session", lambda p, **o: Recorded(calls, p, **o))
    monkeypatch.setattr(recognize, "create_session", lambda p, **o: Recorded(calls, p, **o))
    manifest = load_manifest(DEFAULT_MODEL)
    detect.OnnxDetector("det.onnx", manifest.detection, device="cuda", threads=2)
    recognize.OnnxRecognizer("rec.onnx", manifest.recognition, device="auto", threads=1)
    recognize.OnnxRecognizer("rec.onnx", manifest.recognition)
    assert calls == [
        {"path": "det.onnx", "device": "cuda", "threads": 2},
        {"path": "rec.onnx", "device": "auto", "threads": 1},
        {"path": "rec.onnx", "device": "cpu", "threads": None},
    ]
