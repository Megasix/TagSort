"""HTTP server: every response is checked against schemas/api.v1.openapi.json."""

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from PIL import Image
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012
from starlette.testclient import TestClient

from tagsort import (
    ModelError,
    Profile,
    ProviderTag,
    TagSortError,
    Usage,
    VisionAnswer,
    VisionRequest,
)
from tagsort.fallback.base import BoxFormat
from tagsort.pipeline.local import LineReading, LocalPipeline
from tagsort.server import ServerConfig, create_app
from tests.helpers import load_schema

API = load_schema("api.v1.openapi.json")
REGISTRY = Registry().with_resources(
    [
        ("result.v1.json", Resource.from_contents(load_schema("result.v1.json"))),
        ("api.v1.openapi.json", Resource.from_contents(API, default_specification=DRAFT202012)),
    ]
)
PROFILE = {
    "schema_version": "1.0",
    "name": "Lot",
    "tags_per_individual": 1,
    "tags": [{"id": "primary", "pattern": "GJ\\d{5}"}],
}
TOKEN = "s3cret"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def check(schema: dict[str, Any], body: Any) -> None:
    Draft202012Validator(schema, registry=REGISTRY).validate(body)


def component(name: str) -> dict[str, Any]:
    return {"$ref": f"api.v1.openapi.json#/components/schemas/{name}"}


class StubPipeline(LocalPipeline):
    def __init__(self, model: str = "stub") -> None:
        self.model = model
        self.version = "1.0"

    def read(self, image: Image.Image, profile: Profile) -> list[LineReading]:
        return [
            LineReading(
                quad=((10.0, 10.0), (90.0, 10.0), (90.0, 30.0), (10.0, 30.0)),
                angle=0.0,
                readings=(("GJ07966", 0.97, "primary"),),
                agrees=True,
                crop=Image.new("RGB", (80, 20), "white"),
            )
        ]


@dataclass
class StubFallback:
    name: str = "fake"
    model: str = "fake-1"
    max_side: int = 256
    box_format: BoxFormat = "pixels_xyxy"
    closed: bool = False

    def read(self, request: VisionRequest) -> VisionAnswer:
        tag = ProviderTag("GJ07966", "certain", (), (1, 1, 10, 10), 0)
        return VisionAnswer(tags=(tag,), model="fake-1", usage=Usage())

    def close(self) -> None:
        self.closed = True


def png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (120, 60), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def client(config: ServerConfig | None = None, **factories: Any) -> TestClient:
    app = create_app(
        config or ServerConfig(token=TOKEN, profiles={"lot": Profile.from_dict(PROFILE)}),
        pipeline_factory=factories.get("pipeline", StubPipeline),
        fallback_factory=factories.get("fallback"),
    )
    return TestClient(app, raise_server_exceptions=factories.get("raise_errors", True))


def test_contract_is_valid_openapi_with_json_schemas() -> None:
    assert API["openapi"] == "3.1.0"
    for name, schema in API["components"]["schemas"].items():
        Draft202012Validator.check_schema(schema), name


def test_health_needs_no_token() -> None:
    with client() as c:
        response = c.get("/v1/health")
    assert response.status_code == 200
    check(component("Health"), response.json())
    assert response.json()["fallback"] is None


def test_models_lists_known_models() -> None:
    with client() as c:
        response = c.get("/v1/models", headers=AUTH)
    assert response.status_code == 200
    check(component("Models"), response.json())
    assert [m["name"] for m in response.json()["models"]] == [
        "ppocrv6-tiny",
        "ppocrv6-small",
        "ppocrv6-medium",
    ]


def test_read_with_a_profile_document() -> None:
    with client() as c:
        response = c.post(
            "/v1/read",
            headers=AUTH,
            files={"image": ("p.png", png())},
            data={"profile": json.dumps(PROFILE)},
        )
    assert response.status_code == 200
    body = response.json()
    check({"$ref": "result.v1.json"}, body)
    assert [(t["text"], t["status"]) for t in body["tags"]] == [("GJ07966", "accepted")]


def test_read_with_a_profile_name() -> None:
    with client() as c:
        response = c.post(
            "/v1/read",
            headers=AUTH,
            files={"image": ("p.png", png())},
            data={"profile_name": "lot"},
        )
    assert response.status_code == 200


def test_fallback_is_reported_and_closed() -> None:
    made: list[StubFallback] = []

    def factory(spec: str) -> StubFallback:
        made.append(StubFallback())
        return made[-1]

    config = ServerConfig(token=TOKEN, fallback="fake")
    with client(config, fallback=factory) as c:
        assert c.get("/v1/health").json()["fallback"] == "fake:fake-1"
    assert made[0].closed


@pytest.mark.parametrize(
    ("headers", "files", "data", "status", "code"),
    [
        ({}, {"image": ("p.png", b"x")}, {"profile_name": "lot"}, 401, "unauthorized"),
        (
            {"Authorization": "Bearer nope"},
            {"image": ("p.png", b"x")},
            {"profile_name": "lot"},
            401,
            "unauthorized",
        ),
        (AUTH, {}, {"profile_name": "lot"}, 400, "invalid_request"),
        (AUTH, {"image": ("p.png", b"x")}, {}, 400, "invalid_request"),
        (AUTH, {"image": ("p.png", b"x")}, {"profile_name": "other"}, 404, "unknown_profile"),
        (AUTH, {"image": ("p.png", b"x")}, {"profile": "{"}, 400, "invalid_profile"),
        (
            AUTH,
            {"image": ("p.png", b"not an image")},
            {"profile_name": "lot"},
            400,
            "invalid_image",
        ),
    ],
)
def test_errors(headers: dict[str, str], files: Any, data: Any, status: int, code: str) -> None:
    with client() as c:
        response = c.post("/v1/read", headers=headers, files=files, data=data)
    assert response.status_code == status
    check(component("Error"), response.json())
    assert response.json()["error"]["code"] == code


def test_invalid_pattern_says_where_and_why() -> None:
    bad = {**PROFILE, "tags": [{"id": "primary", "pattern": "GJ\\d+"}]}
    with client() as c:
        response = c.post(
            "/v1/read",
            headers=AUTH,
            files={"image": ("p.png", png())},
            data={"profile": json.dumps(bad)},
        )
    error = response.json()["error"]
    check(component("Error"), response.json())
    assert (response.status_code, error["code"], error["pattern_code"], error["location"]) == (
        400,
        "invalid_pattern",
        "unbounded_quantifier",
        "tags[0].pattern",
    )


def test_not_multipart() -> None:
    with client() as c:
        response = c.post(
            "/v1/read",
            headers={**AUTH, "content-type": "multipart/form-data; boundary=x"},
            content=b"garbage",
        )
    assert response.json()["error"]["code"] in {"invalid_request"}


def test_large_uploads_are_refused() -> None:
    config = ServerConfig(
        token=TOKEN, profiles={"lot": Profile.from_dict(PROFILE)}, max_upload_bytes=1000
    )
    with client(config) as c:
        response = c.post(
            "/v1/read",
            headers=AUTH,
            files={"image": ("p.bin", b"x" * 5000)},
            data={"profile_name": "lot"},
        )
    assert (response.status_code, response.json()["error"]["code"]) == (413, "too_large")


def test_missing_model_answers_503() -> None:
    def broken(model: str) -> LocalPipeline:
        raise ModelError("model 'x' is not downloaded")

    with client(pipeline=broken) as c:
        response = c.post(
            "/v1/read",
            headers=AUTH,
            files={"image": ("p.png", png())},
            data={"profile_name": "lot"},
        )
        assert c.get("/v1/health").status_code == 200
    assert (response.status_code, response.json()["error"]["code"]) == (503, "model_unavailable")


def test_unexpected_errors_are_hidden() -> None:
    class Exploding(StubPipeline):
        def read(self, image: Image.Image, profile: Profile) -> list[LineReading]:
            raise RuntimeError("secret detail")

    with client(pipeline=Exploding, raise_errors=False) as c:
        response = c.post(
            "/v1/read",
            headers=AUTH,
            files={"image": ("p.png", png())},
            data={"profile_name": "lot"},
        )
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal", "message": "internal error"}}


def test_config_from_env(tmp_path: Path) -> None:
    (tmp_path / "museum_a.json").write_text(json.dumps(PROFILE))
    config = ServerConfig.from_env(
        {
            "TAGSORT_API_TOKEN": "t",
            "TAGSORT_PROFILES": str(tmp_path),
            "TAGSORT_MODEL": "ppocrv6-tiny",
            "TAGSORT_FALLBACK": "gemini",
            "TAGSORT_MAX_UPLOAD_MB": "2",
            "TAGSORT_CONCURRENCY": "3",
        }
    )
    assert (config.token, config.model, config.fallback, config.concurrency) == (
        "t",
        "ppocrv6-tiny",
        "gemini",
        3,
    )
    assert config.max_upload_bytes == 2 * 1024 * 1024
    assert list(config.profiles) == ["museum_a"]
    defaults = ServerConfig.from_env({"TAGSORT_ALLOW_NO_TOKEN": "1"})
    assert (defaults.token, defaults.model, defaults.fallback) == (None, "ppocrv6-small", None)
    with pytest.raises(TagSortError, match="TAGSORT_API_TOKEN"):
        ServerConfig.from_env({})


def test_no_token_mode_for_local_development() -> None:
    config = ServerConfig(token=None, profiles={"lot": Profile.from_dict(PROFILE)})
    with client(config) as c:
        assert c.get("/v1/models").status_code == 200


def test_default_fallback_needs_its_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from tagsort.server import _default_fallback

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(TagSortError, match="GEMINI_API_KEY is not set"):
        _default_fallback("gemini")
    with pytest.raises(TagSortError, match="unknown fallback provider"):
        _default_fallback("nope")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    provider = _default_fallback("gemini:gemini-3.5-flash-lite")
    assert provider.model == "gemini-3.5-flash-lite"
