"""HTTP server for applications that call TagSort as a service (``server`` extra).

The contract is ``schemas/api.v1.openapi.json``. Photos are read in memory and never
stored. Configuration comes from the environment, so the same Docker image serves every
deployment:

==========================  ===============================================================
``TAGSORT_API_TOKEN``       Bearer token clients must send. Required unless
                            ``TAGSORT_ALLOW_NO_TOKEN=1`` (local development only).
``TAGSORT_MODEL``           Local model, default ``ppocrv6-small``.
``TAGSORT_FALLBACK``        Optional vision API for doubtful tags, ``provider[:model]``,
                            for example ``gemini``; its key comes from the provider's usual
                            variable (``GEMINI_API_KEY``...). Off by default.
``TAGSORT_PROFILES``        Optional folder of profiles (``*.json``), usable by name.
``TAGSORT_MAX_UPLOAD_MB``   Largest accepted photo, default 25.
``TAGSORT_CONCURRENCY``     Photos read at the same time, default the number of CPUs.
==========================  ===============================================================
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from tagsort._version import __version__
from tagsort.errors import ImageError, PatternError, ProfileError, TagSortError
from tagsort.fallback.base import VisionProvider
from tagsort.models import ModelError, available_models, load_manifest, model_files
from tagsort.pipeline.local import LocalPipeline
from tagsort.pipeline.reader import Reader
from tagsort.profile import Profile

__all__ = ["ServerConfig", "create_app"]

logger = logging.getLogger("tagsort.server")

SERVER_MODEL = "ppocrv6-small"
"""Default model on servers: the best balance of accuracy and speed on lot-02."""

_READER_CACHE = 64
_KEY_VARIABLES = {
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
}


@dataclass(frozen=True)
class ServerConfig:
    """How the server reads. :meth:`from_env` builds it from environment variables."""

    token: str | None
    model: str = SERVER_MODEL
    fallback: str | None = None
    profiles: dict[str, Profile] = field(default_factory=dict)
    max_upload_bytes: int = 25 * 1024 * 1024
    concurrency: int = max(os.cpu_count() or 1, 1)

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> ServerConfig:
        """Read the configuration from the environment.

        Raises:
            TagSortError: If no token is set and ``TAGSORT_ALLOW_NO_TOKEN`` is not ``1``,
                or a profile in ``TAGSORT_PROFILES`` is invalid.
        """
        env = dict(os.environ if environ is None else environ)
        token = env.get("TAGSORT_API_TOKEN") or None
        if token is None and env.get("TAGSORT_ALLOW_NO_TOKEN") != "1":
            raise TagSortError(
                "set TAGSORT_API_TOKEN, or TAGSORT_ALLOW_NO_TOKEN=1 for local development"
            )
        profiles = {}
        folder = env.get("TAGSORT_PROFILES")
        if folder:
            for path in sorted(Path(folder).glob("*.json")):
                profiles[path.stem] = Profile.from_file(path)
        return cls(
            token=token,
            model=env.get("TAGSORT_MODEL") or SERVER_MODEL,
            fallback=env.get("TAGSORT_FALLBACK") or None,
            profiles=profiles,
            max_upload_bytes=int(float(env.get("TAGSORT_MAX_UPLOAD_MB", "25")) * 1024 * 1024),
            concurrency=max(int(env.get("TAGSORT_CONCURRENCY", "0")) or (os.cpu_count() or 1), 1),
        )


class _ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, **extra: str) -> None:
        super().__init__(message)
        self.status = status
        self.body = {"error": {"code": code, "message": message, **extra}}


def create_app(
    config: ServerConfig,
    *,
    pipeline_factory: Callable[[str], LocalPipeline] = LocalPipeline,
    fallback_factory: Callable[[str], VisionProvider] | None = None,
) -> Starlette:
    """Build the HTTP application.

    Args:
        config: How the server reads.
        pipeline_factory: Builds the local pipeline from a model name; for tests.
        fallback_factory: Builds the fallback provider from ``config.fallback``; defaults
            to the providers of the ``api`` extra with keys from the environment.

    Models load when the application starts; a missing model makes every read answer 503.
    """
    state: dict[str, Any] = {}
    readers: OrderedDict[str, Reader] = OrderedDict()
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        try:
            state["pipeline"] = pipeline_factory(config.model)
        except ModelError as error:
            logger.error("model unavailable: %s", error)
            state["model_error"] = str(error)
        if config.fallback:
            state["fallback"] = (fallback_factory or _default_fallback)(config.fallback)
        state["slots"] = asyncio.Semaphore(config.concurrency)
        yield
        fallback = state.get("fallback")
        if fallback is not None and hasattr(fallback, "close"):
            fallback.close()

    def reader_for(profile: Profile) -> Reader:
        key = json.dumps(profile.to_dict(), sort_keys=True)
        with lock:
            reader = readers.get(key)
            if reader is None:
                reader = Reader(profile, backend=state["pipeline"], fallback=state.get("fallback"))
                readers[key] = reader
                if len(readers) > _READER_CACHE:
                    readers.popitem(last=False)
            readers.move_to_end(key)
            return reader

    def check_token(request: Request) -> None:
        if config.token is None:
            return
        header = request.headers.get("authorization", "")
        scheme, _, given = header.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(
            given.encode(), config.token.encode()
        ):
            raise _ApiError(401, "unauthorized", "missing or wrong bearer token")

    async def health(request: Request) -> JSONResponse:
        fallback = state.get("fallback")
        return JSONResponse(
            {
                "status": "ok",
                "engine_version": __version__,
                "model": config.model,
                "fallback": f"{fallback.name}:{fallback.model}" if fallback else None,
            }
        )

    async def models(request: Request) -> JSONResponse:
        check_token(request)
        listed = []
        for name in available_models():
            manifest = load_manifest(name)
            try:
                model_files(name)
                downloaded = True
            except ModelError:
                downloaded = False
            listed.append(
                {
                    "name": name,
                    "size_bytes": manifest.size,
                    "license": manifest.license,
                    "downloaded": downloaded,
                }
            )
        return JSONResponse({"active": config.model, "models": listed})

    async def read(request: Request) -> JSONResponse:
        check_token(request)
        if "pipeline" not in state:
            raise _ApiError(503, "model_unavailable", state.get("model_error", "model not loaded"))
        length = request.headers.get("content-length")
        if (
            length is not None
            and length.isdigit()
            and int(length) > config.max_upload_bytes + 65536
        ):
            raise _ApiError(
                413, "too_large", f"photos are limited to {config.max_upload_bytes} bytes"
            )
        try:
            form = await request.form(max_files=1, max_fields=4)
        except Exception as error:  # malformed multipart bodies raise various errors
            raise _ApiError(
                400, "invalid_request", f"expected multipart/form-data: {error}"
            ) from None
        try:
            # Uploads are spooled to temporary files: always close them.
            upload = form.get("image")
            if not isinstance(upload, UploadFile):
                raise _ApiError(400, "invalid_request", "the image field is missing")
            data = await upload.read(config.max_upload_bytes + 1)
            profile_text, profile_name = form.get("profile"), form.get("profile_name")
        finally:
            await form.close()
        if len(data) > config.max_upload_bytes:
            raise _ApiError(
                413, "too_large", f"photos are limited to {config.max_upload_bytes} bytes"
            )
        profile = _profile(profile_text, profile_name, config.profiles)
        reader = reader_for(profile)
        start = time.perf_counter()
        async with state["slots"]:
            try:
                result = await run_in_threadpool(reader.read, data)
            except ImageError as error:
                raise _ApiError(400, "invalid_image", str(error)) from None
        logger.info(
            "read %d tags in %.0f ms", len(result.tags), (time.perf_counter() - start) * 1000
        )
        return JSONResponse(result.to_dict())

    async def api_error(request: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, _ApiError)
        return JSONResponse(exc.body, status_code=exc.status)

    async def internal_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unexpected error")
        return JSONResponse(
            {"error": {"code": "internal", "message": "internal error"}}, status_code=500
        )

    return Starlette(
        routes=[
            Route("/v1/health", health, methods=["GET"]),
            Route("/v1/models", models, methods=["GET"]),
            Route("/v1/read", read, methods=["POST"]),
        ],
        lifespan=lifespan,
        exception_handlers={_ApiError: api_error, Exception: internal_error},
    )


def _profile(text: object, name: object, known: dict[str, Profile]) -> Profile:
    if isinstance(text, str) and text.strip():
        try:
            return Profile.from_json(text)
        except PatternError as error:
            extra = {"pattern_code": error.code}
            if error.location:
                extra["location"] = error.location
            raise _ApiError(400, "invalid_pattern", error.message, **extra) from None
        except ProfileError as error:
            extra = {"location": error.location} if error.location else {}
            raise _ApiError(400, "invalid_profile", error.message, **extra) from None
    if isinstance(name, str) and name:
        if name not in known:
            raise _ApiError(404, "unknown_profile", f"no profile named {name!r} on this server")
        return known[name]
    raise _ApiError(400, "invalid_request", "give a profile (JSON) or a profile_name")


def _default_fallback(spec: str) -> VisionProvider:
    import tagsort

    name, _, model = spec.partition(":")
    if name not in _KEY_VARIABLES:
        raise TagSortError(f"unknown fallback provider {name!r}")
    key = os.environ.get(_KEY_VARIABLES[name])
    if not key:
        raise TagSortError(f"{_KEY_VARIABLES[name]} is not set for the {name} fallback")
    class_name = {
        "anthropic": "Anthropic",
        "deepseek": "DeepSeek",
        "gemini": "Gemini",
        "openai": "OpenAI",
    }
    provider: VisionProvider = getattr(tagsort, f"{class_name[name]}Provider")(
        api_key=key, model=model or None
    )
    return provider
