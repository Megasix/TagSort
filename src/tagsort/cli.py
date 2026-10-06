"""The ``tagsort`` command.

tagsort models list                 known models, size, license, downloaded or not
tagsort models download [NAME]      download and verify a model (default: the smallest)
tagsort models path                 folder where models are stored
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

from tagsort.errors import TagSortError
from tagsort.models import (
    DEFAULT_MODEL,
    available_models,
    download_model,
    load_manifest,
    model_files,
    models_dir,
)

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; return the exit code."""
    parser = argparse.ArgumentParser(prog="tagsort", description="Read specimen ID tags.")
    commands = parser.add_subparsers(dest="command", required=True)
    read = commands.add_parser("read", help="read photos and print one JSON line per photo")
    read.add_argument("paths", nargs="+", type=Path, help="photos or folders of photos")
    read.add_argument("--profile", required=True, type=Path, help="profile JSON file")
    read.add_argument("--model", default=DEFAULT_MODEL, help="local model (tagsort models list)")
    read.add_argument(
        "--fallback", metavar="PROVIDER[:MODEL]", help="vision API for doubtful tags (crops only)"
    )
    read.add_argument("--workers", type=int, default=4, help="photos read at the same time")
    read.add_argument("--out", type=Path, help="also write <photo>.json files to this folder")

    serve = commands.add_parser("serve", help="start the HTTP server (server extra)")
    serve.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 inside a container")
    serve.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))

    models = commands.add_parser("models", help="manage local models")
    actions = models.add_subparsers(dest="action", required=True)
    actions.add_parser("list", help="list known models")
    download = actions.add_parser("download", help="download and verify a model")
    download.add_argument("name", nargs="?", default=DEFAULT_MODEL)
    actions.add_parser("path", help="print the folder where models are stored")
    args = parser.parse_args(argv)

    if args.command == "read":
        return _read(args)
    if args.command == "serve":
        return _serve(args)
    try:
        if args.action == "path":
            print(models_dir())
        elif args.action == "list":
            for name in available_models():
                manifest = load_manifest(name)
                try:
                    model_files(name)
                    state = "downloaded"
                except TagSortError:
                    state = "not downloaded"
                print(f"{name:16s} {manifest.size / 1e6:6.1f} MB  {manifest.license:11s} {state}")
        else:
            manifest = load_manifest(args.name)
            print(f"Downloading {args.name} ({manifest.size / 1e6:.1f} MB, {manifest.license})")

            def progress(kind: str, done: int, total: int) -> None:
                print(f"\r  {kind}: {done / 1e6:5.1f} / {total / 1e6:.1f} MB", end="", flush=True)

            paths = download_model(args.name, progress=progress)
            print()
            for kind, path in paths.items():
                print(f"  {kind}: {path} (checksum verified)")
    except (TagSortError, ImportError, OSError) as error:
        print(f"tagsort: {error}", file=sys.stderr)
        return 1
    return 0


PHOTO_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")


def _photos(paths: Sequence[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(p for p in path.iterdir() if p.suffix.lower() in PHOTO_SUFFIXES)
        else:
            yield path


def _read(args: argparse.Namespace) -> int:
    from tagsort.pipeline.local import LocalPipeline
    from tagsort.pipeline.reader import Reader
    from tagsort.profile import Profile

    try:
        profile = Profile.from_file(args.profile)
        fallback = None
        if args.fallback:
            from tagsort.server import _default_fallback

            fallback = _default_fallback(args.fallback)
        reader = Reader(profile, backend=LocalPipeline(args.model), fallback=fallback)
    except (TagSortError, ImportError, OSError) as error:
        print(f"tagsort: {error}", file=sys.stderr)
        return 1
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
    failed = 0
    photos = list(_photos(args.paths))
    # Read in parallel, but print in order and keep going when one photo fails.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as pool:
        futures = [pool.submit(reader.read, photo) for photo in photos]
        for photo, future in zip(photos, futures, strict=True):
            line: dict[str, object]
            try:
                result = future.result()
            except (TagSortError, OSError) as error:
                failed += 1
                line = {"file": str(photo), "error": str(error)}
            else:
                line = {"file": str(photo), "result": result.to_dict()}
                if args.out is not None:
                    (args.out / f"{photo.stem}.json").write_text(
                        result.to_json(indent=2) + "\n", encoding="utf-8"
                    )
            print(json.dumps(line, ensure_ascii=False), flush=True)
    return 1 if failed else 0


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from tagsort.server import ServerConfig, create_app
    except ModuleNotFoundError:
        print(
            'tagsort: the server needs the server extra: pip install "tagsort[server]"',
            file=sys.stderr,
        )
        return 1
    try:
        config = ServerConfig.from_env()
    except TagSortError as error:
        print(f"tagsort: {error}", file=sys.stderr)
        return 1
    uvicorn.run(create_app(config), host=args.host, port=args.port, log_level="warning")
    return 0
