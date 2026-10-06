"""The ``tagsort`` command.

tagsort models list                 known models, size, license, downloaded or not
tagsort models download [NAME]      download and verify a model (default: the smallest)
tagsort models path                 folder where models are stored
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

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
    models = commands.add_parser("models", help="manage local models")
    actions = models.add_subparsers(dest="action", required=True)
    actions.add_parser("list", help="list known models")
    download = actions.add_parser("download", help="download and verify a model")
    download.add_argument("name", nargs="?", default=DEFAULT_MODEL)
    actions.add_parser("path", help="print the folder where models are stored")
    args = parser.parse_args(argv)

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
