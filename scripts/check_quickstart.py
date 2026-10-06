"""Run the README quickstart exactly as written, in an empty folder.

Used by CI on a clean machine with the freshly built wheel: the M7 exit criterion is that a
newcomer reads a tag in five minutes using the README.

    python scripts/check_quickstart.py --wheel dist/tagsort-*.whl
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def quickstart() -> str:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"<!-- quickstart -->\n```sh\n(.*?)\n```", readme, re.S)
    if match is None:
        raise SystemExit("README has no <!-- quickstart --> shell block")
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", required=True, help="wheel to install instead of PyPI")
    args = parser.parse_args()
    script = quickstart()
    # Before the release, the package comes from the wheel just built, not from PyPI.
    script = script.replace('pip install "tagsort[api]"', f'pip install "{args.wheel}[api]"')
    with tempfile.TemporaryDirectory() as folder:
        output = subprocess.run(
            ["bash", "-euo", "pipefail", "-c", script],
            cwd=folder,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    line = json.loads(output.strip().splitlines()[-1])
    tags = [(t["text"], t["status"]) for t in line["result"]["tags"]]
    print(tags)
    if tags != [("GJ07966", "accepted")]:
        raise SystemExit(f"unexpected quickstart result: {tags}")
    print("The README quickstart works.")


if __name__ == "__main__":
    main()
