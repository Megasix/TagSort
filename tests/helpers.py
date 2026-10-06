import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
DATA = Path(__file__).parent / "data"


def load_schema(name: str) -> dict[str, Any]:
    schema: dict[str, Any] = json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))
    return schema
