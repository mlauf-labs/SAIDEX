"""Regenerate the golden expectations from the currently installed json-repair.

Run: ``uv run python tests/golden/json_repair/regenerate.py``

Rewrites every ``<name>.json`` / ``<name>.error`` next to its ``<name>.txt``.
Review the resulting ``git diff`` deliberately before committing.
"""

from __future__ import annotations

import json
from pathlib import Path

from saidex.extractor import _extract_json_from_content

HERE = Path(__file__).parent


def main() -> None:
    for txt in sorted(HERE.glob("*.txt")):
        with txt.open(encoding="utf-8", newline="") as fh:
            raw = fh.read()
        json_path = txt.with_suffix(".json")
        error_path = txt.with_suffix(".error")
        json_path.unlink(missing_ok=True)
        error_path.unlink(missing_ok=True)
        try:
            value = _extract_json_from_content(raw)
        except Exception as exc:
            error_path.write_bytes(type(exc).__name__.encode("utf-8"))
            print(f"{txt.stem}: raises {type(exc).__name__}")
        else:
            payload = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
            json_path.write_bytes(payload.encode("utf-8"))
            print(f"{txt.stem}: {json.dumps(value)}")


if __name__ == "__main__":
    main()
