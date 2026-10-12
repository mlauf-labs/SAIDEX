"""Golden tests pinning how broken LLM output is parsed (json-repair behaviour).

The corpus lives in ``tests/golden/json_repair/``; see the README there.
"""

from __future__ import annotations

import builtins
import importlib.metadata
import json
from pathlib import Path

import pytest

from saidex.extractor import _extract_json_from_content

GOLDEN_DIR = Path(__file__).parent / "golden" / "json_repair"

#: json-repair version the goldens were generated with.
GOLDEN_JSON_REPAIR_VERSION = "0.64.0"

_INSTALLED = importlib.metadata.version("json-repair")


def _version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in version.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


pytestmark = pytest.mark.skipif(
    _version_tuple(_INSTALLED) < _version_tuple(GOLDEN_JSON_REPAIR_VERSION),
    reason=(
        f"golden files were generated with json-repair {GOLDEN_JSON_REPAIR_VERSION}; "
        f"installed {_INSTALLED} may repair differently"
    ),
)

_CASES = sorted(p.stem for p in GOLDEN_DIR.glob("*.txt"))


def _read(path: Path) -> str:
    with path.open(encoding="utf-8", newline="") as fh:
        return fh.read()


def test_corpus_is_not_empty() -> None:
    assert len(_CASES) >= 20


@pytest.mark.parametrize("name", _CASES)
def test_golden(name: str) -> None:
    raw = _read(GOLDEN_DIR / f"{name}.txt")
    json_path = GOLDEN_DIR / f"{name}.json"
    error_path = GOLDEN_DIR / f"{name}.error"
    assert json_path.exists() != error_path.exists(), "exactly one of .json/.error expected"

    if error_path.exists():
        exc_type = getattr(builtins, _read(error_path).strip())
        with pytest.raises(exc_type):
            _extract_json_from_content(raw)
    else:
        expected = json.loads(_read(json_path))
        assert _extract_json_from_content(raw) == expected
