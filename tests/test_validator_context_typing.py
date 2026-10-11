"""Static-typing regression tests for ``with_context`` (run through mypy --strict)."""

from __future__ import annotations

from pathlib import Path

from mypy import api

_PREAMBLE = """\
from typing import Any

from pydantic import BaseModel

from saidex import ValidatorContext, extract_data, with_context


class Invoice(BaseModel):
    total: float


class Order(BaseModel):
    qty: int


def on_base(inv: BaseModel, ctx: ValidatorContext) -> None:
    return None


def on_order(order: Order, ctx: ValidatorContext) -> None:
    return None

"""


def _run_mypy(tmp_path: Path, body: str) -> tuple[str, int]:
    snippet = tmp_path / "snippet.py"
    snippet.write_text(_PREAMBLE + body, encoding="utf-8")
    stdout, _stderr, status = api.run(
        [
            "--strict",
            "--no-incremental",
            "--cache-dir",
            str(tmp_path / ".mypy_cache"),
            str(snippet),
        ]
    )
    return stdout, status


def test_shared_context_validator_is_accepted_for_a_concrete_schema(tmp_path: Path) -> None:
    body = """\
shared = with_context(on_base)


async def run(llm: Any) -> None:
    result, _ = await extract_data(llm, Invoice, [], validator=shared)
    reveal_type(result)
"""
    stdout, status = _run_mypy(tmp_path, body)
    assert status == 0, stdout
    assert "Invoice | None" in stdout


def test_context_validator_for_a_different_schema_is_rejected(tmp_path: Path) -> None:
    body = """\
async def run(llm: Any) -> None:
    await extract_data(llm, Invoice, [], validator=with_context(on_order))
"""
    stdout, status = _run_mypy(tmp_path, body)
    assert status == 1, stdout
    assert "arg-type" in stdout
