"""Tests for LangChain callback tracing (chain runs + tool events)."""

from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest
from langchain_core.callbacks.base import AsyncCallbackHandler

from saidex._callbacks import _ChainRun


class RecordingHandler(AsyncCallbackHandler):
    """Async handler that records (event, payload) tuples and run-id linkage."""

    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []
        self.chain_run_id: UUID | None = None
        self.tool_parents: list[UUID | None] = []

    async def on_chain_start(
        self,
        serialized: Any,
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kw: Any,
    ) -> None:
        self.chain_run_id = run_id
        self.events.append(("chain_start", kw.get("name") or (serialized or {}).get("name")))

    async def on_chain_end(self, outputs: Any, **kw: Any) -> None:
        self.events.append(("chain_end", outputs))

    async def on_chain_error(self, error: BaseException, **kw: Any) -> None:
        self.events.append(("chain_error", error))

    async def on_tool_start(
        self,
        serialized: Any,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kw: Any,
    ) -> None:
        self.tool_parents.append(parent_run_id)
        self.events.append(("tool_start", (serialized or {}).get("name"), input_str))

    async def on_tool_end(self, output: Any, **kw: Any) -> None:
        self.events.append(("tool_end", output))

    async def on_tool_error(self, error: BaseException, **kw: Any) -> None:
        self.events.append(("tool_error", error))


@pytest.mark.asyncio
async def test_chain_run_emits_nested_tool_and_chain_events() -> None:
    rec = RecordingHandler()
    trace = await _ChainRun.start([rec], name="saidex.test", inputs={"schema": "X"})

    assert trace.child_callbacks() is not None

    async with trace.tool_span("do_thing", {"a": 1}) as span:
        span.record_output("ok")

    await trace.end({"success": True})

    names = [e[0] for e in rec.events]
    assert names == ["chain_start", "tool_start", "tool_end", "chain_end"]
    assert rec.events[0] == ("chain_start", "saidex.test")
    assert rec.events[1] == ("tool_start", "do_thing", '{"a": 1}')
    assert rec.events[2] == ("tool_end", "ok")
    # Tool span nests under the chain run.
    assert rec.tool_parents == [rec.chain_run_id]


@pytest.mark.asyncio
async def test_chain_run_reports_tool_error() -> None:
    rec = RecordingHandler()
    trace = await _ChainRun.start([rec], name="saidex.test", inputs={})
    boom = RuntimeError("kaboom")

    async with trace.tool_span("do_thing", {}) as span:
        span.record_error(boom)

    await trace.end({})
    assert ("tool_error", boom) in rec.events
    assert not any(e[0] == "tool_end" for e in rec.events)


@pytest.mark.asyncio
async def test_chain_run_is_noop_without_callbacks() -> None:
    for cb in (None, []):
        trace = await _ChainRun.start(cb, name="saidex.test", inputs={})
        assert trace.child_callbacks() is None
        async with trace.tool_span("x", {}) as span:  # must not raise
            span.record_output("y")
        await trace.end({"success": True})  # must not raise
        await trace.error(RuntimeError("x"))  # must not raise


@pytest.mark.asyncio
async def test_broken_handler_is_swallowed() -> None:
    class Broken(AsyncCallbackHandler):
        async def on_chain_start(self, *a: Any, **k: Any) -> None:
            raise RuntimeError("start boom")

        async def on_tool_start(self, *a: Any, **k: Any) -> None:
            raise RuntimeError("tool boom")

        async def on_chain_end(self, *a: Any, **k: Any) -> None:
            raise RuntimeError("end boom")

    trace = await _ChainRun.start([Broken()], name="saidex.test", inputs={})
    async with trace.tool_span("x", {}) as span:  # must not raise
        span.record_output("y")
    await trace.end({})  # must not raise
