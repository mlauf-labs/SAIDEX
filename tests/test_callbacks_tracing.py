"""Tests for LangChain callback tracing (chain runs + tool events)."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from langchain_core.callbacks.base import AsyncCallbackHandler
from pydantic import BaseModel

from saidex import Tool, run_extractor_agent
from saidex._callbacks import _ChainRun, _ToolSpan
from saidex.retry import RetryConfig

_NO_RETRY = RetryConfig(max_retries=0, retry_delays=[])


class _Final(BaseModel):
    result: str


class _Args(BaseModel):
    q: str


def _resp(tool_calls: list[dict[str, Any]] | None = None) -> MagicMock:
    r = MagicMock()
    r.tool_calls = tool_calls or []
    r.invalid_tool_calls = []
    r.content = ""
    return r


def _llm(responses: list[MagicMock]) -> MagicMock:
    bound = MagicMock()
    bound.ainvoke = AsyncMock(side_effect=responses)
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=bound)
    llm.ainvoke = bound.ainvoke
    return llm


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
async def test_module_suppresses_callback_errors(caplog: pytest.LogCaptureFixture) -> None:
    """The module's own try/except swallows callback errors and logs them.

    ``raise_error = True`` forces langchain-core to re-raise past its internal
    catch, so the exception reaches _callbacks.py's own suppression layer
    (langchain's default swallow would let this test pass even with our
    try/except removed).
    """

    class RaisingHandler(AsyncCallbackHandler):
        raise_error = True  # re-raise past langchain's default internal swallow

        async def on_tool_start(self, *a: Any, **k: Any) -> None:
            raise RuntimeError("tool_start boom")

        async def on_chain_end(self, *a: Any, **k: Any) -> None:
            raise RuntimeError("chain_end boom")

    trace = await _ChainRun.start([RaisingHandler()], name="saidex.test", inputs={})
    # Chain started (no on_chain_start override), so the child manager exists.
    assert trace.child_callbacks() is not None

    with caplog.at_level(logging.ERROR, logger="saidex._callbacks"):
        async with trace.tool_span("x", {}) as span:  # on_tool_start raises into our except
            span.record_output("y")
        await trace.end({})  # on_chain_end raises into our except

    messages = [r.getMessage() for r in caplog.records if r.name == "saidex._callbacks"]
    assert any("suppressed" in m for m in messages)  # proves OUR except ran


@pytest.mark.asyncio
async def test_toolspan_and_error_suppress_raising_run(caplog: pytest.LogCaptureFixture) -> None:
    """_ToolSpan._finish and _ChainRun.error swallow a run whose callbacks raise."""

    class RaisingRun:
        async def on_tool_end(self, *a: Any, **k: Any) -> None:
            raise RuntimeError("tool_end boom")

        async def on_chain_error(self, *a: Any, **k: Any) -> None:
            raise RuntimeError("chain_error boom")

    with caplog.at_level(logging.ERROR, logger="saidex._callbacks"):
        span = _ToolSpan(RaisingRun())  # type: ignore[arg-type]
        span.record_output("y")
        await span._finish()  # must not raise

        trace = _ChainRun(RaisingRun(), None)  # type: ignore[arg-type]
        await trace.error(RuntimeError("x"))  # must not raise

    messages = [r.getMessage() for r in caplog.records if r.name == "saidex._callbacks"]
    assert any("suppressed" in m for m in messages)


@pytest.mark.asyncio
async def test_agent_loop_wraps_generations_in_chain_run() -> None:
    rec = RecordingHandler()
    llm = _llm([_resp([{"name": "_Final", "args": {"result": "done"}, "id": "c1"}])])

    result, _ = await run_extractor_agent(
        llm, _Final, [], tools=[], callbacks=[rec], retry_config=_NO_RETRY
    )

    assert result is not None and result.result == "done"
    names = [e[0] for e in rec.events]
    assert names[0] == "chain_start"
    assert rec.events[0] == ("chain_start", "saidex.agent_loop")
    assert names[-1] == "chain_end"
    end_outputs = rec.events[-1][1]
    assert end_outputs["success"] is True
    assert "iterations" in end_outputs and "tool_calls" in end_outputs
    # Generations are wired to nest: the child manager was passed to ainvoke.
    call = llm.ainvoke.call_args
    assert call.kwargs["config"]["callbacks"] is not None


@pytest.mark.asyncio
async def test_agent_loop_emits_tool_span() -> None:
    rec = RecordingHandler()

    async def handler(q: str) -> dict[str, Any]:
        return {"answer": q.upper()}

    tool = Tool(name="lookup", description="d", parameters=_Args, handler=handler)
    llm = _llm(
        [
            _resp([{"name": "lookup", "args": {"q": "hi"}, "id": "c1"}]),
            _resp([{"name": "_Final", "args": {"result": "done"}, "id": "c2"}]),
        ]
    )

    result, stats = await run_extractor_agent(
        llm, _Final, [], tools=[tool], callbacks=[rec], retry_config=_NO_RETRY
    )

    assert result is not None
    names = [e[0] for e in rec.events]
    assert names == ["chain_start", "tool_start", "tool_end", "chain_end"]
    assert rec.events[1] == ("tool_start", "lookup", '{"q": "hi"}')
    assert rec.tool_parents == [rec.chain_run_id]  # nested under the loop


@pytest.mark.asyncio
async def test_agent_loop_handler_crash_emits_tool_error() -> None:
    rec = RecordingHandler()

    async def handler(q: str) -> dict[str, Any]:
        raise RuntimeError("handler down")

    tool = Tool(name="lookup", description="d", parameters=_Args, handler=handler)
    llm = _llm(
        [
            _resp([{"name": "lookup", "args": {"q": "hi"}, "id": "c1"}]),
            _resp([{"name": "_Final", "args": {"result": "recovered"}, "id": "c2"}]),
        ]
    )

    result, _ = await run_extractor_agent(
        llm, _Final, [], tools=[tool], callbacks=[rec], retry_config=_NO_RETRY
    )

    # Loop still completes despite the crashing handler.
    assert result is not None and result.result == "recovered"
    assert any(e[0] == "tool_error" for e in rec.events)
    assert not any(e[0] == "tool_end" for e in rec.events)


@pytest.mark.asyncio
async def test_agent_loop_unknown_tool_emits_tool_end() -> None:
    rec = RecordingHandler()
    llm = _llm(
        [
            _resp([{"name": "ghost", "args": {"q": "x"}, "id": "c1"}]),
            _resp([{"name": "_Final", "args": {"result": "ok"}, "id": "c2"}]),
        ]
    )

    result, _ = await run_extractor_agent(
        llm, _Final, [], tools=[], callbacks=[rec], retry_config=_NO_RETRY
    )

    assert result is not None
    tool_ends = [e for e in rec.events if e[0] == "tool_end"]
    assert tool_ends and "Unknown tool 'ghost'" in tool_ends[0][1]


@pytest.mark.asyncio
async def test_agent_loop_no_callbacks_still_works() -> None:
    async def handler(q: str) -> dict[str, Any]:
        return {"answer": q}

    tool = Tool(name="lookup", description="d", parameters=_Args, handler=handler)
    llm = _llm(
        [
            _resp([{"name": "lookup", "args": {"q": "hi"}, "id": "c1"}]),
            _resp([{"name": "_Final", "args": {"result": "done"}, "id": "c2"}]),
        ]
    )
    result, _ = await run_extractor_agent(llm, _Final, [], tools=[tool], retry_config=_NO_RETRY)
    assert result is not None and result.result == "done"
