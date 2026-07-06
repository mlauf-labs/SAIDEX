# Callback Tool Events Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fire LangChain `on_tool_*` / `on_chain_*` callback events from SAIDEX's agent loop and retry loops so tracers (Langfuse, LangSmith, …) see tool executions and a properly nested run tree — not just the raw LLM generations.

**Architecture:** A new private `_callbacks.py` helper wraps `langchain_core`'s `AsyncCallbackManager` into a small `_ChainRun` object that opens an enclosing chain run, hands a child callback manager to every `llm.ainvoke` (so generations nest), and brackets each helper-tool execution with `on_tool_start`/`on_tool_end`/`on_tool_error`. `extract_data`, `extract_data_list` and `run_extractor_agent` open/close one chain run each; `Tool` gains an internal `_invoke` that surfaces the handler exception so tool errors can be reported. Everything is additive and a no-op when `callbacks` is falsy.

**Tech Stack:** Python 3.10+, `langchain-core` (installed 1.4.0, floor `>=0.2`), `pydantic>=2`, `pytest`/`pytest-asyncio`, `uv`.

## Global Constraints

- Package manager is **uv** — never `pip`. Run tools via `uv run`.
- **No new runtime dependency.** Only `langchain-core` APIs already available (`AsyncCallbackManager.configure`, `on_chain_start`, `get_child`, `on_tool_start`, run-manager `on_chain_end`/`on_chain_error`/`on_tool_end`/`on_tool_error`).
- **Public API unchanged** — `src/saidex/__init__.py` exports are not touched; `_callbacks.py` is private (underscore).
- `mypy --strict` must pass on `src/` — every new function fully annotated.
- Ruff lint rules: `E, F, I, N, W, UP, B, C4, SIM`; line length 100 (`E501` ignored).
- Tests run **without API keys or network** — use the mock-LLM pattern from `tests/test_agent_loop.py`.
- **Observability can never break a run:** every callback invocation in `_callbacks.py` is wrapped in `try/except`, logged at ERROR, and swallowed.
- Conventional Commits; Git Flow — work on branch `feature/callback-tool-events` (already created off `develop`), PR targets `develop`.
- CI gate before PR: `uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/ && uv run mypy src/ && uv run pytest`.

---

### Task 1: `_callbacks.py` — the tracing helper

**Files:**
- Create: `src/saidex/_callbacks.py`
- Test: `tests/test_callbacks_tracing.py`

**Interfaces:**
- Produces:
  - `class _ToolSpan` with `record_output(output: str) -> None`, `record_error(error: BaseException) -> None`.
  - `class _ChainRun` with:
    - `async classmethod start(callbacks: list[Any] | None, *, name: str, inputs: dict[str, Any], metadata: dict[str, Any] | None = None) -> _ChainRun`
    - `child_callbacks() -> AsyncCallbackManager | None`
    - `tool_span(name: str, args: dict[str, Any]) -> AbstractAsyncContextManager[_ToolSpan]` (async context manager)
    - `async end(outputs: dict[str, Any]) -> None`
    - `async error(error: BaseException) -> None`
  - No-op instance (returned when `callbacks` is falsy): `child_callbacks()` returns `None`, `tool_span` yields an inert span, `end`/`error` do nothing.

- [ ] **Step 1: Write the failing test**

Create `tests/test_callbacks_tracing.py`:

```python
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
        self, serialized: Any, inputs: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kw: Any
    ) -> None:
        self.chain_run_id = run_id
        self.events.append(("chain_start", kw.get("name") or (serialized or {}).get("name")))

    async def on_chain_end(self, outputs: Any, **kw: Any) -> None:
        self.events.append(("chain_end", outputs))

    async def on_chain_error(self, error: BaseException, **kw: Any) -> None:
        self.events.append(("chain_error", error))

    async def on_tool_start(
        self, serialized: Any, input_str: str, *, run_id: UUID, parent_run_id: UUID | None = None, **kw: Any
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_callbacks_tracing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'saidex._callbacks'`.

- [ ] **Step 3: Write the implementation**

Create `src/saidex/_callbacks.py`:

```python
"""LangChain callback plumbing for the agent loop and retry loops.

Wraps ``langchain_core``'s :class:`AsyncCallbackManager` into a small
:class:`_ChainRun` that opens one enclosing chain run, hands a child callback
manager to each ``llm.ainvoke`` so generations nest, and brackets each helper
tool execution with ``on_tool_start`` / ``on_tool_end`` / ``on_tool_error``.

Private module — not part of the public API.  When no ``callbacks`` are passed
a no-op instance is returned, so there is zero overhead and byte-for-byte
unchanged behaviour.  Every callback invocation is isolated: a raising handler
is logged and swallowed and can never break the extraction it observes.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from langchain_core.callbacks import AsyncCallbackManager
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    AsyncCallbackManagerForToolRun,
)

logger = logging.getLogger(__name__)


class _ToolSpan:
    """Handle for one tool run.  Record the outcome; the span is closed on exit.

    Call :meth:`record_output` for a normal result or :meth:`record_error` for a
    handler exception.  When neither is called (e.g. the no-op span) closing is a
    no-op.
    """

    def __init__(self, run: AsyncCallbackManagerForToolRun | None) -> None:
        self._run = run
        self._output: str = ""
        self._error: BaseException | None = None

    def record_output(self, output: str) -> None:
        self._output = output

    def record_error(self, error: BaseException) -> None:
        self._error = error

    async def _finish(self) -> None:
        if self._run is None:
            return
        try:
            if self._error is not None:
                await self._run.on_tool_error(self._error)
            else:
                await self._run.on_tool_end(self._output)
        except Exception as exc:  # noqa: BLE001 — observability must not break the run
            logger.error("tool-span end callback raised and was suppressed: %s", exc)


class _ChainRun:
    """One enclosing chain run for an extraction, or a no-op when untraced."""

    def __init__(
        self,
        run: AsyncCallbackManagerForChainRun | None,
        child: AsyncCallbackManager | None,
    ) -> None:
        self._run = run
        self._child = child
        self._closed = False

    @classmethod
    async def start(
        cls,
        callbacks: list[Any] | None,
        *,
        name: str,
        inputs: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> _ChainRun:
        """Open a chain run for *callbacks*, or return a no-op when there are none."""
        if not callbacks:
            return cls(None, None)
        try:
            manager = AsyncCallbackManager.configure(
                inheritable_callbacks=callbacks,
                inheritable_metadata=metadata,
            )
            run = await manager.on_chain_start({"name": name}, inputs, name=name)
            return cls(run, run.get_child())
        except Exception as exc:  # noqa: BLE001 — observability must not break the run
            logger.error("failed to start trace chain run '%s': %s", name, exc)
            return cls(None, None)

    def child_callbacks(self) -> AsyncCallbackManager | None:
        """Child manager to hand to ``llm.ainvoke(config={'callbacks': ...})``."""
        return self._child

    @asynccontextmanager
    async def tool_span(self, name: str, args: dict[str, Any]) -> AsyncIterator[_ToolSpan]:
        """Bracket a tool execution with ``on_tool_start`` / ``on_tool_end``|``on_tool_error``."""
        run: AsyncCallbackManagerForToolRun | None = None
        if self._child is not None:
            try:
                run = await self._child.on_tool_start(
                    {"name": name},
                    json.dumps(args, ensure_ascii=False, default=str),
                    name=name,
                )
            except Exception as exc:  # noqa: BLE001 — observability must not break the run
                logger.error("tool-span start callback raised and was suppressed: %s", exc)
                run = None
        span = _ToolSpan(run)
        try:
            yield span
        finally:
            await span._finish()

    async def end(self, outputs: dict[str, Any]) -> None:
        """Close the chain run with ``on_chain_end``.  Idempotent."""
        if self._run is None or self._closed:
            return
        self._closed = True
        try:
            await self._run.on_chain_end(outputs)
        except Exception as exc:  # noqa: BLE001 — observability must not break the run
            logger.error("chain-end callback raised and was suppressed: %s", exc)

    async def error(self, error: BaseException) -> None:
        """Close the chain run with ``on_chain_error``.  Idempotent."""
        if self._run is None or self._closed:
            return
        self._closed = True
        try:
            await self._run.on_chain_error(error)
        except Exception as exc:  # noqa: BLE001 — observability must not break the run
            logger.error("chain-error callback raised and was suppressed: %s", exc)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_callbacks_tracing.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Lint + type-check the new module**

Run: `uv run ruff check src/saidex/_callbacks.py tests/test_callbacks_tracing.py && uv run ruff format src/saidex/_callbacks.py tests/test_callbacks_tracing.py && uv run mypy src/`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/saidex/_callbacks.py tests/test_callbacks_tracing.py
git commit -m "feat(observability): add private LangChain chain-run/tool-span helper"
```

---

### Task 2: `Tool._invoke` — surface handler exceptions

**Files:**
- Modify: `src/saidex/tools.py` (add `_ToolOutcome` + `_invoke`, refactor `execute`)
- Test: `tests/test_agent_loop.py` (extend)

**Interfaces:**
- Produces:
  - `@dataclass class _ToolOutcome: content: str; handler_error: Exception | None`
  - `Tool._invoke(self, raw_args: dict[str, Any]) -> _ToolOutcome` — runs arg-validation + handler; `content` is the string that goes into the `ToolMessage`; `handler_error` is set **only** when the handler itself raised.
  - `Tool.execute` keeps its exact signature/behaviour: `execute(raw_args) -> str` returns `_invoke(raw_args).content`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent_loop.py`:

```python
@pytest.mark.asyncio
async def test_tool_invoke_success_has_no_error() -> None:
    tool = _make_tool()
    outcome = await tool._invoke({"name": "foo", "parent_id": None})
    assert outcome.handler_error is None
    data = json.loads(outcome.content)
    assert data["name"] == "foo"


@pytest.mark.asyncio
async def test_tool_invoke_invalid_args_is_not_a_handler_error() -> None:
    tool = _make_tool()
    outcome = await tool._invoke({"parent_id": "x"})  # 'name' missing
    assert outcome.handler_error is None
    assert "Invalid arguments" in outcome.content


@pytest.mark.asyncio
async def test_tool_invoke_handler_exception_is_captured() -> None:
    async def _failing(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("boom")

    tool = Tool(name="boom_tool", description=".", parameters=ToolArgs, handler=_failing)
    outcome = await tool._invoke({"name": "x"})
    assert isinstance(outcome.handler_error, RuntimeError)
    assert "boom" in outcome.content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_agent_loop.py -k tool_invoke -v`
Expected: FAIL — `AttributeError: 'Tool' object has no attribute '_invoke'`.

- [ ] **Step 3: Refactor `tools.py`**

In `src/saidex/tools.py`, add the dataclass import and a `_ToolOutcome` type, then replace the body of `execute` with a thin wrapper over a new `_invoke`. Replace the current `execute` method (lines 76-103) with:

```python
    async def _invoke(self, raw_args: dict[str, Any]) -> _ToolOutcome:
        """Validate *raw_args*, run the handler, and capture the outcome.

        ``content`` is always the string destined for the ``ToolMessage``.
        ``handler_error`` is set **only** when the handler itself raised — invalid
        arguments are a normal tool result, not a handler crash.
        """
        from .utils import create_instance_safe  # local import to avoid circular deps

        args_model, error_text = create_instance_safe(self.parameters, **raw_args)
        if error_text or args_model is None:
            return _ToolOutcome(f"Invalid arguments for tool '{self.name}': {error_text}", None)

        try:
            result = await self.handler(**args_model.model_dump())
        except Exception as exc:  # noqa: BLE001
            return _ToolOutcome(f"Tool '{self.name}' raised an error: {exc}", exc)

        if isinstance(result, str):
            return _ToolOutcome(result, None)
        try:
            return _ToolOutcome(json.dumps(result, ensure_ascii=False, default=str), None)
        except (TypeError, ValueError):
            return _ToolOutcome(str(result), None)

    async def execute(self, raw_args: dict[str, Any]) -> str:
        """Validate *raw_args* against ``parameters``, run the handler, return result.

        Args:
            raw_args: Raw argument dict from the LLM tool call.

        Returns:
            A string (JSON or plain text) to use as the ``ToolMessage`` content.
            Handler exceptions are caught and returned as error strings so the
            LLM can self-correct rather than crashing the whole pipeline.
        """
        return (await self._invoke(raw_args)).content
```

Add the `_ToolOutcome` dataclass just above the `Tool` class (after the module imports):

```python
@dataclass
class _ToolOutcome:
    """Result of running one tool: the ``ToolMessage`` content plus any handler error.

    Attributes:
        content: The string appended to the conversation as the tool's result.
        handler_error: The exception the handler raised, or ``None`` when the tool
            ran cleanly or the arguments were merely invalid.
    """

    content: str
    handler_error: Exception | None
```

(`dataclass` and `Any` are already imported at the top of `tools.py`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_agent_loop.py -k "tool_invoke or tool_execute" -v`
Expected: PASS — the new `_invoke` tests **and** the existing `test_tool_execute_*` tests (behaviour unchanged).

- [ ] **Step 5: Lint + type-check**

Run: `uv run ruff check src/saidex/tools.py tests/test_agent_loop.py && uv run mypy src/`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/saidex/tools.py tests/test_agent_loop.py
git commit -m "feat(tools): surface handler exceptions via Tool._invoke"
```

---

### Task 3: Agent-loop chain run (nested generations + chain-end metrics)

**Files:**
- Modify: `src/saidex/extractor.py` (`run_extractor_agent`, `_run_extractor_agent_with_model`, add `_agent_outputs`)
- Test: `tests/test_callbacks_tracing.py` (extend)

**Interfaces:**
- Consumes: `_ChainRun` (Task 1).
- Produces:
  - `_agent_outputs(result: MODEL_T | None, stats: ExtractorRunStats) -> dict[str, Any]`
  - `_run_extractor_agent_with_model(...)` now takes `trace: _ChainRun` instead of `callbacks: list[Any] | None`; LLM calls use `trace.child_callbacks()`.
  - `run_extractor_agent` opens `saidex.agent_loop` chain run, closes it in `_finish`, fires `on_chain_error` on unexpected exceptions.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_callbacks_tracing.py` (imports for the mock helpers reuse the patterns from `tests/test_agent_loop.py`):

```python
from unittest.mock import AsyncMock, MagicMock

from pydantic import BaseModel

from saidex import Tool, run_extractor_agent
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_callbacks_tracing.py -k wraps_generations -v`
Expected: FAIL — `chain_start` event absent (currently no chain run) / `KeyError` on `config`.

- [ ] **Step 3: Add the `_agent_outputs` helper**

In `src/saidex/extractor.py`, add near the top of the "Internal implementation" section (just before `_run_extractor_agent_with_model`, line ~932):

```python
def _agent_outputs(result: Any, stats: ExtractorRunStats) -> dict[str, Any]:
    """Chain-run outputs for the agent loop: the result plus run-level metrics."""
    return {
        "result": result,
        "success": stats.success,
        "failure_reason": stats.failure_reason,
        "iterations": stats.iterations,
        "tool_calls": stats.tool_calls,
        "validation_retries": stats.validation_retries,
        "fallback_used": stats.fallback_used,
        "problem_fields": list(stats.problem_fields),
        "format_errors": stats.format_errors,
    }
```

Add the import at the top of the file (with the other local imports, after line 32):

```python
from ._callbacks import _ChainRun
```

- [ ] **Step 4: Open/close the chain run in `run_extractor_agent`**

In `run_extractor_agent`, after `grounding_source = _source_text_from_messages(original_messages)` (line ~830), insert:

```python
    trace = await _ChainRun.start(
        callbacks,
        name="saidex.agent_loop",
        inputs={
            "schema": schema.__name__,
            "mode": final_answer_mode.value,
            "tools": [t.name for t in tools],
            **({"text": grounding_source} if capture_source_text else {}),
        },
    )
```

Change the `_finish` inner function so it closes the chain run — replace its body's `return result, stats` tail so the whole function reads:

```python
    async def _finish(
        result: MODEL_T | None, stats: ExtractorRunStats
    ) -> tuple[MODEL_T | None, ExtractorRunStats]:
        source_text = _source_text_from_messages(original_messages) if capture_source_text else None
        if source_text is not None:
            stats = replace(stats, source_text=source_text)
        await _emit_completion(
            on_complete,
            ExtractionEvent(
                schema_name=schema.__name__,
                result=result,
                stats=stats,
                messages=original_messages,
                source_text=source_text,
            ),
        )
        await trace.end(_agent_outputs(result, stats))
        return result, stats
```

Wrap the body from the first `_run_extractor_agent_with_model` call to the final `return` in a `try/except` that reports an unexpected error. Concretely, change the two call sites to pass `trace=trace` instead of `callbacks=callbacks`, and wrap the existing logic:

```python
    try:
        result, primary_stats = await _run_extractor_agent_with_model(
            llm_model=llm_model,
            schema=schema,
            messages=list(original_messages),
            tools=tools,
            final_answer_mode=final_answer_mode,
            trace=trace,
            max_iterations=max_iterations,
            max_validation_retries=max_validation_retries,
            model_label="primary",
            retry_config=effective_retry_config,
            validator=validator,
            source_text=grounding_source,
        )

        if result is not None:
            return await _finish(result, primary_stats)

        if fallback_llm_model is not None:
            # ... existing fallback block unchanged, EXCEPT the inner call passes
            #     trace=trace (not callbacks=callbacks) ...
            result, fallback_stats = await _run_extractor_agent_with_model(
                llm_model=fallback_llm_model,
                schema=schema,
                messages=fallback_messages,
                tools=tools,
                final_answer_mode=final_answer_mode,
                trace=trace,
                max_iterations=max_iterations,
                max_validation_retries=max_validation_retries,
                model_label="fallback",
                retry_config=effective_retry_config,
                validator=validator,
                source_text=grounding_source,
            )
            # ... existing combined-stats construction unchanged ...
            return await _finish(result, combined)

        # ... existing "no fallback configured" logging unchanged ...
        return await _finish(None, primary_stats)
    except Exception as exc:
        await trace.error(exc)
        raise
```

> Keep every existing line inside the fallback block (the `logger.info`, `fallback_messages` construction, `combined` stats, success/failure logging) exactly as it is today — the only edits are: (a) `callbacks=callbacks` → `trace=trace` at both call sites, (b) the surrounding `try/except`, (c) `_finish` now calls `trace.end`.

- [ ] **Step 5: Thread `trace` through `_run_extractor_agent_with_model`**

Change the signature (line ~932): replace the parameter `callbacks: list[Any] | None,` with `trace: _ChainRun,` (same position).

Replace the `_invoke` closure inside the loop (lines ~992-995):

```python
            async def _invoke() -> Any:
                child = trace.child_callbacks()
                if child is not None:
                    return await llm.ainvoke(messages, config={"callbacks": child})
                return await llm.ainvoke(messages)
```

(The helper-tool execution block is updated in Task 4 — leave it calling `tool_obj.execute(...)` for now so this task stays green.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_callbacks_tracing.py tests/test_agent_loop.py -v`
Expected: PASS — new `wraps_generations` test plus all existing agent-loop tests.

- [ ] **Step 7: Lint + type-check**

Run: `uv run ruff check src/saidex/extractor.py && uv run mypy src/`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add src/saidex/extractor.py tests/test_callbacks_tracing.py
git commit -m "feat(observability): wrap the agent loop in a chain run"
```

---

### Task 4: Agent-loop tool spans

**Files:**
- Modify: `src/saidex/extractor.py` (`_run_extractor_agent_with_model` — helper-tool block)
- Test: `tests/test_callbacks_tracing.py` (extend)

**Interfaces:**
- Consumes: `_ChainRun.tool_span` (Task 1), `Tool._invoke` / `_ToolOutcome` (Task 2), the `trace` parameter (Task 3).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_callbacks_tracing.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_callbacks_tracing.py -k "tool_span or handler_crash or unknown_tool" -v`
Expected: FAIL — no `tool_start`/`tool_error` events (tools still run via bare `execute`).

- [ ] **Step 3: Wrap the helper-tool execution**

In `_run_extractor_agent_with_model`, replace the helper-tool branch (the `else:` block at lines ~1276-1296, starting `total_tool_calls += 1`) with:

```python
            # ── Helper tool ───────────────────────────────────────────────
            else:
                total_tool_calls += 1
                tool_obj = tool_by_name.get(tc_name)
                if tool_obj is None:
                    logger.warning("%s: agent loop called unknown tool '%s'", model_label, tc_name)
                    unknown_msg = (
                        f"Unknown tool '{tc_name}'. Available tools: {', '.join(tool_by_name)}."
                    )
                    async with trace.tool_span(tc_name, tc_args) as span:
                        span.record_output(unknown_msg)
                    tool_result_messages.append(
                        ToolMessage(content=unknown_msg, tool_call_id=tc_id)
                    )
                    continue

                async with trace.tool_span(tc_name, tc_args) as span:
                    outcome = await tool_obj._invoke(tc_args)
                    if outcome.handler_error is not None:
                        span.record_error(outcome.handler_error)
                    else:
                        span.record_output(outcome.content)
                    result_str = outcome.content
                logger.debug(
                    "%s: agent loop tool '%s' → %s", model_label, tc_name, result_str[:120]
                )
                tool_result_messages.append(ToolMessage(content=result_str, tool_call_id=tc_id))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_callbacks_tracing.py tests/test_agent_loop.py -v`
Expected: PASS.

- [ ] **Step 5: Lint + type-check**

Run: `uv run ruff check src/saidex/extractor.py tests/test_callbacks_tracing.py && uv run mypy src/`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/saidex/extractor.py tests/test_callbacks_tracing.py
git commit -m "feat(observability): emit tool spans for agent-loop tool calls"
```

---

### Task 5: `extract_data` / `extract_data_list` chain runs

**Files:**
- Modify: `src/saidex/extractor.py` (`extract_data`, `extract_data_list`, `_try_with_model`, add `_extract_outputs`)
- Test: `tests/test_callbacks_tracing.py` (extend)

**Interfaces:**
- Consumes: `_ChainRun` (Task 1).
- Produces:
  - `_extract_outputs(result: Any, stats: ExtractDataStats) -> dict[str, Any]`
  - `extract_data(...)` gains a private param `_trace: _ChainRun | None = None`; opens `saidex.extract_data` when `_trace is None`, otherwise reuses the parent trace (and does not close it).
  - `_try_with_model(...)` takes `trace: _ChainRun` instead of `callbacks: list[Any] | None`.
  - `extract_data_list(...)` opens `saidex.extract_data_list` and passes `_trace=` to the inner `extract_data`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_callbacks_tracing.py`:

```python
from saidex import IbanStr, extract_data, extract_data_list
from langchain_core.messages import HumanMessage


class _Bank(BaseModel):
    holder: str
    iban: IbanStr


def _plain_llm(responses: list[MagicMock]) -> MagicMock:
    bound = MagicMock()
    bound.ainvoke = AsyncMock(side_effect=responses)
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=bound)
    llm.ainvoke = bound.ainvoke
    return llm


@pytest.mark.asyncio
async def test_extract_data_wraps_retries_in_one_chain_run() -> None:
    rec = RecordingHandler()
    # First attempt: bad IBAN → validation retry. Second: valid.
    r1 = _resp([{"name": "_Bank", "args": {"holder": "ACME", "iban": "not-an-iban"}, "id": "a"}])
    r2 = _resp([{"name": "_Bank", "args": {"holder": "ACME", "iban": "DE89370400440532013000"}, "id": "b"}])
    llm = _plain_llm([r1, r2])

    result, stats = await extract_data(
        llm, _Bank, [HumanMessage(content="pay ACME")], callbacks=[rec], retry_config=_NO_RETRY
    )

    assert result is not None
    starts = [e for e in rec.events if e[0] == "chain_start"]
    ends = [e for e in rec.events if e[0] == "chain_end"]
    assert starts == [("chain_start", "saidex.extract_data")]  # exactly one chain run
    assert len(ends) == 1
    assert ends[0][1]["success"] is True
    assert ends[0][1]["total_retries"] == stats.total_retries


@pytest.mark.asyncio
async def test_extract_data_list_uses_single_chain_run() -> None:
    rec = RecordingHandler()
    llm = _plain_llm(
        [_resp([{"name": "_FinalList", "args": {"items": [{"result": "a"}]}, "id": "x"}])]
    )
    # Container tool name is "<schema>List"; patch response name accordingly.
    llm.bind_tools().ainvoke.side_effect = [
        _resp([{"name": "_FinalList", "args": {"items": [{"result": "a"}]}, "id": "x"}])
    ]

    items, _ = await extract_data_list(
        llm, _Final, [HumanMessage(content="list them")], callbacks=[rec], retry_config=_NO_RETRY
    )

    assert items is not None and len(items) == 1
    starts = [e for e in rec.events if e[0] == "chain_start"]
    assert starts == [("chain_start", "saidex.extract_data_list")]  # not the inner extract_data
```

> Note for the implementer: the container schema built by `_build_list_container` is named `<schema>List` (here `_FinalList`); the mock's tool-call `name` must match. If matching proves brittle with the mock, assert only on the **number and names of `chain_start` events** (exactly one, `saidex.extract_data_list`) — the invariant under test is "no duplicate inner chain run", which does not depend on a successful final parse.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_callbacks_tracing.py -k "wraps_retries or single_chain_run" -v`
Expected: FAIL — no `chain_start` events for `extract_data` / `extract_data_list`.

- [ ] **Step 3: Add `_extract_outputs`**

In `src/saidex/extractor.py`, add near `_agent_outputs` (Task 3):

```python
def _extract_outputs(result: Any, stats: ExtractDataStats) -> dict[str, Any]:
    """Chain-run outputs for extract_data / extract_data_list: result + metrics."""
    return {
        "result": result,
        "success": stats.success,
        "failure_reason": stats.failure_reason,
        "total_retries": stats.total_retries,
        "primary_retries": stats.primary_retries,
        "fallback_retries": stats.fallback_retries,
        "fallback_used": stats.fallback_used,
        "item_count": stats.item_count,
        "problem_fields": list(stats.problem_fields),
        "format_errors": stats.format_errors,
    }
```

- [ ] **Step 4: Wire `extract_data`**

Add the private parameter to `extract_data`'s signature (alongside `_notify_observers`, line ~159):

```python
    _notify_observers: bool = True,
    _trace: _ChainRun | None = None,
```

After `grounding_source = _source_text_from_messages(original_messages)` (line ~246), insert:

```python
    trace = _trace if _trace is not None else await _ChainRun.start(
        callbacks,
        name="saidex.extract_data",
        inputs={
            "schema": schema.__name__,
            "mode": mode.value,
            **({"text": grounding_source} if capture_source_text else {}),
        },
    )
    owns_trace = _trace is None
```

In the `_finish` inner function, add before `return result, stats`:

```python
        if owns_trace:
            await trace.end(_extract_outputs(result, stats))
```

Wrap the body from the first `_try_with_model` call through the final `return await _finish(...)` in:

```python
    try:
        primary = await _try_with_model(
            llm_model=llm_model,
            schema=schema,
            messages=list(original_messages),
            trace=trace,
            max_retries=max_primary_retries,
            model_label="primary",
            retry_config=effective_retry_config,
            mode=mode,
            validator=validator,
            source_text=grounding_source,
        )
        # ... existing primary/fallback/return logic unchanged, EXCEPT the fallback
        #     _try_with_model call also uses trace=trace instead of callbacks=callbacks ...
    except Exception as exc:
        if owns_trace:
            await trace.error(exc)
        raise
```

Change **both** `_try_with_model(...)` call sites: `callbacks=callbacks` → `trace=trace`.

- [ ] **Step 5: Thread `trace` through `_try_with_model`**

Change the signature (line ~1347): replace `callbacks: list[Any] | None,` with `trace: _ChainRun,`.

Replace the `_invoke` closure (lines ~1391-1394):

```python
            async def _invoke() -> Any:
                child = trace.child_callbacks()
                if child is not None:
                    return await llm.ainvoke(messages, config={"callbacks": child})
                return await llm.ainvoke(messages)
```

- [ ] **Step 6: Wire `extract_data_list`**

In `extract_data_list`, after `container = _build_list_container(schema)` (line ~556), open a chain run and wrap the inner call. Replace the inner `extract_data(...)` call (lines ~576-589) and the trailing `_emit_completion` so the function opens/passes/closes one trace:

```python
    grounding_source = _source_text_from_messages(list(messages))
    trace = await _ChainRun.start(
        callbacks,
        name="saidex.extract_data_list",
        inputs={
            "schema": schema.__name__,
            "mode": mode.value,
            **({"text": grounding_source} if capture_source_text else {}),
        },
    )
    try:
        result, stats = await extract_data(
            llm_model=llm_model,
            schema=container,
            messages=messages,
            mode=mode,
            callbacks=None,
            fallback_llm_model=fallback_llm_model,
            max_primary_retries=max_primary_retries,
            max_fallback_retries=max_fallback_retries,
            retry_config=retry_config,
            capture_source_text=capture_source_text,
            validator=container_validator,
            _notify_observers=False,
            _trace=trace,
        )
        # ... existing item_issues / stats reshaping unchanged ...
        item_issues = tuple(replace(i, schema_name=schema.__name__) for i in stats.field_issues)
        stats = replace(stats, schema_name=schema.__name__, field_issues=item_issues)
        items: list[MODEL_T] | None = None if result is None else cast(Any, result).items
        if items is not None:
            stats = replace(stats, item_count=len(items))

        await _emit_completion(
            on_complete,
            ExtractionEvent(
                schema_name=schema.__name__,
                result=cast("BaseModel | list[BaseModel] | None", items),
                stats=stats,
                messages=list(messages),
                source_text=stats.source_text,
            ),
        )
        await trace.end(_extract_outputs(items, stats))
        return items, stats
    except Exception as exc:
        await trace.error(exc)
        raise
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_callbacks_tracing.py tests/test_extractor.py tests/test_batch.py -v`
Expected: PASS — new chain-run tests plus all existing extractor/batch tests.

- [ ] **Step 8: Lint + type-check**

Run: `uv run ruff check src/saidex/extractor.py tests/test_callbacks_tracing.py && uv run mypy src/`
Expected: no errors.

- [ ] **Step 9: Commit**

```bash
git add src/saidex/extractor.py tests/test_callbacks_tracing.py
git commit -m "feat(observability): wrap extract_data and batch runs in chain runs"
```

---

### Task 6: Sync-path test, full suite, docs & PR

**Files:**
- Test: `tests/test_callbacks_tracing.py` (add sync-path test)
- Modify: `docs/langfuse-tracing.md`, `docs/observability.md`, `docs/agent-loop.md`

**Interfaces:**
- Consumes: everything above. No new production code.

- [ ] **Step 1: Add the sync-wrapper test**

Add to `tests/test_callbacks_tracing.py`:

```python
from saidex import extract_data_with_tools_sync


def test_sync_wrapper_fires_callbacks() -> None:
    rec = RecordingHandler()
    llm = _llm([_resp([{"name": "_Final", "args": {"result": "done"}, "id": "c1"}])])

    result, _ = extract_data_with_tools_sync(
        llm, _Final, "do it", tools=[], callbacks=[rec], retry_config=_NO_RETRY
    )

    assert result is not None
    names = [e[0] for e in rec.events]
    assert names[0] == "chain_start" and names[-1] == "chain_end"
```

- [ ] **Step 2: Run the full test suite with coverage**

Run: `uv run pytest --cov --cov-report=term-missing`
Expected: PASS — entire suite green; `src/saidex/_callbacks.py` covered.

- [ ] **Step 3: Update `docs/langfuse-tracing.md` (scenario 3)**

Replace the "**What the trace shows:**" paragraph at the end of section "## 3. The agent loop" (lines ~243-246) with:

```markdown
**What the trace shows:** one `saidex.agent_loop` span for the whole loop. Under
it sit each tool-deciding LLM generation **and** a dedicated span for every tool
your model calls — `get_order_status` appears as its own `on_tool_start` /
`on_tool_end` span with the arguments as input and the tool result as output. If
a tool handler raises, its span is marked as an error (`on_tool_error`) while the
loop keeps running. The final-answer schema call is not a tool span — it is the
loop's output, attached to the `saidex.agent_loop` span. Reconcile the tree with
`stats.iterations` and `stats.tool_calls`.
```

Also update the intro of "## 1. A single extraction" — after "Every LLM call SAIDEX makes is then recorded automatically." add:

```markdown
Each extraction is wrapped in a single enclosing span (`saidex.extract_data`,
`saidex.extract_data_list`, or `saidex.agent_loop`) so retries and — in the agent
loop — tool executions nest under one logical run instead of appearing side by
side.
```

- [ ] **Step 4: Update `docs/observability.md` (Callbacks section)**

Replace the "## Callbacks" body (lines ~12-23) so it reads:

```markdown
Both `extract_data_from_text` and `extract_data` accept a `callbacks`
parameter.  Pass any list of LangChain `BaseCallbackHandler` instances to
get automatic tracing.  SAIDEX wraps each extraction in one enclosing chain-run
span (`saidex.extract_data`, `saidex.extract_data_list`, or `saidex.agent_loop`)
and nests every LLM call — including retry attempts — under it.  In the agent
loop, each helper-tool execution additionally fires `on_tool_start` /
`on_tool_end` (or `on_tool_error` when the handler raises), so tools appear as
their own spans:

```python
result, stats = await extract_data(
    llm,
    MySchema,
    messages,
    callbacks=[my_handler],
)
```
```

- [ ] **Step 5: Update `docs/agent-loop.md`**

Find the section covering observability/tracing in `docs/agent-loop.md` (search for "callbacks" or "trace"). Add — or extend the existing tracing note with — this paragraph:

```markdown
### Tracing tool calls

When you pass `callbacks=[...]`, the agent loop is wrapped in a single
`saidex.agent_loop` span and **every helper-tool execution is traced** as its own
`on_tool_start` / `on_tool_end` span (or `on_tool_error` if the handler raises),
nested under the loop alongside the LLM generations. See
[Langfuse Tracing](langfuse-tracing.md#3-the-agent-loop) for a full walkthrough.
```

If `docs/agent-loop.md` has no tracing section, add the block above under a new `## Observability` heading near the end, before any "Related" section.

- [ ] **Step 6: Full CI gate**

Run:
```bash
uv run ruff check src/ tests/ && \
uv run ruff format --check src/ tests/ && \
uv run mypy src/ && \
uv run pytest --cov --cov-report=term-missing
```
Expected: all four green.

- [ ] **Step 7: Commit docs**

```bash
git add docs/langfuse-tracing.md docs/observability.md docs/agent-loop.md tests/test_callbacks_tracing.py
git commit -m "docs(observability): document tool-call tracing in the agent loop"
```

- [ ] **Step 8: Push and open the PR**

```bash
git push -u origin feature/callback-tool-events
gh pr create --base develop --title "feat(observability): trace tool calls and wrap runs in chain spans" --fill
```

PR body must explain *what* (fire LangChain `on_tool_*` / `on_chain_*` events so tracers see tool executions and a nested run tree) and *why* (Langfuse/LangSmith previously saw only raw generations; the docs already promised tool visibility). Note the behaviour change: **trace shape changes** (generations now nest under a SAIDEX chain-run span; tool executions are new spans) while the **public API is unchanged** — not a breaking change. Reference the design spec `docs/superpowers/specs/2026-07-06-callback-tool-events-design.md`.

---

## Self-Review

**Spec coverage:**
- Fire `on_tool_start` / `on_tool_end` / `on_tool_error` for helper tools → Task 4. ✅
- Wrap agent loop + `extract_data` / `extract_data_list` in chain runs → Tasks 3, 5. ✅
- Tracer-agnostic, reuse `callbacks=`, no new dependency → Task 1 (only `langchain-core`). ✅
- Observability never breaks a run → Task 1 isolation tests + integration test in Task 4 (`handler_crash`) + Task 1 `broken_handler`. ✅
- Handler exception → `on_tool_error` with real exception → Tasks 2 + 4. ✅
- Final-answer schema call is chain output, not a tool span → Task 3 (`_agent_outputs`), no tool span fired for `final_tool_name`. ✅
- PII-light inputs (text only when `capture_source_text`) → Tasks 3, 5. ✅
- Chain-end metrics mirror `on_complete` → `_agent_outputs` / `_extract_outputs`. ✅
- `extract_data_list` single chain run via `_trace` → Task 5. ✅
- Sync path works → Task 6 test. ✅
- Docs updated (langfuse-tracing, observability, agent-loop) → Task 6. ✅
- `feat(...)` commits → Commitizen changelog; no hand-edit of `CHANGELOG.md`. ✅

**Placeholder scan:** No `TBD`/`TODO`; every code step shows complete code. ✅

**Type consistency:** `_ChainRun.start` / `child_callbacks` / `tool_span` / `end` / `error`, `_ToolSpan.record_output` / `record_error`, `Tool._invoke` → `_ToolOutcome(content, handler_error)`, `_agent_outputs` / `_extract_outputs` — names and signatures identical across the task that defines them and the tasks that consume them. The `trace: _ChainRun` parameter replaces `callbacks` in both `_run_extractor_agent_with_model` (Task 3) and `_try_with_model` (Task 5) consistently. ✅
