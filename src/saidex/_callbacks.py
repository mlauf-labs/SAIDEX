"""LangChain callback plumbing for the agent loop and retry loops.

Wraps ``langchain_core``'s :class:`AsyncCallbackManager` into a small
:class:`_ChainRun` that opens one enclosing chain run, hands a child callback
manager to each ``llm.ainvoke`` so generations nest, and brackets each helper
tool execution with ``on_tool_start`` / ``on_tool_end`` / ``on_tool_error``.

Private module — not part of the public API.  When no ``callbacks`` are passed
a no-op instance is returned, so there is zero overhead and byte-for-byte
unchanged behaviour.  Every callback invocation is isolated: a raising handler
is logged and swallowed and can never break the extraction it observes.  Note
that isolation is per *event*, not per handler: langchain-core dispatches one
event to all handlers together, so a handler configured with
``raise_error=True`` that raises can suppress that event (and its span) for
every handler in the list.  With the default ``raise_error=False`` langchain
swallows handler errors itself and other handlers are unaffected.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from typing import Any, TypeVar

from langchain_core.callbacks import (
    AsyncCallbackManager,
    AsyncCallbackManagerForChainRun,
    AsyncCallbackManagerForToolRun,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


async def _suppressed(awaitable: Awaitable[_T], what: str) -> _T | None:
    """Await *awaitable*, logging and swallowing any exception it raises.

    Args:
        awaitable: The callback invocation to shield the extraction from.
        what: Human-readable label for the log line.

    Returns:
        The awaitable's result, or ``None`` when it raised.
    """
    try:
        return await awaitable
    except Exception as exc:  # noqa: BLE001 — observability must not break the run
        logger.error("%s raised and was suppressed: %s", what, exc)
        return None


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
        if self._error is not None:
            await _suppressed(self._run.on_tool_error(self._error), "tool-span end callback")
        else:
            await _suppressed(self._run.on_tool_end(self._output), "tool-span end callback")


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
        """Open a chain run for *callbacks*, or return a no-op when there are none.

        When opening the chain run itself fails (e.g. a handler with
        ``raise_error=True`` raising in ``on_chain_start``), tracing degrades to
        the un-nested pre-chain-run behaviour instead of going dark: the
        configured manager is kept as the child, so LLM generations and tool
        spans still reach every handler — they just have no enclosing parent.
        """
        if not callbacks:
            return cls(None, None)
        manager: AsyncCallbackManager | None = None
        try:
            manager = AsyncCallbackManager.configure(
                inheritable_callbacks=callbacks,
                inheritable_metadata=metadata,
            )
            run = await manager.on_chain_start({"name": name}, inputs, name=name)
            return cls(run, run.get_child())
        except Exception as exc:  # noqa: BLE001 — observability must not break the run
            logger.error("failed to start trace chain run '%s': %s", name, exc)
            return cls(None, manager)

    def child_callbacks(self) -> AsyncCallbackManager | None:
        """Child manager to hand to ``llm.ainvoke(config={'callbacks': ...})``."""
        return self._child

    async def invoke_llm(self, llm: Any, messages: Any) -> Any:
        """Invoke *llm* with the child callback manager when tracing is active.

        The single choke point through which the agent loop and the retry loop
        run their LLM calls, so the generations nest under this chain run.  LLM
        errors propagate unchanged — only observability is layered on here.

        Args:
            llm: The (bound) chat model to invoke.
            messages: The message list to pass to ``ainvoke``.

        Returns:
            Whatever ``llm.ainvoke`` returns.
        """
        if self._child is not None:
            return await llm.ainvoke(messages, config={"callbacks": self._child})
        return await llm.ainvoke(messages)

    @asynccontextmanager
    async def tool_span(self, name: str, args: dict[str, Any]) -> AsyncIterator[_ToolSpan]:
        """Bracket a tool execution with ``on_tool_start`` / ``on_tool_end``|``on_tool_error``.

        An exception escaping the ``async with`` body before an outcome was
        recorded (including :class:`BaseException` such as a task cancellation)
        closes the span via ``on_tool_error`` and re-raises — never as a
        false-success ``on_tool_end``.
        """
        run: AsyncCallbackManagerForToolRun | None = None
        if self._child is not None:
            try:
                input_str = json.dumps(args, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                input_str = str(args)
            run = await _suppressed(
                self._child.on_tool_start({"name": name}, input_str, name=name),
                "tool-span start callback",
            )
        span = _ToolSpan(run)
        try:
            yield span
        except BaseException as exc:
            span.record_error(exc)
            raise
        finally:
            await span._finish()

    async def end(self, outputs: dict[str, Any]) -> None:
        """Close the chain run with ``on_chain_end``.  Idempotent."""
        if self._run is None or self._closed:
            return
        self._closed = True
        await _suppressed(self._run.on_chain_end(outputs), "chain-end callback")

    async def error(self, error: BaseException) -> None:
        """Close the chain run with ``on_chain_error``.  Idempotent."""
        if self._run is None or self._closed:
            return
        self._closed = True
        await _suppressed(self._run.on_chain_error(error), "chain-error callback")
