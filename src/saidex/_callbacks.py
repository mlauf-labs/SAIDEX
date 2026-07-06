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
