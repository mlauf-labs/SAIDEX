"""Scoped and global observers for extraction and tool-node statistics.

`collect_stats` is a scoped context manager — usable with either ``async with``
or plain ``with`` — that captures the stats of every extraction or
``SaidexToolNode`` run inside its block, so callers need not thread a stats
object through intermediate signatures.  `on_extraction` and `on_tool_node`
register process-wide listeners for observability integrations (logging,
tracing, metrics).
"""

from __future__ import annotations

import contextlib
import inspect
import logging
from collections.abc import Awaitable, Callable, Iterator
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import (
        ExtractDataStats,
        ExtractionEvent,
        ExtractorRunStats,
        ToolNodeEvent,
        ToolNodeStats,
    )

logger = logging.getLogger(__name__)

#: Signature of an ``on_extraction`` listener: a sync **or** async callable that
#: receives the completed :class:`~saidex.models.ExtractionEvent`.
ExtractionListener = Callable[["ExtractionEvent"], "Awaitable[None] | None"]

#: Signature of an ``on_tool_node`` listener: a sync **or** async callable that
#: receives the completed :class:`~saidex.models.ToolNodeEvent`.
ToolNodeListener = Callable[["ToolNodeEvent"], "Awaitable[None] | None"]


class StatsSink:
    """Collects the stats of every extraction or tool-node run inside a
    ``collect_stats`` block.

    You do not construct this directly — :func:`collect_stats` yields one.  Only
    each run's stats object is stored (never its messages, result, or source
    text), so a sink stays PII-light by default.
    """

    def __init__(self) -> None:
        self._stats: list[ExtractDataStats | ExtractorRunStats | ToolNodeStats] = []

    def _record(self, event: ExtractionEvent | ToolNodeEvent) -> None:
        """Append one completed run's stats.  Never raises."""
        self._stats.append(event.stats)

    def all(self) -> list[ExtractDataStats | ExtractorRunStats | ToolNodeStats]:
        """Return the collected stats, oldest first.

        Returns:
            A new list holding one stats object per extraction or tool-node
            invocation that completed inside the ``collect_stats`` block, in
            chronological order.
        """
        return list(self._stats)

    def __len__(self) -> int:
        return len(self._stats)

    def __iter__(self) -> Iterator[ExtractDataStats | ExtractorRunStats | ToolNodeStats]:
        return iter(list(self._stats))


_active_sinks: ContextVar[tuple[StatsSink, ...]] = ContextVar("saidex_active_sinks", default=())
_global_listeners: list[ExtractionListener] = []
_tool_node_listeners: list[ToolNodeListener] = []


class _StatsCollector:
    """Context manager returned by :func:`collect_stats`.

    Implements both the sync and async context-manager protocols so the same
    object works with ``with`` and ``async with``.
    """

    def __init__(self) -> None:
        self._sink = StatsSink()
        self._token: Token[tuple[StatsSink, ...]] | None = None

    def _push(self) -> StatsSink:
        self._token = _active_sinks.set((*_active_sinks.get(), self._sink))
        return self._sink

    def _pop(self) -> None:
        if self._token is not None:
            _active_sinks.reset(self._token)
            self._token = None

    def __enter__(self) -> StatsSink:
        return self._push()

    def __exit__(self, *exc: object) -> None:
        self._pop()

    async def __aenter__(self) -> StatsSink:
        return self._push()

    async def __aexit__(self, *exc: object) -> None:
        self._pop()


def collect_stats() -> _StatsCollector:
    """Create a scoped sink for extraction statistics.

    The returned object is a context manager — usable with either ``async with``
    or plain ``with`` — that yields a :class:`StatsSink`.  Every extraction that
    completes inside the block has its stats appended to that sink, with no need
    to capture return values or thread a stats object through call signatures.

    Nesting works: an inner ``collect_stats`` collects only its own extractions,
    while any enclosing ``collect_stats`` collects those too.

    Returns:
        A :class:`_StatsCollector` yielding a :class:`StatsSink` on entry.
    """
    return _StatsCollector()


class Subscription:
    """Handle returned by :func:`on_extraction` and :func:`on_tool_node`, used
    to remove the listener.

    Removing the listener can be done three equivalent, idempotent ways: call
    the object, call :meth:`unsubscribe`, or use it as a context manager (the
    listener is removed on block exit).
    """

    def __init__(self, callback: Any, registry: list[Any] | None = None) -> None:
        self._callback = callback
        self._registry: list[Any] = _global_listeners if registry is None else registry
        self._active = True

    def unsubscribe(self) -> None:
        """Remove the registered listener.  Safe to call more than once."""
        if self._active:
            self._active = False
            with contextlib.suppress(ValueError):
                self._registry.remove(self._callback)

    def __call__(self) -> None:
        self.unsubscribe()

    def __enter__(self) -> Subscription:
        return self

    def __exit__(self, *exc: object) -> None:
        self.unsubscribe()


def on_extraction(callback: ExtractionListener) -> Subscription:
    """Register a process-wide listener invoked after every extraction.

    The listener fires once per top-level extraction, on both success and
    failure, receiving the full :class:`~saidex.models.ExtractionEvent` (stats,
    result, messages, source text).  It may be sync or async.  Any exception it
    raises is logged and swallowed so an observer can never break the run.

    Args:
        callback: A sync or async callable receiving the completed
            :class:`~saidex.models.ExtractionEvent`.

    Returns:
        A :class:`Subscription` that removes the listener when called, when its
        :meth:`Subscription.unsubscribe` method runs, or on context-manager exit.
    """
    _global_listeners.append(callback)
    return Subscription(callback)


def on_tool_node(callback: ToolNodeListener) -> Subscription:
    """Register a process-wide listener invoked after every ``SaidexToolNode`` run.

    The listener fires once per node invocation, receiving the completed
    :class:`~saidex.models.ToolNodeEvent`.  It may be sync or async.  Any
    exception it raises is logged and swallowed so an observer can never break
    the graph run.  ``on_extraction`` listeners do **not** receive these events.

    Args:
        callback: A sync or async callable receiving the completed
            :class:`~saidex.models.ToolNodeEvent`.

    Returns:
        A :class:`Subscription` that removes the listener when called, when its
        :meth:`Subscription.unsubscribe` method runs, or on context-manager exit.
    """
    _tool_node_listeners.append(callback)
    return Subscription(callback, _tool_node_listeners)


async def dispatch_to_observers(event: ExtractionEvent) -> None:
    """Notify every active sink and global listener of a completed extraction.

    Invoked once per extraction from the single completion choke point.  Returns
    immediately when neither a sink nor a listener is active, so inactive
    observers add no measurable overhead.  A failing observer is isolated: its
    exception is logged with a traceback and swallowed so it neither breaks the
    extraction nor stops the remaining observers from being notified.

    Args:
        event: The completed extraction's event.
    """
    sinks = _active_sinks.get()
    if not sinks and not _global_listeners:
        return
    for sink in sinks:
        try:
            sink._record(event)
        except Exception as exc:  # noqa: BLE001 — observers must not break the run
            logger.error("stats sink failed to record and was skipped: %s", exc, exc_info=True)
    for listener in tuple(_global_listeners):
        try:
            outcome = listener(event)
            if inspect.isawaitable(outcome):
                await outcome
        except Exception as exc:  # noqa: BLE001 — observers must not break the run
            logger.error("on_extraction listener raised and was suppressed: %s", exc, exc_info=True)


async def dispatch_tool_node_event(event: ToolNodeEvent) -> None:
    """Notify every active sink and ``on_tool_node`` listener of a node run.

    Mirrors :func:`dispatch_to_observers` for tool-node events: sinks record
    the stats object, global tool-node listeners receive the full event, and a
    failing observer is logged and isolated.

    Args:
        event: The completed node invocation's event.
    """
    sinks = _active_sinks.get()
    if not sinks and not _tool_node_listeners:
        return
    for sink in sinks:
        try:
            sink._record(event)
        except Exception as exc:  # noqa: BLE001 — observers must not break the run
            logger.error("stats sink failed to record and was skipped: %s", exc, exc_info=True)
    for listener in tuple(_tool_node_listeners):
        try:
            outcome = listener(event)
            if inspect.isawaitable(outcome):
                await outcome
        except Exception as exc:  # noqa: BLE001 — observers must not break the run
            logger.error("on_tool_node listener raised and was suppressed: %s", exc, exc_info=True)
