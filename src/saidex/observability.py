"""Scoped observer for extraction statistics.

`collect_stats` is a scoped context manager — usable with either ``async with``
or plain ``with`` — that captures the stats of every extraction inside its
block, so callers need not thread a stats object through intermediate
signatures.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import ExtractDataStats, ExtractionEvent, ExtractorRunStats


class StatsSink:
    """Collects the stats of every extraction inside a ``collect_stats`` block.

    You do not construct this directly — :func:`collect_stats` yields one.  Only
    each extraction's stats object is stored (never its messages, result, or
    source text), so a sink stays PII-light by default.
    """

    def __init__(self) -> None:
        self._stats: list[ExtractDataStats | ExtractorRunStats] = []

    def _record(self, event: ExtractionEvent) -> None:
        """Append one completed extraction's stats.  Never raises."""
        self._stats.append(event.stats)

    def all(self) -> list[ExtractDataStats | ExtractorRunStats]:
        """Return the collected stats, oldest first.

        Returns:
            A new list holding one stats object per extraction that completed
            inside the ``collect_stats`` block, in chronological order.
        """
        return list(self._stats)

    def __len__(self) -> int:
        return len(self._stats)

    def __iter__(self) -> Iterator[ExtractDataStats | ExtractorRunStats]:
        return iter(list(self._stats))


_active_sinks: ContextVar[tuple[StatsSink, ...]] = ContextVar("saidex_active_sinks", default=())


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


async def dispatch_to_observers(event: ExtractionEvent) -> None:
    """Notify every active sink of a completed extraction.

    Invoked once per extraction from the single completion choke point.  Returns
    immediately when no sink is active, so inactive observers add no measurable
    overhead.

    Args:
        event: The completed extraction's event.
    """
    sinks = _active_sinks.get()
    if not sinks:
        return
    for sink in sinks:
        sink._record(event)
