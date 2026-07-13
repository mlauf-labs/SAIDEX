"""Tests for ToolCallStats/ToolNodeStats/ToolNodeEvent and their observability plumbing.

These types intentionally have no langgraph dependency — they must be
importable from core `saidex` without the extra installed.
"""

from __future__ import annotations

import pytest

from saidex import (
    FieldIssue,
    ToolCallStats,
    ToolNodeEvent,
    ToolNodeStats,
    collect_stats,
    on_extraction,
    on_tool_node,
)
from saidex.observability import dispatch_tool_node_event

# ---------------------------------------------------------------------------
# Dataclass shape
# ---------------------------------------------------------------------------


def _issue(field: str = "a") -> FieldIssue:
    return FieldIssue(
        schema_name="add",
        field_path=field,
        category="type",
        error_type="int_parsing",
        message="not an int",
        attempt=0,
    )


def test_tool_node_stats_aggregates() -> None:
    stats = ToolNodeStats(
        calls=(
            ToolCallStats(tool_name="add", tool_call_id="c1", outcome="executed"),
            ToolCallStats(
                tool_name="add",
                tool_call_id="c2",
                outcome="corrected",
                correction_retries=1,
            ),
            ToolCallStats(
                tool_name="add",
                tool_call_id="c3",
                outcome="feedback",
                field_issues=(_issue(),),
            ),
            ToolCallStats(tool_name="add", tool_call_id="c4", outcome="executed", repaired=True),
        )
    )
    assert stats.executed_count == 2
    assert stats.corrected_count == 1
    assert stats.feedback_count == 1
    assert stats.repaired_count == 1
    assert stats.field_issues == (_issue(),)


def test_tool_node_event_carries_stats() -> None:
    stats = ToolNodeStats()
    event = ToolNodeEvent(node_name="tools", stats=stats)
    assert event.node_name == "tools"
    assert event.stats is stats


# ---------------------------------------------------------------------------
# Sink + listener plumbing
# ---------------------------------------------------------------------------


def _event() -> ToolNodeEvent:
    return ToolNodeEvent(
        node_name="tools",
        stats=ToolNodeStats(
            calls=(ToolCallStats(tool_name="add", tool_call_id="c1", outcome="executed"),)
        ),
    )


@pytest.mark.asyncio
async def test_collect_stats_captures_tool_node_stats() -> None:
    async with collect_stats() as sink:
        await dispatch_tool_node_event(_event())
    assert len(sink) == 1
    stats = sink.all()[0]
    assert isinstance(stats, ToolNodeStats)
    assert stats.executed_count == 1


@pytest.mark.asyncio
async def test_on_tool_node_listener_fires_and_unsubscribes() -> None:
    received: list[ToolNodeEvent] = []
    sub = on_tool_node(received.append)
    try:
        await dispatch_tool_node_event(_event())
    finally:
        sub.unsubscribe()
    await dispatch_tool_node_event(_event())
    assert len(received) == 1
    assert received[0].node_name == "tools"


@pytest.mark.asyncio
async def test_on_extraction_listeners_do_not_receive_tool_events() -> None:
    received: list[object] = []
    sub = on_extraction(received.append)
    try:
        await dispatch_tool_node_event(_event())
    finally:
        sub.unsubscribe()
    assert received == []


@pytest.mark.asyncio
async def test_failing_tool_node_listener_is_suppressed() -> None:
    def boom(event: ToolNodeEvent) -> None:
        raise RuntimeError("listener bug")

    received: list[ToolNodeEvent] = []
    sub_bad = on_tool_node(boom)
    sub_ok = on_tool_node(received.append)
    try:
        await dispatch_tool_node_event(_event())
    finally:
        sub_bad.unsubscribe()
        sub_ok.unsubscribe()
    assert len(received) == 1


@pytest.mark.asyncio
async def test_failing_sink_does_not_break_dispatch_or_other_sinks() -> None:
    def _boom(event: object) -> None:
        raise RuntimeError("sink recording must never break the tool-node run")

    async with collect_stats() as outer:
        outer._record = _boom  # force the outer sink to raise on record
        async with collect_stats() as inner:
            # Must not raise despite the outer sink's _record blowing up.
            await dispatch_tool_node_event(_event())

    # The healthy inner sink still recorded its event ...
    assert len(inner) == 1
    # ... and the (poisoned) outer sink was skipped rather than crashing dispatch.
    assert len(outer) == 0
