"""Tests for ToolCallStats/ToolNodeStats/ToolNodeEvent and their observability plumbing.

These types intentionally have no langgraph dependency — they must be
importable from core `saidex` without the extra installed.
"""

from __future__ import annotations

from saidex import FieldIssue, ToolCallStats, ToolNodeEvent, ToolNodeStats

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
