"""Cross-run field-issue aggregation for the benchmark sweep.

Kept separate from :mod:`benchmarks._base` so it depends only on ``saidex`` —
no LLM/provider packages — and can therefore be unit-tested without a model
backend.  It turns the per-run :class:`~saidex.FieldIssue` data collected during
a sweep into a :class:`~saidex.FieldIssueSummary` plus render helpers for the
console and the Markdown report.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from typing import Any, Protocol

from saidex import (
    ExtractDataStats,
    FieldIssue,
    FieldIssueSummary,
    summarize_field_issues,
)


class _ResultLike(Protocol):
    """The subset of ``BenchmarkResult`` this module needs."""

    success: bool
    schema_name: str
    field_issues: tuple[FieldIssue, ...]


def results_to_stats(results: Iterable[_ResultLike]) -> list[ExtractDataStats]:
    """Project benchmark results onto the stats objects the summary consumes."""
    return [
        ExtractDataStats(
            success=r.success,
            schema_name=r.schema_name,
            field_issues=tuple(r.field_issues),
        )
        for r in results
    ]


def summarize_results(results: Iterable[_ResultLike]) -> FieldIssueSummary:
    """Aggregate benchmark results into a per-schema field-issue summary."""
    return summarize_field_issues(results_to_stats(results))


def summary_to_dict(summary: FieldIssueSummary) -> dict[str, Any]:
    """Serialise a :class:`~saidex.FieldIssueSummary` for the JSON results file."""
    return asdict(summary)


def format_console_summary(summary: FieldIssueSummary, *, top_n: int = 5) -> list[str]:
    """Render a compact, per-schema overview of the worst problem fields.

    Args:
        summary: The aggregated summary.
        top_n: Maximum number of problem fields to show per schema.

    Returns:
        A list of printable lines (empty when there are no issues at all).
    """
    schemas_with_issues = [s for s in summary.schemas if s.field_problems]
    if not schemas_with_issues:
        return []

    lines: list[str] = []
    lines.append(f"{'━' * 72}")
    lines.append("  FIELD ISSUE ANALYSIS (which fields failed validation, per schema)")
    lines.append(f"{'━' * 72}")
    for schema in schemas_with_issues:
        lines.append(
            f"  {schema.schema_name}: {schema.success_rate:.0%} success "
            f"({schema.failed_runs}/{schema.total_runs} runs failed)"
        )
        for fp in schema.field_problems[:top_n]:
            top_error = max(fp.by_error_type, key=lambda k: fp.by_error_type[k])
            lines.append(
                f"    {fp.field_path:<24} "
                f"hits={fp.total_occurrences:<3} "
                f"failed_runs={fp.failed_runs_with_problem:<3} "
                f"recovery={fp.recovery_rate:.0%}  "
                f"top_error={top_error}"
            )
    lines.append("")
    return lines
