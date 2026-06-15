"""Aggregate field-level extraction issues across many runs.

The extraction functions return per-run stats carrying a structured
:class:`~saidex.models.FieldIssue` list.  Run enough varied inputs — a benchmark
sweep, a labelled evaluation set — and a pattern emerges: *which* fields a model
keeps getting wrong, *why*, and whether it recovers.  :func:`summarize_field_issues`
turns a pile of runs into that picture, grouped per schema and per field, so the
weak field descriptions/prompts can be found and improved.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from .models import ExtractDataStats, ExtractorRunStats, FieldIssue

# Maximum number of distinct example values retained per field.
_MAX_SAMPLES = 5

# Weight applied to failed-run involvement when scoring a field's severity.
_SEVERITY_FAILED_WEIGHT = 10.0

StatsOrIssue = ExtractDataStats | ExtractorRunStats | FieldIssue


@dataclass(frozen=True)
class FieldProblemStat:
    """How one field (within one schema) behaved across the analysed runs.

    Attributes:
        schema_name: Schema the field belongs to.
        field_path: Path to the field, e.g. ``"items -> 0 -> price"``.
        total_occurrences: Total number of issues seen for this field,
            counting every attempt of every run plus any bare issues.
        runs_with_problem: Number of runs in which this field failed at least
            once.  Bare :class:`~saidex.models.FieldIssue` inputs do not count
            as runs and are excluded here.
        failed_runs_with_problem: Subset of ``runs_with_problem`` whose run
            ultimately failed.
        occurrences_on_successful_runs: Issues for this field that occurred in
            runs which still succeeded (self-corrected).
        recovery_rate: Fraction of ``runs_with_problem`` that still succeeded.
            ``0.0`` when there were no runs (bare issues only).
        by_category: Issue count per coarse category.
        by_error_type: Issue count per raw Pydantic error type.
        first_error_attempt_avg: Mean of the earliest attempt index at which the
            field failed (per run); lower means the model tends to get it wrong
            up front.
        sample_received: Up to five distinct offending values, first seen first.
        severity: Sort score — dominated by ``failed_runs_with_problem`` and
            tie-broken by ``total_occurrences``.  Higher is more urgent.
    """

    schema_name: str
    field_path: str
    total_occurrences: int
    runs_with_problem: int
    failed_runs_with_problem: int
    occurrences_on_successful_runs: int
    recovery_rate: float
    by_category: dict[str, int] = field(default_factory=dict)
    by_error_type: dict[str, int] = field(default_factory=dict)
    first_error_attempt_avg: float = 0.0
    sample_received: tuple[str, ...] = ()
    severity: float = 0.0


@dataclass(frozen=True)
class SchemaProblemSummary:
    """Per-schema rollup of field problems.

    Attributes:
        schema_name: The schema these problems belong to.
        total_runs: Number of runs (stats objects) seen for this schema.  Bare
            issues do not count as runs.
        failed_runs: How many of those runs failed.
        success_rate: ``(total_runs - failed_runs) / total_runs``; ``0.0`` when
            there were no runs (bare issues only).
        field_problems: Field stats sorted by descending severity.
    """

    schema_name: str
    total_runs: int
    failed_runs: int
    success_rate: float
    field_problems: tuple[FieldProblemStat, ...] = ()


@dataclass(frozen=True)
class FieldIssueSummary:
    """Top-level result of :func:`summarize_field_issues`.

    Attributes:
        schemas: Per-schema summaries, most problematic schema first.
    """

    schemas: tuple[SchemaProblemSummary, ...] = ()

    def to_markdown(self) -> str:
        """Render this summary as a deterministic Markdown report.

        Convenience wrapper around :func:`render_field_issue_report`.

        Returns:
            A Markdown document with one section per schema, each listing its
            problem fields (most severe first) as a table.
        """
        return render_field_issue_report(self)


# ---------------------------------------------------------------------------
# Internal accumulators
# ---------------------------------------------------------------------------


@dataclass
class _FieldAccumulator:
    """Mutable per-field tally used while scanning runs."""

    total_occurrences: int = 0
    runs_with_problem: int = 0
    failed_runs_with_problem: int = 0
    occurrences_on_successful_runs: int = 0
    by_category: Counter[str] = field(default_factory=Counter)
    by_error_type: Counter[str] = field(default_factory=Counter)
    first_attempts: list[int] = field(default_factory=list)
    samples: list[str] = field(default_factory=list)

    def add_samples(self, issues: Iterable[FieldIssue]) -> None:
        for issue in issues:
            if issue.received is None or len(self.samples) >= _MAX_SAMPLES:
                continue
            if issue.received not in self.samples:
                self.samples.append(issue.received)


def summarize_field_issues(runs: Iterable[StatsOrIssue]) -> FieldIssueSummary:
    """Aggregate field issues from many runs into a per-schema summary.

    Accepts a heterogeneous iterable: extraction stats objects
    (:class:`~saidex.models.ExtractDataStats` /
    :class:`~saidex.models.ExtractorRunStats`) and/or bare
    :class:`~saidex.models.FieldIssue` instances.  Stats objects contribute run
    context (success/failure, run counts); bare issues contribute only to the
    occurrence-based metrics, since they carry no run outcome.

    Args:
        runs: The runs and/or loose issues to aggregate.  Each item is grouped
            by its ``schema_name``.

    Returns:
        A :class:`FieldIssueSummary` with one
        :class:`SchemaProblemSummary` per distinct schema, most problematic
        schema first, each holding its fields sorted by descending severity.
    """
    # Per schema: run records (success, issues) and the set of seen schema names.
    run_records: dict[str, list[tuple[bool, list[FieldIssue]]]] = defaultdict(list)
    bare_issues: dict[str, list[FieldIssue]] = defaultdict(list)
    schema_names: set[str] = set()

    for item in runs:
        if isinstance(item, FieldIssue):
            bare_issues[item.schema_name].append(item)
            schema_names.add(item.schema_name)
        else:
            run_records[item.schema_name].append((item.success, list(item.field_issues)))
            schema_names.add(item.schema_name)

    summaries: list[SchemaProblemSummary] = []
    for schema_name in schema_names:
        records = run_records.get(schema_name, [])
        loose = bare_issues.get(schema_name, [])
        total_runs = len(records)
        failed_runs = sum(1 for success, _ in records if not success)

        accumulators: dict[str, _FieldAccumulator] = defaultdict(_FieldAccumulator)

        for success, issues in records:
            by_field: dict[str, list[FieldIssue]] = defaultdict(list)
            for issue in issues:
                by_field[issue.field_path].append(issue)
            for field_path, field_issues in by_field.items():
                acc = accumulators[field_path]
                acc.total_occurrences += len(field_issues)
                acc.runs_with_problem += 1
                if not success:
                    acc.failed_runs_with_problem += 1
                else:
                    acc.occurrences_on_successful_runs += len(field_issues)
                acc.first_attempts.append(min(i.attempt for i in field_issues))
                for issue in field_issues:
                    acc.by_category[issue.category] += 1
                    acc.by_error_type[issue.error_type] += 1
                acc.add_samples(field_issues)

        for issue in loose:
            acc = accumulators[issue.field_path]
            acc.total_occurrences += 1
            acc.first_attempts.append(issue.attempt)
            acc.by_category[issue.category] += 1
            acc.by_error_type[issue.error_type] += 1
            acc.add_samples([issue])

        field_problems = [
            _finalize_field(schema_name, field_path, acc)
            for field_path, acc in accumulators.items()
        ]
        field_problems.sort(key=lambda f: (-f.severity, f.field_path))

        success_rate = (total_runs - failed_runs) / total_runs if total_runs else 0.0
        summaries.append(
            SchemaProblemSummary(
                schema_name=schema_name,
                total_runs=total_runs,
                failed_runs=failed_runs,
                success_rate=success_rate,
                field_problems=tuple(field_problems),
            )
        )

    summaries.sort(
        key=lambda s: (
            -s.failed_runs,
            -sum(f.total_occurrences for f in s.field_problems),
            s.schema_name,
        )
    )
    return FieldIssueSummary(schemas=tuple(summaries))


def _finalize_field(schema_name: str, field_path: str, acc: _FieldAccumulator) -> FieldProblemStat:
    """Convert a mutable accumulator into an immutable :class:`FieldProblemStat`."""
    recovery_rate = (
        (acc.runs_with_problem - acc.failed_runs_with_problem) / acc.runs_with_problem
        if acc.runs_with_problem
        else 0.0
    )
    first_error_attempt_avg = (
        sum(acc.first_attempts) / len(acc.first_attempts) if acc.first_attempts else 0.0
    )
    severity = acc.failed_runs_with_problem * _SEVERITY_FAILED_WEIGHT + acc.total_occurrences
    return FieldProblemStat(
        schema_name=schema_name,
        field_path=field_path,
        total_occurrences=acc.total_occurrences,
        runs_with_problem=acc.runs_with_problem,
        failed_runs_with_problem=acc.failed_runs_with_problem,
        occurrences_on_successful_runs=acc.occurrences_on_successful_runs,
        recovery_rate=recovery_rate,
        by_category=dict(acc.by_category),
        by_error_type=dict(acc.by_error_type),
        first_error_attempt_avg=first_error_attempt_avg,
        sample_received=tuple(acc.samples),
        severity=severity,
    )


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

# Maximum number of sample values rendered in a report cell.
_REPORT_MAX_SAMPLES = 3

_REPORT_HEADER = (
    "| Field | Category | Top error | Hits | In failed runs | Recovery | Samples |\n"
    "| --- | --- | --- | ---: | ---: | ---: | --- |"
)


def render_field_issue_report(summary: FieldIssueSummary) -> str:
    """Render a :class:`FieldIssueSummary` as a deterministic Markdown report.

    Produces one section per schema — a heading with the success rate and
    failed-run count, followed by a table of problem fields sorted by descending
    severity (most failure-causing first).  Output is stable for the same input,
    so it is safe to snapshot or diff across benchmark runs.

    Args:
        summary: The summary to render.

    Returns:
        A Markdown document.  When the summary is empty, a short placeholder is
        returned instead of an empty string.
    """
    lines: list[str] = ["# Field issue report", ""]

    if not summary.schemas:
        lines.append("_No field issues recorded._")
        return "\n".join(lines)

    for schema in summary.schemas:
        lines.append(
            f"## {schema.schema_name} — {schema.success_rate:.0%} success "
            f"({schema.failed_runs}/{schema.total_runs} runs failed)"
        )
        lines.append("")
        if not schema.field_problems:
            lines.append("_No field issues recorded._")
            lines.append("")
            continue
        lines.append(_REPORT_HEADER)
        for fp in schema.field_problems:
            lines.append(_render_field_row(fp))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _render_field_row(fp: FieldProblemStat) -> str:
    """Render one :class:`FieldProblemStat` as a Markdown table row."""
    category = _dominant(fp.by_category)
    top_error = _dominant(fp.by_error_type)
    samples = ", ".join(f"`{_escape_cell(s)}`" for s in fp.sample_received[:_REPORT_MAX_SAMPLES])
    return (
        f"| {_escape_cell(fp.field_path)} "
        f"| {category} "
        f"| {top_error} "
        f"| {fp.total_occurrences} "
        f"| {fp.failed_runs_with_problem} "
        f"| {fp.recovery_rate:.0%} "
        f"| {samples} |"
    )


def _dominant(counts: dict[str, int]) -> str:
    """Return the most frequent key, breaking ties alphabetically."""
    if not counts:
        return "—"
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _escape_cell(text: str) -> str:
    """Make *text* safe for a single Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ").replace("\r", "")
