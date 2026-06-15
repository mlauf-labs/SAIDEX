"""Return types and configuration enums for structured output extraction."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ExtractionMode(str, Enum):
    """How the LLM is asked to produce structured output.

    Attributes:
        TOOL_CALLING: Bind the schema as an OpenAI-style tool and force the
            model to call it (``llm.bind_tools(..., tool_choice=...)``).
            Requires a model that supports tool/function calling.  This is the
            default and most reliable mode.
        JSON: Inject the schema's JSON Schema into the prompt and ask the model
            to reply with a single raw JSON object.  The response content is
            parsed directly.  Works with any chat model — including those that
            do **not** support tool calling — at the cost of slightly less
            structural guarantees.
    """

    TOOL_CALLING = "tool_calling"
    JSON = "json"


@dataclass(frozen=True)
class FieldIssue:
    """A single field-level problem observed during one extraction attempt.

    Captures *which* field failed validation, *why* (both the human-readable
    category and the raw Pydantic error type), and *when* (which retry attempt).
    Carrying :attr:`schema_name` on every issue means a bare ``list[FieldIssue]``
    can be grouped per schema without any surrounding stats object.

    Attributes:
        schema_name: Name of the schema the issue belongs to.
        field_path: Dotted/arrow path to the offending field, e.g.
            ``"items -> 0 -> price"``.
        category: Coarse, human-facing bucket — one of ``"missing"``,
            ``"type"``, ``"enum"``, ``"value"``, ``"format"`` or ``"other"``.
        error_type: The raw Pydantic error type, e.g. ``"int_parsing"``.
        message: Human-readable description of the problem.
        attempt: Zero-based retry attempt in which the error occurred.
        received: Truncated string form of the value the LLM produced, or
            ``None`` when not available (e.g. a missing field).
    """

    schema_name: str
    field_path: str
    category: str
    error_type: str
    message: str
    attempt: int
    received: str | None = None


@dataclass(frozen=True)
class _ExtractionStatsBase:
    """Fields shared by every extraction-stats type.

    Both :class:`ExtractDataStats` and :class:`ExtractorRunStats` inherit these
    so the success/quality signals have the same names and types regardless of
    which extraction entry point produced them.

    Attributes:
        success: Whether the run produced a validated instance.  Note that a
            successful run may still carry :attr:`field_issues` from earlier,
            self-corrected attempts.
        failure_reason: Why the run failed, or ``None`` on success.  One of
            ``"validation_exhausted"``, ``"parse_error"``, ``"llm_error"`` or
            ``"no_tool_call"``.
        schema_name: Name of the schema that was extracted.
        format_errors: Number of pure parse/tool-call failures (malformed JSON
            or tool calls) — as opposed to schema-validation mismatches.
        field_issues: Every field-level validation problem seen across all
            attempts, in chronological order.  Populated even when
            :attr:`success` is ``True`` (self-corrected errors).
    """

    success: bool = False
    failure_reason: str | None = None
    schema_name: str = ""
    format_errors: int = 0
    field_issues: tuple[FieldIssue, ...] = ()

    @property
    def problem_fields(self) -> tuple[str, ...]:
        """Unique field paths that had at least one issue, in first-seen order."""
        seen: dict[str, None] = {}
        for issue in self.field_issues:
            seen.setdefault(issue.field_path, None)
        return tuple(seen)


@dataclass(frozen=True)
class ExtractDataStats(_ExtractionStatsBase):
    """Statistics from an ``extract_data`` call.

    Tracks how many retries were needed, whether the fallback model was used,
    whether the run ultimately succeeded, and which fields gave the model
    trouble along the way.

    Attributes:
        primary_retries: Number of retries against the primary model.
        fallback_retries: Number of retries against the fallback model.
        fallback_used: Whether the fallback model was invoked at all.
        item_count: Number of items returned by a batch (``extract_data_list``)
            call.  Always ``0`` for single-item extraction.

    In addition to the fields below, every attribute of
    :class:`_ExtractionStatsBase` (``success``, ``failure_reason``,
    ``schema_name``, ``format_errors``, ``field_issues``, ``problem_fields``)
    is available.
    """

    primary_retries: int = 0
    fallback_retries: int = 0
    fallback_used: bool = False
    item_count: int = 0

    @property
    def total_retries(self) -> int:
        """Total retry count across primary and fallback."""
        return self.primary_retries + self.fallback_retries


@dataclass(frozen=True)
class ExtractorRunStats(_ExtractionStatsBase):
    """Statistics from an :func:`extract_data_with_tools` / :func:`run_extractor_agent` call.

    Attributes:
        iterations: Total LLM invocations performed in the agent loop.
        tool_calls: Total helper-tool calls executed (not counting final-answer calls).
        validation_retries: Number of times the final-answer schema failed validation
            and the loop continued.
        fallback_used: Whether the fallback model was invoked.

    In addition to the fields below, every attribute of
    :class:`_ExtractionStatsBase` (``success``, ``failure_reason``,
    ``schema_name``, ``format_errors``, ``field_issues``, ``problem_fields``)
    is available.
    """

    iterations: int = 0
    tool_calls: int = 0
    validation_retries: int = 0
    fallback_used: bool = False

    def __add__(self, other: Any) -> "ExtractorRunStats":
        """Merge two stats instances (used when combining primary + fallback)."""
        if not isinstance(other, ExtractorRunStats):
            return NotImplemented
        return ExtractorRunStats(
            iterations=self.iterations + other.iterations,
            tool_calls=self.tool_calls + other.tool_calls,
            validation_retries=self.validation_retries + other.validation_retries,
            fallback_used=self.fallback_used or other.fallback_used,
            success=self.success or other.success,
            failure_reason=other.failure_reason if other.failure_reason else self.failure_reason,
            schema_name=self.schema_name or other.schema_name,
            format_errors=self.format_errors + other.format_errors,
            field_issues=(*self.field_issues, *other.field_issues),
        )
