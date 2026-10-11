"""Return types and configuration enums for structured output extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Literal

if TYPE_CHECKING:
    from langchain_core.messages.base import BaseMessage
    from pydantic import BaseModel


class ExtractionMode(str, Enum):  # noqa: UP042 — StrEnum would change str(mode) output
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
class ToolCallConfig:
    """How the schema tool is bound in :attr:`ExtractionMode.TOOL_CALLING`.

    The defaults reproduce OpenAI's most reliable setup: force the model to call
    the schema tool, disable parallel calls, and switch on strict structured
    outputs.  Several OpenAI-*compatible* gateways (Moonshot/Kimi, some
    Qwen/DeepSeek deployments) reject one or more of these with an HTTP 400, so
    every flag can be overridden — or dropped from the request entirely.

    ``None`` means **omit the keyword argument**, which is not the same as
    sending ``False``: some gateways already fail on the mere presence of
    ``strict``, while others accept the parameter but refuse the value ``True``.

    Ready-made presets:

    - :attr:`OPENAI` — the defaults below.
    - :attr:`COMPATIBLE` — ``tool_choice="auto"`` and no ``strict`` /
      ``parallel_tool_calls`` at all; the safe setting for limited gateways.

    Attributes:
        tool_choice: ``"forced"`` sends the schema tool's name (the model *must*
            call it), ``"auto"`` lets the model decide, ``None`` omits the
            keyword argument.
        strict: Value for the ``strict`` keyword (OpenAI structured outputs), or
            ``None`` to omit it.
        parallel_tool_calls: Value for the ``parallel_tool_calls`` keyword, or
            ``None`` to omit it.
        auto_relax: When ``True`` (default) and the provider rejects the bound
            flags — an HTTP 400 naming one of them, or a ``TypeError`` from a
            ``bind_tools`` implementation that does not accept them — the schema
            tool is re-bound **once** with :attr:`COMPATIBLE` and the attempt
            continues, without consuming a validation retry.
    """

    tool_choice: Literal["forced", "auto"] | None = "forced"
    strict: bool | None = True
    parallel_tool_calls: bool | None = False
    auto_relax: bool = True

    #: Default flags — forced ``tool_choice``, ``strict=True``, no parallel calls.
    OPENAI: ClassVar[ToolCallConfig]
    #: Relaxed flags for OpenAI-compatible gateways with limited support.
    COMPATIBLE: ClassVar[ToolCallConfig]


ToolCallConfig.OPENAI = ToolCallConfig()
ToolCallConfig.COMPATIBLE = ToolCallConfig(
    tool_choice="auto",
    strict=None,
    parallel_tool_calls=None,
)


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
            ``"type"``, ``"enum"``, ``"value"``, ``"format"``, ``"grounding"``
            (a value not found in the source text) or ``"other"``.
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


#: ``field_path`` of an external-validator issue that is not tied to a field.
EXTERNAL_VALIDATOR_PATH = "<external validator>"

#: ``field_path`` of the issue recorded when an ``on_exhausted`` hook returns an
#: instance that fails schema validation.
ON_EXHAUSTED_PATH = "<on_exhausted>"


@dataclass(frozen=True)
class ValidationIssue:
    """One finding reported by an external ``validator`` callable.

    Return a sequence of these from a validator (instead of a plain string) to
    report several independent problems at once.  Each issue becomes its own
    :class:`FieldIssue` in the stats, and ``"retry"`` issues are fed back to the
    model grouped by field path, like Pydantic and ``FieldCheck`` failures.

    Attributes:
        message: Human-readable description of the problem, shown to the model.
        field_path: Path of the offending field, e.g. ``"body"`` or
            ``"items -> 0 -> price"``.  Defaults to ``"<external validator>"``
            for findings that concern the instance as a whole.
        category: Coarse bucket, same vocabulary as :attr:`FieldIssue.category`.
        error_type: Free-form rule id, e.g. ``"leaks_internal_note"`` — use it
            to count violations per rule across runs.
        on_mismatch: ``"retry"`` (default) rejects the attempt and consumes a
            retry; ``"flag"`` only records the issue and accepts the instance.
        received: Optional short string form of the offending value.
    """

    message: str
    field_path: str = EXTERNAL_VALIDATOR_PATH
    category: str = "value"
    error_type: str = "external_validator"
    on_mismatch: Literal["retry", "flag"] = "retry"
    received: str | None = None

    def __post_init__(self) -> None:
        """Reject an ``on_mismatch`` value the engine would not understand."""
        if self.on_mismatch not in ("retry", "flag"):
            raise ValueError(f"on_mismatch must be 'retry' or 'flag', got {self.on_mismatch!r}")


@dataclass(frozen=True)
class ValidatorContext:
    """Where one external-validator call sits in the run's retry budget.

    Passed as the second argument to validators wrapped with
    :func:`~saidex.with_context`, so a validator can, for example, be strict
    in early rounds and lenient on the final one.

    Attributes:
        attempt: 1-based position of this attempt in the current model's
            budget.  Attempts used up by parse errors count too, so this is
            not the number of validator calls.
        max_attempts: Size of the current model's budget
            (``max_primary_retries`` / ``max_fallback_retries``, or
            ``max_validation_retries + 1`` in the agent loop).
        is_final: ``True`` only on the last allowed attempt of the whole run:
            the fallback's last attempt when a fallback model will run,
            otherwise the primary's.  In the agent loop, ``max_iterations``
            can end a run before any attempt is final.  If the model sends
            several final answers in one response, each of them may be checked
            with ``is_final=True``.
        model_label: ``"primary"`` or ``"fallback"``.
        is_fallback: Whether the fallback model produced the candidate.
    """

    attempt: int
    max_attempts: int
    is_final: bool
    model_label: str
    is_fallback: bool


class FailureReason(StrEnum):
    """Why an extraction run ended without a result.

    A ``StrEnum``: every member *is* its string value, so comparisons such as
    ``stats.failure_reason == "parse_error"`` keep working and the value
    serialises to JSON as a plain string.

    The reason describes only the run's **last** attempt.  An earlier attempt
    may still have produced a schema-valid candidate that a later check
    rejected.  See the *Failure reasons* page in the documentation for what
    each value means and whether a later retry makes sense.

    Attributes:
        VALIDATION_EXHAUSTED: The last attempt produced parseable output that
            failed schema validation, source grounding or the external
            validator, and no retries were left.
        PARSE_ERROR: The last attempt's output could not be parsed into
            arguments — malformed JSON or an invalid tool call.
        LLM_ERROR: The model call failed after all network retries, or the
            provider rejected the tool binding.
        NO_TOOL_CALL: Agent loop with tool-calling final answers only: the
            model replied with plain text instead of calling a tool, and no
            retries were left.
    """

    VALIDATION_EXHAUSTED = "validation_exhausted"
    PARSE_ERROR = "parse_error"
    LLM_ERROR = "llm_error"
    NO_TOOL_CALL = "no_tool_call"


class ModelErrorKind(StrEnum):
    """Provider-independent category of the error that ended a model run.

    Set on ``stats.llm_error_kind`` when ``failure_reason`` is
    :attr:`FailureReason.LLM_ERROR`.  Derived by
    :func:`saidex.classify_model_error` from langchain-core's standard model
    exceptions (langchain-core ≥ 1.6) and, for older versions, from the
    ``openai`` / ``httpx`` exception types.

    Attributes:
        RATE_LIMIT: The provider rejected the request because of a rate limit (HTTP 429).
        TIMEOUT: The request timed out.
        CONNECTION: The provider could not be reached.
        SERVER_ERROR: The provider reported a server failure (HTTP 5xx).
        AUTHENTICATION: Authentication failed (HTTP 401).
        PERMISSION_DENIED: The credentials lack permission (HTTP 403).
        INVALID_REQUEST: The provider rejected the request as invalid (HTTP 400),
            including tool-binding flags it does not accept.
        NOT_FOUND: The requested model was not found (HTTP 404).
        CONTEXT_OVERFLOW: The input exceeded the model's context window.
        OTHER: Anything not covered above.
    """

    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    SERVER_ERROR = "server_error"
    AUTHENTICATION = "authentication"
    PERMISSION_DENIED = "permission_denied"
    INVALID_REQUEST = "invalid_request"
    NOT_FOUND = "not_found"
    CONTEXT_OVERFLOW = "context_overflow"
    OTHER = "other"


def _combine_failure_reasons(
    primary: FailureReason | None,
    fallback: FailureReason | None,
    *,
    success: bool,
) -> FailureReason | None:
    """Combine the failure reasons of a primary and a fallback model run.

    The single rule used by every entry point: a successful combined run has no
    reason; otherwise the fallback's reason wins and the primary's is used when
    the fallback reported none (e.g. it was configured with zero attempts).
    """
    if success:
        return None
    return fallback if fallback is not None else primary


def _combine_llm_error_kinds(
    primary_reason: FailureReason | None,
    primary_kind: ModelErrorKind | None,
    fallback_reason: FailureReason | None,
    fallback_kind: ModelErrorKind | None,
    *,
    success: bool,
) -> ModelErrorKind | None:
    """Pick the error kind belonging to the combined failure reason.

    Mirrors :func:`_combine_failure_reasons`: the fallback's kind when the
    fallback reported a reason, otherwise the primary's.  A run's kind is only
    set when its own reason is ``llm_error``, so the result is ``None`` unless
    the combined reason is ``llm_error``.
    """
    if success:
        return None
    return fallback_kind if fallback_reason is not None else primary_kind


def _combine_candidates(
    primary: _ExtractionStatsBase,
    fallback: _ExtractionStatsBase,
    *,
    success: bool,
) -> tuple[BaseModel | list[BaseModel] | None, tuple[FieldIssue, ...]]:
    """Pick the run-level ``(last_candidate, last_rejection)`` pair.

    Success clears both; otherwise the fallback's pair wins when it has a
    candidate, and the primary's is kept when it has none.
    """
    if success:
        return None, ()
    if fallback.last_candidate is not None:
        return fallback.last_candidate, fallback.last_rejection
    return primary.last_candidate, primary.last_rejection


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
        failure_reason: Why the run failed, as a :class:`FailureReason`, or
            ``None`` on success.  Compares equal to its plain string value.  See the *Failure reasons* page in the documentation.
        schema_name: Name of the schema that was extracted.
        format_errors: Number of pure parse/tool-call failures (malformed JSON
            or tool calls) — as opposed to schema-validation mismatches.
        field_issues: Every field-level validation problem seen across all
            attempts, in chronological order.  Populated even when
            :attr:`success` is ``True`` (self-corrected errors).
        source_text: The input text the run was performed on, or ``None``.  Only
            populated when the caller opts in via ``capture_source_text=True``
            (off by default for PII/memory reasons).
        last_candidate: On failure, the most recent instance that passed schema
            validation but was rejected by source grounding or the external
            validator — kept even when later attempts failed differently.  A
            ``list`` of items for the batch variants.  ``None`` on a clean
            success or when no attempt produced a schema-valid instance.  A
            salvaged run keeps it unchanged, as the audit record of what the
            ``on_exhausted`` hook received.  Contains extracted source data:
            treat it like ``source_text`` (PII).
        last_rejection: The ``"retry"``-level issues that rejected
            :attr:`last_candidate`; ``()`` when there is none (always on a
            clean success).
        salvaged: ``True`` when the run's retries were exhausted and an
            ``on_exhausted`` hook turned the last candidate into the result.
            Such a run reports ``success=True`` and ``failure_reason=None``;
            check this flag to tell it apart from a clean success.
        llm_error_kind: Why the model call failed, as a :class:`ModelErrorKind`,
            when :attr:`failure_reason` is ``llm_error``; otherwise ``None``.
    """

    success: bool = False
    failure_reason: FailureReason | None = None
    schema_name: str = ""
    format_errors: int = 0
    field_issues: tuple[FieldIssue, ...] = ()
    source_text: str | None = None
    # Excluded from hash/repr: models and lists are unhashable, and repr lands in logs (PII).
    last_candidate: BaseModel | list[BaseModel] | None = field(default=None, hash=False, repr=False)
    last_rejection: tuple[FieldIssue, ...] = ()
    salvaged: bool = False
    llm_error_kind: ModelErrorKind | None = None

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
    ``schema_name``, ``format_errors``, ``field_issues``, ``source_text``,
    ``last_candidate``, ``last_rejection``, ``salvaged``, ``llm_error_kind``,
    ``problem_fields``)
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
    ``schema_name``, ``format_errors``, ``field_issues``, ``source_text``,
    ``last_candidate``, ``last_rejection``, ``salvaged``, ``llm_error_kind``,
    ``problem_fields``)
    is available.
    """

    iterations: int = 0
    tool_calls: int = 0
    validation_retries: int = 0
    fallback_used: bool = False

    def __add__(self, other: Any) -> ExtractorRunStats:
        """Merge two stats instances (used when combining primary + fallback)."""
        if not isinstance(other, ExtractorRunStats):
            return NotImplemented
        success = self.success or other.success
        salvaged = self.salvaged or other.salvaged
        # A salvaged run keeps its audit pair, so only a clean success clears it.
        last_candidate, last_rejection = _combine_candidates(
            self, other, success=success and not salvaged
        )
        return ExtractorRunStats(
            iterations=self.iterations + other.iterations,
            tool_calls=self.tool_calls + other.tool_calls,
            validation_retries=self.validation_retries + other.validation_retries,
            fallback_used=self.fallback_used or other.fallback_used,
            success=success,
            failure_reason=_combine_failure_reasons(
                self.failure_reason, other.failure_reason, success=success
            ),
            schema_name=self.schema_name or other.schema_name,
            format_errors=self.format_errors + other.format_errors,
            field_issues=(*self.field_issues, *other.field_issues),
            last_candidate=last_candidate,
            last_rejection=last_rejection,
            salvaged=salvaged,
            llm_error_kind=_combine_llm_error_kinds(
                self.failure_reason,
                self.llm_error_kind,
                other.failure_reason,
                other.llm_error_kind,
                success=success,
            ),
        )


@dataclass(frozen=True)
class ExtractionEvent:
    """Payload handed to an ``on_complete`` hook at the end of one extraction.

    The hook fires exactly once per top-level call, on both success and failure,
    after all retries and any fallback. It bundles the run's outcome with the
    conversation so observability backends (e.g. Langfuse) can attach the
    extraction-level verdict — which LangChain ``callbacks`` never see — to the
    same trace as the LLM calls.

    Attributes:
        schema_name: Name of the schema that was extracted.
        result: The validated instance, a list of instances (batch), or ``None``
            when the run failed.
        stats: The stats object returned to the caller.
        messages: The full message list as sent to the model.  Always present,
            so the hook can derive its own context even when ``source_text`` is
            not captured.
        source_text: The input text, when ``capture_source_text=True`` was
            passed; otherwise ``None``.
    """

    schema_name: str
    result: BaseModel | list[BaseModel] | None
    stats: ExtractDataStats | ExtractorRunStats
    messages: list[BaseMessage]
    source_text: str | None = None

    @property
    def last_candidate(self) -> BaseModel | list[BaseModel] | None:
        """Shortcut for ``stats.last_candidate`` — see :class:`_ExtractionStatsBase`."""
        return self.stats.last_candidate

    @property
    def last_rejection(self) -> tuple[FieldIssue, ...]:
        """Shortcut for ``stats.last_rejection``."""
        return self.stats.last_rejection


#: The four outcomes a SaidexToolNode-processed tool call can have. See
#: :attr:`ToolCallStats.outcome` for what each value means.
ToolCallOutcome = Literal["executed", "corrected", "feedback", "dropped"]


@dataclass(frozen=True)
class ToolCallStats:
    """Outcome of one tool call processed by a ``SaidexToolNode``.

    Attributes:
        tool_name: Name of the tool the call targeted.
        tool_call_id: The call id from the model output, or ``None`` when the
            provider did not assign one at all.  This can happen on entries
            from either ``tool_calls`` or ``invalid_tool_calls`` (LangChain
            types ``id`` as optional on both) — such a call cannot be answered
            with a ``ToolMessage`` and is dropped rather than executed or
            answered; see ``outcome="dropped"``.
        outcome: What ultimately happened — ``"executed"`` (ran, possibly after
            deterministic repair), ``"corrected"`` (ran after an LLM correction
            cycle), ``"feedback"`` (not run; a corrective ``ToolMessage`` was
            emitted instead) or ``"dropped"`` (not run and not answered — the
            call carried no ``tool_call_id`` at all, so no ``ToolMessage``
            could be produced for it either).
        repaired: Whether the call was deterministically recovered from
            ``invalid_tool_calls`` (think-tag stripping / json-repair).
        prevalidated: Whether the args were validated against the tool's
            Pydantic schema before execution.  ``False`` covers several
            distinct cases with different consequences: an unknown tool name
            or a tool with a non-Pydantic (dict) schema is passed through to
            the executor unchanged, whereas an unrepairable malformed call or
            a dropped id-less call is **not** passed through at all — see
            ``outcome``.
        correction_retries: LLM attempts consumed by the correction cycle
            (``0`` unless ``on_invalid="correct"`` ran for this call).
        field_issues: Field-level validation problems observed before the
            policy was applied.
    """

    tool_name: str
    tool_call_id: str | None
    outcome: ToolCallOutcome
    repaired: bool = False
    prevalidated: bool = True
    correction_retries: int = 0
    field_issues: tuple[FieldIssue, ...] = ()


@dataclass(frozen=True)
class ToolNodeStats:
    """Statistics from one ``SaidexToolNode`` invocation.

    Attributes:
        calls: Per-call outcomes, in original tool-call order
            (``tool_calls`` first, then ``invalid_tool_calls``).
    """

    calls: tuple[ToolCallStats, ...] = ()

    @property
    def executed_count(self) -> int:
        """Calls that executed without an LLM correction cycle."""
        return sum(1 for c in self.calls if c.outcome == "executed")

    @property
    def corrected_count(self) -> int:
        """Calls that executed after an LLM correction cycle."""
        return sum(1 for c in self.calls if c.outcome == "corrected")

    @property
    def feedback_count(self) -> int:
        """Calls that were answered with a corrective feedback message."""
        return sum(1 for c in self.calls if c.outcome == "feedback")

    @property
    def repaired_count(self) -> int:
        """Calls deterministically recovered from ``invalid_tool_calls``."""
        return sum(1 for c in self.calls if c.repaired)

    @property
    def field_issues(self) -> tuple[FieldIssue, ...]:
        """All field-level issues across calls, in call order."""
        return tuple(issue for c in self.calls for issue in c.field_issues)


@dataclass(frozen=True)
class ToolNodeEvent:
    """Payload dispatched to observers after one ``SaidexToolNode`` invocation.

    Attributes:
        node_name: The node's ``name`` (as shown in the graph).
        stats: The invocation's :class:`ToolNodeStats`.
    """

    node_name: str
    stats: ToolNodeStats
