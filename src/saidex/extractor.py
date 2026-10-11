"""Core extraction logic: LLM → validated Pydantic model."""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import json
import logging
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypeAlias,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import pydantic
from json_repair import repair_json
from langchain_core.callbacks.base import Callbacks
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import AliasChoices, BaseModel, Field, RootModel, ValidationError, create_model
from pydantic.fields import FieldInfo
from typing_extensions import is_typeddict

from ._callbacks import _ChainRun
from ._reasoning import (
    ReasoningEscalation,
    _is_reasoning_effort_rejection,
    _level_for_attempt,
    _resolve_reasoning_levels,
    _with_reasoning_effort,
)
from .grounding import _collect_field_checks
from .models import (
    EXTERNAL_VALIDATOR_PATH,
    ON_EXHAUSTED_PATH,
    ExtractDataStats,
    ExtractionEvent,
    ExtractionMode,
    ExtractorRunStats,
    FailureReason,
    FieldIssue,
    ModelErrorKind,
    ToolCallConfig,
    ValidationIssue,
    ValidatorContext,
    _combine_candidates,
    _combine_failure_reasons,
    _combine_llm_error_kinds,
)
from .observability import dispatch_to_observers
from .retry import DEFAULT_RETRY_CONFIG, RetryConfig, classify_model_error, with_retry
from .utils import _format_validation_error, create_instance_with_issues

if TYPE_CHECKING:
    from .tools import Tool

logger = logging.getLogger(__name__)

MODEL_T = TypeVar("MODEL_T", bound=BaseModel)
# WithContext only consumes the model, so a validator typed for a base schema is
# usable wherever a subclass schema is expected.
_MODEL_CONTRA = TypeVar("_MODEL_CONTRA", bound=BaseModel, contravariant=True)

#: Signature of an ``on_complete`` hook: an async callable receiving a single
#: :class:`~saidex.models.ExtractionEvent`.
OnComplete = Callable[[ExtractionEvent], Awaitable[None]]

#: Signature of an ``on_exhausted`` hook: called once after every retry (and
#: the fallback) failed, with the last schema-valid candidate and the issues
#: that rejected it.  Return a salvaged instance to end the run successfully
#: (``stats.salvaged=True``), or ``None`` to let it fail.  Sync or async.
OnExhausted = Callable[
    [MODEL_T, tuple[FieldIssue, ...]], "MODEL_T | None | Awaitable[MODEL_T | None]"
]

#: List-variant counterpart of :data:`OnExhausted`: receives the last rejected
#: ``list`` of items and returns the items to keep (``[]`` is a valid result),
#: or ``None`` to let the run fail.
OnExhaustedList = Callable[
    [list[MODEL_T], tuple[FieldIssue, ...]],
    "list[MODEL_T] | None | Awaitable[list[MODEL_T] | None]",
]

#: What an external ``validator`` may return: ``None`` (or any falsy value) to
#: accept, a non-empty string to reject with one message, or one / several
#: :class:`~saidex.models.ValidationIssue` objects for structured findings.  A
#: sequence may also hold plain strings, each becoming one unlocated issue.
ValidatorResult: TypeAlias = str | ValidationIssue | Sequence[str | ValidationIssue] | None

#: Signature of a context-aware validator: like a plain validator, plus a
#: :class:`~saidex.models.ValidatorContext` as second argument.  Wrap it with
#: :func:`with_context` before passing it as ``validator=``.
ContextValidator = Callable[
    [MODEL_T, ValidatorContext], "ValidatorResult | Awaitable[ValidatorResult]"
]


@dataclass(frozen=True)
class WithContext(Generic[_MODEL_CONTRA]):
    """An external validator that wants a :class:`~saidex.models.ValidatorContext`.

    Build it with :func:`with_context` and pass it wherever ``validator=`` is
    accepted.  An explicit wrapper instead of signature inspection, which is
    unreliable for ``functools.partial``, callable objects and parameters with
    defaults.

    Attributes:
        fn: The wrapped ``(instance, ctx)`` callable, sync or async.
    """

    fn: Callable[[_MODEL_CONTRA, ValidatorContext], ValidatorResult | Awaitable[ValidatorResult]]


def with_context(fn: ContextValidator[MODEL_T]) -> WithContext[MODEL_T]:
    """Mark *fn* as a validator that receives the attempt context.

    Args:
        fn: A sync or async callable taking ``(instance, ctx)`` and returning a
            :data:`ValidatorResult` (or raising to reject), exactly like a
            plain validator.

    Returns:
        A :class:`WithContext` to pass as ``validator=``.

    Example::

        def strict_then_lenient(inv: Invoice, ctx: ValidatorContext) -> ValidatorResult:
            if inv.total == sum(line.amount for line in inv.lines):
                return None
            return ValidationIssue(
                message="total does not match the line items",
                field_path="total",
                on_mismatch="flag" if ctx.is_final else "retry",
            )

        result, stats = await extract_data(
            llm, Invoice, messages, validator=with_context(strict_then_lenient)
        )
    """
    return WithContext(fn)


#: Signature of an external ``validator``: a sync **or** async callable that
#: receives the freshly validated model instance and returns a
#: :data:`ValidatorResult` (or raises to reject), or a :class:`WithContext`
#: built by :func:`with_context` whose callable also receives a
#: :class:`~saidex.models.ValidatorContext`.  Every reported issue lands in
#: ``stats.field_issues``; ``"retry"`` issues re-enter the field-level retry loop
#: with per-field feedback, ``"flag"`` issues are only recorded.
Validator = (
    Callable[[MODEL_T], "ValidatorResult | Awaitable[ValidatorResult]"] | WithContext[MODEL_T]
)


class _InvalidValidatorResultError(TypeError):
    """A validator returned something that is not a :data:`ValidatorResult`.

    A distinct subclass so :func:`_call_validator` can let it escape its broad
    ``except``: the list wrapper runs item validators *inside* the container
    validator, and a plain ``TypeError`` there would be mistaken for a rejection.
    """


def _coerce_validator_result(outcome: object) -> list[ValidationIssue]:
    """Normalise a validator's return value into a list of issues.

    Falsy values were accepted in v0.7.0 as well, so they stay accepted here
    before any type check (``x.bad and "msg"`` returns ``False`` when fine).
    Plain strings inside a sequence are kept for the v0.7.0 pattern of
    returning a list of error messages.
    """
    if not outcome:
        return []
    if isinstance(outcome, str):
        return [ValidationIssue(message=outcome)]
    if isinstance(outcome, ValidationIssue):
        return [outcome]
    if isinstance(outcome, Sequence):
        items = list(outcome)
        if all(isinstance(i, (str, ValidationIssue)) for i in items):
            return [
                ValidationIssue(message=i) if isinstance(i, str) else i
                for i in items
                if not isinstance(i, str) or i
            ]
    raise _InvalidValidatorResultError(
        "external validator must return None, a str, a ValidationIssue or a sequence of "
        f"str / ValidationIssue objects; got {outcome!r}"
    )


async def _call_validator(
    validator: Validator[MODEL_T],
    instance: MODEL_T,
    schema_name: str,
    context: ValidatorContext,
) -> list[ValidationIssue]:
    """Run *validator* on *instance* and return its findings (empty = accepted).

    A :class:`WithContext` validator additionally receives *context*.  A raised
    exception is a deliberate rejection and becomes one issue.  An
    invalid return value raises :class:`TypeError` — it is a programming error,
    so it is re-raised (also from a nested validator, as in the list wrapper)
    and never consumes a retry.
    """
    try:
        if isinstance(validator, WithContext):
            outcome = validator.fn(instance, context)
        else:
            outcome = validator(instance)
        if inspect.isawaitable(outcome):
            outcome = await outcome
    except _InvalidValidatorResultError:
        raise
    except Exception as exc:  # noqa: BLE001 — any raise is a deliberate rejection
        message = (
            str(exc) or f"{exc.__class__.__name__} raised by external validator for {schema_name}"
        )
        return [ValidationIssue(message=message)]
    return _coerce_validator_result(outcome)


def _validator_feedback(issues: Sequence[ValidationIssue]) -> str | None:
    """Render the ``"retry"`` issues as feedback text, or ``None`` if there are none.

    A single issue with no field path and no received value is rendered as its
    bare message, which keeps the v0.7.0 feedback for string validators.
    """
    retry = [i for i in issues if i.on_mismatch == "retry"]
    if not retry:
        return None
    if (
        len(retry) == 1
        and retry[0].field_path == EXTERNAL_VALIDATOR_PATH
        and retry[0].received is None
    ):
        return retry[0].message
    by_path: dict[str, list[ValidationIssue]] = {}
    for issue in retry:
        by_path.setdefault(issue.field_path, []).append(issue)
    lines: list[str] = []
    for path, group in by_path.items():
        prefix = "" if path == EXTERNAL_VALIDATOR_PATH else f"'{path}': "
        for issue in group:
            lines.append(f"  - {prefix}{issue.message}")
            if issue.received is not None:
                lines.append(f"    Received: '{issue.received}'")
    return "\n".join(lines)


def _to_field_issues(issues: Sequence[ValidationIssue], schema_name: str) -> list[FieldIssue]:
    """Convert validator findings to :class:`FieldIssue` records with ``attempt=0``."""
    return [
        FieldIssue(
            schema_name=schema_name,
            field_path=issue.field_path,
            category=issue.category,
            error_type=issue.error_type,
            message=issue.message,
            attempt=0,
            received=issue.received,
        )
        for issue in issues
    ]


#: ``model_validate(..., by_name=True)`` exists from Pydantic 2.11 on and
#: accepts field names for every alias form (``AliasChoices``, ``AliasPath``).
_VALIDATE_BY_NAME = "by_name" in inspect.signature(BaseModel.model_validate).parameters


def _input_key(name: str, info: FieldInfo) -> str:
    """Return the input key under which field *name* re-validates.

    The field name when ``by_name`` validation is available; otherwise the
    first string validation alias (Pydantic < 2.11 accepts only aliases unless
    the schema sets ``populate_by_name``).
    """
    if _VALIDATE_BY_NAME:
        return name
    alias = info.validation_alias
    if isinstance(alias, AliasChoices):
        alias = next((c for c in alias.choices if isinstance(c, str)), None)
    if isinstance(alias, str):
        return alias
    return info.alias or name


def _field_values(value: Any) -> Any:
    """Re-express *value* as validation input built from its stored field values.

    Models are read field by field instead of serialized, so field serializers,
    ``exclude=True`` and serialization aliases cannot change what is
    re-validated.  Computed fields are not model fields and are left out, at
    every depth, so a nested ``extra="forbid"`` model does not reject them.
    """
    if isinstance(value, RootModel):
        return _field_values(value.root)
    if isinstance(value, BaseModel):
        stored = value.__dict__
        data = {
            _input_key(name, info): _field_values(stored[name])
            for name, info in type(value).model_fields.items()
            if name in stored
        }
        for key, extra in (value.__pydantic_extra__ or {}).items():
            data[key] = _field_values(extra)
        return data
    if isinstance(value, dict):
        return {key: _field_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_field_values(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_field_values(item) for item in value)
    return value


def _revalidate(schema: type[MODEL_T], outcome: BaseModel) -> tuple[MODEL_T | None, str | None]:
    """Validate an ``on_exhausted`` result against *schema* from scratch.

    The primary attempt validates the stored field values (see
    :func:`_field_values`).  If that fails, the ``model_dump(round_trip=True)``
    output is tried as well, for schemas whose validators expect serialized
    input.  Validators may raise any exception on already-parsed values, so
    every exception counts as a failed attempt here.

    Returns:
        ``(instance, None)`` when valid, otherwise ``(None, error_text)`` with
        the primary attempt's error.
    """
    data = _field_values(outcome)
    try:
        if _VALIDATE_BY_NAME:
            return schema.model_validate(data, by_name=True), None
        return schema.model_validate(data), None
    except ValidationError as exc:
        error_text = _format_validation_error(schema, exc)
    except Exception as exc:  # noqa: BLE001 — see docstring
        error_text = f"{exc.__class__.__name__}: {exc}"
    try:
        dumped = outcome.model_dump(by_alias=True, round_trip=True, warnings=False)
        return schema.model_validate(dumped), None
    except Exception:  # noqa: BLE001 — see docstring
        return None, error_text


async def _salvage(
    on_exhausted: OnExhausted[MODEL_T],
    schema: type[MODEL_T],
    candidate: MODEL_T,
    rejection: tuple[FieldIssue, ...],
    *,
    attempt: int,
) -> tuple[MODEL_T | None, FieldIssue | None]:
    """Run an ``on_exhausted`` hook and re-validate what it returns.

    The hook receives a deep copy of *candidate*.  The returned instance is
    re-validated against *schema* only (see :func:`_revalidate`): the hook owns
    the decision, so grounding and the external validator are not re-run.
    Re-validation catches instances built with ``model_copy(update=...)`` or
    ``model_construct``, which bypass validation.

    Args:
        on_exhausted: The caller's hook.
        schema: The schema the result must satisfy.
        candidate: The last schema-valid candidate.
        rejection: The issues that rejected *candidate*.
        attempt: Attempt index stamped onto an invalid-result issue.

    Returns:
        ``(instance, None)`` when salvaged, ``(None, None)`` when the hook
        declined, ``(None, issue)`` when its result failed schema validation.

    Raises:
        TypeError: The hook returned neither ``None`` nor a model instance.
    """
    # A deep copy, so a hook that edits in place cannot rewrite the audit record
    # (``stats.last_candidate``), which still references the original.
    outcome = on_exhausted(candidate.model_copy(deep=True), rejection)
    if inspect.isawaitable(outcome):
        outcome = await outcome
    if outcome is None:
        return None, None
    if not isinstance(outcome, BaseModel):
        raise TypeError(
            f"on_exhausted must return a {schema.__name__} instance or None; got {outcome!r}"
        )
    instance, error_text = _revalidate(schema, outcome)
    if instance is not None:
        return instance, None
    logger.warning("on_exhausted returned an invalid %s: %s", schema.__name__, error_text)
    return None, FieldIssue(
        schema_name=schema.__name__,
        field_path=ON_EXHAUSTED_PATH,
        category="other",
        error_type="on_exhausted_invalid",
        message=error_text or "on_exhausted returned an invalid instance",
        attempt=attempt,
    )


def _item_issue_path(index: int, field_path: str) -> str:
    """Locate an item validator's finding inside the ``…List`` container."""
    if field_path == EXTERNAL_VALIDATOR_PATH:
        return f"items -> {index}"
    return f"items -> {index} -> {field_path}"


def issue_item_index(issue: FieldIssue, list_field: str = "items") -> int | None:
    """Return the list index an issue points at, or ``None``.

    Batch extraction reports item issues with paths like ``items -> 3`` or
    ``items -> 3 -> quote``; for a list field inside your own schema the path
    starts with that field's name (``findings -> 3 -> quote``).  Use this in an
    ``on_exhausted`` hook to drop the items that were rejected.

    Args:
        issue: The issue to inspect, e.g. from ``stats.last_rejection``.
        list_field: Name of the list field; ``"items"`` for the batch variants.

    Returns:
        The 0-based item index, or ``None`` when the path does not start with
        ``<list_field> -> <index>``.
    """
    head, sep, rest = issue.field_path.partition(" -> ")
    if head != list_field or not sep:
        return None
    index = rest.split(" -> ", 1)[0]
    return int(index) if index.isdigit() else None


async def _run_external_validator(
    validator: Validator[MODEL_T] | None,
    instance: MODEL_T,
    schema_name: str,
    context: ValidatorContext,
) -> tuple[list[FieldIssue], list[FieldIssue], str | None]:
    """Run an optional external *validator* against a validated *instance*.

    Follows the same ``(issues, rejecting, feedback)`` contract as
    :func:`~saidex.grounding._collect_field_checks`: every finding
    (``"retry"`` and ``"flag"``) is returned as a :class:`FieldIssue` with
    ``attempt=0`` for the caller to re-stamp, and feedback is built only from
    ``"retry"`` findings.

    Args:
        validator: The caller-supplied validation callable, or ``None``.
        instance: The instance that already passed Pydantic validation.
        schema_name: Stamped onto each issue and used for the fallback message
            of an exception that carries no text of its own.
        context: Attempt context, handed to :class:`WithContext` validators.

    Returns:
        ``(issues, rejecting, feedback)``. ``rejecting`` is the ``"retry"``-level
        subset of ``issues``. ``feedback`` is ``None`` when the instance is
        accepted — no issues, or only ``"flag"`` issues.

    Raises:
        TypeError: The validator returned something other than a
            :data:`ValidatorResult`.
    """
    if validator is None:
        return [], [], None
    issues = await _call_validator(validator, instance, schema_name, context)
    rejecting = [i for i in issues if i.on_mismatch == "retry"]
    return (
        _to_field_issues(issues, schema_name),
        _to_field_issues(rejecting, schema_name),
        _validator_feedback(issues),
    )


@dataclass(frozen=True)
class _CandidateCheck:
    """Outcome of the post-validation checks on one schema-valid candidate.

    Attributes:
        issues: Every finding (``"retry"`` and ``"flag"``), stamped with the
            attempt index — grounding findings first, then validator findings.
        rejecting: The ``"retry"``-level subset of ``issues`` that rejected the
            candidate; empty when accepted.
        feedback: Feedback text for the model, or ``None`` when accepted.
        stage: The check that rejected the candidate, or ``None`` when accepted.
    """

    issues: list[FieldIssue]
    rejecting: list[FieldIssue]
    feedback: str | None
    stage: Literal["grounding", "validator"] | None


async def _check_candidate(
    instance: MODEL_T,
    *,
    attempt: int,
    context: ValidatorContext,
    source_text: str | None,
    validator: Validator[MODEL_T] | None,
    schema_name: str,
) -> _CandidateCheck:
    """Run source grounding, then the external validator, on a schema-valid *instance*.

    The one post-validation pipeline shared by :func:`_try_with_model` and both
    final-answer branches of the agent loop.  The validator runs only when
    grounding accepted the instance, so it never sees a value that is not in
    the source text.

    Args:
        instance: The instance that already passed Pydantic validation.
        attempt: Attempt index stamped onto every issue.
        context: Attempt context, handed to :class:`WithContext` validators.
        source_text: Text to ground against; falsy disables grounding.
        validator: The caller's external validator, or ``None``.
        schema_name: Stamped onto each issue.

    Returns:
        A :class:`_CandidateCheck`; ``feedback is None`` means accepted.
    """
    grounding_issues, grounding_rejecting, grounding_feedback = _collect_field_checks(
        instance, source_text, schema_name
    )
    issues = [replace(i, attempt=attempt) for i in grounding_issues]
    if grounding_feedback is not None:
        rejecting = [replace(i, attempt=attempt) for i in grounding_rejecting]
        return _CandidateCheck(issues, rejecting, grounding_feedback, "grounding")
    validator_issues, validator_rejecting, validator_feedback = await _run_external_validator(
        validator, instance, schema_name, context
    )
    issues.extend(replace(i, attempt=attempt) for i in validator_issues)
    if validator_feedback is not None:
        rejecting = [replace(i, attempt=attempt) for i in validator_rejecting]
        return _CandidateCheck(issues, rejecting, validator_feedback, "validator")
    return _CandidateCheck(issues, [], None, None)


def _validator_context(
    *,
    attempt_index: int,
    max_attempts: int,
    model_label: str,
    is_last_model: bool,
    reasoning_effort: str | None = None,
) -> ValidatorContext:
    """Build the :class:`ValidatorContext` for the attempt at *attempt_index* (0-based)."""
    attempt = attempt_index + 1
    return ValidatorContext(
        attempt=attempt,
        max_attempts=max_attempts,
        is_final=is_last_model and attempt >= max_attempts,
        model_label=model_label,
        is_fallback=model_label == "fallback",
        reasoning_effort=reasoning_effort,
    )


def _final_answer_rejection(
    stage: Literal["grounding", "validator"] | None,
    feedback: str,
    final_tool_name: str,
    *,
    at_limit: bool,
) -> str:
    """ToolMessage text for a tool-calling final answer rejected after schema validation."""
    if stage == "grounding":
        if at_limit:
            return f"Grounding failed: {feedback}"
        return f"{feedback}\n\nPlease call '{final_tool_name}' again with corrected values."
    if at_limit:
        return f"External validation failed: {feedback}"
    return (
        f"The final answer is structurally valid but failed an additional "
        f"validation check:\n\n{feedback}\n\n"
        f"Please call '{final_tool_name}' again with corrected values."
    )


def _source_text_from_messages(messages: list[BaseMessage]) -> str:
    """Concatenate the human-turn text content of *messages*.

    Used to populate :attr:`ExtractionEvent.source_text` when the caller opts in
    via ``capture_source_text``.  Multimodal (list) content keeps only its text
    parts; non-text parts (e.g. images) are skipped.
    """
    parts: list[str] = []
    for message in messages:
        if message.__class__.__name__ != "HumanMessage":
            continue
        content = message.content
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
    return "\n".join(p for p in parts if p)


async def _emit_completion(
    on_complete: OnComplete | None, event: ExtractionEvent, *, dispatch: bool = True
) -> None:
    """Invoke an ``on_complete`` hook and notify scoped/global observers.

    A failing hook or observer must never break the extraction it is watching,
    so their exceptions are logged and swallowed.  ``dispatch`` is ``False`` for
    internal sub-extractions (the ``…List`` container extraction inside
    :func:`extract_data_list`) so observers see only the top-level call.
    """
    if on_complete is not None:
        try:
            await on_complete(event)
        except Exception as exc:  # noqa: BLE001 — observers must not break the run
            logger.error("on_complete hook raised and was suppressed: %s", exc)
    if dispatch:
        await dispatch_to_observers(event)


async def extract_data(
    llm_model: Any,
    schema: type[MODEL_T],
    messages: list[BaseMessage],
    *,
    mode: ExtractionMode = ExtractionMode.TOOL_CALLING,
    tool_config: ToolCallConfig | None = None,
    callbacks: Callbacks = None,
    fallback_llm_model: Any = None,
    max_primary_retries: int = 3,
    max_fallback_retries: int = 3,
    retry_config: RetryConfig | None = None,
    on_complete: OnComplete | None = None,
    capture_source_text: bool = False,
    validator: Validator[MODEL_T] | None = None,
    on_exhausted: OnExhausted[MODEL_T] | None = None,
    reasoning_escalation: ReasoningEscalation | None = None,
    fallback_reasoning_escalation: ReasoningEscalation | None = None,
    # Private: suppresses observer dispatch for the internal container
    # sub-extraction inside extract_data_list. Not part of the public API.
    _notify_observers: bool = True,
    # Private: reuses a parent chain run (from extract_data_list) instead of
    # opening a new one. Not part of the public API.
    _trace: _ChainRun | None = None,
) -> tuple[MODEL_T | None, ExtractDataStats]:
    """Extract a validated Pydantic model from a LangChain message list.

    Depending on *mode*, the function either drives the LLM via tool-calling
    (the default) or asks it to reply with a raw JSON object.  In both cases,
    when the model returns invalid JSON or the data fails Pydantic validation,
    the detailed error is fed back as a new
    :class:`~langchain_core.messages.HumanMessage` so the model can self-correct.

    Extraction modes
    ----------------
    - :attr:`~saidex.ExtractionMode.TOOL_CALLING` (default):
      requires a tool/function-calling capable model.  Most reliable.
    - :attr:`~saidex.ExtractionMode.JSON`: injects the schema's
      JSON Schema into the prompt and parses the response content as JSON.
      Works with **any** chat model, including those without tool calling.

    Two-phase strategy
    ------------------
    1. **Primary model** — up to *max_primary_retries* validation-retry attempts.
    2. **Fallback model** (optional) — if the primary model exhausts all retries,
       the fallback model gets *max_fallback_retries* fresh attempts starting from
       the original messages.

    Args:
        llm_model: Any LangChain-compatible chat model.  In
            :attr:`~saidex.ExtractionMode.TOOL_CALLING` mode it
            must support ``.bind_tools()``; in
            :attr:`~saidex.ExtractionMode.JSON` mode only
            ``.ainvoke()`` is required.
        schema: The Pydantic ``BaseModel`` subclass to populate.
        messages: Conversation history passed to the model.  Must contain at
            least one message describing what to extract.
        mode: Which extraction strategy to use.  Defaults to
            :attr:`~saidex.ExtractionMode.TOOL_CALLING`.
        tool_config: Which flags to send with ``bind_tools`` in
            :attr:`~saidex.ExtractionMode.TOOL_CALLING` mode.  Defaults to
            :attr:`~saidex.ToolCallConfig.OPENAI` (forced ``tool_choice``,
            ``strict=True``, no parallel calls).  Pass
            :attr:`~saidex.ToolCallConfig.COMPATIBLE` for OpenAI-compatible
            gateways that reject those flags.  Ignored in
            :attr:`~saidex.ExtractionMode.JSON` mode.
        callbacks: Optional LangChain callback handlers (or an existing callback
            manager), e.g. for tracing with LangSmith or Langfuse.
        fallback_llm_model: Optional second model tried when the primary model
            fails all retries.  Typically a larger or more capable model.
        max_primary_retries: Maximum validation-retry attempts for the primary
            model (default 3).
        max_fallback_retries: Maximum validation-retry attempts for the fallback
            model (default 3).
        retry_config: Network-level retry config (rate limits, timeouts).
            Defaults to :data:`~saidex.retry.DEFAULT_RETRY_CONFIG`.
        on_complete: Optional async hook invoked exactly once when the run ends
            (success or failure), receiving an
            :class:`~saidex.models.ExtractionEvent`.  Exceptions it raises are
            logged and suppressed so they cannot break the extraction.
        capture_source_text: When ``True``, the human-turn text of *messages* is
            stored on ``stats.source_text``, passed to *on_complete*, and included
            in the chain-run trace inputs delivered to any attached *callbacks*
            (e.g. a Langfuse/LangSmith handler).  Off by default to avoid
            retaining — or exporting to a tracing backend — potentially
            sensitive input.
        validator: Optional external validation callable (sync or async).  After
            the model passes Pydantic validation it is handed to *validator*,
            which accepts it by returning ``None`` or rejects it by returning a
            non-empty error string or raising.  To report several findings at
            once, return a list of :class:`~saidex.models.ValidationIssue`
            objects: each lands in ``stats.field_issues`` with its own field
            path, ``"retry"`` issues reject the attempt, and ``"flag"`` issues
            are only recorded.  A rejection re-enters the same field-level retry
            loop with the findings surfaced to the model, so it can correct
            cross-field, stateful, or reference-data problems that Pydantic
            alone cannot express.  An invalid return type raises ``TypeError``.
        on_exhausted: Optional hook (sync or async) called once when every
            attempt, including the fallback's, has failed and a schema-valid
            candidate exists (``stats.last_candidate``).  It receives that
            candidate and ``stats.last_rejection`` and returns a salvaged
            instance (the run then succeeds with ``stats.salvaged=True``) or
            ``None``.  The hook gets a deep copy, so editing it in place never
            changes ``stats.last_candidate``.  The returned instance is
            re-validated against *schema* only, not by grounding or
            *validator*, from its stored field values (computed fields left
            out), so aliases and field serializers do not interfere; an invalid
            result keeps the run failed and adds an ``on_exhausted_invalid``
            issue.  Exceptions propagate.
        reasoning_escalation: Opt-in reasoning-effort escalation for the primary
            model.  A list of levels (e.g. ``["low", "medium", "high"]``) binds
            ``reasoning_effort=levels[min(n - 1, len(levels) - 1)]`` on 1-based
            attempt *n* (the last level repeats once the list runs out); ``True`` uses the levels listed in the
            model's ``profile["reasoning_effort_levels"]`` (no escalation when
            the profile lists none).  Parse and validation failures each
            consume a level.  Raises ``ValueError`` before any model call for a
            plain string, an empty list, non-string entries, or a level the
            model profile does not list.  Requires an integration that accepts
            ``reasoning_effort`` (e.g. OpenAI-compatible chat models).
        fallback_reasoning_escalation: Same as *reasoning_escalation*, for the
            fallback model.  The fallback never inherits the primary's levels.

    Returns:
        A tuple of ``(model_instance, stats)``:

        - *model_instance* is the validated :class:`~pydantic.BaseModel` instance,
          or ``None`` when all attempts failed.
        - *stats* is a :class:`~saidex.models.ExtractDataStats`
          object.  Check ``stats.success`` for the outcome and
          ``stats.total_retries`` for the retry count.

    Example::

        result, stats = await extract_data(llm, MySchema, messages)
        if not stats.success:
            print(f"Extraction failed ({stats.failure_reason}) after "
                  f"{stats.total_retries} retries; problem fields: {stats.problem_fields}")

        # Without tool calling:
        result, stats = await extract_data(
            llm, MySchema, messages, mode=ExtractionMode.JSON
        )
    """
    primary_levels = _resolve_reasoning_levels(
        llm_model, reasoning_escalation, param="reasoning_escalation"
    )
    fallback_levels = (
        _resolve_reasoning_levels(
            fallback_llm_model, fallback_reasoning_escalation, param="fallback_reasoning_escalation"
        )
        if fallback_llm_model is not None
        else ()
    )
    effective_retry_config = retry_config or DEFAULT_RETRY_CONFIG

    original_messages = list(messages)
    # Source text for field grounding — derived transiently from the human turn,
    # independent of ``capture_source_text`` (which controls retention in stats).
    grounding_source = _source_text_from_messages(original_messages)

    logger.debug(
        "Starting structured output extraction for %s (mode=%s)",
        schema.__name__,
        mode.value,
    )

    trace = (
        _trace
        if _trace is not None
        else await _ChainRun.start(
            callbacks,
            name="saidex.extract_data",
            inputs=_trace_inputs(
                schema.__name__, mode.value, grounding_source, capture_source_text
            ),
        )
    )
    owns_trace = _trace is None

    async def _finish(
        result: MODEL_T | None, stats: ExtractDataStats
    ) -> tuple[MODEL_T | None, ExtractDataStats]:
        source_text = _source_text_from_messages(original_messages) if capture_source_text else None
        if source_text is not None:
            stats = replace(stats, source_text=source_text)
        await _emit_completion(
            on_complete,
            ExtractionEvent(
                schema_name=schema.__name__,
                result=result,
                stats=stats,
                messages=original_messages,
                source_text=source_text,
            ),
            dispatch=_notify_observers,
        )
        if owns_trace:
            await trace.end(_extract_outputs(result, stats))
        return result, stats

    async def _fail(
        stats: ExtractDataStats, candidate: MODEL_T | None
    ) -> tuple[MODEL_T | None, ExtractDataStats]:
        if on_exhausted is None or candidate is None:
            return await _finish(None, stats)
        salvaged, invalid = await _salvage(
            on_exhausted, schema, candidate, stats.last_rejection, attempt=stats.total_retries
        )
        if salvaged is not None:
            logger.info("on_exhausted salvaged a result for %s.", schema.__name__)
            return await _finish(
                salvaged,
                replace(
                    stats, success=True, failure_reason=None, salvaged=True, llm_error_kind=None
                ),
            )
        if invalid is not None:
            stats = replace(stats, field_issues=(*stats.field_issues, invalid))
        return await _finish(None, stats)

    try:
        primary = await _try_with_model(
            llm_model=llm_model,
            schema=schema,
            messages=list(original_messages),
            trace=trace,
            max_retries=max_primary_retries,
            model_label="primary",
            retry_config=effective_retry_config,
            mode=mode,
            tool_config=tool_config,
            validator=validator,
            source_text=grounding_source,
            is_last_model=fallback_llm_model is None or max_fallback_retries <= 0,
            reasoning_levels=primary_levels,
        )

        if primary.instance is not None:
            return await _finish(
                primary.instance,
                ExtractDataStats(
                    primary_retries=primary.retries,
                    success=True,
                    schema_name=schema.__name__,
                    format_errors=primary.format_errors,
                    field_issues=tuple(primary.field_issues),
                    reasoning_effort=primary.reasoning_effort,
                ),
            )

        if fallback_llm_model is not None:
            logger.info(
                "Primary model exhausted %d retries for %s — switching to fallback.",
                primary.retries,
                schema.__name__,
            )
            fallback_messages = list(original_messages)
            fallback_messages.append(
                HumanMessage(
                    content=(
                        f"A previous attempt to produce structured output for "
                        f"'{schema.__name__}' failed after {primary.retries} retries. "
                        f"Please try again carefully, ensuring your response strictly "
                        f"follows the required JSON schema."
                    )
                )
            )

            fallback = await _try_with_model(
                llm_model=fallback_llm_model,
                schema=schema,
                messages=fallback_messages,
                trace=trace,
                max_retries=max_fallback_retries,
                model_label="fallback",
                retry_config=effective_retry_config,
                mode=mode,
                tool_config=tool_config,
                validator=validator,
                source_text=grounding_source,
                is_last_model=True,
                reasoning_levels=fallback_levels,
            )

            combined_issues = (*primary.field_issues, *fallback.field_issues)
            combined_format_errors = primary.format_errors + fallback.format_errors

            if fallback.instance is not None:
                logger.info("Fallback model succeeded for %s.", schema.__name__)
                return await _finish(
                    fallback.instance,
                    ExtractDataStats(
                        primary_retries=primary.retries,
                        fallback_retries=fallback.retries,
                        fallback_used=True,
                        success=True,
                        schema_name=schema.__name__,
                        format_errors=combined_format_errors,
                        field_issues=combined_issues,
                        reasoning_effort=fallback.reasoning_effort,
                    ),
                )

            logger.error(
                "Both primary and fallback models failed for %s (%d + %d = %d total retries).",
                schema.__name__,
                primary.retries,
                fallback.retries,
                primary.retries + fallback.retries,
            )
            last_candidate, last_rejection = _last_candidate(primary, fallback)
            return await _fail(
                ExtractDataStats(
                    primary_retries=primary.retries,
                    fallback_retries=fallback.retries,
                    fallback_used=True,
                    schema_name=schema.__name__,
                    failure_reason=_combine_failure_reasons(
                        primary.failure_reason, fallback.failure_reason, success=False
                    ),
                    llm_error_kind=_combine_llm_error_kinds(
                        primary.failure_reason,
                        primary.llm_error_kind,
                        fallback.failure_reason,
                        fallback.llm_error_kind,
                        success=False,
                    ),
                    format_errors=combined_format_errors,
                    field_issues=combined_issues,
                    last_candidate=last_candidate,
                    last_rejection=last_rejection,
                    reasoning_effort=fallback.reasoning_effort,
                ),
                last_candidate,
            )

        logger.error(
            "Structured output failed for %s after %d retries (no fallback configured).",
            schema.__name__,
            primary.retries,
        )
        last_candidate, last_rejection = _last_candidate(primary)
        return await _fail(
            ExtractDataStats(
                primary_retries=primary.retries,
                schema_name=schema.__name__,
                failure_reason=primary.failure_reason,
                llm_error_kind=primary.llm_error_kind,
                format_errors=primary.format_errors,
                field_issues=tuple(primary.field_issues),
                last_candidate=last_candidate,
                last_rejection=last_rejection,
                reasoning_effort=primary.reasoning_effort,
            ),
            last_candidate,
        )
    except BaseException as exc:
        # BaseException so a task cancellation (asyncio.CancelledError) still
        # closes the chain span instead of leaking it open in the tracer.
        if owns_trace:
            await trace.error(exc)
        raise


async def extract_data_from_text(
    llm_model: Any,
    schema: type[MODEL_T],
    text: str,
    *,
    mode: ExtractionMode = ExtractionMode.TOOL_CALLING,
    tool_config: ToolCallConfig | None = None,
    system_prompt: str | None = None,
    callbacks: Callbacks = None,
    fallback_llm_model: Any = None,
    max_primary_retries: int = 3,
    max_fallback_retries: int = 3,
    retry_config: RetryConfig | None = None,
    on_complete: OnComplete | None = None,
    capture_source_text: bool = False,
    validator: Validator[MODEL_T] | None = None,
    on_exhausted: OnExhausted[MODEL_T] | None = None,
    reasoning_escalation: ReasoningEscalation | None = None,
    fallback_reasoning_escalation: ReasoningEscalation | None = None,
) -> tuple[MODEL_T | None, ExtractDataStats]:
    """Extract a validated Pydantic model from a plain text string.

    Convenience wrapper around :func:`extract_data` that converts
    *text* into a single :class:`~langchain_core.messages.HumanMessage` (with
    an optional :class:`~langchain_core.messages.SystemMessage` prepended).

    Args:
        llm_model: Any LangChain-compatible chat model.
        schema: The Pydantic ``BaseModel`` subclass to populate.
        text: The text to analyse.
        mode: Which extraction strategy to use (tool calling or raw JSON).
            Defaults to :attr:`~saidex.ExtractionMode.TOOL_CALLING`.
        tool_config: Tool-binding flags for tool-calling mode — see
            :func:`extract_data`.
        system_prompt: Optional system instruction prepended to the message
            list.  When omitted a generic extraction prompt is used.
        callbacks: Optional LangChain callback handlers (or an existing callback manager).
        fallback_llm_model: Optional fallback model.
        max_primary_retries: Validation retries for the primary model.
        max_fallback_retries: Validation retries for the fallback model.
        retry_config: Network-level retry configuration.
        on_complete: Optional async completion hook — see :func:`extract_data`.
        capture_source_text: When ``True``, *text* is stored on
            ``stats.source_text``, passed to *on_complete*, and included in the
            trace inputs sent to attached *callbacks* — see :func:`extract_data`.
        validator: Optional external validation callable — see :func:`extract_data`.
        on_exhausted: Optional salvage hook — see :func:`extract_data`.
        reasoning_escalation: Opt-in reasoning-effort escalation for the primary
            model — see :func:`extract_data`.
        fallback_reasoning_escalation: Same, for the fallback model — see
            :func:`extract_data`.

    Returns:
        Same as :func:`extract_data`.

    Example::

        result, stats = await extract_data_from_text(
            llm,
            PersonSchema,
            "Alice is 30 years old and works as an engineer.",
        )
    """
    effective_system = system_prompt or (
        f"You are a precise data-extraction assistant. "
        f"Extract the requested information from the user's text and populate "
        f"the '{schema.__name__}' schema exactly. "
        f"Return only the structured data — no explanations."
    )

    messages: list[BaseMessage] = [
        SystemMessage(content=effective_system),
        HumanMessage(content=text),
    ]

    return await extract_data(
        llm_model=llm_model,
        schema=schema,
        messages=messages,
        mode=mode,
        tool_config=tool_config,
        callbacks=callbacks,
        fallback_llm_model=fallback_llm_model,
        max_primary_retries=max_primary_retries,
        max_fallback_retries=max_fallback_retries,
        retry_config=retry_config,
        on_complete=on_complete,
        capture_source_text=capture_source_text,
        validator=validator,
        on_exhausted=on_exhausted,
        reasoning_escalation=reasoning_escalation,
        fallback_reasoning_escalation=fallback_reasoning_escalation,
    )


def _build_list_container(schema: type[MODEL_T]) -> type[BaseModel]:
    """Build a single-field wrapper model holding ``items: list[schema]``.

    Batch extraction reuses the entire single-item machinery (tool calling /
    JSON mode, validation, field-level error feedback, retries, fallback) by
    asking the model to fill one container schema whose only field is a list of
    the target items.  Per-item validation errors surface with paths like
    ``items -> 0 -> price`` so the existing error formatter guides the model
    straight to the offending record.

    Args:
        schema: The Pydantic ``BaseModel`` subclass describing a single item.

    Returns:
        A dynamically created ``BaseModel`` subclass with one required field
        ``items`` of type ``list[schema]``.
    """
    container: type[BaseModel] = create_model(
        f"{schema.__name__}List",
        items=(
            list[schema],  # type: ignore[valid-type]
            Field(description=f"All '{schema.__name__}' items found in the input, in order."),
        ),
    )
    container.__doc__ = f"A list of '{schema.__name__}' items extracted from the input."
    return container


async def extract_data_list(
    llm_model: Any,
    schema: type[MODEL_T],
    messages: list[BaseMessage],
    *,
    mode: ExtractionMode = ExtractionMode.TOOL_CALLING,
    tool_config: ToolCallConfig | None = None,
    callbacks: Callbacks = None,
    fallback_llm_model: Any = None,
    max_primary_retries: int = 3,
    max_fallback_retries: int = 3,
    retry_config: RetryConfig | None = None,
    on_complete: OnComplete | None = None,
    capture_source_text: bool = False,
    validator: Validator[MODEL_T] | None = None,
    on_exhausted: OnExhaustedList[MODEL_T] | None = None,
    reasoning_escalation: ReasoningEscalation | None = None,
    fallback_reasoning_escalation: ReasoningEscalation | None = None,
) -> tuple[list[MODEL_T] | None, ExtractDataStats]:
    """Extract a list of validated Pydantic models from a message list.

    The batch counterpart of :func:`extract_data`.  Use it when a single
    document contains several repeated records — invoice line items, multiple
    people in a transcript, several products on a page — and you want them all
    in one LLM call, precisely typed as ``list[ModelT]``.

    Internally the *schema* is wrapped in a one-field container model
    (``items: list[schema]``) and driven through the exact same extraction
    pipeline as :func:`extract_data`, so tool-calling and
    :attr:`~saidex.ExtractionMode.JSON` modes, per-item Pydantic validation,
    field-level error feedback, validation retries and the optional fallback
    model all apply unchanged.

    Args:
        llm_model: Any LangChain-compatible chat model.
        schema: The Pydantic ``BaseModel`` subclass describing **one** item.
        messages: Conversation history passed to the model.  Must contain at
            least one message describing what to extract.
        mode: Which extraction strategy to use.  Defaults to
            :attr:`~saidex.ExtractionMode.TOOL_CALLING`.
        tool_config: Tool-binding flags for tool-calling mode — see
            :func:`extract_data`.
        callbacks: Optional LangChain callback handlers (or an existing callback manager).
        fallback_llm_model: Optional fallback model.
        max_primary_retries: Validation retries for the primary model.
        max_fallback_retries: Validation retries for the fallback model.
        retry_config: Network-level retry configuration.
        on_complete: Optional async completion hook — see :func:`extract_data`.
            Fires once with the list result and the item schema name.
        capture_source_text: When ``True``, the human-turn text is stored on
            ``stats.source_text``, passed to *on_complete*, and included in the
            trace inputs sent to attached *callbacks* — see :func:`extract_data`.
        validator: Optional external validation callable run on **each** extracted
            item (sync or async).  Each finding is recorded per item with its
            field path prefixed by the item index (``items -> 2`` or, for a
            :class:`~saidex.models.ValidationIssue` with ``field_path="amount"``,
            ``items -> 2 -> amount``); a ``"retry"`` finding on any item
            re-enters the shared retry loop.  Wrap it with :func:`with_context` to
            receive the attempt context; every item of one attempt gets the same
            context.  See :func:`extract_data`.
        on_exhausted: Optional salvage hook receiving the last rejected ``list`` of
            items and the issues that rejected it (paths ``items -> i``, item
            schema name).  Return the items to keep (``[]`` is allowed) or
            ``None``; every element must be a *schema* instance, anything else
            (a ``dict`` included) raises ``TypeError``.  The items are copies,
            and the kept list is re-validated like a single result — see
            :func:`extract_data`.
        reasoning_escalation: Opt-in reasoning-effort escalation for the primary
            model — see :func:`extract_data`.
        fallback_reasoning_escalation: Same, for the fallback model — see
            :func:`extract_data`.

    Returns:
        A tuple of ``(items, stats)``:

        - *items* is a ``list`` of validated *schema* instances (possibly
          empty when the document contains no records), or ``None`` when all
          attempts failed.
        - *stats* is a :class:`~saidex.models.ExtractDataStats` whose
          ``item_count`` reflects how many items were returned.

    Example::

        lines, stats = await extract_data_list(llm, InvoiceLine, messages)
        if lines is not None:
            print(f"Extracted {stats.item_count} line items")
    """
    container = _build_list_container(schema)

    container_validator: Validator[BaseModel] | None = None
    if validator is not None:
        item_validator = validator

        async def _validate_each_item(
            container_instance: BaseModel, context: ValidatorContext
        ) -> list[ValidationIssue]:
            """Run the item *validator* over every container item, locating each finding."""
            issues: list[ValidationIssue] = []
            for index, item in enumerate(cast(Any, container_instance).items):
                for issue in await _call_validator(item_validator, item, schema.__name__, context):
                    issues.append(
                        replace(issue, field_path=_item_issue_path(index, issue.field_path))
                    )
            return issues

        container_validator = WithContext(_validate_each_item)

    container_hook: OnExhausted[BaseModel] | None = None
    if on_exhausted is not None:
        item_hook = on_exhausted

        async def _salvage_items(
            container_instance: BaseModel, rejection: tuple[FieldIssue, ...]
        ) -> BaseModel | None:
            """Hand the caller's hook the items, then re-wrap what it keeps."""
            items = list(cast(Any, container_instance).items)
            item_rejection = tuple(replace(i, schema_name=schema.__name__) for i in rejection)
            kept = item_hook(items, item_rejection)
            if inspect.isawaitable(kept):
                kept = await kept
            if kept is None:
                return None
            if not isinstance(kept, list):
                raise TypeError(
                    f"on_exhausted must return a list of {schema.__name__} or None; got {kept!r}"
                )
            wrong = next((k for k in kept if not isinstance(k, schema)), None)
            if wrong is not None:
                raise TypeError(
                    f"on_exhausted must return a list of {schema.__name__} instances; "
                    f"got an element {wrong!r}"
                )
            return container.model_construct(items=kept)

        container_hook = _salvage_items

    # Computed lazily: the list-level source text only feeds the trace inputs
    # (the inner extract_data derives its own copy for grounding).
    grounding_source = _source_text_from_messages(messages) if capture_source_text else ""
    trace = await _ChainRun.start(
        callbacks,
        name="saidex.extract_data_list",
        inputs=_trace_inputs(schema.__name__, mode.value, grounding_source, capture_source_text),
    )
    try:
        # The inner call must not fire the hook or notify observers: it would
        # report the internal ``…List`` container and the wrapper instance
        # instead of the list result. It also reuses this trace instead of
        # opening its own, so the batch shows a single chain run.
        result, stats = await extract_data(
            llm_model=llm_model,
            schema=container,
            messages=messages,
            mode=mode,
            tool_config=tool_config,
            callbacks=None,
            fallback_llm_model=fallback_llm_model,
            max_primary_retries=max_primary_retries,
            max_fallback_retries=max_fallback_retries,
            retry_config=retry_config,
            capture_source_text=capture_source_text,
            validator=container_validator,
            on_exhausted=container_hook,
            reasoning_escalation=reasoning_escalation,
            fallback_reasoning_escalation=fallback_reasoning_escalation,
            _notify_observers=False,
            _trace=trace,
        )
        # Report the item schema name (not the internal ``…List`` container) so
        # the stats group under the schema the caller actually passed.
        item_issues = tuple(
            replace(
                i,
                schema_name=schema.__name__,
                message=i.message.replace(f"'{container.__name__}'", f"'list[{schema.__name__}]'")
                if i.error_type == "on_exhausted_invalid"
                else i.message,
            )
            for i in stats.field_issues
        )
        container_candidate = stats.last_candidate
        stats = replace(
            stats,
            schema_name=schema.__name__,
            field_issues=item_issues,
            last_candidate=(
                None if container_candidate is None else list(cast(Any, container_candidate).items)
            ),
            last_rejection=tuple(
                replace(i, schema_name=schema.__name__) for i in stats.last_rejection
            ),
        )
        items: list[MODEL_T] | None = None if result is None else cast(Any, result).items
        if items is not None:
            stats = replace(stats, item_count=len(items))

        await _emit_completion(
            on_complete,
            ExtractionEvent(
                schema_name=schema.__name__,
                result=cast("BaseModel | list[BaseModel] | None", items),
                stats=stats,
                messages=list(messages),
                source_text=stats.source_text,
            ),
        )
        await trace.end(_extract_outputs(items, stats))
        return items, stats
    except BaseException as exc:
        # BaseException so a task cancellation (asyncio.CancelledError) still
        # closes the chain span instead of leaking it open in the tracer.
        await trace.error(exc)
        raise


async def extract_data_list_from_text(
    llm_model: Any,
    schema: type[MODEL_T],
    text: str,
    *,
    mode: ExtractionMode = ExtractionMode.TOOL_CALLING,
    tool_config: ToolCallConfig | None = None,
    system_prompt: str | None = None,
    callbacks: Callbacks = None,
    fallback_llm_model: Any = None,
    max_primary_retries: int = 3,
    max_fallback_retries: int = 3,
    retry_config: RetryConfig | None = None,
    on_complete: OnComplete | None = None,
    capture_source_text: bool = False,
    validator: Validator[MODEL_T] | None = None,
    on_exhausted: OnExhaustedList[MODEL_T] | None = None,
    reasoning_escalation: ReasoningEscalation | None = None,
    fallback_reasoning_escalation: ReasoningEscalation | None = None,
) -> tuple[list[MODEL_T] | None, ExtractDataStats]:
    """Extract a list of validated Pydantic models from a plain text string.

    Convenience wrapper around :func:`extract_data_list` that converts *text*
    into a single :class:`~langchain_core.messages.HumanMessage` (with an
    optional :class:`~langchain_core.messages.SystemMessage` prepended).  The
    default system prompt instructs the model to return **every** matching
    record rather than a single one.

    Args:
        llm_model: Any LangChain-compatible chat model.
        schema: The Pydantic ``BaseModel`` subclass describing **one** item.
        text: The text to analyse.
        mode: Which extraction strategy to use (tool calling or raw JSON).
            Defaults to :attr:`~saidex.ExtractionMode.TOOL_CALLING`.
        tool_config: Tool-binding flags for tool-calling mode — see
            :func:`extract_data`.
        system_prompt: Optional system instruction prepended to the message
            list.  When omitted a generic list-extraction prompt is used.
        callbacks: Optional LangChain callback handlers (or an existing callback manager).
        fallback_llm_model: Optional fallback model.
        max_primary_retries: Validation retries for the primary model.
        max_fallback_retries: Validation retries for the fallback model.
        retry_config: Network-level retry configuration.
        on_complete: Optional async completion hook — see :func:`extract_data`.
        capture_source_text: When ``True``, *text* is stored on
            ``stats.source_text``, passed to *on_complete*, and included in the
            trace inputs sent to attached *callbacks* — see :func:`extract_data`.
        validator: Optional external validation callable run on each item — see
            :func:`extract_data_list`.
        on_exhausted: Optional salvage hook for the item list — see
            :func:`extract_data_list`.
        reasoning_escalation: Opt-in reasoning-effort escalation for the primary
            model — see :func:`extract_data`.
        fallback_reasoning_escalation: Same, for the fallback model — see
            :func:`extract_data`.

    Returns:
        Same as :func:`extract_data_list`.

    Example::

        items, stats = await extract_data_list_from_text(
            llm,
            InvoiceLine,
            "2x Widget @ 9.99, 1x Gadget @ 19.99",
        )
    """
    effective_system = system_prompt or (
        f"You are a precise data-extraction assistant. "
        f"Extract every '{schema.__name__}' record present in the user's text and "
        f"return them all as a list — one entry per record, in the order they appear. "
        f"Do not merge, deduplicate, or omit records. "
        f"Return only the structured data — no explanations."
    )

    messages: list[BaseMessage] = [
        SystemMessage(content=effective_system),
        HumanMessage(content=text),
    ]

    return await extract_data_list(
        llm_model=llm_model,
        schema=schema,
        messages=messages,
        mode=mode,
        tool_config=tool_config,
        callbacks=callbacks,
        fallback_llm_model=fallback_llm_model,
        max_primary_retries=max_primary_retries,
        max_fallback_retries=max_fallback_retries,
        retry_config=retry_config,
        on_complete=on_complete,
        capture_source_text=capture_source_text,
        validator=validator,
        on_exhausted=on_exhausted,
        reasoning_escalation=reasoning_escalation,
        fallback_reasoning_escalation=fallback_reasoning_escalation,
    )


async def extract_data_with_tools(
    llm_model: Any,
    schema: type[MODEL_T],
    text: str,
    *,
    tools: list[Tool],
    final_answer_mode: ExtractionMode = ExtractionMode.TOOL_CALLING,
    system_prompt: str | None = None,
    callbacks: Callbacks = None,
    fallback_llm_model: Any = None,
    max_iterations: int = 12,
    max_validation_retries: int = 3,
    retry_config: RetryConfig | None = None,
    on_complete: OnComplete | None = None,
    capture_source_text: bool = False,
    validator: Validator[MODEL_T] | None = None,
    on_exhausted: OnExhausted[MODEL_T] | None = None,
    reasoning_escalation: ReasoningEscalation | None = None,
    fallback_reasoning_escalation: ReasoningEscalation | None = None,
) -> tuple[MODEL_T | None, ExtractorRunStats]:
    """Run an agentic tool-loop to produce a validated Pydantic model.

    The LLM may call any of the caller-supplied *tools* freely in a loop (e.g. to
    create resources, look up data) and finally either call the *schema* as a
    "final answer" tool (``TOOL_CALLING`` mode) or emit a plain JSON object
    (``JSON`` mode).  The loop continues until the LLM produces a valid final
    answer or the iteration / validation-retry budget is exhausted.

    Args:
        llm_model: Any LangChain-compatible chat model that supports ``bind_tools``
            and ``ainvoke``.
        schema: The Pydantic ``BaseModel`` subclass for the final answer.
        text: The human-turn text describing the task (becomes a
            :class:`~langchain_core.messages.HumanMessage`).
        tools: Helper tools the LLM may call any number of times before
            producing its final answer.
        final_answer_mode: How the final answer is collected.
            ``TOOL_CALLING`` (default): the *schema* is bound as an extra tool
            and the loop ends when the LLM calls it.
            ``JSON``: only helper tools are bound; when the LLM stops calling
            tools and emits plain text, the content is parsed as JSON.
        system_prompt: Optional system instruction.  A generic instruction is
            used when omitted.
        callbacks: Optional LangChain callback handlers (or an existing callback manager).
        fallback_llm_model: Optional fallback model tried when the primary
            exhausts its iteration or validation budget.
        max_iterations: Maximum LLM invocations per model attempt (default 12).
        max_validation_retries: Max times the final-answer schema may fail
            validation before giving up (default 3).
        retry_config: Network-level retry configuration.
        on_complete: Optional async completion hook — see :func:`extract_data`.
        capture_source_text: When ``True``, *text* is stored on
            ``stats.source_text``, passed to *on_complete*, and included in the
            trace inputs sent to attached *callbacks* — see :func:`extract_data`.
        validator: Optional external validation callable applied to the final
            answer (sync or async).  A rejection keeps the agent loop running
            with the message surfaced to the model — see :func:`extract_data`.
            Wrapped with :func:`with_context`, it receives a
            :class:`~saidex.models.ValidatorContext` whose ``max_attempts`` is
            ``max_validation_retries + 1``; ``max_iterations`` can end the loop
            before any attempt is final.
        on_exhausted: Optional salvage hook — see :func:`extract_data`.
        reasoning_escalation: Opt-in reasoning-effort escalation for the primary
            model — see :func:`extract_data`.  Levels advance per final-answer
            attempt (parse or validation failure); helper-tool iterations keep
            the current level.  In ``JSON`` mode the level is bound on the raw
            model.
        fallback_reasoning_escalation: Same, for the fallback model — see
            :func:`extract_data`.

    Returns:
        ``(model_instance, ExtractorRunStats)`` — *model_instance* is ``None`` on
        failure.

    Example::

        result, stats = await extract_data_with_tools(
            llm,
            FolderDecision,
            document_summary,
            tools=[create_folder_tool],
            system_prompt=rendered_prompt,
        )
    """
    effective_system = system_prompt or (
        f"You are a precise assistant with tools at your disposal. "
        f"Use the provided tools as needed, then produce a final answer "
        f"as a '{schema.__name__}' object. "
        f"Call tools before giving your final answer when additional actions are required."
    )
    messages: list[BaseMessage] = [
        SystemMessage(content=effective_system),
        HumanMessage(content=text),
    ]
    return await run_extractor_agent(
        llm_model=llm_model,
        schema=schema,
        messages=messages,
        tools=tools,
        final_answer_mode=final_answer_mode,
        callbacks=callbacks,
        fallback_llm_model=fallback_llm_model,
        max_iterations=max_iterations,
        max_validation_retries=max_validation_retries,
        retry_config=retry_config,
        on_complete=on_complete,
        capture_source_text=capture_source_text,
        validator=validator,
        on_exhausted=on_exhausted,
        reasoning_escalation=reasoning_escalation,
        fallback_reasoning_escalation=fallback_reasoning_escalation,
    )


async def run_extractor_agent(
    llm_model: Any,
    schema: type[MODEL_T],
    messages: list[BaseMessage],
    *,
    tools: list[Tool],
    final_answer_mode: ExtractionMode = ExtractionMode.TOOL_CALLING,
    callbacks: Callbacks = None,
    fallback_llm_model: Any = None,
    max_iterations: int = 12,
    max_validation_retries: int = 3,
    retry_config: RetryConfig | None = None,
    on_complete: OnComplete | None = None,
    capture_source_text: bool = False,
    validator: Validator[MODEL_T] | None = None,
    on_exhausted: OnExhausted[MODEL_T] | None = None,
    reasoning_escalation: ReasoningEscalation | None = None,
    fallback_reasoning_escalation: ReasoningEscalation | None = None,
) -> tuple[MODEL_T | None, ExtractorRunStats]:
    """Low-level agent loop that accepts a full message list.

    Identical to :func:`extract_data_with_tools` but receives an already-built
    ``messages`` list instead of building ``[SystemMessage, HumanMessage]``
    internally.  Use this when you need precise control over the conversation
    history (e.g. multi-turn scenarios or injecting previous context).

    Supports an optional *fallback_llm_model*: when the primary model
    exhausts its iteration or validation budget without producing a valid
    answer, the fallback model receives a fresh conversation starting from
    the original messages with a brief hint.

    Args:
        on_complete: Optional async completion hook — see :func:`extract_data`.
        capture_source_text: When ``True``, the human-turn text of *messages* is
            stored on ``stats.source_text``, passed to *on_complete*, and included
            in the trace inputs sent to attached *callbacks* — see :func:`extract_data`.
        validator: Optional external validation callable applied to the final
            answer — see :func:`extract_data_with_tools`.
        on_exhausted: Optional salvage hook — see :func:`extract_data`.
        reasoning_escalation: Opt-in reasoning-effort escalation for the primary
            model — see :func:`extract_data`.  Levels advance per final-answer
            attempt (parse or validation failure); helper-tool iterations keep
            the current level.  In ``JSON`` mode the level is bound on the raw
            model.
        fallback_reasoning_escalation: Same, for the fallback model — see
            :func:`extract_data`.

    Returns:
        ``(model_instance, ExtractorRunStats)`` — *model_instance* is ``None`` on
        failure.
    """
    primary_levels = _resolve_reasoning_levels(
        llm_model, reasoning_escalation, param="reasoning_escalation"
    )
    fallback_levels = (
        _resolve_reasoning_levels(
            fallback_llm_model, fallback_reasoning_escalation, param="fallback_reasoning_escalation"
        )
        if fallback_llm_model is not None
        else ()
    )
    effective_retry_config = retry_config or DEFAULT_RETRY_CONFIG
    original_messages = list(messages)
    grounding_source = _source_text_from_messages(original_messages)

    trace = await _ChainRun.start(
        callbacks,
        name="saidex.agent_loop",
        inputs=_trace_inputs(
            schema.__name__,
            final_answer_mode.value,
            grounding_source,
            capture_source_text,
            tools=[t.name for t in tools],
        ),
    )

    async def _finish(
        result: MODEL_T | None, stats: ExtractorRunStats
    ) -> tuple[MODEL_T | None, ExtractorRunStats]:
        source_text = _source_text_from_messages(original_messages) if capture_source_text else None
        if source_text is not None:
            stats = replace(stats, source_text=source_text)
        await _emit_completion(
            on_complete,
            ExtractionEvent(
                schema_name=schema.__name__,
                result=result,
                stats=stats,
                messages=original_messages,
                source_text=source_text,
            ),
        )
        await trace.end(_agent_outputs(result, stats))
        return result, stats

    async def _fail(stats: ExtractorRunStats) -> tuple[MODEL_T | None, ExtractorRunStats]:
        candidate = cast("MODEL_T | None", stats.last_candidate)
        if on_exhausted is None or candidate is None:
            return await _finish(None, stats)
        salvaged, invalid = await _salvage(
            on_exhausted,
            schema,
            candidate,
            stats.last_rejection,
            attempt=stats.validation_retries,
        )
        if salvaged is not None:
            logger.info("on_exhausted salvaged a result for agent loop (%s).", schema.__name__)
            return await _finish(
                salvaged,
                replace(
                    stats, success=True, failure_reason=None, salvaged=True, llm_error_kind=None
                ),
            )
        if invalid is not None:
            stats = replace(stats, field_issues=(*stats.field_issues, invalid))
        return await _finish(None, stats)

    try:
        result, primary_stats = await _run_extractor_agent_with_model(
            llm_model=llm_model,
            schema=schema,
            messages=list(original_messages),
            tools=tools,
            final_answer_mode=final_answer_mode,
            trace=trace,
            max_iterations=max_iterations,
            max_validation_retries=max_validation_retries,
            model_label="primary",
            retry_config=effective_retry_config,
            validator=validator,
            source_text=grounding_source,
            reasoning_levels=primary_levels,
            is_last_model=fallback_llm_model is None,
        )

        if result is not None:
            return await _finish(result, primary_stats)

        if fallback_llm_model is not None:
            logger.info(
                "Primary model exhausted budget for agent loop (%s) — switching to fallback.",
                schema.__name__,
            )
            fallback_messages = list(original_messages)
            fallback_messages.append(
                HumanMessage(
                    content=(
                        f"A previous attempt to complete the task for '{schema.__name__}' "
                        f"did not produce a valid final answer after {primary_stats.iterations} "
                        f"iterations. Please try again carefully."
                    )
                )
            )
            result, fallback_stats = await _run_extractor_agent_with_model(
                llm_model=fallback_llm_model,
                schema=schema,
                messages=fallback_messages,
                tools=tools,
                final_answer_mode=final_answer_mode,
                trace=trace,
                max_iterations=max_iterations,
                max_validation_retries=max_validation_retries,
                model_label="fallback",
                retry_config=effective_retry_config,
                validator=validator,
                source_text=grounding_source,
                reasoning_levels=fallback_levels,
                is_last_model=True,
            )
            succeeded = result is not None
            last_candidate, last_rejection = _combine_candidates(
                primary_stats, fallback_stats, success=succeeded
            )
            combined = ExtractorRunStats(
                iterations=primary_stats.iterations + fallback_stats.iterations,
                tool_calls=primary_stats.tool_calls + fallback_stats.tool_calls,
                validation_retries=(
                    primary_stats.validation_retries + fallback_stats.validation_retries
                ),
                fallback_used=True,
                success=succeeded,
                schema_name=schema.__name__,
                failure_reason=_combine_failure_reasons(
                    primary_stats.failure_reason,
                    fallback_stats.failure_reason,
                    success=succeeded,
                ),
                llm_error_kind=_combine_llm_error_kinds(
                    primary_stats.failure_reason,
                    primary_stats.llm_error_kind,
                    fallback_stats.failure_reason,
                    fallback_stats.llm_error_kind,
                    success=succeeded,
                ),
                format_errors=primary_stats.format_errors + fallback_stats.format_errors,
                field_issues=(*primary_stats.field_issues, *fallback_stats.field_issues),
                last_candidate=last_candidate,
                last_rejection=last_rejection,
                reasoning_effort=fallback_stats.reasoning_effort,
            )
            if succeeded:
                logger.info("Fallback model succeeded for agent loop (%s).", schema.__name__)
                return await _finish(result, combined)
            logger.error(
                "Both models failed for agent loop (%s). Total iterations: %d.",
                schema.__name__,
                combined.iterations,
            )
            return await _fail(combined)

        logger.error(
            "Agent loop failed for %s after %d iterations (no fallback configured).",
            schema.__name__,
            primary_stats.iterations,
        )
        return await _fail(primary_stats)
    except BaseException as exc:
        # BaseException so a task cancellation (asyncio.CancelledError) still
        # closes the chain span instead of leaking it open in the tracer.
        await trace.error(exc)
        raise


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------


def _trace_inputs(
    schema_name: str,
    mode_value: str,
    grounding_source: str,
    capture_source_text: bool,
    **extra: Any,
) -> dict[str, Any]:
    """Chain-run inputs shared by every extraction entry point.

    The raw source text is included only when the caller opted in via
    ``capture_source_text`` — the same privacy gate that governs
    ``stats.source_text``.
    """
    inputs: dict[str, Any] = {"schema": schema_name, "mode": mode_value, **extra}
    if capture_source_text:
        inputs["text"] = grounding_source
    return inputs


def _common_outputs(result: Any, stats: ExtractDataStats | ExtractorRunStats) -> dict[str, Any]:
    """Chain-run output fields shared by both stats types."""
    return {
        "result": result,
        "success": stats.success,
        "failure_reason": None if stats.failure_reason is None else stats.failure_reason.value,
        "llm_error_kind": None if stats.llm_error_kind is None else stats.llm_error_kind.value,
        "fallback_used": stats.fallback_used,
        "has_last_candidate": stats.last_candidate is not None,
        "salvaged": stats.salvaged,
        "problem_fields": list(stats.problem_fields),
        "format_errors": stats.format_errors,
        "reasoning_effort": stats.reasoning_effort,
    }


def _agent_outputs(result: Any, stats: ExtractorRunStats) -> dict[str, Any]:
    """Chain-run outputs for the agent loop: the result plus run-level metrics."""
    return {
        **_common_outputs(result, stats),
        "iterations": stats.iterations,
        "tool_calls": stats.tool_calls,
        "validation_retries": stats.validation_retries,
    }


def _extract_outputs(result: Any, stats: ExtractDataStats) -> dict[str, Any]:
    """Chain-run outputs for extract_data / extract_data_list: result + metrics."""
    return {
        **_common_outputs(result, stats),
        "total_retries": stats.total_retries,
        "primary_retries": stats.primary_retries,
        "fallback_retries": stats.fallback_retries,
        "item_count": stats.item_count,
    }


async def _run_extractor_agent_with_model(
    llm_model: Any,
    schema: type[MODEL_T],
    messages: list[BaseMessage],
    tools: list[Tool],
    final_answer_mode: ExtractionMode,
    trace: _ChainRun,
    max_iterations: int,
    max_validation_retries: int,
    model_label: str,
    retry_config: RetryConfig,
    is_last_model: bool,
    validator: Validator[MODEL_T] | None = None,
    source_text: str | None = None,
    reasoning_levels: tuple[str, ...] = (),
) -> tuple[MODEL_T | None, ExtractorRunStats]:
    """Execute the agent loop with one model; return (instance, stats)."""
    from .tools import Tool as ToolType  # noqa: F401 – used for type clarity only

    tool_by_name: dict[str, Tool] = {t.name: t for t in tools}

    # Build OpenAI-style tool definitions for all helper tools.
    helper_tool_defs = [t.to_openai_tool() for t in tools]

    # Determine how the final answer is collected and build the bound LLM.
    if final_answer_mode is ExtractionMode.TOOL_CALLING:
        final_tool = convert_to_openai_tool(schema)
        final_tool_name: str = final_tool["function"]["name"]
        all_tool_defs = [*helper_tool_defs, final_tool]
        llm = llm_model.bind_tools(all_tool_defs, tool_choice="auto")
    else:
        # JSON mode: only bind helper tools; when the LLM stops calling tools,
        # we parse its text output as JSON into the schema.
        final_tool_name = ""
        llm = llm_model.bind_tools(helper_tool_defs, tool_choice="auto") if tools else llm_model
        # Inject the JSON Schema instruction once, before the loop begins.
        messages = list(messages)
        messages.append(HumanMessage(content=_build_json_instructions(schema)))

    iterations = 0
    total_tool_calls = 0
    validation_retries = 0
    agent_issues: list[FieldIssue] = []
    format_errors = 0
    failure_reason: FailureReason | None = None
    llm_error_kind: ModelErrorKind | None = None
    candidate: MODEL_T | None = None
    rejection: tuple[FieldIssue, ...] = ()
    current_level: str | None = None

    def _stats(success: bool, reason: FailureReason | None) -> ExtractorRunStats:
        return ExtractorRunStats(
            iterations=iterations,
            tool_calls=total_tool_calls,
            validation_retries=validation_retries,
            success=success,
            schema_name=schema.__name__,
            failure_reason=None if success else reason,
            llm_error_kind=(
                None if success or reason is not FailureReason.LLM_ERROR else llm_error_kind
            ),
            format_errors=format_errors,
            field_issues=tuple(agent_issues),
            last_candidate=None if success else candidate,
            last_rejection=() if success else rejection,
            reasoning_effort=current_level,
        )

    while iterations < max_iterations:
        # ── LLM invocation with network-level retries ─────────────────────
        try:
            current_level = _level_for_attempt(reasoning_levels, validation_retries)
            invoke_result = await with_retry(
                operation=partial(
                    trace.invoke_llm, _with_reasoning_effort(llm, current_level), messages
                ),
                config=retry_config,
                operation_name=f"agent_loop({model_label})",
            )
            response = invoke_result.result
            if invoke_result.retry_count:
                logger.info(
                    "%s: agent loop invoke succeeded after %d network retry(ies).",
                    model_label,
                    invoke_result.retry_count,
                )
        except Exception as exc:
            logger.error("%s: agent loop LLM invocation error: %s", model_label, exc)
            failure_reason = FailureReason.LLM_ERROR
            llm_error_kind = _classify_invocation_error(exc, current_level, model_label)
            break

        iterations += 1

        # Always append the assistant response to maintain conversation integrity.
        messages.append(response)

        tool_calls: list[dict[str, Any]] = list(getattr(response, "tool_calls", None) or [])
        invalid_tool_calls: list[dict[str, Any]] = list(
            getattr(response, "invalid_tool_calls", None) or []
        )

        # ── No tool calls ─────────────────────────────────────────────────
        if not tool_calls and not invalid_tool_calls:
            if final_answer_mode is ExtractionMode.JSON:
                # LLM is done calling tools — parse the response text as JSON.
                args, format_error = _parse_json_response(response, schema.__name__, model_label)
                if format_error is not None:
                    format_errors += 1
                    failure_reason = FailureReason.PARSE_ERROR
                    if validation_retries >= max_validation_retries:
                        break
                    messages.append(HumanMessage(content=format_error))
                    validation_retries += 1
                    continue
                assert args is not None
                instance, attempt_issues, error_text = create_instance_with_issues(schema, **args)
                if error_text:
                    agent_issues.extend(
                        replace(i, attempt=validation_retries) for i in attempt_issues
                    )
                    failure_reason = FailureReason.VALIDATION_EXHAUSTED
                    if validation_retries >= max_validation_retries:
                        break
                    logger.warning(
                        "%s: agent loop JSON final answer validation failed: %s",
                        model_label,
                        error_text,
                    )
                    messages.append(
                        HumanMessage(
                            content=(
                                f"The JSON output does not match the required schema.\n\n"
                                f"{error_text}\n\n"
                                f"Please correct it and return only the fixed JSON object."
                            )
                        )
                    )
                    validation_retries += 1
                    continue
                assert instance is not None
                check = await _check_candidate(
                    instance,
                    attempt=validation_retries,
                    context=_validator_context(
                        attempt_index=validation_retries,
                        max_attempts=max_validation_retries + 1,
                        model_label=model_label,
                        is_last_model=is_last_model,
                        reasoning_effort=current_level,
                    ),
                    source_text=source_text,
                    validator=validator,
                    schema_name=schema.__name__,
                )
                agent_issues.extend(check.issues)
                if check.feedback is not None:
                    candidate, rejection = instance, tuple(check.rejecting)
                    failure_reason = FailureReason.VALIDATION_EXHAUSTED
                    if validation_retries >= max_validation_retries:
                        break
                    if check.stage == "grounding":
                        logger.warning(
                            "%s: agent loop JSON final answer failed grounding: %s",
                            model_label,
                            check.feedback,
                        )
                        messages.append(HumanMessage(content=check.feedback))
                    else:
                        logger.warning(
                            "%s: agent loop JSON final answer failed external validation: %s",
                            model_label,
                            check.feedback,
                        )
                        messages.append(
                            HumanMessage(
                                content=(
                                    f"The JSON output is structurally valid but failed an "
                                    f"additional validation check:\n\n{check.feedback}\n\n"
                                    f"Please correct it and return only the fixed JSON object."
                                )
                            )
                        )
                    validation_retries += 1
                    continue
                logger.debug("%s: agent loop JSON final answer validated.", model_label)
                return instance, _stats(success=True, reason=None)
            else:
                # TOOL_CALLING mode but no tool calls — ask the LLM to use a tool.
                failure_reason = FailureReason.NO_TOOL_CALL
                if validation_retries >= max_validation_retries:
                    break
                tool_names = ", ".join([t.name for t in tools] + [final_tool_name])
                messages.append(
                    HumanMessage(
                        content=(
                            f"You must call one of the available tools to proceed. "
                            f"Available tools: {tool_names}. "
                            f"When you have completed all necessary actions, call "
                            f"'{final_tool_name}' to submit your final answer."
                        )
                    )
                )
                validation_retries += 1
                continue

        # ── Invalid tool calls without any valid ones ─────────────────────
        if invalid_tool_calls and not tool_calls:
            format_errors += 1
            failure_reason = FailureReason.PARSE_ERROR
            if validation_retries >= max_validation_retries:
                break
            bad = invalid_tool_calls[0]
            # Try stripping chain-of-thought tags first (Qwen3, DeepSeek-R1).
            raw_args = bad.get("args", "")
            if isinstance(raw_args, str) and _THINKING_RE.search(raw_args):
                clean_args = _strip_thinking_tags(raw_args)
                # Re-process as valid tool calls below if parsing succeeds.
                with contextlib.suppress(ValueError, json.JSONDecodeError):
                    tool_calls = [
                        {
                            "name": bad.get("name", ""),
                            "args": json.loads(clean_args),
                            "id": bad.get("id", ""),
                        }
                    ]
            if not tool_calls:
                messages.append(
                    HumanMessage(
                        content=(
                            f"Your tool call was malformed and could not be parsed. "
                            f"Error: {bad.get('error', 'unknown')}. "
                            f"Please try again with a valid tool call."
                        )
                    )
                )
                validation_retries += 1
                continue

        # ── Process valid tool calls ───────────────────────────────────────
        tool_result_messages: list[ToolMessage] = []
        found_final_answer: MODEL_T | None = None
        final_answer_valid = False
        budget_spent = False

        for tc in tool_calls:
            tc_name: str = tc.get("name", "")
            tc_id: str = tc.get("id", "") or tc_name
            tc_args: dict[str, Any] = tc.get("args") or {}

            # ── Final answer tool ─────────────────────────────────────────
            if tc_name == final_tool_name and final_answer_mode is ExtractionMode.TOOL_CALLING:
                # Try to recover thinking-tagged args.
                if isinstance(tc_args, str) and _THINKING_RE.search(tc_args):
                    clean = _strip_thinking_tags(tc_args)
                    with contextlib.suppress(ValueError, json.JSONDecodeError):
                        tc_args = json.loads(clean)

                instance, attempt_issues, error_text = create_instance_with_issues(
                    schema, **tc_args
                )
                if error_text:
                    agent_issues.extend(
                        replace(i, attempt=validation_retries) for i in attempt_issues
                    )
                    failure_reason = FailureReason.VALIDATION_EXHAUSTED
                    if validation_retries >= max_validation_retries:
                        budget_spent = True
                        tool_result_messages.append(
                            ToolMessage(
                                content=f"Schema validation failed: {error_text}",
                                tool_call_id=tc_id,
                            )
                        )
                    else:
                        tool_result_messages.append(
                            ToolMessage(
                                content=(
                                    f"The final answer did not match the required schema.\n\n"
                                    f"{error_text}\n\n"
                                    f"Please call '{final_tool_name}' again with corrected values."
                                ),
                                tool_call_id=tc_id,
                            )
                        )
                        validation_retries += 1
                else:
                    assert instance is not None
                    check = await _check_candidate(
                        instance,
                        attempt=validation_retries,
                        context=_validator_context(
                            attempt_index=validation_retries,
                            max_attempts=max_validation_retries + 1,
                            model_label=model_label,
                            is_last_model=is_last_model,
                            reasoning_effort=current_level,
                        ),
                        source_text=source_text,
                        validator=validator,
                        schema_name=schema.__name__,
                    )
                    agent_issues.extend(check.issues)
                    if check.feedback is None:
                        tool_result_messages.append(
                            ToolMessage(content="Final answer accepted.", tool_call_id=tc_id)
                        )
                        found_final_answer = instance
                        final_answer_valid = True
                    else:
                        candidate, rejection = instance, tuple(check.rejecting)
                        failure_reason = FailureReason.VALIDATION_EXHAUSTED
                        at_limit = validation_retries >= max_validation_retries
                        tool_result_messages.append(
                            ToolMessage(
                                content=_final_answer_rejection(
                                    check.stage,
                                    check.feedback,
                                    final_tool_name,
                                    at_limit=at_limit,
                                ),
                                tool_call_id=tc_id,
                            )
                        )
                        if at_limit:
                            budget_spent = True
                        else:
                            validation_retries += 1

            # ── Helper tool ───────────────────────────────────────────────
            else:
                total_tool_calls += 1
                tool_obj = tool_by_name.get(tc_name)
                if tool_obj is None:
                    logger.warning("%s: agent loop called unknown tool '%s'", model_label, tc_name)
                    unknown_msg = (
                        f"Unknown tool '{tc_name}'. Available tools: {', '.join(tool_by_name)}."
                    )
                    async with trace.tool_span(tc_name, tc_args) as span:
                        span.record_output(unknown_msg)
                    tool_result_messages.append(
                        ToolMessage(content=unknown_msg, tool_call_id=tc_id)
                    )
                    continue

                async with trace.tool_span(tc_name, tc_args) as span:
                    outcome = await tool_obj._invoke(tc_args)
                    if outcome.handler_error is not None:
                        span.record_error(outcome.handler_error)
                    else:
                        span.record_output(outcome.content)
                    result_str = outcome.content
                logger.debug(
                    "%s: agent loop tool '%s' → %s", model_label, tc_name, result_str[:120]
                )
                tool_result_messages.append(ToolMessage(content=result_str, tool_call_id=tc_id))

        # Append all ToolMessages to history before the next LLM invocation.
        messages.extend(tool_result_messages)

        if final_answer_valid and found_final_answer is not None:
            logger.debug(
                "%s: agent loop final answer validated for %s.",
                model_label,
                schema.__name__,
            )
            return found_final_answer, _stats(success=True, reason=None)

        if budget_spent:
            # Same contract as the JSON final-answer path: a final answer
            # rejected with no validation retries left ends this model's run.
            break

    logger.error(
        "%s: agent loop exhausted budget for %s "
        "(iterations=%d, tool_calls=%d, validation_retries=%d).",
        model_label,
        schema.__name__,
        iterations,
        total_tool_calls,
        validation_retries,
    )
    return None, _stats(success=False, reason=failure_reason or FailureReason.VALIDATION_EXHAUSTED)


#: pydantic's ``MISSING`` default sentinel (pydantic >= 2.14), else ``None``.
_MISSING: Any = getattr(pydantic, "MISSING", None)

#: Schemas (by qualified name) whose non-strict fallback was already warned about.
_WARNED_FALLBACK: set[str] = set()

#: Keyword arguments of ``bind_tools`` that limited providers are known to reject.
_TOOL_FLAG_NAMES = ("strict", "tool_choice", "parallel_tool_calls")

#: Substrings marking a *client* error (as opposed to a transport hiccup).  A
#: rejection is only treated as a flag problem when one of these appears next to
#: a flag name, so an unrelated failure never triggers a relaxed re-bind.
_BAD_REQUEST_MARKERS = (
    "400",
    "bad request",
    "invalid_request",
    "invalid request",
    "unsupported",
    "unrecognized",
    "not supported",
    "unknown parameter",
    "unexpected keyword",
)


def _strip_defaults(node: Any, *, in_properties: bool = False) -> Any:
    """Recursively drop JSON-Schema ``default`` keywords, keeping fields *named* ``default``.

    A removed non-null default is appended to the node's ``description`` so the
    model still sees the intended value.
    """
    if isinstance(node, dict):
        out = {
            key: _strip_defaults(value, in_properties=key == "properties" and not in_properties)
            for key, value in node.items()
            if in_properties or key != "default"
        }
        if not in_properties and node.get("default") is not None:
            note = f"Default: {json.dumps(node['default'])}"
            existing = out.get("description")
            out["description"] = f"{existing} {note}" if existing else note
        return out
    if isinstance(node, list):
        return [_strip_defaults(item) for item in node]
    return node


def _annotation_incompatibility(annotation: Any, stack: list[type]) -> str | None:
    """Reason a type annotation cannot be expressed in a strict schema, or ``None``."""
    if annotation is Any or annotation is object:
        return "an Any/object-typed field"
    if is_typeddict(annotation):
        return "a TypedDict-typed field (optional keys cannot be expressed)"
    if inspect.isclass(annotation):
        if issubclass(annotation, BaseModel) or dataclasses.is_dataclass(annotation):
            return _model_incompatibility(annotation, stack)
        if issubclass(annotation, Mapping):
            return "a dict/Mapping-typed field"
        return None
    origin = get_origin(annotation)
    if origin is None or origin is Literal:
        return None
    if inspect.isclass(origin) and issubclass(origin, Mapping):
        return "a dict/Mapping-typed field"
    for arg in get_args(annotation):
        reason = _annotation_incompatibility(arg, stack)
        if reason is not None:
            return reason
    return None


def _model_fields(model: type) -> list[tuple[str, Any, Any]]:
    """``(name, annotation, default)`` for a pydantic model or dataclass."""
    if issubclass(model, BaseModel):
        return [(n, i.annotation, i.default) for n, i in model.model_fields.items()]
    try:
        hints = get_type_hints(model)
    except Exception:
        hints = {}
    return [
        (f.name, hints.get(f.name, f.type), f.default) for f in dataclasses.fields(cast(Any, model))
    ]


def _model_incompatibility(model: type, stack: list[type]) -> str | None:
    if model in stack:
        return f"{model.__name__} is recursive (reachable from itself)"
    stack.append(model)
    try:
        for name, annotation, default in _model_fields(model):
            if _MISSING is not None and default is _MISSING:
                return f"field '{name}' of {model.__name__} has a MISSING default"
            reason = _annotation_incompatibility(annotation, stack)
            if reason is not None:
                return f"field '{name}' of {model.__name__} is {reason}"
    finally:
        stack.pop()
    return None


def _strict_incompatibility(schema: type[BaseModel]) -> str | None:
    """Why *schema* cannot be sent as a strict tool, or ``None`` when it can.

    Strict mode makes every property required and forbids free-form keys, so
    ``dict[...]`` / ``Any`` / ``TypedDict`` fields (at any depth) would collapse
    to ``{}`` or lose optional keys, ``MISSING`` defaults could never be omitted,
    and recursive models cannot be expressed.

    Args:
        schema: The extraction schema.

    Returns:
        A human-readable reason, or ``None`` if strict binding is safe.
    """
    return _model_incompatibility(schema, [])


def _schema_tool(schema: type[BaseModel], *, strict: bool) -> dict[str, Any]:
    """Build the OpenAI-style tool definition of *schema*, strict or plain.

    Args:
        schema: The extraction schema.
        strict: Request OpenAI strict structured outputs.  The ``default``
            keyword, which strict mode rejects, is removed.

    Returns:
        The tool definition.
    """
    if not strict:
        return convert_to_openai_tool(schema)
    tool = convert_to_openai_tool(schema, strict=True)
    tool["function"]["parameters"] = _strip_defaults(tool["function"]["parameters"])
    return tool


def _bind_schema_tool(
    llm_model: Any,
    schema: type[BaseModel],
    tool_name: str,
    config: ToolCallConfig,
    model_label: str = "",
    *,
    force_plain: bool = False,
) -> tuple[Any, bool]:
    """Bind the tool for *schema* to *llm_model*, sending only the flags *config* asks for.

    A ``None`` field means the keyword argument is left out of the request
    entirely — some OpenAI-compatible gateways reject the mere presence of
    ``strict`` or ``parallel_tool_calls``.  With ``strict=True`` the schema itself
    is converted strictly, unless it uses shapes strict mode cannot express, in
    which case the plain schema is bound and a warning is logged.

    Args:
        llm_model: The chat model to bind against.
        schema: The extraction schema.
        tool_name: The tool's function name, used for a forced ``tool_choice``.
        config: Which flags to send, and with what values.
        model_label: Model name used in the fallback warning.
        force_plain: Bind the plain (non-strict) tool even if *config* asks for
            strict, keeping every other flag.

    Returns:
        The bound chat model and whether the bound tool is a strict one.
    """
    strict_schema = config.strict is True and not force_plain
    if strict_schema:
        reason = _strict_incompatibility(schema)
        if reason is not None:
            key = f"{schema.__module__}.{schema.__qualname__}"
            level = logging.DEBUG if key in _WARNED_FALLBACK else logging.WARNING
            _WARNED_FALLBACK.add(key)
            logger.log(
                level,
                "%s: %s cannot be sent as a strict tool schema (%s) — binding it non-strict. "
                "Use ToolCallConfig.COMPATIBLE or ExtractionMode.JSON to silence this.",
                model_label,
                schema.__name__,
                reason,
            )
            strict_schema = False
    tool = _schema_tool(schema, strict=strict_schema)
    kwargs: dict[str, Any] = {}
    if config.tool_choice == "forced":
        kwargs["tool_choice"] = tool_name
    elif config.tool_choice == "auto":
        kwargs["tool_choice"] = "auto"
    if config.strict is not None:
        kwargs["strict"] = config.strict and strict_schema
    if config.parallel_tool_calls is not None:
        kwargs["parallel_tool_calls"] = config.parallel_tool_calls
    return llm_model.bind_tools([tool], **kwargs), strict_schema


def _classify_invocation_error(
    exc: BaseException, level: str | None, model_label: str
) -> ModelErrorKind:
    """Classify an LLM invocation error, attributing a rejected ``reasoning_effort`` bind."""
    if level is not None and _is_reasoning_effort_rejection(exc):
        logger.error(
            "%s: model does not accept reasoning_effort=%s (unsupported by this integration "
            "or conflicting with its own reasoning settings); disable reasoning_escalation "
            "for this model.",
            model_label,
            level,
        )
        return ModelErrorKind.INVALID_REQUEST
    return classify_model_error(exc)


def _is_tool_flag_rejection(exc: BaseException) -> bool:
    """Whether *exc* looks like the provider refusing one of the tool-binding flags.

    Covers both failure sites: an HTTP 400 from a gateway that does not support
    ``strict`` / a forced ``tool_choice`` / ``parallel_tool_calls``, and a
    ``TypeError`` from a ``bind_tools`` implementation that never accepted the
    keyword argument in the first place.
    """
    text = str(exc).lower()
    if not any(flag in text for flag in _TOOL_FLAG_NAMES):
        return False
    return _is_client_error(exc)


def _is_client_error(exc: BaseException) -> bool:
    """Whether *exc* looks like the provider refusing the request (HTTP 400-ish)."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 400:
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _BAD_REQUEST_MARKERS)


def _is_strict_schema_rejection(exc: BaseException) -> bool:
    """A client error that is not specifically about a tool-binding flag."""
    if not _is_client_error(exc):
        return False
    text = str(exc).lower()
    return "schema" in text or not any(flag in text for flag in _TOOL_FLAG_NAMES)


@dataclass(frozen=True)
class _ModelAttempt(Generic[MODEL_T]):
    """Outcome of one single-model extraction attempt loop.

    Attributes:
        instance: The validated instance, or ``None`` if every attempt failed.
        retries: Number of validation/format retries consumed.
        field_issues: Structured field problems seen across all attempts, each
            tagged with the attempt index in which it occurred.
        format_errors: Count of pure parse/tool-call (format) failures.
        failure_reason: Why the loop ended without success, or ``None`` on success.
        candidate: The most recent schema-valid instance rejected by grounding
            or the external validator, kept across later failures.
        rejection: The retry-level issues that rejected :attr:`candidate`.
        llm_error_kind: Kind of the model error when ``failure_reason`` is ``llm_error``.
        reasoning_effort: Level bound for the last invocation.
    """

    instance: MODEL_T | None
    retries: int
    field_issues: list[FieldIssue]
    format_errors: int
    failure_reason: FailureReason | None
    candidate: MODEL_T | None = None
    rejection: tuple[FieldIssue, ...] = ()
    llm_error_kind: ModelErrorKind | None = None
    reasoning_effort: str | None = None


def _last_candidate(
    *attempts: _ModelAttempt[MODEL_T],
) -> tuple[MODEL_T | None, tuple[FieldIssue, ...]]:
    """The most recent schema-valid candidate across model runs, with its rejection.

    Later runs win; a run without a candidate (e.g. a fallback that only
    produced parse errors) does not overwrite an earlier one.
    """
    for attempt in reversed(attempts):
        if attempt.candidate is not None:
            return attempt.candidate, attempt.rejection
    return None, ()


async def _try_with_model(
    llm_model: Any,
    schema: type[MODEL_T],
    messages: list[BaseMessage],
    trace: _ChainRun,
    max_retries: int,
    model_label: str,
    retry_config: RetryConfig,
    mode: ExtractionMode,
    is_last_model: bool,
    tool_config: ToolCallConfig | None = None,
    validator: Validator[MODEL_T] | None = None,
    source_text: str | None = None,
    reasoning_levels: tuple[str, ...] = (),
) -> _ModelAttempt[MODEL_T]:
    """Attempt structured extraction with one model, with validation retries.

    Works in either :attr:`ExtractionMode.TOOL_CALLING` or
    :attr:`ExtractionMode.JSON` mode.  Returns a :class:`_ModelAttempt` capturing
    the instance (or ``None``), the retry count, the structured field issues seen
    across attempts, the pure parse/format-error count, and a failure reason.

    *is_last_model* is ``False`` when another model runs if this one fails; it
    decides ``ValidatorContext.is_final``.  *reasoning_levels* are the
    ``reasoning_effort`` levels bound per attempt (the last repeats); ``()``
    disables escalation.
    """
    flags = tool_config or ToolCallConfig.OPENAI
    #: Set once the schema tool has been re-bound with relaxed flags, so a
    #: provider that keeps refusing cannot spin the loop forever.
    relaxed = False
    relax_rebind: Callable[[], Any] | None = None
    #: Whether the currently bound schema tool is a strict one, and whether it
    #: was already swapped for the plain tool (a separate one-shot relaxation).
    strict_bound = False
    strict_relaxed = False

    if mode is ExtractionMode.TOOL_CALLING:
        schema_label: str = convert_to_openai_tool(schema)["function"]["name"]
        if flags.auto_relax:

            def relax_rebind() -> Any:
                return _bind_schema_tool(
                    llm_model, schema, schema_label, ToolCallConfig.COMPATIBLE
                )[0]

        try:
            llm, strict_bound = _bind_schema_tool(
                llm_model, schema, schema_label, flags, model_label
            )
        except TypeError as exc:
            if relax_rebind is None or not _is_tool_flag_rejection(exc):
                logger.error(
                    "%s: bind_tools rejected the tool-calling flags for %s: %s",
                    model_label,
                    schema.__name__,
                    exc,
                )
                return _ModelAttempt(
                    None,
                    0,
                    [],
                    0,
                    FailureReason.LLM_ERROR,
                    llm_error_kind=ModelErrorKind.INVALID_REQUEST,
                )
            logger.warning(
                "%s: model does not accept the tool-binding flags for %s (%s) — "
                "re-binding with relaxed flags.",
                model_label,
                schema.__name__,
                exc,
            )
            relaxed = True
            strict_bound = False
            try:
                llm = relax_rebind()
            except TypeError as relaxed_exc:
                logger.error(
                    "%s: relaxed bind_tools also failed for %s: %s",
                    model_label,
                    schema.__name__,
                    relaxed_exc,
                )
                return _ModelAttempt(
                    None,
                    0,
                    [],
                    0,
                    FailureReason.LLM_ERROR,
                    llm_error_kind=ModelErrorKind.INVALID_REQUEST,
                )
    else:
        schema_label = schema.__name__
        llm = llm_model
        # Inject the JSON Schema + output instructions once, before the loop.
        messages.append(HumanMessage(content=_build_json_instructions(schema)))

    retries = 0
    remaining = max_retries
    issues: list[FieldIssue] = []
    format_errors = 0
    failure_reason: FailureReason | None = None
    candidate: MODEL_T | None = None
    rejection: tuple[FieldIssue, ...] = ()
    last_level: str | None = None

    while remaining > 0:
        level = _level_for_attempt(reasoning_levels, retries)
        last_level = level
        # --- LLM invocation with network-level retries ---
        try:
            invoke_result = await with_retry(
                operation=partial(trace.invoke_llm, _with_reasoning_effort(llm, level), messages),
                config=retry_config,
                operation_name=f"structured_output({model_label})",
            )
            response = invoke_result.result
            if invoke_result.retry_count:
                logger.info(
                    "%s: LLM invoke succeeded after %d network retry(ies).",
                    model_label,
                    invoke_result.retry_count,
                )
        except Exception as exc:
            if (
                flags.auto_relax
                and strict_bound
                and not strict_relaxed
                and _is_strict_schema_rejection(exc)
            ):
                logger.warning(
                    "%s: provider rejected the strict tool schema for %s (%s) — "
                    "re-binding the plain schema and retrying.",
                    model_label,
                    schema.__name__,
                    exc,
                )
                strict_relaxed = True
                try:
                    llm, strict_bound = _bind_schema_tool(
                        llm_model, schema, schema_label, flags, model_label, force_plain=True
                    )
                except TypeError as plain_exc:
                    logger.error(
                        "%s: plain bind_tools also failed for %s: %s",
                        model_label,
                        schema.__name__,
                        plain_exc,
                    )
                    return _ModelAttempt(
                        None,
                        retries,
                        issues,
                        format_errors,
                        FailureReason.LLM_ERROR,
                        candidate,
                        rejection,
                    )
                continue
            if relax_rebind is not None and not relaxed and _is_tool_flag_rejection(exc):
                logger.warning(
                    "%s: provider rejected the tool-calling flags for %s (%s) — "
                    "re-binding with relaxed flags and retrying.",
                    model_label,
                    schema.__name__,
                    exc,
                )
                relaxed = True
                strict_bound = False
                try:
                    llm = relax_rebind()
                except TypeError as relaxed_exc:
                    logger.error(
                        "%s: relaxed bind_tools also failed for %s: %s",
                        model_label,
                        schema.__name__,
                        relaxed_exc,
                    )
                    return _ModelAttempt(
                        None,
                        retries,
                        issues,
                        format_errors,
                        FailureReason.LLM_ERROR,
                        candidate,
                        rejection,
                        llm_error_kind=ModelErrorKind.INVALID_REQUEST,
                        reasoning_effort=last_level,
                    )
                # Deliberately no ``remaining -= 1``: the relaxed re-bind is a
                # transport-level correction, not a failed extraction attempt.
                continue
            logger.error(
                "%s: LLM invocation error for %s: %s",
                model_label,
                schema.__name__,
                exc,
            )
            return _ModelAttempt(
                None,
                retries,
                issues,
                format_errors,
                FailureReason.LLM_ERROR,
                candidate,
                rejection,
                _classify_invocation_error(exc, level, model_label),
                reasoning_effort=last_level,
            )

        # --- Extract candidate arguments (mode-specific) ---
        if mode is ExtractionMode.TOOL_CALLING:
            args, format_error = _parse_tool_response(response, schema_label, model_label)
        else:
            args, format_error = _parse_json_response(response, schema_label, model_label)

        if format_error is not None:
            messages.append(HumanMessage(content=format_error))
            format_errors += 1
            failure_reason = FailureReason.PARSE_ERROR
            retries += 1
            remaining -= 1
            continue

        # --- Pydantic validation (shared) ---
        assert args is not None
        try:
            instance, attempt_issues, error_text = create_instance_with_issues(schema, **args)
        except Exception as exc:
            logger.error("%s: unexpected parse error for %s: %s", model_label, schema.__name__, exc)
            return _ModelAttempt(
                None,
                retries,
                issues,
                format_errors,
                FailureReason.PARSE_ERROR,
                candidate,
                rejection,
                reasoning_effort=last_level,
            )

        if error_text:
            logger.warning(
                "%s: validation failed for %s (attempt %d/%d):\n%s",
                model_label,
                schema.__name__,
                retries + 1,
                max_retries,
                error_text,
            )
            issues.extend(replace(i, attempt=retries) for i in attempt_issues)
            failure_reason = FailureReason.VALIDATION_EXHAUSTED
            messages.append(
                HumanMessage(
                    content=(
                        f"The output does not match the required schema. "
                        f"Please correct it based on the following errors:\n\n"
                        f"{error_text}\n\n"
                        f"Return a corrected response that addresses every issue above."
                    )
                )
            )
            retries += 1
            remaining -= 1
            continue

        assert instance is not None
        check = await _check_candidate(
            instance,
            attempt=retries,
            context=_validator_context(
                attempt_index=retries,
                max_attempts=max_retries,
                model_label=model_label,
                is_last_model=is_last_model,
                reasoning_effort=level,
            ),
            source_text=source_text,
            validator=validator,
            schema_name=schema.__name__,
        )
        # Record every finding (including advisory ``flag`` ones); only a
        # rejection produces feedback and re-enters the loop.
        issues.extend(check.issues)
        if check.feedback is not None:
            candidate, rejection = instance, tuple(check.rejecting)
            failure_reason = FailureReason.VALIDATION_EXHAUSTED
            if check.stage == "grounding":
                logger.warning(
                    "%s: grounding failed for %s (attempt %d/%d):\n%s",
                    model_label,
                    schema.__name__,
                    retries + 1,
                    max_retries,
                    check.feedback,
                )
                messages.append(HumanMessage(content=check.feedback))
            else:
                logger.warning(
                    "%s: external validation failed for %s (attempt %d/%d): %s",
                    model_label,
                    schema.__name__,
                    retries + 1,
                    max_retries,
                    check.feedback,
                )
                messages.append(
                    HumanMessage(
                        content=(
                            f"The extracted data is structurally valid but failed an additional "
                            f"validation check:\n\n{check.feedback}\n\n"
                            f"Return a corrected response that resolves the problem above."
                        )
                    )
                )
            retries += 1
            remaining -= 1
            continue

        logger.debug("%s: successfully extracted %s.", model_label, schema.__name__)
        return _ModelAttempt(
            instance, retries, issues, format_errors, None, reasoning_effort=last_level
        )

    return _ModelAttempt(
        None,
        retries,
        issues,
        format_errors,
        failure_reason,
        candidate,
        rejection,
        reasoning_effort=last_level,
    )


# ---------------------------------------------------------------------------
# Response parsing — tool calling
# ---------------------------------------------------------------------------


def _parse_tool_response(
    response: Any,
    tool_name: str,
    model_label: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Extract tool-call arguments from a tool-calling response.

    Returns ``(args, None)`` when a valid tool call is present, or
    ``(None, feedback_message)`` describing what the model must fix.
    """
    has_valid_calls = bool(getattr(response, "tool_calls", None))
    has_invalid_calls = bool(getattr(response, "invalid_tool_calls", None))

    if has_valid_calls and not has_invalid_calls:
        args: dict[str, Any] = response.tool_calls[0]["args"]
        return args, None

    error_detail = ""
    if has_invalid_calls:
        bad = response.invalid_tool_calls[0]
        # Some models (Qwen3, DeepSeek-R1) embed thinking blocks inside the
        # tool-call args string, producing invalid JSON.  Strip the blocks and
        # retry before giving up.
        raw_args = bad.get("args", "")
        if isinstance(raw_args, str) and _THINKING_RE.search(raw_args):
            clean_args = _strip_thinking_tags(raw_args)
            try:
                parsed_args: dict[str, Any] = json.loads(clean_args)
                logger.info(
                    "%s: recovered tool-call args for %s after stripping thinking tags.",
                    model_label,
                    tool_name,
                )
                return parsed_args, None
            except (ValueError, json.JSONDecodeError):
                pass
        error_detail = (
            f"\n\nInvalid tool call detected:"
            f"\n  Error:         {bad.get('error', 'unknown')}"
            f"\n  Malformed args: {bad.get('args', 'N/A')}"
        )
        logger.warning(
            "%s: invalid tool call for %s: %s",
            model_label,
            tool_name,
            bad.get("error", "")[:200],
        )
    else:
        logger.warning(
            "%s: no tool_calls in response for %s. Content: %s",
            model_label,
            tool_name,
            str(getattr(response, "content", ""))[:200],
        )

    feedback = (
        f"Your response did not contain a valid tool call for the "
        f"'{tool_name}' schema. Please try again and call the tool "
        f"with valid JSON that matches the schema exactly."
        f"{error_detail}"
    )
    return None, feedback


# ---------------------------------------------------------------------------
# Response parsing — raw JSON
# ---------------------------------------------------------------------------


def _build_json_instructions(schema: type[BaseModel]) -> str:
    """Build the prompt that instructs the model to emit a raw JSON object."""
    schema_json = json.dumps(schema.model_json_schema(), indent=2, ensure_ascii=False)
    return (
        "Respond with a SINGLE JSON object that strictly conforms to the JSON "
        "Schema below. Output ONLY the raw JSON object — no markdown code "
        "fences, no comments, and no explanatory text before or after it.\n\n"
        f"JSON Schema for '{schema.__name__}':\n{schema_json}"
    )


def _parse_json_response(
    response: Any,
    schema_name: str,
    model_label: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a raw-JSON response's content into an argument dict.

    Returns ``(args, None)`` on success, or ``(None, feedback_message)`` when
    the content was not parseable as a JSON object.
    """
    content = getattr(response, "content", "")
    try:
        parsed = _extract_json_from_content(content)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning(
            "%s: could not parse JSON from response for %s: %s",
            model_label,
            schema_name,
            exc,
        )
        return None, (
            f"Your response could not be parsed as JSON ({exc}). "
            f"Return ONLY a single valid JSON object that matches the "
            f"'{schema_name}' schema — no code fences and no extra text."
        )

    if not isinstance(parsed, dict):
        logger.warning(
            "%s: JSON response for %s was not an object (got %s).",
            model_label,
            schema_name,
            type(parsed).__name__,
        )
        return None, (
            f"Your response was valid JSON but not a JSON object. "
            f"Return a single JSON object with the fields of the "
            f"'{schema_name}' schema."
        )

    return parsed, None


#: Block types that never carry the answer.  Reasoning models emit these next to
#: the real answer block; parsing their text would feed the model's private
#: chain of thought to the JSON parser.  Matched *before* the text key, because
#: some providers put the reasoning under a ``text`` key as well.
_NON_ANSWER_BLOCK_TYPES = frozenset({"reasoning", "thinking", "redacted_thinking"})


def _extract_json_from_content(content: Any) -> Any:
    """Extract and parse a JSON value from raw LLM response content.

    Handles plain JSON, JSON wrapped in markdown code fences, JSON surrounded
    by stray explanatory text, responses that start with a chain-of-thought
    ``<think>…</think>`` block (Qwen3, DeepSeek-R1, …), and truncated or
    otherwise malformed JSON (missing commas, unclosed brackets, etc.) via
    ``json-repair`` as a last-resort fallback.

    List-shaped content is flattened by taking every block that exposes a
    ``text`` key and is not a reasoning block.  Block *type* names differ per
    provider — LangChain's ``"text"``, the Responses API's ``"output_text"`` —
    so the text key, not the type name, decides what counts as an answer.
    """
    if isinstance(content, list):
        # Multimodal / chunked content — concatenate the text parts.
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                if part.get("type") in _NON_ANSWER_BLOCK_TYPES:
                    continue
                text_parts.append(str(part["text"]))
        text = "\n".join(text_parts)
    else:
        text = str(content)

    # Strip chain-of-thought reasoning blocks first so their stray `{`, `,` etc.
    # cannot confuse the outermost-block fallback below.
    text = _strip_thinking_tags(text)
    text = _strip_code_fences(text.strip())
    if not text:
        raise ValueError("empty response content")

    try:
        return json.loads(text)
    except json.JSONDecodeError as json_exc:
        # Fallback 1: isolate the outermost { ... } block and retry.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        # Fallback 2: use json-repair to fix truncated / malformed LLM output
        # (missing commas, unclosed strings/arrays/objects, stray text, …).
        logger.warning(
            "Standard JSON parsing failed; attempting json-repair on %d-char content.",
            len(text),
        )
        repaired = repair_json(text, return_objects=True)
        if isinstance(repaired, (dict, list)):
            return repaired
        raise ValueError(
            f"json-repair could not produce a dict/list from content: {text[:200]!r}"
        ) from json_exc


def _strip_code_fences(text: str) -> str:
    """Remove a surrounding markdown code fence (```` ```json ... ``` ````)."""
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


# Matches <think>...</think> and <thinking>...</thinking> reasoning blocks emitted
# by chain-of-thought models (Qwen3, DeepSeek-R1, o1, …). The blocks are stripped
# before JSON / tool-call parsing so stray `{`, `,` etc. inside the reasoning do
# not confuse the JSON parser.
_THINKING_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


def _strip_thinking_tags(text: str) -> str:
    """Remove chain-of-thought ``<think>…</think>`` blocks from model output."""
    return _THINKING_RE.sub("", text).strip()
