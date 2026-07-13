"""Core extraction logic: LLM → validated Pydantic model."""

from __future__ import annotations

import contextlib
import inspect
import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from json_repair import repair_json
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field, create_model

from ._callbacks import _ChainRun
from .grounding import collect_field_check_issues
from .models import (
    ExtractDataStats,
    ExtractionEvent,
    ExtractionMode,
    ExtractorRunStats,
    FieldIssue,
)
from .observability import dispatch_to_observers
from .retry import DEFAULT_RETRY_CONFIG, RetryConfig, with_retry
from .utils import create_instance_with_issues

if TYPE_CHECKING:
    from .tools import Tool

logger = logging.getLogger(__name__)

MODEL_T = TypeVar("MODEL_T", bound=BaseModel)

#: Signature of an ``on_complete`` hook: an async callable receiving a single
#: :class:`~saidex.models.ExtractionEvent`.
OnComplete = Callable[[ExtractionEvent], Awaitable[None]]

#: Signature of an external ``validator``: a sync **or** async callable that
#: receives the freshly validated model instance and either accepts it (returns
#: ``None`` / an empty string) or rejects it (returns a non-empty error message
#: or raises an exception).  A rejection re-enters the existing field-level retry
#: loop with the message surfaced to the model, exactly like a Pydantic failure.
Validator = Callable[[MODEL_T], "str | None | Awaitable[str | None]"]


def _external_validator_issue(schema_name: str, message: str, attempt: int) -> FieldIssue:
    """Build a :class:`~saidex.models.FieldIssue` for an external-validator rejection."""
    return FieldIssue(
        schema_name=schema_name,
        field_path="<external validator>",
        category="value",
        error_type="external_validator",
        message=message,
        attempt=attempt,
        received=None,
    )


async def _run_external_validator(
    validator: Validator[MODEL_T] | None,
    instance: MODEL_T,
    schema_name: str,
) -> str | None:
    """Run an optional external *validator* against a validated *instance*.

    The validator may be sync or async.  It signals rejection either by raising
    any exception or by returning a non-empty string; both are turned into a
    human-readable message suitable for feeding back to the model.

    Args:
        validator: The caller-supplied validation callable, or ``None``.
        instance: The instance that already passed Pydantic validation.
        schema_name: Used to build a fallback message when an exception carries
            no text of its own.

    Returns:
        ``None`` when the instance is accepted, otherwise the rejection message.
    """
    if validator is None:
        return None
    try:
        outcome = validator(instance)
        if inspect.isawaitable(outcome):
            outcome = await outcome
    except Exception as exc:  # noqa: BLE001 — any raise is a deliberate rejection
        return (
            str(exc) or f"{exc.__class__.__name__} raised by external validator for {schema_name}"
        )
    if outcome:
        return str(outcome)
    return None


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
    callbacks: list[Any] | None = None,
    fallback_llm_model: Any = None,
    max_primary_retries: int = 3,
    max_fallback_retries: int = 3,
    retry_config: RetryConfig | None = None,
    on_complete: OnComplete | None = None,
    capture_source_text: bool = False,
    validator: Validator[MODEL_T] | None = None,
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
        callbacks: Optional list of LangChain callback handlers (e.g. for
            tracing with LangSmith or Langfuse).
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
            non-empty error string or raising.  A rejection re-enters the same
            field-level retry loop with the message surfaced to the model, so it
            can correct cross-field, stateful, or reference-data problems that
            Pydantic alone cannot express.

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
            validator=validator,
            source_text=grounding_source,
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
                validator=validator,
                source_text=grounding_source,
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
                    ),
                )

            logger.error(
                "Both primary and fallback models failed for %s (%d + %d = %d total retries).",
                schema.__name__,
                primary.retries,
                fallback.retries,
                primary.retries + fallback.retries,
            )
            return await _finish(
                None,
                ExtractDataStats(
                    primary_retries=primary.retries,
                    fallback_retries=fallback.retries,
                    fallback_used=True,
                    schema_name=schema.__name__,
                    failure_reason=fallback.failure_reason or primary.failure_reason,
                    format_errors=combined_format_errors,
                    field_issues=combined_issues,
                ),
            )

        logger.error(
            "Structured output failed for %s after %d retries (no fallback configured).",
            schema.__name__,
            primary.retries,
        )
        return await _finish(
            None,
            ExtractDataStats(
                primary_retries=primary.retries,
                schema_name=schema.__name__,
                failure_reason=primary.failure_reason,
                format_errors=primary.format_errors,
                field_issues=tuple(primary.field_issues),
            ),
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
    system_prompt: str | None = None,
    callbacks: list[Any] | None = None,
    fallback_llm_model: Any = None,
    max_primary_retries: int = 3,
    max_fallback_retries: int = 3,
    retry_config: RetryConfig | None = None,
    on_complete: OnComplete | None = None,
    capture_source_text: bool = False,
    validator: Validator[MODEL_T] | None = None,
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
        system_prompt: Optional system instruction prepended to the message
            list.  When omitted a generic extraction prompt is used.
        callbacks: Optional LangChain callback handlers.
        fallback_llm_model: Optional fallback model.
        max_primary_retries: Validation retries for the primary model.
        max_fallback_retries: Validation retries for the fallback model.
        retry_config: Network-level retry configuration.
        on_complete: Optional async completion hook — see :func:`extract_data`.
        capture_source_text: When ``True``, *text* is stored on
            ``stats.source_text``, passed to *on_complete*, and included in the
            trace inputs sent to attached *callbacks* — see :func:`extract_data`.
        validator: Optional external validation callable — see :func:`extract_data`.

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
        callbacks=callbacks,
        fallback_llm_model=fallback_llm_model,
        max_primary_retries=max_primary_retries,
        max_fallback_retries=max_fallback_retries,
        retry_config=retry_config,
        on_complete=on_complete,
        capture_source_text=capture_source_text,
        validator=validator,
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
    callbacks: list[Any] | None = None,
    fallback_llm_model: Any = None,
    max_primary_retries: int = 3,
    max_fallback_retries: int = 3,
    retry_config: RetryConfig | None = None,
    on_complete: OnComplete | None = None,
    capture_source_text: bool = False,
    validator: Validator[MODEL_T] | None = None,
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
        callbacks: Optional LangChain callback handlers.
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
            item (sync or async).  Any item it rejects re-enters the shared retry
            loop with a per-item message (e.g. ``items -> 2: …``) surfaced to the
            model.  See :func:`extract_data`.

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

        async def _validate_each_item(container_instance: BaseModel) -> str | None:
            """Run the item *validator* over every container item, joining failures."""
            errors: list[str] = []
            for index, item in enumerate(cast(Any, container_instance).items):
                item_error = await _run_external_validator(item_validator, item, schema.__name__)
                if item_error is not None:
                    errors.append(f"items -> {index}: {item_error}")
            return "\n".join(errors) if errors else None

        container_validator = _validate_each_item

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
            callbacks=None,
            fallback_llm_model=fallback_llm_model,
            max_primary_retries=max_primary_retries,
            max_fallback_retries=max_fallback_retries,
            retry_config=retry_config,
            capture_source_text=capture_source_text,
            validator=container_validator,
            _notify_observers=False,
            _trace=trace,
        )
        # Report the item schema name (not the internal ``…List`` container) so
        # the stats group under the schema the caller actually passed.
        item_issues = tuple(replace(i, schema_name=schema.__name__) for i in stats.field_issues)
        stats = replace(stats, schema_name=schema.__name__, field_issues=item_issues)
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
    system_prompt: str | None = None,
    callbacks: list[Any] | None = None,
    fallback_llm_model: Any = None,
    max_primary_retries: int = 3,
    max_fallback_retries: int = 3,
    retry_config: RetryConfig | None = None,
    on_complete: OnComplete | None = None,
    capture_source_text: bool = False,
    validator: Validator[MODEL_T] | None = None,
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
        system_prompt: Optional system instruction prepended to the message
            list.  When omitted a generic list-extraction prompt is used.
        callbacks: Optional LangChain callback handlers.
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
        callbacks=callbacks,
        fallback_llm_model=fallback_llm_model,
        max_primary_retries=max_primary_retries,
        max_fallback_retries=max_fallback_retries,
        retry_config=retry_config,
        on_complete=on_complete,
        capture_source_text=capture_source_text,
        validator=validator,
    )


async def extract_data_with_tools(
    llm_model: Any,
    schema: type[MODEL_T],
    text: str,
    *,
    tools: list[Tool],
    final_answer_mode: ExtractionMode = ExtractionMode.TOOL_CALLING,
    system_prompt: str | None = None,
    callbacks: list[Any] | None = None,
    fallback_llm_model: Any = None,
    max_iterations: int = 12,
    max_validation_retries: int = 3,
    retry_config: RetryConfig | None = None,
    on_complete: OnComplete | None = None,
    capture_source_text: bool = False,
    validator: Validator[MODEL_T] | None = None,
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
        callbacks: Optional LangChain callback handlers.
        fallback_llm_model: Optional fallback model tried when the primary
            exhausts its iteration budget.
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
    )


async def run_extractor_agent(
    llm_model: Any,
    schema: type[MODEL_T],
    messages: list[BaseMessage],
    *,
    tools: list[Tool],
    final_answer_mode: ExtractionMode = ExtractionMode.TOOL_CALLING,
    callbacks: list[Any] | None = None,
    fallback_llm_model: Any = None,
    max_iterations: int = 12,
    max_validation_retries: int = 3,
    retry_config: RetryConfig | None = None,
    on_complete: OnComplete | None = None,
    capture_source_text: bool = False,
    validator: Validator[MODEL_T] | None = None,
) -> tuple[MODEL_T | None, ExtractorRunStats]:
    """Low-level agent loop that accepts a full message list.

    Identical to :func:`extract_data_with_tools` but receives an already-built
    ``messages`` list instead of building ``[SystemMessage, HumanMessage]``
    internally.  Use this when you need precise control over the conversation
    history (e.g. multi-turn scenarios or injecting previous context).

    Supports an optional *fallback_llm_model*: when the primary model
    exhausts its iteration budget without producing a valid answer, the
    fallback model receives a fresh conversation starting from the original
    messages with a brief hint.

    Args:
        on_complete: Optional async completion hook — see :func:`extract_data`.
        capture_source_text: When ``True``, the human-turn text of *messages* is
            stored on ``stats.source_text``, passed to *on_complete*, and included
            in the trace inputs sent to attached *callbacks* — see :func:`extract_data`.
        validator: Optional external validation callable applied to the final
            answer — see :func:`extract_data_with_tools`.

    Returns:
        ``(model_instance, ExtractorRunStats)`` — *model_instance* is ``None`` on
        failure.
    """
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
            )
            succeeded = result is not None
            combined = ExtractorRunStats(
                iterations=primary_stats.iterations + fallback_stats.iterations,
                tool_calls=primary_stats.tool_calls + fallback_stats.tool_calls,
                validation_retries=(
                    primary_stats.validation_retries + fallback_stats.validation_retries
                ),
                fallback_used=True,
                success=succeeded,
                schema_name=schema.__name__,
                failure_reason=None if succeeded else fallback_stats.failure_reason,
                format_errors=primary_stats.format_errors + fallback_stats.format_errors,
                field_issues=(*primary_stats.field_issues, *fallback_stats.field_issues),
            )
            if succeeded:
                logger.info("Fallback model succeeded for agent loop (%s).", schema.__name__)
            else:
                logger.error(
                    "Both models failed for agent loop (%s). Total iterations: %d.",
                    schema.__name__,
                    combined.iterations,
                )
            return await _finish(result, combined)

        logger.error(
            "Agent loop failed for %s after %d iterations (no fallback configured).",
            schema.__name__,
            primary_stats.iterations,
        )
        return await _finish(None, primary_stats)
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
        "failure_reason": stats.failure_reason,
        "fallback_used": stats.fallback_used,
        "problem_fields": list(stats.problem_fields),
        "format_errors": stats.format_errors,
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
    validator: Validator[MODEL_T] | None = None,
    source_text: str | None = None,
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
    failure_reason: str | None = None

    def _stats(success: bool, reason: str | None) -> ExtractorRunStats:
        return ExtractorRunStats(
            iterations=iterations,
            tool_calls=total_tool_calls,
            validation_retries=validation_retries,
            success=success,
            schema_name=schema.__name__,
            failure_reason=None if success else reason,
            format_errors=format_errors,
            field_issues=tuple(agent_issues),
        )

    while iterations < max_iterations:
        # ── LLM invocation with network-level retries ─────────────────────
        try:
            invoke_result = await with_retry(
                operation=partial(trace.invoke_llm, llm, messages),
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
            failure_reason = "llm_error"
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
                    failure_reason = "parse_error"
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
                    failure_reason = "validation_exhausted"
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
                grounding_issues, grounding_error = collect_field_check_issues(
                    instance, source_text, schema.__name__
                )
                agent_issues.extend(
                    replace(i, attempt=validation_retries) for i in grounding_issues
                )
                if grounding_error is not None:
                    failure_reason = "validation_exhausted"
                    if validation_retries >= max_validation_retries:
                        break
                    logger.warning(
                        "%s: agent loop JSON final answer failed grounding: %s",
                        model_label,
                        grounding_error,
                    )
                    messages.append(HumanMessage(content=grounding_error))
                    validation_retries += 1
                    continue
                validator_error = await _run_external_validator(
                    validator, instance, schema.__name__
                )
                if validator_error is not None:
                    agent_issues.append(
                        _external_validator_issue(
                            schema.__name__, validator_error, validation_retries
                        )
                    )
                    failure_reason = "validation_exhausted"
                    if validation_retries >= max_validation_retries:
                        break
                    logger.warning(
                        "%s: agent loop JSON final answer failed external validation: %s",
                        model_label,
                        validator_error,
                    )
                    messages.append(
                        HumanMessage(
                            content=(
                                f"The JSON output is structurally valid but failed an "
                                f"additional validation check:\n\n{validator_error}\n\n"
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
                failure_reason = "no_tool_call"
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
            failure_reason = "parse_error"
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
                    failure_reason = "validation_exhausted"
                    if validation_retries >= max_validation_retries:
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
                    grounding_issues, grounding_error = collect_field_check_issues(
                        instance, source_text, schema.__name__
                    )
                    agent_issues.extend(
                        replace(i, attempt=validation_retries) for i in grounding_issues
                    )
                    if grounding_error is not None:
                        failure_reason = "validation_exhausted"
                        if validation_retries >= max_validation_retries:
                            tool_result_messages.append(
                                ToolMessage(
                                    content=f"Grounding failed: {grounding_error}",
                                    tool_call_id=tc_id,
                                )
                            )
                        else:
                            tool_result_messages.append(
                                ToolMessage(
                                    content=(
                                        f"{grounding_error}\n\n"
                                        f"Please call '{final_tool_name}' again with corrected "
                                        f"values."
                                    ),
                                    tool_call_id=tc_id,
                                )
                            )
                            validation_retries += 1
                        continue
                    validator_error = await _run_external_validator(
                        validator, instance, schema.__name__
                    )
                    if validator_error is not None:
                        agent_issues.append(
                            _external_validator_issue(
                                schema.__name__, validator_error, validation_retries
                            )
                        )
                        failure_reason = "validation_exhausted"
                        if validation_retries >= max_validation_retries:
                            tool_result_messages.append(
                                ToolMessage(
                                    content=f"External validation failed: {validator_error}",
                                    tool_call_id=tc_id,
                                )
                            )
                        else:
                            tool_result_messages.append(
                                ToolMessage(
                                    content=(
                                        f"The final answer is structurally valid but failed an "
                                        f"additional validation check:\n\n{validator_error}\n\n"
                                        f"Please call '{final_tool_name}' again with corrected "
                                        f"values."
                                    ),
                                    tool_call_id=tc_id,
                                )
                            )
                            validation_retries += 1
                    else:
                        tool_result_messages.append(
                            ToolMessage(content="Final answer accepted.", tool_call_id=tc_id)
                        )
                        found_final_answer = instance
                        final_answer_valid = True

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

        # Check if we already hit the validation limit for a bad final answer.
        if validation_retries >= max_validation_retries and not final_answer_valid:
            # Give the model one last iteration only when tool_calls were present
            # (otherwise we already broke out above).
            pass

    logger.error(
        "%s: agent loop exhausted budget for %s "
        "(iterations=%d, tool_calls=%d, validation_retries=%d).",
        model_label,
        schema.__name__,
        iterations,
        total_tool_calls,
        validation_retries,
    )
    return None, _stats(success=False, reason=failure_reason or "validation_exhausted")


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
    """

    instance: MODEL_T | None
    retries: int
    field_issues: list[FieldIssue]
    format_errors: int
    failure_reason: str | None


async def _try_with_model(
    llm_model: Any,
    schema: type[MODEL_T],
    messages: list[BaseMessage],
    trace: _ChainRun,
    max_retries: int,
    model_label: str,
    retry_config: RetryConfig,
    mode: ExtractionMode,
    validator: Validator[MODEL_T] | None = None,
    source_text: str | None = None,
) -> _ModelAttempt[MODEL_T]:
    """Attempt structured extraction with one model, with validation retries.

    Works in either :attr:`ExtractionMode.TOOL_CALLING` or
    :attr:`ExtractionMode.JSON` mode.  Returns a :class:`_ModelAttempt` capturing
    the instance (or ``None``), the retry count, the structured field issues seen
    across attempts, the pure parse/format-error count, and a failure reason.
    """
    if mode is ExtractionMode.TOOL_CALLING:
        tool = convert_to_openai_tool(schema)
        schema_label: str = tool["function"]["name"]
        llm = llm_model.bind_tools(
            [tool],
            tool_choice=schema_label,
            parallel_tool_calls=False,
            strict=True,
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
    failure_reason: str | None = None

    while remaining > 0:
        # --- LLM invocation with network-level retries ---
        try:
            invoke_result = await with_retry(
                operation=partial(trace.invoke_llm, llm, messages),
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
            logger.error(
                "%s: LLM invocation error for %s: %s",
                model_label,
                schema.__name__,
                exc,
            )
            return _ModelAttempt(None, retries, issues, format_errors, "llm_error")

        # --- Extract candidate arguments (mode-specific) ---
        if mode is ExtractionMode.TOOL_CALLING:
            args, format_error = _parse_tool_response(response, schema_label, model_label)
        else:
            args, format_error = _parse_json_response(response, schema_label, model_label)

        if format_error is not None:
            messages.append(HumanMessage(content=format_error))
            format_errors += 1
            failure_reason = "parse_error"
            retries += 1
            remaining -= 1
            continue

        # --- Pydantic validation (shared) ---
        assert args is not None
        try:
            instance, attempt_issues, error_text = create_instance_with_issues(schema, **args)
        except Exception as exc:
            logger.error("%s: unexpected parse error for %s: %s", model_label, schema.__name__, exc)
            return _ModelAttempt(None, retries, issues, format_errors, "parse_error")

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
            failure_reason = "validation_exhausted"
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
        grounding_issues, grounding_error = collect_field_check_issues(
            instance, source_text, schema.__name__
        )
        # Record every grounding issue (including advisory ``flag`` ones); only a
        # ``retry`` failure produces feedback and re-enters the loop below.
        issues.extend(replace(i, attempt=retries) for i in grounding_issues)
        if grounding_error is not None:
            logger.warning(
                "%s: grounding failed for %s (attempt %d/%d):\n%s",
                model_label,
                schema.__name__,
                retries + 1,
                max_retries,
                grounding_error,
            )
            failure_reason = "validation_exhausted"
            messages.append(HumanMessage(content=grounding_error))
            retries += 1
            remaining -= 1
            continue

        validator_error = await _run_external_validator(validator, instance, schema.__name__)
        if validator_error is not None:
            logger.warning(
                "%s: external validation failed for %s (attempt %d/%d): %s",
                model_label,
                schema.__name__,
                retries + 1,
                max_retries,
                validator_error,
            )
            issues.append(_external_validator_issue(schema.__name__, validator_error, retries))
            failure_reason = "validation_exhausted"
            messages.append(
                HumanMessage(
                    content=(
                        f"The extracted data is structurally valid but failed an additional "
                        f"validation check:\n\n{validator_error}\n\n"
                        f"Return a corrected response that resolves the problem above."
                    )
                )
            )
            retries += 1
            remaining -= 1
            continue

        logger.debug("%s: successfully extracted %s.", model_label, schema.__name__)
        return _ModelAttempt(instance, retries, issues, format_errors, None)

    return _ModelAttempt(None, retries, issues, format_errors, failure_reason)


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
