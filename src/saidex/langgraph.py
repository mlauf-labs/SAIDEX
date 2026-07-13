"""``SaidexToolNode`` — a validating, correcting drop-in for LangGraph's ``ToolNode``.

Requires the optional ``saidex[langgraph]`` extra.  The node repairs
``invalid_tool_calls`` (think-tag stripping, json-repair), validates tool-call
arguments against each tool's Pydantic schema *before* execution, and applies a
configurable policy to invalid calls: structured feedback, an LLM correction
cycle, or fail-fast raising.  Valid calls are executed by an internal stock
``langgraph.prebuilt.ToolNode``, so parallel execution, ``handle_tool_errors``,
``Command`` returns and injected state keep working unchanged.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Callable, Sequence
from typing import Any, Literal

from langchain_core.callbacks.base import Callbacks
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel

try:
    from langgraph.prebuilt import ToolNode
except ImportError as exc:  # keep langgraph an optional dependency
    raise ImportError(
        "saidex.langgraph requires the 'langgraph' package. "
        "Install it with: pip install 'saidex[langgraph]'."
    ) from exc

from json_repair import repair_json
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as _as_tool

from .extractor import _strip_code_fences, _strip_thinking_tags, extract_data
from .models import (
    ExtractionMode,
    FieldIssue,
    ToolCallOutcome,
    ToolCallStats,
    ToolNodeEvent,
    ToolNodeStats,
)
from .observability import dispatch_tool_node_event
from .retry import RetryConfig
from .sync import _run_sync
from .utils import create_instance_with_issues

__all__ = [
    "SaidexToolNode",
    "ToolCallValidationError",
]

logger = logging.getLogger(__name__)

#: Sentinel meaning "let the stock ToolNode keep its own default".
_UNSET: Any = object()

OnInvalid = Literal["feedback", "correct", "raise"]

#: Network-level retry config for the correction cycle's ``extract_data`` call.
#: The cycle already has its own bounded retry budget (``max_correction_retries``
#: validation attempts) — silently layering transport-level retries on top of
#: that would make a failing correction slow and non-deterministic instead of
#: falling straight through to the feedback policy, and would let a mock
#: model's raised exceptions be retried instead of surfacing immediately in
#: tests. Corrections never retry on network errors either way.
_NO_NETWORK_RETRY = RetryConfig(
    max_retries=0, retry_delays=[], retryable_exceptions=(), rate_limit_exceptions=()
)


@dataclasses.dataclass
class _CallPlan:
    """Per-call working state for one node invocation."""

    call: dict[str, Any]
    from_invalid: bool = False
    raw_entry: dict[str, Any] | None = None
    raw_args: str | None = None
    repaired: bool = False
    recoverable: bool = True
    prevalidated: bool = False
    schema: type[BaseModel] | None = None
    error_text: str | None = None
    issues: tuple[FieldIssue, ...] = ()
    executable: bool = False
    #: ``""`` is the pre-dispatch default; the dispatch loop in ``_arun``
    #: finalizes every plan to one of the four real ``ToolCallOutcome``
    #: values before stats are built — see ``_finalized_outcome``, which
    #: turns that invariant into a type-checked narrowing at the point
    #: ``ToolCallStats`` is constructed.
    outcome: ToolCallOutcome | Literal[""] = ""
    correction_retries: int = 0
    #: False when the call carries no ``id`` at all. A ``ToolMessage`` requires
    #: a ``tool_call_id``, so an id-less call has no way to be answered — it
    #: must not get a fabricated-id feedback message, and must be dropped (not
    #: kept as an orphaned malformed entry) when sanitizing.
    answerable: bool = True
    #: True when the call has an id but no ``name`` — it CAN be answered (the
    #: id is real), but no tool name may be fabricated into the feedback
    #: message or the sanitized history.
    missing_name: bool = False


class ToolCallValidationError(Exception):
    """Raised by ``on_invalid="raise"`` before any tool call is executed.

    Attributes:
        failures: One ``(tool_name, tool_call_id, error_text)`` triple per
            invalid call, in original call order.
    """

    def __init__(self, failures: Sequence[tuple[str, str | None, str]]) -> None:
        self.failures = tuple(failures)
        lines = [
            f"- {name} (id={call_id}): {error.splitlines()[0]}"
            for name, call_id, error in self.failures
        ]
        super().__init__(
            f"Tool-call validation failed for {len(self.failures)} call(s):\n" + "\n".join(lines)
        )


class SaidexToolNode(Runnable[Any, Any]):
    """Drop-in replacement for ``langgraph.prebuilt.ToolNode`` with validation.

    Accepts the same ``tools`` sequence and produces the same ``ToolMessage``
    outputs as the stock node, so it replaces it with a one-line change::

        graph.add_node("tools", SaidexToolNode(tools))

    **Malformed entries with a missing ``id``/``name``.** LangChain types
    ``id`` as optional (``str | None``) on *both* ``ToolCall`` (the
    ``tool_calls`` list) and ``InvalidToolCall`` (``invalid_tool_calls``), and
    types ``name`` as optional only on ``InvalidToolCall``. So a provider can
    in principle emit a call — well-formed or malformed — carrying no id at
    all, or a malformed entry carrying an id but no name. Because a
    ``ToolMessage`` *requires* a real ``tool_call_id``, the two situations are
    handled differently:

    - **Has an ``id``, no ``name`` (``invalid_tool_calls`` only):** the call
      CAN be answered. A feedback ``ToolMessage`` is emitted with the real
      ``tool_call_id`` and no fabricated ``name`` (never invented) explaining
      that the call carried no tool name and must be re-issued with one. The
      raw entry is kept in the sanitized message's ``invalid_tool_calls`` so
      it still matches that feedback message.
    - **No ``id`` at all (``tool_calls`` or ``invalid_tool_calls``):** the
      call CANNOT be answered — there is no ``tool_call_id`` to construct a
      ``ToolMessage`` with, so none is emitted, and the call is not executed
      either (executing it would still leave it unanswered). A
      ``logging.Logger.warning`` records the drop. When ``sanitize_messages``
      is enabled, the entry is removed from the sanitized ``AIMessage``'s
      ``tool_calls``/``invalid_tool_calls`` (whichever it came from) so
      history carries neither an orphaned tool result nor an unanswerable
      call. With ``sanitize_messages=False`` the entry is left exactly as the
      model produced it, matching stock ``ToolNode`` behavior — that is the
      caller's choice.

    **A tool's schema validator raises an unexpected exception type.**
    Pre-validation only treats ``ValueError``/``AssertionError`` raised inside
    a Pydantic ``@field_validator``/``@model_validator(mode="before")`` as a
    correctable validation failure (Pydantic itself only converts those two
    into a ``ValidationError``). If such a validator raises anything else —
    e.g. a bare ``KeyError`` — that is a bug in the tool's schema, not a
    correctable model mistake, and is intentionally allowed to propagate out
    of the whole node rather than being laundered into a ``ToolMessage``
    feedback the model can't meaningfully act on. Stock ``ToolNode``'s
    ``handle_tool_errors`` would instead turn it into an error result for
    just that one call — write validators that only raise
    ``ValueError``/``AssertionError`` if that softer, per-call behavior is
    required.

    **The correction cycle's retry budget is validation-only.**
    ``max_correction_retries`` bounds LLM *validation* attempts, but
    correction LLM calls do not perform network-level (transport) retries —
    the cycle uses a zero-retry :class:`~saidex.retry.RetryConfig` so a
    failing correction degrades deterministically instead of retrying
    silently on top of its own retry budget. A transient provider error
    (e.g. a 429) on a correction call therefore does not retry at the
    transport level either: it degrades straight to the ``"feedback"``
    policy for that call rather than crashing the node.

    Args:
        tools: The tools available for execution — ``BaseTool`` instances or
            plain callables, exactly as accepted by the stock ``ToolNode``.
        on_invalid: Policy for calls whose arguments fail validation or cannot
            be recovered: ``"feedback"`` (default) answers with a corrective
            ``ToolMessage``; ``"correct"`` runs a bounded LLM correction cycle
            before executing; ``"raise"`` raises
            :class:`ToolCallValidationError` before executing anything.
        correction_model: Chat model driving the ``"correct"`` cycle.  Required
            when ``on_invalid="correct"``.
        max_correction_retries: LLM attempts per call in the correction cycle.
        correction_mode: Extraction mode for the correction cycle — use
            :attr:`~saidex.ExtractionMode.JSON` for models without tool calling.
        strip_thinking: Strip ``<think>`` blocks from raw invalid-call args
            before parsing.
        json_repair: Apply ``json-repair`` to raw invalid-call args that do not
            parse as JSON.
        sanitize_messages: Return an updated copy of the ``AIMessage`` (same
            ``id``) whose tool calls carry the repaired/corrected args, so the
            standard ``add_messages`` reducer replaces the malformed message in
            state.  Set to ``False`` for message channels with a plain append
            reducer.
        name: Node name, forwarded to the inner ``ToolNode``.
        tags: Tags, forwarded to the inner ``ToolNode``.
        handle_tool_errors: Forwarded to the inner ``ToolNode`` when given;
            otherwise the stock default stays active.
        messages_key: State key holding the message list.
    """

    def __init__(
        self,
        tools: Sequence[BaseTool | Callable[..., Any]],
        *,
        on_invalid: OnInvalid = "feedback",
        correction_model: Any = None,
        max_correction_retries: int = 2,
        correction_mode: ExtractionMode = ExtractionMode.TOOL_CALLING,
        strip_thinking: bool = True,
        json_repair: bool = True,
        sanitize_messages: bool = True,
        name: str = "tools",
        tags: list[str] | None = None,
        handle_tool_errors: Any = _UNSET,
        messages_key: str = "messages",
    ) -> None:
        if on_invalid not in ("feedback", "correct", "raise"):
            raise ValueError(
                f"on_invalid must be 'feedback', 'correct' or 'raise', got {on_invalid!r}"
            )
        if on_invalid == "correct" and correction_model is None:
            raise ValueError("on_invalid='correct' requires a correction_model")

        inner_kwargs: dict[str, Any] = {"name": name, "tags": tags, "messages_key": messages_key}
        if handle_tool_errors is not _UNSET:
            inner_kwargs["handle_tool_errors"] = handle_tool_errors
        self._inner = ToolNode(tools, **inner_kwargs)

        self.name = name
        self._on_invalid: OnInvalid = on_invalid
        self._correction_model = correction_model
        self._max_correction_retries = max_correction_retries
        self._correction_mode = correction_mode
        self._strip_thinking = strip_thinking
        self._json_repair = json_repair
        self._sanitize_messages = sanitize_messages
        self._messages_key = messages_key
        self._tools_by_name = _build_registry(tools)

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        """Synchronous entry point; bridges to :meth:`ainvoke` on a fresh event loop.

        See :meth:`ainvoke` for the full pipeline description (input shapes
        accepted, output shape produced). Not usable from inside an already
        running event loop — call :meth:`ainvoke` there instead.

        Args:
            input: Graph state — a message list, a dict containing
                ``messages_key``, or a state object exposing it as an attribute.
            config: LangChain run configuration; forwarded to the inner
                ``ToolNode`` and the correction cycle so tracing nests properly.

        Returns:
            Tool results in the same shape the stock ``ToolNode`` produces, as
            described in :meth:`ainvoke`.

        Raises:
            RuntimeError: If called from inside an already running event loop.
        """
        return _run_sync(
            self._arun(input, config, **kwargs),
            sync_name="SaidexToolNode.invoke",
            async_name="SaidexToolNode.ainvoke",
        )

    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        """Repair, validate and execute the last ``AIMessage``'s tool calls.

        Args:
            input: Graph state — a message list, a dict containing
                ``messages_key``, or a state object exposing it as an attribute.
            config: LangChain run configuration; forwarded to the inner
                ``ToolNode`` and the correction cycle so tracing nests properly.

        Returns:
            Tool results in the same shape the stock ``ToolNode`` produces
            (list input → message list, otherwise ``{messages_key: [...]}``),
            plus policy feedback messages and, when ``sanitize_messages``
            applies, the updated ``AIMessage``.
        """
        return await self._arun(input, config, **kwargs)

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    async def _arun(self, input: Any, config: RunnableConfig | None, **kwargs: Any) -> Any:
        messages, shape = _extract_messages(input, self._messages_key)
        ai_message, ai_index = _last_ai_message(messages)
        plans = self._plan_calls(ai_message)

        if self._on_invalid == "raise":
            failures = [
                (plan.call["name"], plan.call.get("id"), plan.error_text or "invalid tool call")
                for plan in plans
                if not plan.executable
            ]
            if failures:
                raise ToolCallValidationError(failures)

        if self._on_invalid == "correct":
            callbacks: Callbacks = (config or {}).get("callbacks")
            for plan in plans:
                # Unanswerable (id-less) plans are never corrected: there is
                # no tool_call_id to answer with even if the correction
                # succeeds, so a "corrected" id-less plan would still have to
                # be dropped — see the id-less handling in _plan_calls.
                if not plan.executable and plan.answerable:
                    await self._correct_plan(plan, callbacks)

        feedback: list[BaseMessage] = []
        for plan in plans:
            if plan.executable:
                if not plan.outcome:
                    plan.outcome = "executed"
            elif plan.answerable:
                plan.outcome = "feedback"
                feedback.append(self._feedback_message(plan))
            else:
                plan.outcome = "dropped"

        executable_calls = [plan.call for plan in plans if plan.executable]
        inner_output: Any = None
        if executable_calls:
            # A correction mutates plan.call["args"] in place without touching
            # plan.repaired (that flag means "deterministically recovered from
            # invalid_tool_calls" — see ToolCallStats.repaired), so it must be
            # counted here explicitly. Otherwise a corrected call originating
            # from tool_calls (same id count, repaired=False) would look
            # unchanged and the inner node would run the ORIGINAL bad args.
            changed = any(plan.repaired or plan.outcome == "corrected" for plan in plans) or len(
                executable_calls
            ) != len(ai_message.tool_calls)
            if changed:
                delegated = ai_message.model_copy(
                    update={"tool_calls": executable_calls, "invalid_tool_calls": []}
                )
                inner_input = _substitute_message(
                    input, messages, delegated, ai_index, shape, self._messages_key
                )
            else:
                inner_input = input
            inner_output = await self._inner.ainvoke(inner_input, config, **kwargs)

        extra: list[BaseMessage] = list(feedback)
        sanitized = self._sanitize_message(ai_message, plans)
        if sanitized is not None:
            extra.append(sanitized)

        order = {
            call_id: index
            for index, call_id in enumerate(
                [c.get("id") for c in ai_message.tool_calls]
                + [c.get("id") for c in ai_message.invalid_tool_calls]
            )
            if call_id is not None
        }

        stats = ToolNodeStats(
            calls=tuple(
                ToolCallStats(
                    tool_name=str(plan.call.get("name") or ""),
                    tool_call_id=plan.call.get("id"),
                    outcome=_finalized_outcome(plan.outcome),
                    repaired=plan.repaired,
                    prevalidated=plan.prevalidated,
                    correction_retries=plan.correction_retries,
                    field_issues=plan.issues,
                )
                for plan in plans
            )
        )
        # self.name is always a str (the constructor defaults it to "tools"),
        # but Runnable.name is typed str | None on the base class — the `or`
        # satisfies mypy without weakening the runtime guarantee.
        await dispatch_tool_node_event(ToolNodeEvent(node_name=self.name or "tools", stats=stats))

        return _merge_output(inner_output, extra, shape, self._messages_key, order)

    async def _correct_plan(self, plan: _CallPlan, callbacks: Callbacks) -> None:
        """Run the LLM correction cycle for one invalid, answerable call, mutating *plan*.

        On success the plan becomes executable with the corrected args and
        ``plan.outcome`` is set to ``"corrected"``; on failure the plan is left
        exactly as invalid as it was, so the feedback policy still applies —
        a failed correction must never crash the node. The correction
        conversation (prompt, retries, intermediate responses) is private to
        this call and never reaches graph state: only the corrected args (via
        ``plan.call``) survive, never the messages that produced them.

        Args:
            plan: The invalid, answerable plan to correct (must not already be
                executable — callers only invoke this for plans that failed
                pre-validation or could not be parsed at all).
            callbacks: Callbacks pulled from the node's ``RunnableConfig`` so
                correction LLM calls nest under the node's tracing span.
        """
        name = plan.call["name"]
        schema = plan.schema or self._pydantic_schema_for(name)
        if schema is None:
            # No known schema to correct against (e.g. a malformed call that
            # never carried a tool name) — leave the plan invalid; feedback
            # applies.
            return
        raw = (
            plan.raw_args
            if plan.raw_args is not None
            else json.dumps(plan.call["args"], ensure_ascii=False, default=str)
        )
        prompt = (
            f"An AI agent produced an invalid call to the tool '{name}'.\n\n"
            f"Invalid arguments:\n{raw}\n\n"
            + (f"Validation errors:\n{plan.error_text}\n\n" if plan.error_text else "")
            + "Return the corrected tool arguments. Preserve the original intent; "
            "change only what is needed to satisfy the schema."
        )
        try:
            instance, stats = await extract_data(
                self._correction_model,
                schema,
                [HumanMessage(content=prompt)],
                mode=self._correction_mode,
                max_primary_retries=self._max_correction_retries,
                retry_config=_NO_NETWORK_RETRY,
                callbacks=callbacks,
                _notify_observers=False,
            )
            plan.correction_retries = stats.primary_retries
            if instance is None:
                logger.warning(
                    "SaidexToolNode: correction cycle failed for tool call %r (id=%r)",
                    name,
                    plan.call.get("id"),
                )
                return
            # exclude_unset=True reports only what the correction model actually
            # supplied, not schema defaults it never touched — the tool then
            # sees the same shape of args a well-formed call would have
            # produced. Stays inside this try: a tool schema with a custom
            # serializer can raise here too, and that must degrade to the
            # feedback policy exactly like an extract_data failure — not
            # crash the node.
            corrected_args = instance.model_dump(exclude_unset=True)
        except Exception as exc:
            # extract_data degrades most failures to (None, stats) internally,
            # but a model that rejects tool binding outright (e.g. bind_tools
            # raising NotImplementedError for a chat model without tool-calling
            # support — exactly the case correction_mode=JSON exists for) is
            # only guarded for TypeError inside extractor._try_with_model and
            # otherwise propagates. A correction failure must never crash the
            # node, so catch broadly here and fall through to the feedback
            # policy — but Exception, not BaseException, so cancellation
            # (asyncio.CancelledError) and KeyboardInterrupt still propagate.
            logger.warning(
                "SaidexToolNode: correction cycle raised for tool call %r (id=%r): %s",
                name,
                plan.call.get("id"),
                exc,
            )
            return
        plan.call["args"] = corrected_args
        # Deliberately NOT plan.repaired = True: that flag means
        # "deterministically recovered from invalid_tool_calls" (see
        # ToolCallStats.repaired) — an LLM correction is a distinct outcome
        # category ("corrected", set below), not a repair.
        plan.recoverable = True
        plan.prevalidated = True
        plan.executable = True
        plan.outcome = "corrected"

    def _plan_calls(self, message: AIMessage) -> list[_CallPlan]:
        """Classify every call on *message* into an executable/invalid plan."""
        plans: list[_CallPlan] = []
        for call in message.tool_calls:
            plans.append(_CallPlan(call=dict(call)))
        for entry in message.invalid_tool_calls:
            raw = entry.get("args") or ""
            name = entry.get("name")
            # Empty string normalizes to None here too, matching the
            # truthiness of the repair gate right below and the falsy-based
            # drop check further down — an entry with id="" is treated as
            # having no id everywhere, not just wherever this variable is read.
            entry_id = entry.get("id") or None
            repaired_args = (
                _repair_raw_args(
                    raw,
                    strip_thinking=self._strip_thinking,
                    use_json_repair=self._json_repair,
                )
                if name and entry_id
                else None
            )
            if repaired_args is None:
                plans.append(
                    _CallPlan(
                        call={
                            "name": name or "",
                            "args": {},
                            "id": entry_id,
                            "type": "tool_call",
                        },
                        from_invalid=True,
                        raw_entry=dict(entry),
                        raw_args=raw,
                        recoverable=False,
                        # An id but no name CAN be answered (real tool_call_id
                        # to respond to); no id means there is nothing to
                        # respond to at all — see the general id check below,
                        # which also covers this without duplicating it here.
                        missing_name=bool(entry_id) and not name,
                        error_text=str(entry.get("error") or "malformed tool call"),
                    )
                )
                continue
            plans.append(
                _CallPlan(
                    call={
                        "name": name,
                        "args": repaired_args,
                        "id": entry_id,
                        "type": "tool_call",
                    },
                    from_invalid=True,
                    raw_entry=dict(entry),
                    raw_args=raw,
                    repaired=True,
                )
            )

        for plan in plans:
            if not plan.call.get("id"):
                # A ToolMessage requires a tool_call_id, so a call with no id
                # — or an empty string, treated the same — has no way to be
                # answered, regardless of whether it came from tool_calls or
                # invalid_tool_calls (both allow id: str | None). Do not
                # fabricate an empty-id ToolMessage; the sanitizer drops the
                # entry (see _sanitize_message) so history carries neither an
                # orphan result nor an unanswerable call.
                logger.warning(
                    "SaidexToolNode: dropping a tool call with no id (name=%r); "
                    "a ToolMessage cannot be produced without a tool_call_id, "
                    "so this call will not be executed, answered, or retained "
                    "in sanitized history",
                    plan.call.get("name"),
                )
                plan.answerable = False
                plan.recoverable = False
                continue
            if not plan.recoverable:
                continue
            schema = self._pydantic_schema_for(plan.call["name"])
            if schema is None:
                # Unknown tool or non-Pydantic schema: the executor's own
                # validation and error handling stay authoritative.
                plan.executable = True
                continue
            plan.schema = schema
            plan.prevalidated = True
            try:
                _, issues, error_text = create_instance_with_issues(schema, **plan.call["args"])
            except TypeError as exc:
                # Only a malformed-args TypeError (non-mapping args, or a dict
                # with non-string keys) is something a model can be asked to
                # correct. Anything else is a genuine bug and must propagate
                # rather than being laundered into model-facing feedback.
                issues, error_text = [], f"arguments not valid: {exc}"
            if error_text is None:
                plan.executable = True
            else:
                plan.issues = tuple(issues)
                plan.error_text = error_text
        return plans

    def _pydantic_schema_for(self, name: str) -> type[BaseModel] | None:
        """Return tool *name*'s Pydantic args schema, or ``None`` if unknown/non-Pydantic."""
        tool = self._tools_by_name.get(name)
        schema = getattr(tool, "tool_call_schema", None) if tool is not None else None
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema
        return None

    def _feedback_message(self, plan: _CallPlan) -> ToolMessage:
        """Build the corrective ``ToolMessage`` for an invalid call.

        Only called for *answerable* plans (``plan.answerable`` is True), which
        guarantees ``plan.call["id"]`` is a real id — see ``_CallPlan.answerable``.
        An id-less call cannot be answered at all and never reaches here.
        """
        call_id = plan.call["id"]
        if plan.missing_name:
            # The id is real but no tool name was ever provided, so there is
            # nothing to fabricate a ``name`` from — omit it rather than
            # inventing a placeholder that would misrepresent history.
            content = (
                "Your tool call carried no tool name and could not be "
                "identified, so it was NOT executed.\n"
                f"Provider error: {plan.error_text}\n"
                f"Raw arguments received:\n{(plan.raw_args or '')[:500]}\n"
                "Call the tool again, including a valid tool name."
            )
            return ToolMessage(content=content, tool_call_id=call_id, status="error")

        name = plan.call["name"]
        if not plan.recoverable:
            schema = self._pydantic_schema_for(name)
            schema_hint = ""
            if schema is not None:
                fields = ", ".join(schema.model_fields)
                schema_hint = f"\nExpected arguments for '{name}': {fields}."
            content = (
                f"The arguments of your call to tool '{name}' could not be parsed "
                f"as a single JSON object and the call was NOT executed. Arguments "
                f"must be a JSON object — a JSON list or a bare scalar value (e.g. "
                f"a string or number) is not acceptable, even if it is valid JSON.\n"
                f"Provider error: {plan.error_text}\n"
                f"Raw arguments received:\n{(plan.raw_args or '')[:500]}\n"
                f"{schema_hint}\n"
                f"Call the tool again with a single valid JSON object as arguments."
            )
        else:
            content = (
                f"The arguments for tool call '{name}' did not match the tool's "
                f"schema and the call was NOT executed.\n\n{plan.error_text}\n\n"
                f"Call the tool again with corrected arguments that fix every "
                f"issue above."
            )
        return ToolMessage(
            content=content,
            name=name,
            tool_call_id=call_id,
            status="error",
        )

    def _sanitize_message(self, message: AIMessage, plans: list[_CallPlan]) -> AIMessage | None:
        """Return an updated ``AIMessage`` only when its calls actually changed.

        Rebuilds the ``tool_calls``/``invalid_tool_calls`` the message would
        carry after repair/validation and compares them structurally against
        the original — an id-less entry, whether it originated in
        ``tool_calls`` or ``invalid_tool_calls``, is dropped entirely (nothing
        can answer it, see ``_CallPlan.answerable``), while every other
        from-invalid, non-executable entry keeps its raw form so it still
        matches the feedback ``ToolMessage`` that answered it. Returning
        ``None`` when nothing actually differs avoids emitting an inert copy
        (same content, new identity) — e.g. when the only thing that happened
        was repairing an entry into a dict that is still schema-invalid, so
        its raw, unrepaired form ends up right back in ``invalid_tool_calls``.
        """
        if not self._sanitize_messages or message.id is None:
            return None
        tool_calls = [plan.call for plan in plans if not plan.from_invalid and plan.answerable]
        tool_calls += [plan.call for plan in plans if plan.from_invalid and plan.executable]
        invalid_tool_calls = [
            plan.raw_entry
            for plan in plans
            if plan.from_invalid
            and not plan.executable
            and plan.answerable
            and plan.raw_entry is not None
        ]
        if tool_calls == list(message.tool_calls) and invalid_tool_calls == list(
            message.invalid_tool_calls
        ):
            return None
        return message.model_copy(
            update={"tool_calls": tool_calls, "invalid_tool_calls": invalid_tool_calls}
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _build_registry(
    tools: Sequence[BaseTool | Callable[..., Any]],
) -> dict[str, BaseTool]:
    """Map tool names to ``BaseTool`` instances for schema lookup.

    Plain callables are converted with :func:`langchain_core.tools.tool`, the
    same conversion the stock node applies.  A tool that cannot be converted is
    skipped: its calls are passed through unvalidated and the inner node's own
    handling applies, so registry gaps only reduce pre-validation coverage.
    """
    registry: dict[str, BaseTool] = {}
    for candidate in tools:
        try:
            base = candidate if isinstance(candidate, BaseTool) else _as_tool(candidate)
        except Exception as exc:  # degrade to pass-through
            logger.warning("could not build schema registry entry for %r: %s", candidate, exc)
            continue
        registry[base.name] = base
    return registry


def _finalized_outcome(outcome: ToolCallOutcome | Literal[""]) -> ToolCallOutcome:
    """Narrow a finalized ``_CallPlan.outcome`` to the public ``ToolCallOutcome``.

    ``""`` only ever exists as ``_CallPlan.outcome``'s pre-dispatch default —
    the dispatch loop in ``_arun`` sets every plan's outcome to one of the
    four real values before stats are built (see the loop right after the
    ``on_invalid == "correct"`` branch). This turns that invariant into an
    explicit, type-checked guarantee instead of silently letting an
    unfinalized ``""`` leak into the public ``ToolCallStats`` API.
    """
    if not outcome:
        raise AssertionError("SaidexToolNode: _CallPlan.outcome was never finalized")
    return outcome


def _extract_messages(input: Any, messages_key: str) -> tuple[list[BaseMessage], str]:
    """Return the message list and the input shape (``list``/``dict``/``object``)."""
    if isinstance(input, list):
        return input, "list"
    if isinstance(input, dict):
        messages = input.get(messages_key)
        if messages is None:
            raise ValueError(f"No message list found under state key {messages_key!r}")
        return list(messages), "dict"
    messages = getattr(input, messages_key, None)
    if messages is None:
        raise ValueError(f"No message list found on state attribute {messages_key!r}")
    return list(messages), "object"


def _last_ai_message(messages: Sequence[BaseMessage]) -> tuple[AIMessage, int]:
    """Return the last ``AIMessage`` in ``messages`` and its index.

    Scans backward, mirroring stock ``ToolNode._parse_input`` (which finds the
    last ``AIMessage`` anywhere in the list, not necessarily the final
    element) so state with trailing non-AI messages after the tool-calling
    ``AIMessage`` still executes exactly as it would with the stock node. The
    index is returned so callers can target that message specifically (e.g. a
    future sanitizer that must swap it in place) rather than assuming it is
    ``messages[-1]``.
    """
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, AIMessage):
            return message, index
    raise ValueError("SaidexToolNode expects an AIMessage in state")


def _repair_raw_args(
    raw: str, *, strip_thinking: bool, use_json_repair: bool
) -> dict[str, Any] | None:
    """Deterministically recover an args dict from a malformed raw string."""
    text = _strip_thinking_tags(raw) if strip_thinking else raw
    text = _strip_code_fences(text.strip())
    if not text:
        return None
    try:
        parsed: Any = json.loads(text)
    except ValueError:  # json.JSONDecodeError is a ValueError subclass
        if not use_json_repair:
            return None
        parsed = repair_json(text, return_objects=True)
    return parsed if isinstance(parsed, dict) else None


def _substitute_message(
    input: Any,
    messages: Sequence[BaseMessage],
    replacement: AIMessage,
    index: int,
    shape: str,
    messages_key: str,
) -> Any:
    """Rebuild *input* with the message at *index* swapped for *replacement*.

    Keeps every other state field intact so ``InjectedState`` tools observe the
    real state. *index* must target the same ``AIMessage`` that
    :func:`_last_ai_message` located — it is not necessarily the last message
    in *messages*.
    """
    new_messages: list[BaseMessage] = list(messages)
    new_messages[index] = replacement
    if shape == "list":
        return new_messages
    if shape == "dict":
        return {**input, messages_key: new_messages}
    if isinstance(input, BaseModel):
        return input.model_copy(update={messages_key: new_messages})
    if dataclasses.is_dataclass(input) and not isinstance(input, type):
        return dataclasses.replace(input, **{messages_key: new_messages})
    raise TypeError(
        f"Unsupported state type for SaidexToolNode: {type(input).__name__}. "
        "Use a message list, a dict, a Pydantic model or a dataclass."
    )


def _order_tool_messages(messages: list[BaseMessage], order: dict[str, int]) -> list[BaseMessage]:
    """Stable-sort ``ToolMessage``s into original call order; others keep position."""

    def key(indexed: tuple[int, BaseMessage]) -> tuple[int, int]:
        position, message = indexed
        if isinstance(message, ToolMessage) and message.tool_call_id in order:
            return (order[message.tool_call_id], position)
        return (len(order), position)

    return [message for _, message in sorted(enumerate(messages), key=key)]


def _extra_update(
    extra_messages: list[BaseMessage], shape: str, messages_key: str, order: dict[str, int]
) -> Any:
    """Order policy messages into original call order, then shape them to match
    the caller's input (bare list vs. ``{messages_key: [...]}``) — the same
    convention stock ``ToolNode._combine_tool_outputs`` uses for the
    non-``Command`` outputs it mixes into a ``Command``-bearing list.
    """
    ordered = _order_tool_messages(list(extra_messages), order)
    return ordered if shape == "list" else {messages_key: ordered}


def _merge_output(
    inner_output: Any,
    extra_messages: list[BaseMessage],
    shape: str,
    messages_key: str,
    order: dict[str, int],
) -> Any:
    """Combine the inner node's output with policy messages, mirroring shape.

    ``Command`` objects (and any other non-message elements the inner node
    returns) are passed through unchanged; policy messages are ordered and
    shaped via :func:`_extra_update` in that case, so the original-call-order
    guarantee and the list/dict input shape both survive a ``Command`` return.
    """
    if inner_output is None:
        merged: list[BaseMessage] = _order_tool_messages(list(extra_messages), order)
        return merged if shape == "list" else {messages_key: merged}
    if isinstance(inner_output, dict):
        combined = list(inner_output.get(messages_key, [])) + list(extra_messages)
        return {**inner_output, messages_key: _order_tool_messages(combined, order)}
    if isinstance(inner_output, list):
        if all(isinstance(element, BaseMessage) for element in inner_output):
            return _order_tool_messages([*inner_output, *extra_messages], order)
        if extra_messages:
            return [*inner_output, _extra_update(extra_messages, shape, messages_key, order)]
        return inner_output
    if extra_messages:
        return [inner_output, _extra_update(extra_messages, shape, messages_key, order)]
    return inner_output
