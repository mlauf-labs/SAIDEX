"""Framework-agnostic repair, validation and correction of tool calls.

The core behind :class:`saidex.langgraph.SaidexToolNode`: it plans every tool
call of one ``AIMessage`` (deterministic repair, Pydantic pre-validation),
runs the optional LLM correction cycle, builds feedback messages, sanitizes
the message and builds stats.  It never executes tools and never imports
``langgraph``, so other integrations can reuse it.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Callable, Sequence
from typing import Any, Literal

from json_repair import repair_json
from langchain_core.callbacks.base import Callbacks
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as _as_tool
from pydantic import BaseModel

from .extractor import _strip_code_fences, _strip_thinking_tags, extract_data
from .models import ExtractionMode, FieldIssue, ToolCallOutcome, ToolCallStats, ToolNodeStats
from .retry import RetryConfig
from .utils import create_instance_with_issues

logger = logging.getLogger(__name__)

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
class CallPlan:
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
    #: ``""`` is the pre-dispatch default; ``ToolCallPipeline.resolve``
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
    """Raised by ``on_unresolved="raise"`` before any tool call is executed.

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


class ToolCallPipeline:
    """Plan, correct and answer the tool calls of one ``AIMessage``.

    Holds the per-node configuration; every method works on the
    :class:`CallPlan` list of a single invocation.  It never executes tools.

    Args:
        tools: The tools the calls may target, as accepted by LangGraph's ``ToolNode``.
        correction_model: Chat model for the LLM correction cycle, or ``None``.
        max_correction_retries: LLM attempts per call in the correction cycle.
        correction_mode: Extraction mode of the correction cycle.
        strip_thinking: Strip ``<think>`` blocks from raw invalid-call args.
        json_repair: Apply json-repair to raw invalid-call args that do not parse.
        on_unresolved: What happens to calls that stay invalid: ``"feedback"``
            answers them with an error ToolMessage, ``"raise"`` raises
            ToolCallValidationError before any call executes.
    """

    def __init__(
        self,
        tools: Sequence[BaseTool | Callable[..., Any]],
        *,
        correction_model: Any,
        max_correction_retries: int,
        correction_mode: ExtractionMode,
        strip_thinking: bool,
        json_repair: bool,
        on_unresolved: Literal["feedback", "raise"] = "feedback",
    ) -> None:
        self._correction_model = correction_model
        self._max_correction_retries = max_correction_retries
        self._correction_mode = correction_mode
        self._strip_thinking = strip_thinking
        self._json_repair = json_repair
        self._on_unresolved = on_unresolved
        self._tools_by_name = _build_registry(tools)

    @property
    def correction_enabled(self) -> bool:
        """Whether invalid calls go through the LLM correction cycle."""
        return self._correction_model is not None

    def plan(self, message: AIMessage) -> list[CallPlan]:
        """Classify every call on *message* into an executable/invalid plan."""
        plans: list[CallPlan] = []
        for call in message.tool_calls:
            plans.append(CallPlan(call=dict(call)))
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
                    CallPlan(
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
                CallPlan(
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
                # entry (see sanitize) so history carries neither an
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
            schema = self.schema_for(plan.call["name"])
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

    async def correct_one(self, plan: CallPlan, callbacks: Callbacks) -> None:
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
        schema = plan.schema or self.schema_for(name)
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

    def resolve(self, plans: list[CallPlan]) -> list[ToolMessage]:
        """Apply ``on_unresolved`` and finalize every plan's ``outcome``.

        Args:
            plans: The planned (and possibly corrected) calls.

        Returns:
            One error ``ToolMessage`` per call answered with feedback, in call order.

        Raises:
            ToolCallValidationError: With ``on_unresolved="raise"``, before any
                call executes, when at least one call is not executable.
        """
        if self._on_unresolved == "raise":
            failures = [
                (plan.call["name"], plan.call.get("id"), plan.error_text or "invalid tool call")
                for plan in plans
                if not plan.executable
            ]
            if failures:
                raise ToolCallValidationError(failures)
        feedback: list[ToolMessage] = []
        for plan in plans:
            if plan.executable:
                if not plan.outcome:
                    plan.outcome = "executed"
            elif plan.answerable:
                plan.outcome = "feedback"
                feedback.append(self.feedback_message(plan))
            else:
                plan.outcome = "dropped"
        return feedback

    def schema_for(self, name: str) -> type[BaseModel] | None:
        """Return tool *name*'s Pydantic args schema, or ``None`` if unknown/non-Pydantic."""
        tool = self._tools_by_name.get(name)
        schema = getattr(tool, "tool_call_schema", None) if tool is not None else None
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema
        return None

    def feedback_message(self, plan: CallPlan) -> ToolMessage:
        """Build the corrective ``ToolMessage`` for an invalid call.

        Only called for *answerable* plans (``plan.answerable`` is True), which
        guarantees ``plan.call["id"]`` is a real id — see ``CallPlan.answerable``.
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
            schema = self.schema_for(name)
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

    def sanitize(self, message: AIMessage, plans: list[CallPlan]) -> AIMessage | None:
        """Return an updated ``AIMessage`` only when its calls actually changed.

        Rebuilds the ``tool_calls``/``invalid_tool_calls`` the message would
        carry after repair/validation and compares them structurally against
        the original — an id-less entry, whether it originated in
        ``tool_calls`` or ``invalid_tool_calls``, is dropped entirely (nothing
        can answer it, see ``CallPlan.answerable``), while every other
        from-invalid, non-executable entry keeps its raw form so it still
        matches the feedback ``ToolMessage`` that answered it. Returning
        ``None`` when nothing actually differs avoids emitting an inert copy
        (same content, new identity) — e.g. when the only thing that happened
        was repairing an entry into a dict that is still schema-invalid, so
        its raw, unrepaired form ends up right back in ``invalid_tool_calls``.
        """
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

    def build_stats(self, plans: list[CallPlan]) -> ToolNodeStats:
        """Build the invocation's stats from finalized *plans*.

        Args:
            plans: Plans whose ``outcome`` has been finalized.

        Returns:
            One :class:`ToolCallStats` per plan, in call order.
        """
        return ToolNodeStats(
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
    """Narrow a finalized ``CallPlan.outcome`` to the public ``ToolCallOutcome``.

    ``""`` only ever exists as ``CallPlan.outcome``'s pre-dispatch default —
    ``ToolCallPipeline.resolve`` sets every plan's outcome to one of the
    four real values before stats are built. This turns that invariant into an
    explicit, type-checked guarantee instead of silently letting an
    unfinalized ``""`` leak into the public ``ToolCallStats`` API.
    """
    if not outcome:
        raise AssertionError("SaidexToolNode: CallPlan.outcome was never finalized")
    return outcome


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
