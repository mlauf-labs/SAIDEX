"""Framework-agnostic repair, validation and correction of tool calls.

The core behind :class:`saidex.langgraph.SaidexToolNode`: it plans every tool
call of one ``AIMessage`` (deterministic repair, Pydantic pre-validation),
runs the optional LLM correction cycle, builds feedback messages, sanitizes
the message and builds stats.  It never executes tools and never imports
``langgraph``, so other integrations can reuse it.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal, TypeAlias

from json_repair import repair_json
from langchain_core.callbacks.base import Callbacks
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as _as_tool
from pydantic import AliasChoices, BaseModel

from .extractor import (
    ValidatorResult,
    WithContext,
    _InvalidValidatorResultError,
    _run_external_validator,
    _strip_code_fences,
    _strip_thinking_tags,
    extract_data,
)
from .models import (
    ExtractionMode,
    FieldIssue,
    ToolCallOutcome,
    ToolCallStats,
    ToolCallValidatorContext,
    ToolNodeStats,
    UnresolvedToolCall,
    ValidatorContext,
)
from .retry import RetryConfig
from .utils import create_instance_with_issues

logger = logging.getLogger(__name__)

#: What an ``on_unresolved`` hook returns: salvaged arguments (``dict``), a
#: custom feedback text (``str``), or ``None`` for the default feedback.
UnresolvedDecision: TypeAlias = dict[str, Any] | str | None

#: ``on_unresolved`` policy: ``"feedback"``, ``"raise"``, or a sync or async
#: hook receiving an :class:`~saidex.models.UnresolvedToolCall`.
OnUnresolved: TypeAlias = (
    Literal["feedback", "raise"]
    | Callable[[UnresolvedToolCall], UnresolvedDecision | Awaitable[UnresolvedDecision]]
)

#: Signature of a ``SaidexToolNode`` tool validator: receives the validated
#: arguments model and a :class:`~saidex.models.ToolCallValidatorContext`, and
#: returns a :data:`~saidex.ValidatorResult` (or raises to reject). Sync or async.
ToolValidator: TypeAlias = Callable[
    [Any, ToolCallValidatorContext], ValidatorResult | Awaitable[ValidatorResult]
]


@dataclasses.dataclass(frozen=True)
class CallContext:
    """Where the planned ``AIMessage`` sits in the graph state.

    Attributes:
        messages: The full message list from the state.
        ai_index: Index of the ``AIMessage`` whose calls are processed.
        state: The raw state as the node received it.
    """

    messages: tuple[BaseMessage, ...]
    ai_index: int
    state: Any

    @property
    def conversation(self) -> tuple[BaseMessage, ...]:
        """Messages up to and including the planned ``AIMessage``."""
        return self.messages[: self.ai_index + 1]


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
    #: True when the arguments passed Pydantic validation but a tool validator
    #: rejected them — only changes the wording of the feedback message.
    rejected_by_validator: bool = False
    #: ``""`` is the pre-dispatch default; ``ToolCallPipeline.resolve``
    #: finalizes every plan to one of the real ``ToolCallOutcome``
    #: values before stats are built — see ``_finalized_outcome``, which
    #: turns that invariant into a type-checked narrowing at the point
    #: ``ToolCallStats`` is constructed.
    outcome: ToolCallOutcome | Literal[""] = ""
    correction_retries: int = 0
    correction_attempted: bool = False
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
        calls: One :class:`~saidex.models.UnresolvedToolCall` per answerable
            invalid call (calls without an id are counted in ``failures`` only).
    """

    def __init__(
        self,
        failures: Sequence[tuple[str, str | None, str]],
        *,
        calls: Sequence[UnresolvedToolCall] = (),
    ) -> None:
        self.failures = tuple(failures)
        self.calls = tuple(calls)
        lines = [
            f"- {name} (id={call_id}): {error.splitlines()[0]}"
            for name, call_id, error in self.failures
        ]
        super().__init__(
            f"Tool-call validation failed for {len(self.failures)} call(s):\n" + "\n".join(lines)
        )


def _input_keys(schema: type[BaseModel]) -> set[str]:
    """Every key under which *schema* accepts input: field names plus string aliases."""
    keys: set[str] = set()
    for name, info in schema.model_fields.items():
        keys.add(name)
        if info.alias:
            keys.add(info.alias)
        alias = info.validation_alias
        if isinstance(alias, str):
            keys.add(alias)
        elif isinstance(alias, AliasChoices):
            keys.update(choice for choice in alias.choices if isinstance(choice, str))
    return keys


class ToolCallPipeline:
    """Plan, correct and answer the tool calls of one ``AIMessage``.

    Holds the per-node configuration; every method works on the
    :class:`CallPlan` list of a single invocation.  It never executes tools.

    Args:
        tools: The tools the calls may target, as accepted by LangGraph's ``ToolNode``.
        correction_model: Chat model for the LLM correction cycle, or ``None``.
        max_correction_retries: LLM attempts per call in the correction cycle.
        correction_mode: Extraction mode of the correction cycle.
        correct_missing_fields: Let the correction model fill missing required
            fields from the conversation instead of sending such calls to the agent.
        strip_thinking: Strip ``<think>`` blocks from raw invalid-call args.
        json_repair: Apply json-repair to raw invalid-call args that do not parse.
        on_unresolved: What happens to calls that stay invalid: ``"feedback"``
            answers them with an error ToolMessage, ``"raise"`` raises
            ToolCallValidationError before any call executes, and a sync or
            async hook receives each one as an
            :class:`~saidex.models.UnresolvedToolCall` and returns salvaged
            args, a custom feedback text, or ``None`` for the default feedback.
        validators: Per-tool validators keyed by tool name; they run after
            Pydantic validation in every phase.
    """

    def __init__(
        self,
        tools: Sequence[BaseTool | Callable[..., Any]],
        *,
        correction_model: Any,
        max_correction_retries: int,
        correction_mode: ExtractionMode,
        correct_missing_fields: bool = False,
        strip_thinking: bool,
        json_repair: bool,
        on_unresolved: OnUnresolved = "feedback",
        validators: Mapping[str, ToolValidator] | None = None,
    ) -> None:
        self._correction_model = correction_model
        self._max_correction_retries = max_correction_retries
        self._correction_mode = correction_mode
        self._correct_missing_fields = correct_missing_fields
        self._strip_thinking = strip_thinking
        self._json_repair = json_repair
        self._on_unresolved = on_unresolved
        self._validators: dict[str, ToolValidator] = dict(validators or {})
        self._tools_by_name = _build_registry(tools)

    @property
    def correction_enabled(self) -> bool:
        """Whether invalid calls go through the LLM correction cycle."""
        return self._correction_model is not None

    @property
    def tool_names(self) -> frozenset[str]:
        """Names of every tool in the schema registry."""
        return frozenset(self._tools_by_name)

    def is_correctable(self, plan: CallPlan) -> bool:
        """Whether the LLM correction cycle should try to fix *plan*.

        A missing required field is left to the agent, which has the context
        the correction model lacks — unless the arguments carry keys the schema
        does not know (the field is most likely just misnamed) or
        ``correct_missing_fields`` is set.

        Args:
            plan: A planned call.

        Returns:
            ``True`` when the call is invalid, answerable and worth correcting.
        """
        if (
            not self.correction_enabled
            or plan.executable
            or not plan.answerable
            or plan.missing_name
        ):
            return False
        schema = plan.schema or self.schema_for(plan.call["name"])
        if schema is None:
            return False
        if not plan.recoverable or plan.rejected_by_validator or self._correct_missing_fields:
            return True
        if not any(issue.category == "missing" for issue in plan.issues):
            return True
        return bool(set(plan.call["args"]) - _input_keys(schema))

    async def plan(self, message: AIMessage, ctx: CallContext) -> list[CallPlan]:
        """Classify every call on *message* into an executable/invalid plan.

        Args:
            message: The ``AIMessage`` whose calls are planned.
            ctx: Where the AIMessage sits; handed to tool validators.

        Returns:
            One :class:`CallPlan` per call, in call order.
        """
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
                instance, issues, error_text = create_instance_with_issues(
                    schema, **plan.call["args"]
                )
            except TypeError as exc:
                # Only a malformed-args TypeError (non-mapping args, or a dict
                # with non-string keys) is something a model can be asked to
                # correct. Anything else is a genuine bug and must propagate
                # rather than being laundered into model-facing feedback.
                instance, issues, error_text = None, [], f"arguments not valid: {exc}"
            if instance is None:
                plan.issues = tuple(issues)
                plan.error_text = error_text
                continue
            validator_issues, feedback = await self._run_tool_validator(
                plan, instance, ctx, phase="initial"
            )
            plan.issues = tuple(validator_issues)
            if feedback is None:
                plan.executable = True
            else:
                plan.error_text = feedback
                plan.rejected_by_validator = True
        return plans

    async def _run_tool_validator(
        self,
        plan: CallPlan,
        instance: BaseModel,
        ctx: CallContext,
        *,
        phase: Literal["initial", "salvage"],
    ) -> tuple[list[FieldIssue], str | None]:
        """Run *plan*'s tool validator outside the correction cycle.

        Args:
            plan: The plan whose arguments *instance* was built from.
            instance: The arguments model that already passed Pydantic validation.
            ctx: Where the AIMessage sits; handed to the validator.
            phase: ``"initial"`` for the model's own arguments, ``"salvage"``
                for arguments returned by an ``on_unresolved`` hook.

        Returns:
            Every finding as a FieldIssue, and the feedback text built from the
            ``"retry"`` findings (``None`` when the arguments are accepted).
        """
        fn = self._validators.get(plan.call["name"])
        if fn is None:
            return [], None
        is_final = phase == "salvage" or not self.correction_enabled
        tool_ctx = self._validator_context(plan, ctx, phase=phase, is_final=is_final)

        def check(args: BaseModel, _attempt: ValidatorContext) -> Any:
            return fn(args, tool_ctx)

        issues, _, feedback = await _run_external_validator(
            WithContext(check),
            instance,
            type(instance).__name__,
            ValidatorContext(
                attempt=1,
                max_attempts=1,
                is_final=is_final,
                model_label="primary",
                is_fallback=False,
            ),
        )
        return issues, feedback

    def _correction_validator(
        self, plan: CallPlan, ctx: CallContext
    ) -> WithContext[BaseModel] | None:
        """Adapt *plan*'s tool validator for ``extract_data`` in the correction cycle."""
        fn = self._validators.get(plan.call["name"])
        if fn is None:
            return None

        def check(args: BaseModel, attempt: ValidatorContext) -> Any:
            return fn(
                args,
                self._validator_context(
                    plan,
                    ctx,
                    phase="correction",
                    attempt=attempt.attempt,
                    max_attempts=attempt.max_attempts,
                    is_final=attempt.is_final,
                ),
            )

        return WithContext(check)

    def _validator_context(
        self,
        plan: CallPlan,
        ctx: CallContext,
        *,
        phase: Literal["initial", "correction", "salvage"],
        attempt: int = 1,
        max_attempts: int = 1,
        is_final: bool,
    ) -> ToolCallValidatorContext:
        """Build the context a tool validator receives."""
        return ToolCallValidatorContext(
            tool_name=plan.call["name"],
            tool_call_id=str(plan.call["id"]),
            messages=ctx.conversation,
            state=ctx.state,
            phase=phase,
            attempt=attempt,
            max_attempts=max_attempts,
            is_final=is_final,
        )

    async def correct_one(self, plan: CallPlan, ctx: CallContext, callbacks: Callbacks) -> None:
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
            ctx: Where the AIMessage sits; handed to the tool validator.
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
        plan.correction_attempted = True
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
                validator=self._correction_validator(plan, ctx),
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
        except _InvalidValidatorResultError:
            # A validator returning the wrong type is a programming error in the
            # caller's code, not a failed correction — same as in extract_data.
            raise
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

    async def resolve(self, plans: list[CallPlan], ctx: CallContext) -> list[ToolMessage]:
        """Apply ``on_unresolved`` and finalize every plan's ``outcome``.

        Args:
            plans: The planned (and possibly corrected) calls.
            ctx: Where the ``AIMessage`` sits; handed to a hook.

        Returns:
            One error ``ToolMessage`` per call answered with feedback, in call order.

        Raises:
            ToolCallValidationError: With ``on_unresolved="raise"``, before any
                call executes, when at least one call is not executable.
            TypeError: A hook returned something other than ``dict``, ``str`` or ``None``.
        """
        if self._on_unresolved == "raise":
            failing = [plan for plan in plans if not plan.executable]
            if failing:
                raise ToolCallValidationError(
                    [
                        (
                            plan.call["name"],
                            plan.call.get("id"),
                            plan.error_text or "invalid tool call",
                        )
                        for plan in failing
                    ],
                    calls=[self._unresolved(plan, ctx) for plan in failing if plan.answerable],
                )
        feedback: list[ToolMessage] = []
        for plan in plans:
            if plan.executable:
                if not plan.outcome:
                    plan.outcome = "executed"
            elif not plan.answerable:
                plan.outcome = "dropped"
            else:
                message = await self._apply_policy(plan, ctx)
                if message is not None:
                    plan.outcome = "feedback"
                    feedback.append(message)
        return feedback

    async def _apply_policy(self, plan: CallPlan, ctx: CallContext) -> ToolMessage | None:
        """Answer one unresolved call; ``None`` means the hook salvaged it."""
        policy = self._on_unresolved
        if isinstance(policy, str):
            return self.feedback_message(plan)
        decision = policy(self._unresolved(plan, ctx))
        if inspect.isawaitable(decision):
            decision = await decision
        if decision is None:
            return self.feedback_message(plan)
        if isinstance(decision, str):
            return self._custom_feedback(plan, decision)
        if isinstance(decision, dict):
            if await self._salvage(plan, decision, ctx):
                return None
            return self.feedback_message(plan)
        raise TypeError(f"on_unresolved must return a dict, a str or None; got {decision!r}")

    async def _salvage(self, plan: CallPlan, args: dict[str, Any], ctx: CallContext) -> bool:
        """Re-validate hook-supplied *args*; on success make *plan* executable with them."""
        name = plan.call["name"]
        schema = plan.schema or (self.schema_for(name) if name else None)
        if schema is None:
            logger.warning(
                "on_unresolved returned arguments for tool call %r (id=%r), but the tool has "
                "no known Pydantic schema to validate them; answering with feedback",
                name,
                plan.call["id"],
            )
            return False
        try:
            instance, _, error_text = create_instance_with_issues(schema, **args)
        except TypeError as exc:
            instance, error_text = None, f"arguments not valid: {exc}"
        if instance is not None:
            _, error_text = await self._run_tool_validator(plan, instance, ctx, phase="salvage")
        if error_text is not None:
            logger.warning(
                "on_unresolved returned invalid arguments for tool call %r (id=%r): %s",
                name,
                plan.call["id"],
                error_text,
            )
            return False
        plan.call["args"] = dict(args)
        plan.recoverable = True
        plan.prevalidated = True
        plan.executable = True
        plan.outcome = "salvaged"
        return True

    def _custom_feedback(self, plan: CallPlan, content: str) -> ToolMessage:
        """Error ``ToolMessage`` carrying a hook's own feedback text."""
        if plan.missing_name:
            return ToolMessage(content=content, tool_call_id=plan.call["id"], status="error")
        return ToolMessage(
            content=content, name=plan.call["name"], tool_call_id=plan.call["id"], status="error"
        )

    def _unresolved(self, plan: CallPlan, ctx: CallContext) -> UnresolvedToolCall:
        """Public view of an answerable, still invalid *plan*."""
        return UnresolvedToolCall(
            tool_name=str(plan.call.get("name") or ""),
            tool_call_id=str(plan.call["id"]),
            args=dict(plan.call["args"]) if plan.recoverable else None,
            raw_args=plan.raw_args,
            issues=plan.issues,
            error_text=plan.error_text,
            correction_attempted=plan.correction_attempted,
            correction_retries=plan.correction_retries,
            messages=ctx.conversation,
            state=ctx.state,
        )

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
            reason = (
                "were rejected by validation"
                if plan.rejected_by_validator
                else "did not match the tool's schema"
            )
            content = (
                f"The arguments for tool call '{name}' {reason} and the call was NOT "
                f"executed.\n\n{plan.error_text}\n\n"
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
                    correction_attempted=plan.correction_attempted,
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
    real values before stats are built. This turns that invariant into an
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
