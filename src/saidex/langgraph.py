"""``SaidexToolNode`` — a validating, correcting drop-in for LangGraph's ``ToolNode``.

Requires the optional ``saidex[langgraph]`` extra.  The node repairs
``invalid_tool_calls`` (think-tag stripping, json-repair) and validates tool-call
arguments *before* execution — against each tool's Pydantic schema and optional
per-tool validators.  An invalid call is then either corrected by SAIDEX (with a
``correction_model``; the correction never enters graph state) or answered with
structured feedback so the agent corrects it.  Calls that stay invalid follow
``on_unresolved``: feedback, raising, or a custom hook.  Valid calls are executed
by an internal stock ``langgraph.prebuilt.ToolNode``, so parallel execution,
``handle_tool_errors``, ``Command`` returns and injected state keep working
unchanged.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel

try:
    from langgraph.errors import GraphBubbleUp
    from langgraph.prebuilt import ToolNode
except ImportError as exc:  # keep langgraph an optional dependency
    raise ImportError(
        "saidex.langgraph requires the 'langgraph' package. "
        "Install it with: pip install 'saidex[langgraph]'."
    ) from exc

from langchain_core.tools import BaseTool

from ._tool_calls import (
    CallContext,
    CallPlan,
    OnUnresolved,
    ToolCallPipeline,
    ToolCallValidationError,
    ToolValidator,
)
from .models import ExtractionMode, ToolCallValidatorContext, ToolNodeEvent, UnresolvedToolCall
from .observability import dispatch_tool_node_event
from .sync import _run_blocking

__all__ = [
    "OnUnresolved",
    "SaidexToolNode",
    "ToolCallValidationError",
    "ToolCallValidatorContext",
    "ToolValidator",
    "UnresolvedToolCall",
]

#: Sentinel meaning "let the stock ToolNode keep its own default".
_UNSET: Any = object()

OnInvalid = Literal["feedback", "correct", "raise"]


@dataclasses.dataclass
class _Prepared:
    """Everything the pre-execution phase decided for one invocation."""

    shape: str
    ai_message: AIMessage
    plans: list[CallPlan]
    feedback: list[BaseMessage]
    execute: bool
    inner_input: Any


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
    transport level either: the call is left unresolved and follows
    ``on_unresolved`` rather than crashing the node.

    **Send-API fan-out payloads bypass the pipeline entirely.** LangGraph's
    ``Send("tools", tool_call)`` fan-out delivers either a bare
    ``list[ToolCall]`` or a ``ToolCallWithContext`` dict — both carry one
    already-extracted ``ToolCall`` with no surrounding ``AIMessage`` to
    repair, validate, correct or sanitize. These two shapes are detected up
    front (see :func:`_is_send_api_payload`) and delegated straight to the
    internal stock ``ToolNode`` (``invoke`` or ``ainvoke``) with
    ``config``/``kwargs`` passed through unmodified; the result is returned
    unchanged. No ``ToolNodeEvent`` is emitted on this path — no ``CallPlan``
    ever exists to report, consistent with "we did not process any call" (the
    same convention the ``on_unresolved="raise"`` path follows).

    Args:
        tools: The tools available for execution — ``BaseTool`` instances or
            plain callables, exactly as accepted by the stock ``ToolNode``.
        correction_model: Chat model that corrects invalid calls inside the
            node. ``None`` (default) leaves correction to the agent: invalid
            calls are answered with structured feedback and the agent re-issues
            them in its next turn. When set, the correction conversation never
            enters graph state.
        max_correction_retries: LLM attempts per call in the correction cycle;
            must be at least 1 when ``correction_model`` is set.
        correction_mode: Extraction mode for the correction cycle — use
            :attr:`~saidex.ExtractionMode.JSON` for models without tool calling.
        correct_missing_fields: By default a call missing a required field is
            answered with feedback so the agent fills it in; set True to let the
            correction model fill it from the conversation instead. A call whose
            arguments contain keys the schema does not know is always corrected
            (likely a misnamed field).
        correction_context_messages: How many messages before the AIMessage the
            correction model sees, as text (system messages excluded, each cut to
            2,000 characters), plus the AIMessage's own text. 0 disables the context.
        on_unresolved: What happens to calls that stay invalid: ``"feedback"``
            (default) answers them with an error ``ToolMessage``; ``"raise"``
            raises :class:`ToolCallValidationError` before any call executes;
            or a hook (sync or async) receiving an :class:`UnresolvedToolCall`
            that returns salvaged args (dict, re-validated, then executed), a
            custom feedback text (str) or None (default feedback); an exception
            raised by the hook propagates.
        validators: Per-tool validators keyed by tool name. Each is called as
            ``fn(args, ctx)`` with the validated arguments model and a
            :class:`ToolCallValidatorContext`, sync or async, and returns a
            :data:`~saidex.ValidatorResult` like ``extract_data``'s ``validator=``.
            They run after Pydantic validation — on the model's own args, inside
            the correction cycle and on salvaged args — and never at execution
            time. ``interrupt()`` inside a validator pauses the graph as usual.
        strip_thinking: Strip ``<think>`` blocks from raw invalid-call args
            before parsing.
        json_repair: Apply ``json-repair`` to raw invalid-call args that do not
            parse as JSON.
        sanitize_messages: Return an updated copy of the ``AIMessage`` (same
            ``id``), so the standard ``add_messages`` reducer replaces the
            malformed message in state.  For every repaired, corrected or
            salvaged call, ``tool_calls``, the matching tool-call content blocks
            (``tool_use``, ``tool_call``, ``invalid_tool_call``,
            ``function_call``) and OpenAI's raw ``additional_kwargs["tool_calls"]``
            carry the final args.  Calls answered with feedback stay untouched,
            since their feedback message references the call id.  Set to
            ``False`` for message channels with a plain append reducer, where
            the update would duplicate the message.
        name: Node name, forwarded to the inner ``ToolNode``.  Must be
            non-empty (whitespace-only names are rejected too), since it is
            also used verbatim as ``ToolNodeEvent.node_name``.
        tags: Tags, forwarded to the inner ``ToolNode``.
        handle_tool_errors: Forwarded to the inner ``ToolNode`` when given;
            otherwise the stock default stays active.
        messages_key: State key holding the message list.
        on_invalid: Deprecated v1 policy. ``"feedback"`` disables correction,
            ``"correct"`` enables it, ``"raise"`` means ``on_unresolved="raise"``
            without correction. Emits a ``DeprecationWarning``.
    """

    def __init__(
        self,
        tools: Sequence[BaseTool | Callable[..., Any]],
        *,
        correction_model: Any = None,
        max_correction_retries: int = 2,
        correction_mode: ExtractionMode = ExtractionMode.TOOL_CALLING,
        correct_missing_fields: bool = False,
        correction_context_messages: int = 10,
        on_unresolved: OnUnresolved = "feedback",
        validators: Mapping[str, ToolValidator] | None = None,
        strip_thinking: bool = True,
        json_repair: bool = True,
        sanitize_messages: bool = True,
        name: str = "tools",
        tags: list[str] | None = None,
        handle_tool_errors: Any = _UNSET,
        messages_key: str = "messages",
        on_invalid: OnInvalid | None = None,
    ) -> None:
        if on_invalid is not None:
            correction_model, on_unresolved = _apply_deprecated_on_invalid(
                on_invalid, correction_model, on_unresolved
            )
        if not (on_unresolved in ("feedback", "raise") or callable(on_unresolved)):
            raise ValueError(
                f"on_unresolved must be 'feedback', 'raise' or a callable, got {on_unresolved!r}"
            )
        if correction_model is not None and max_correction_retries < 1:
            raise ValueError(
                "max_correction_retries must be at least 1 when a correction_model is set, "
                f"got {max_correction_retries!r}"
            )
        if correction_context_messages < 0:
            raise ValueError(
                "correction_context_messages must not be negative, "
                f"got {correction_context_messages!r}"
            )
        if not name.strip():
            raise ValueError(f"name must not be empty or whitespace-only, got {name!r}")

        inner_kwargs: dict[str, Any] = {"name": name, "tags": tags, "messages_key": messages_key}
        if handle_tool_errors is not _UNSET:
            inner_kwargs["handle_tool_errors"] = handle_tool_errors
        self._inner = ToolNode(tools, **inner_kwargs)

        self.name = name
        # Runnable.name is declared str | None on the base class, so mypy
        # cannot narrow self.name to str even though the constructor above
        # guarantees it — this private mirror carries the precise type for
        # ToolNodeEvent.node_name (which is str, not str | None) without an
        # `or "tools"` fallback that would silently rename an explicitly
        # empty name (see the constructor's ValueError above).
        self._node_name: str = name
        self._sanitize_messages = sanitize_messages
        self._messages_key = messages_key
        self._pipeline = ToolCallPipeline(
            tools,
            correction_model=correction_model,
            max_correction_retries=max_correction_retries,
            correction_mode=correction_mode,
            correct_missing_fields=correct_missing_fields,
            correction_context_messages=correction_context_messages,
            strip_thinking=strip_thinking,
            json_repair=json_repair,
            on_unresolved=on_unresolved,
            validators=validators,
            passthrough_exceptions=(GraphBubbleUp,),
        )
        for tool_name in validators or {}:
            if tool_name not in self._pipeline.tool_names:
                raise ValueError(
                    f"validators has no tool named {tool_name!r}; "
                    f"known tools: {sorted(self._pipeline.tool_names)}"
                )
            if self._pipeline.schema_for(tool_name) is None:
                raise ValueError(
                    f"validators[{tool_name!r}]: the tool has no Pydantic argument schema, "
                    "so there is no model instance to validate"
                )

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        """Synchronous entry point.

        Repair, validation, correction and ``on_unresolved`` run through a
        blocking bridge (also inside an already running event loop, e.g.
        ``graph.invoke`` from a Jupyter cell). Valid calls then execute on the
        stock ``ToolNode``'s sync path: a thread pool bounded by
        ``config["max_concurrency"]``, so ``max_concurrency=1`` runs them one
        after another. Async-only tools fail here exactly as with the stock node.

        Args:
            input: Graph state — a message list, a dict containing
                ``messages_key``, or a state object exposing it as an attribute.
            config: LangChain run configuration; forwarded to the inner
                ``ToolNode`` and the correction cycle so tracing nests properly.

        Returns:
            Tool results in the same shape the stock ``ToolNode`` produces, as
            described in :meth:`ainvoke`.
        """
        if _is_send_api_payload(input):
            return self._inner.invoke(input, config, **kwargs)
        prepared = _run_blocking(self._prepare(input, config))
        inner_output = (
            self._inner.invoke(prepared.inner_input, config, **kwargs) if prepared.execute else None
        )
        output, event = self._finish(prepared, inner_output)
        _run_blocking(dispatch_tool_node_event(event))
        return output

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
            plus feedback messages and, when ``sanitize_messages``
            applies, the updated ``AIMessage``.
        """
        if _is_send_api_payload(input):
            # No AIMessage exists in either shape — nothing to repair, validate
            # or sanitize — so this is a pure pass-through. See the class
            # docstring's "Send-API fan-out payloads" note for the stats
            # decision (no ToolNodeEvent on this path).
            return await self._inner.ainvoke(input, config, **kwargs)
        prepared = await self._prepare(input, config)
        inner_output = (
            await self._inner.ainvoke(prepared.inner_input, config, **kwargs)
            if prepared.execute
            else None
        )
        output, event = self._finish(prepared, inner_output)
        await dispatch_tool_node_event(event)
        return output

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    async def _prepare(self, input: Any, config: RunnableConfig | None) -> _Prepared:
        """Plan, correct and resolve the calls — everything before execution."""
        messages, shape = _extract_messages(input, self._messages_key)
        ai_message, ai_index = _last_ai_message(messages)
        ctx = CallContext(messages=tuple(messages), ai_index=ai_index, state=input)
        plans = await self._pipeline.plan(ai_message, ctx)
        await self._pipeline.correct(
            plans,
            ctx,
            callbacks=(config or {}).get("callbacks"),
            max_concurrency=(config or {}).get("max_concurrency"),
        )
        feedback: list[BaseMessage] = list(await self._pipeline.resolve(plans, ctx))

        executable_calls = [plan.call for plan in plans if plan.executable]
        inner_input: Any = None
        if executable_calls:
            # A correction or salvage mutates plan.call["args"] in place
            # without touching plan.repaired (that flag means
            # "deterministically recovered from invalid_tool_calls"), so both
            # outcomes must be counted here explicitly. Otherwise such a call
            # from tool_calls (same id count, repaired=False) would look
            # unchanged and the inner node would run the ORIGINAL bad args.
            changed = any(
                plan.repaired or plan.outcome in ("corrected", "salvaged") for plan in plans
            ) or len(executable_calls) != len(ai_message.tool_calls)
            if changed:
                delegated = ai_message.model_copy(
                    update={"tool_calls": executable_calls, "invalid_tool_calls": []}
                )
                inner_input = _substitute_message(
                    input, messages, delegated, ai_index, shape, self._messages_key
                )
            else:
                inner_input = input
        return _Prepared(
            shape=shape,
            ai_message=ai_message,
            plans=plans,
            feedback=feedback,
            execute=bool(executable_calls),
            inner_input=inner_input,
        )

    def _finish(self, prepared: _Prepared, inner_output: Any) -> tuple[Any, ToolNodeEvent]:
        """Merge executor output with feedback and the sanitized message; build the event."""
        extra: list[BaseMessage] = list(prepared.feedback)
        if self._sanitize_messages and prepared.ai_message.id is not None:
            sanitized = self._pipeline.sanitize(prepared.ai_message, prepared.plans)
            if sanitized is not None:
                extra.append(sanitized)
        order = {
            call_id: index
            for index, call_id in enumerate(
                [c.get("id") for c in prepared.ai_message.tool_calls]
                + [c.get("id") for c in prepared.ai_message.invalid_tool_calls]
            )
            if call_id is not None
        }
        event = ToolNodeEvent(
            node_name=self._node_name, stats=self._pipeline.build_stats(prepared.plans)
        )
        output = _merge_output(inner_output, extra, prepared.shape, self._messages_key, order)
        return output, event


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _apply_deprecated_on_invalid(
    on_invalid: Any, correction_model: Any, on_unresolved: OnUnresolved
) -> tuple[Any, OnUnresolved]:
    """Translate the deprecated v1 ``on_invalid`` policy into the v2 parameters."""
    warnings.warn(
        "SaidexToolNode(on_invalid=...) is deprecated and will be removed in a future "
        "release. Pass correction_model= to let SAIDEX correct invalid calls, and "
        "on_unresolved= for calls that stay invalid.",
        DeprecationWarning,
        stacklevel=3,
    )
    if on_invalid not in ("feedback", "correct", "raise"):
        raise ValueError(f"on_invalid must be 'feedback', 'correct' or 'raise', got {on_invalid!r}")
    if on_unresolved != "feedback":
        raise ValueError(
            "on_invalid cannot be combined with on_unresolved; use on_unresolved alone"
        )
    if on_invalid == "correct":
        if correction_model is None:
            raise ValueError("on_invalid='correct' requires a correction_model")
        return correction_model, "feedback"
    if on_invalid == "raise":
        return None, "raise"
    return None, "feedback"


def _is_send_api_payload(input: Any) -> bool:
    """True for LangGraph's Send-API fan-out shapes (``Send("tools", tool_call)``).

    Detected with the exact predicates stock ``ToolNode._parse_input`` uses
    (langgraph's ``prebuilt/tool_node.py``): a bare ``list[ToolCall]`` — its
    last element is a dict tagged ``type: "tool_call"`` — or a
    ``ToolCallWithContext`` dict tagged ``__type: "tool_call_with_context"``.
    Both carry a single already-extracted ``ToolCall`` with no surrounding
    ``AIMessage``, so there is nothing for SaidexToolNode's repair/validate/
    correct/sanitize pipeline to do — callers delegate straight to the stock
    executor instead. Mirrors stock's own guard against an empty list ending
    up here: an empty list is not this shape and falls through to the normal
    message-list handling in :func:`_extract_messages`.
    """
    if isinstance(input, list) and input and isinstance(input[-1], dict):
        return input[-1].get("type") == "tool_call"
    return isinstance(input, dict) and input.get("__type") == "tool_call_with_context"


def _is_rebuildable_object_state(input: Any) -> bool:
    """True when *input* is a state object :func:`_substitute_message` can
    rebuild in place (a Pydantic model or a dataclass instance) once a repair
    or correction needs to rewrite its ``AIMessage``.
    """
    return isinstance(input, BaseModel) or (
        dataclasses.is_dataclass(input) and not isinstance(input, type)
    )


def _extract_messages(input: Any, messages_key: str) -> tuple[list[BaseMessage], str]:
    """Return the message list and the input shape (``list``/``dict``/``object``).

    An "object" shape is rejected up front unless it is something
    :func:`_substitute_message` actually knows how to rebuild (a Pydantic
    model or a dataclass) — a plain class exposing ``messages_key`` as an
    attribute would otherwise be accepted here and only fail later, and only
    once some call's repair or correction needs to rewrite the ``AIMessage``
    in place (see :func:`_substitute_message`'s own ``TypeError``). Rejecting
    it here instead makes the failure immediate and consistent regardless of
    what a given batch of tool calls happens to need, rather than a state
    shape that "works" until the day a call needs repairing.
    """
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
    if not _is_rebuildable_object_state(input):
        raise TypeError(
            f"Unsupported state type for SaidexToolNode: {type(input).__name__}. "
            "Use a message list, a dict, a Pydantic model or a dataclass."
        )
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
    # Unreachable in practice: _extract_messages only ever returns shape
    # "object" for input that _is_rebuildable_object_state() already accepted
    # (a BaseModel or a dataclass), so both branches above should always
    # match. Kept as a defensive fallback in case that invariant is ever
    # broken by a future refactor.
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
    """Order feedback messages into original call order, then shape them to match
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
    """Combine the inner node's output with feedback messages, mirroring shape.

    ``Command`` objects (and any other non-message elements the inner node
    returns) are passed through unchanged; feedback messages are ordered and
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
