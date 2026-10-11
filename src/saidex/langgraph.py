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
import logging
import warnings
from collections.abc import Callable, Sequence
from typing import Any, Literal

from langchain_core.callbacks.base import Callbacks
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel

try:
    from langgraph.prebuilt import ToolNode
except ImportError as exc:  # keep langgraph an optional dependency
    raise ImportError(
        "saidex.langgraph requires the 'langgraph' package. "
        "Install it with: pip install 'saidex[langgraph]'."
    ) from exc

from langchain_core.tools import BaseTool

from ._tool_calls import CallContext, OnUnresolved, ToolCallPipeline, ToolCallValidationError
from .models import ExtractionMode, ToolNodeEvent, UnresolvedToolCall
from .observability import dispatch_tool_node_event
from .sync import _run_blocking

__all__ = [
    "OnUnresolved",
    "SaidexToolNode",
    "ToolCallValidationError",
    "UnresolvedToolCall",
]

logger = logging.getLogger(__name__)

#: Sentinel meaning "let the stock ToolNode keep its own default".
_UNSET: Any = object()

OnInvalid = Literal["feedback", "correct", "raise"]


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

    **Send-API fan-out payloads bypass the pipeline entirely.** LangGraph's
    ``Send("tools", tool_call)`` fan-out delivers either a bare
    ``list[ToolCall]`` or a ``ToolCallWithContext`` dict — both carry one
    already-extracted ``ToolCall`` with no surrounding ``AIMessage`` to
    repair, validate, correct or sanitize. These two shapes are detected up
    front (see :func:`_is_send_api_payload`) and delegated straight to the
    internal stock ``ToolNode`` with ``config``/``kwargs`` passed through
    unmodified; the result is returned unchanged. No ``ToolNodeEvent`` is
    emitted on this path — no ``CallPlan`` ever exists to report, consistent
    with "we did not process any call" (the same convention the ``"raise"``
    policy's fail-fast path follows).

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
        on_unresolved: What happens to calls that stay invalid: ``"feedback"``
            (default) answers them with an error ``ToolMessage``; ``"raise"``
            raises :class:`ToolCallValidationError` before any call executes;
            or a hook (sync or async) receiving an :class:`UnresolvedToolCall`
            that returns salvaged args (dict, re-validated, then executed), a
            custom feedback text (str) or None (default feedback); an exception
            raised by the hook propagates.
        strip_thinking: Strip ``<think>`` blocks from raw invalid-call args
            before parsing.
        json_repair: Apply ``json-repair`` to raw invalid-call args that do not
            parse as JSON.
        sanitize_messages: Return an updated copy of the ``AIMessage`` (same
            ``id``) whose tool calls carry the repaired/corrected args, so the
            standard ``add_messages`` reducer replaces the malformed message in
            state.  Set to ``False`` for message channels with a plain append
            reducer.  Only ``tool_calls``/``invalid_tool_calls`` are rewritten,
            never ``AIMessage.content`` — fully effective for providers that
            serialize from ``tool_calls`` (e.g. ``langchain-openai``), but
            content-block providers (Anthropic-style) may still re-serialize
            the original, unsanitized ``tool_use`` block from ``content``.
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
        on_unresolved: OnUnresolved = "feedback",
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
            strip_thinking=strip_thinking,
            json_repair=json_repair,
            on_unresolved=on_unresolved,
        )

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        """Synchronous entry point; runs the pipeline through a blocking bridge.

        See :meth:`ainvoke` for the input and output shapes.  Works inside an
        already running event loop too (e.g. ``graph.invoke`` from a Jupyter
        cell) — the pipeline then runs on a short-lived worker thread.

        Args:
            input: Graph state — a message list, a dict containing
                ``messages_key``, or a state object exposing it as an attribute.
            config: LangChain run configuration; forwarded to the inner
                ``ToolNode`` and the correction cycle so tracing nests properly.

        Returns:
            Tool results in the same shape the stock ``ToolNode`` produces, as
            described in :meth:`ainvoke`.
        """
        return _run_blocking(self._arun(input, config, **kwargs))

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
        if _is_send_api_payload(input):
            # No AIMessage exists in either shape — nothing to repair, validate
            # or sanitize — so this is a pure pass-through. See the class
            # docstring's "Send-API fan-out payloads" note for the stats
            # decision (no ToolNodeEvent on this path).
            return await self._inner.ainvoke(input, config, **kwargs)
        messages, shape = _extract_messages(input, self._messages_key)
        ai_message, ai_index = _last_ai_message(messages)
        ctx = CallContext(messages=tuple(messages), ai_index=ai_index, state=input)
        plans = self._pipeline.plan(ai_message)

        if self._pipeline.correction_enabled:
            callbacks: Callbacks = (config or {}).get("callbacks")
            for plan in plans:
                # Unanswerable (id-less) plans are never corrected: there is
                # no tool_call_id to answer with even if the correction
                # succeeds, so a "corrected" id-less plan would still have to
                # be dropped — see the id-less handling in ToolCallPipeline.plan.
                if not plan.executable and plan.answerable:
                    await self._pipeline.correct_one(plan, callbacks)

        feedback: list[BaseMessage] = list(await self._pipeline.resolve(plans, ctx))

        executable_calls = [plan.call for plan in plans if plan.executable]
        inner_output: Any = None
        if executable_calls:
            # A correction mutates plan.call["args"] in place without touching
            # plan.repaired (that flag means "deterministically recovered from
            # invalid_tool_calls" — see ToolCallStats.repaired), so it must be
            # counted here explicitly. Otherwise a corrected call originating
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
            inner_output = await self._inner.ainvoke(inner_input, config, **kwargs)

        extra: list[BaseMessage] = list(feedback)
        sanitized = (
            self._pipeline.sanitize(ai_message, plans)
            if self._sanitize_messages and ai_message.id is not None
            else None
        )
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

        stats = self._pipeline.build_stats(plans)
        await dispatch_tool_node_event(ToolNodeEvent(node_name=self._node_name, stats=stats))

        return _merge_output(inner_output, extra, shape, self._messages_key, order)


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
