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

from json_repair import repair_json
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as _as_tool

from .extractor import _strip_code_fences, _strip_thinking_tags
from .models import ExtractionMode, FieldIssue
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
    outcome: str = ""
    correction_retries: int = 0


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

        feedback: list[BaseMessage] = []
        for plan in plans:
            if plan.executable:
                plan.outcome = "executed"
            else:
                plan.outcome = "feedback"
                feedback.append(self._feedback_message(plan))

        executable_calls = [plan.call for plan in plans if plan.executable]
        inner_output: Any = None
        if executable_calls:
            changed = any(plan.repaired for plan in plans) or len(executable_calls) != len(
                ai_message.tool_calls
            )
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
        return _merge_output(inner_output, extra, shape, self._messages_key, order)

    def _plan_calls(self, message: AIMessage) -> list[_CallPlan]:
        """Classify every call on *message* into an executable/invalid plan."""
        plans: list[_CallPlan] = []
        for call in message.tool_calls:
            plans.append(_CallPlan(call=dict(call)))
        for entry in message.invalid_tool_calls:
            raw = entry.get("args") or ""
            name = entry.get("name")
            repaired_args = (
                _repair_raw_args(
                    raw,
                    strip_thinking=self._strip_thinking,
                    use_json_repair=self._json_repair,
                )
                if name
                else None
            )
            if repaired_args is None:
                plans.append(
                    _CallPlan(
                        call={
                            "name": name or "unknown",
                            "args": {},
                            "id": entry.get("id"),
                            "type": "tool_call",
                        },
                        from_invalid=True,
                        raw_entry=dict(entry),
                        raw_args=raw,
                        recoverable=False,
                        error_text=str(entry.get("error") or "malformed tool call"),
                    )
                )
                continue
            plans.append(
                _CallPlan(
                    call={
                        "name": name,
                        "args": repaired_args,
                        "id": entry.get("id"),
                        "type": "tool_call",
                    },
                    from_invalid=True,
                    raw_entry=dict(entry),
                    raw_args=raw,
                    repaired=True,
                )
            )

        for plan in plans:
            if not plan.recoverable:
                continue
            tool = self._tools_by_name.get(plan.call["name"])
            schema = getattr(tool, "tool_call_schema", None) if tool is not None else None
            if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
                # Unknown tool or non-Pydantic schema: the executor's own
                # validation and error handling stay authoritative.
                plan.executable = True
                continue
            plan.schema = schema
            plan.prevalidated = True
            try:
                _, issues, error_text = create_instance_with_issues(schema, **plan.call["args"])
            except Exception as exc:  # e.g. non-mapping args
                issues, error_text = [], f"arguments not valid: {exc}"
            if error_text is None:
                plan.executable = True
            else:
                plan.issues = tuple(issues)
                plan.error_text = error_text
        return plans

    def _feedback_message(self, plan: _CallPlan) -> ToolMessage:
        """Build the corrective ``ToolMessage`` for an invalid call."""
        name = plan.call["name"]
        if not plan.recoverable:
            tool = self._tools_by_name.get(name)
            schema_hint = ""
            if tool is not None and isinstance(getattr(tool, "tool_call_schema", None), type):
                fields = ", ".join(tool.tool_call_schema.model_fields)  # type: ignore[union-attr]
                schema_hint = f"\nExpected arguments for '{name}': {fields}."
            content = (
                f"The arguments of your call to tool '{name}' could not be parsed "
                f"as JSON and the call was NOT executed.\n"
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
            tool_call_id=plan.call.get("id") or "",
            status="error",
        )

    def _sanitize_message(self, message: AIMessage, plans: list[_CallPlan]) -> AIMessage | None:
        """Return an updated ``AIMessage`` when repair/correction changed calls."""
        if not self._sanitize_messages or message.id is None:
            return None
        if not any(plan.repaired or plan.outcome == "corrected" for plan in plans):
            return None
        tool_calls = [plan.call for plan in plans if not plan.from_invalid]
        tool_calls += [plan.call for plan in plans if plan.from_invalid and plan.executable]
        invalid_tool_calls = [
            plan.raw_entry
            for plan in plans
            if plan.from_invalid and not plan.executable and plan.raw_entry is not None
        ]
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
    except (ValueError, json.JSONDecodeError):
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


def _merge_output(
    inner_output: Any,
    extra_messages: list[BaseMessage],
    shape: str,
    messages_key: str,
    order: dict[str, int],
) -> Any:
    """Combine the inner node's output with policy messages, mirroring shape.

    ``Command`` objects (and any other non-message elements the inner node
    returns) are passed through unchanged; policy messages are grouped into a
    ``{messages_key: [...]}`` update in that case.
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
            return [*inner_output, {messages_key: extra_messages}]
        return inner_output
    if extra_messages:
        return [inner_output, {messages_key: extra_messages}]
    return inner_output
