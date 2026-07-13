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

import logging
from collections.abc import Callable, Sequence
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig

try:
    from langgraph.prebuilt import ToolNode
except ImportError as exc:  # keep langgraph an optional dependency
    raise ImportError(
        "saidex.langgraph requires the 'langgraph' package. "
        "Install it with: pip install 'saidex[langgraph]'."
    ) from exc

from langchain_core.tools import BaseTool
from langchain_core.tools import tool as _as_tool

from .models import ExtractionMode
from .sync import _run_sync

__all__ = [
    "SaidexToolNode",
    "ToolCallValidationError",
]

logger = logging.getLogger(__name__)

#: Sentinel meaning "let the stock ToolNode keep its own default".
_UNSET: Any = object()

OnInvalid = Literal["feedback", "correct", "raise"]


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
        """Synchronously process the state.  See :meth:`ainvoke`.

        Bridges to the async implementation on a fresh event loop; inside an
        already running loop a ``RuntimeError`` points to :meth:`ainvoke`.
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
        messages, _shape = _extract_messages(input, self._messages_key)
        _last_ai_message(messages)
        return await self._inner.ainvoke(input, config, **kwargs)


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
        except Exception as exc:  # noqa: BLE001 — degrade to pass-through
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


def _last_ai_message(messages: Sequence[BaseMessage]) -> AIMessage:
    """Return the trailing ``AIMessage``, mirroring the stock node's contract."""
    if messages and isinstance(messages[-1], AIMessage):
        return messages[-1]
    raise ValueError("SaidexToolNode expects the last message in state to be an AIMessage")
