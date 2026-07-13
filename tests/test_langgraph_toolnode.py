"""Tests for saidex.langgraph.SaidexToolNode (no network, no API keys)."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402
from langgraph.runtime import Runtime  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from saidex.langgraph import SaidexToolNode  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

EXECUTED: list[str] = []

# The installed langgraph release requires an explicit Runtime whenever a
# ToolNode (stock or wrapped) is invoked outside a compiled graph — bare
# `ToolNode(...).ainvoke(state)` otherwise raises "Missing required config
# key". Supply a no-op default via the public `langgraph.runtime.Runtime`
# API so both sides of the parity comparison can actually execute.
_NO_GRAPH_RUNTIME: Runtime[None] = Runtime()


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    EXECUTED.append(f"add:{a}+{b}")
    return a + b


@tool
def greet(name: str) -> str:
    """Greet a person by name."""
    EXECUTED.append(f"greet:{name}")
    return f"Hello {name}"


@pytest.fixture(autouse=True)
def _clear_executed() -> None:
    EXECUTED.clear()


def _call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _ai(
    tool_calls: list[dict[str, Any]] | None = None,
    invalid: list[dict[str, Any]] | None = None,
    msg_id: str | None = "ai-1",
) -> AIMessage:
    return AIMessage(
        content="",
        id=msg_id,
        tool_calls=tool_calls or [],
        invalid_tool_calls=invalid or [],
    )


def _tool_messages(result: Any) -> list[ToolMessage]:
    messages = result["messages"] if isinstance(result, dict) else result
    return [m for m in messages if isinstance(m, ToolMessage)]


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_ctor_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="on_invalid"):
        SaidexToolNode([add], on_invalid="explode")  # type: ignore[arg-type]


def test_ctor_requires_correction_model_for_correct() -> None:
    with pytest.raises(ValueError, match="correction_model"):
        SaidexToolNode([add], on_invalid="correct")


# ---------------------------------------------------------------------------
# Happy-path parity with stock ToolNode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_matches_stock_toolnode() -> None:
    message = _ai([_call("add", {"a": 1, "b": 2}, "c1")])
    state = {"messages": [message]}

    ours = await SaidexToolNode([add]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)
    stock = await ToolNode([add]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)

    ours_tm = _tool_messages(ours)
    stock_tm = _tool_messages(stock)
    assert len(ours_tm) == len(stock_tm) == 1
    assert ours_tm[0].content == stock_tm[0].content
    assert ours_tm[0].tool_call_id == "c1"
    assert ours_tm[0].name == "add"
    assert EXECUTED == ["add:1+2", "add:1+2"]


@pytest.mark.asyncio
async def test_multiple_valid_calls_all_execute() -> None:
    state = {
        "messages": [
            _ai(
                [
                    _call("add", {"a": 1, "b": 2}, "c1"),
                    _call("greet", {"name": "Ada"}, "c2"),
                ]
            )
        ]
    }
    result = await SaidexToolNode([add, greet]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)
    tool_messages = _tool_messages(result)
    assert [m.tool_call_id for m in tool_messages] == ["c1", "c2"]


@pytest.mark.asyncio
async def test_list_input_returns_list_shaped_output() -> None:
    result = await SaidexToolNode([add]).ainvoke(
        [_ai([_call("add", {"a": 2, "b": 3}, "c1")])], runtime=_NO_GRAPH_RUNTIME
    )
    tool_messages = _tool_messages(result)
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "5"


@pytest.mark.asyncio
async def test_object_input_state() -> None:
    class GraphState(BaseModel):
        messages: list[Any]
        counter: int = 0

    state = GraphState(messages=[_ai([_call("add", {"a": 4, "b": 4}, "c1")])], counter=7)
    result = await SaidexToolNode([add]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)
    assert _tool_messages(result)[0].content == "8"


@pytest.mark.asyncio
async def test_custom_messages_key() -> None:
    node = SaidexToolNode([add], messages_key="chat")
    result = await node.ainvoke(
        {"chat": [_ai([_call("add", {"a": 1, "b": 1}, "c1")])]}, runtime=_NO_GRAPH_RUNTIME
    )
    assert "chat" in result
    assert _tool_messages(result["chat"])[0].content == "2"  # type: ignore[index]


def test_sync_invoke_works() -> None:
    result = SaidexToolNode([add]).invoke(
        {"messages": [_ai([_call("add", {"a": 1, "b": 5}, "c1")])]}, runtime=_NO_GRAPH_RUNTIME
    )
    assert _tool_messages(result)[0].content == "6"


@pytest.mark.asyncio
async def test_sync_invoke_inside_running_loop_raises() -> None:
    node = SaidexToolNode([add])
    with pytest.raises(RuntimeError, match="ainvoke"):
        node.invoke({"messages": [_ai([_call("add", {"a": 1, "b": 1}, "c1")])]})


@pytest.mark.asyncio
async def test_no_tool_calls_matches_stock_toolnode() -> None:
    # The installed langgraph release returns an empty tool-message list for a
    # trailing AIMessage with no tool calls rather than raising, so this
    # verifies our node mirrors that (rather than asserting a raise that the
    # stock node no longer performs).
    state = {"messages": [AIMessage(content="hi")]}
    ours = await SaidexToolNode([add]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)
    stock = await ToolNode([add]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)
    assert _tool_messages(ours) == _tool_messages(stock) == []


@pytest.mark.asyncio
async def test_last_message_not_ai_raises() -> None:
    with pytest.raises(ValueError, match="AIMessage"):
        await SaidexToolNode([add]).ainvoke({"messages": [HumanMessage(content="hi")]})


@pytest.mark.asyncio
async def test_unknown_tool_passes_through_to_stock_error() -> None:
    state = {"messages": [_ai([_call("nope", {"x": 1}, "c1")])]}
    result = await SaidexToolNode([add]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)
    tool_message = _tool_messages(result)[0]
    assert tool_message.tool_call_id == "c1"
    assert tool_message.status == "error"
    assert "not a valid tool" in str(tool_message.content)
