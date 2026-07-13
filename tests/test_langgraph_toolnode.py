"""Tests for saidex.langgraph.SaidexToolNode (no network, no API keys)."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from langgraph.graph.state import CompiledStateGraph  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402
from langgraph.runtime import Runtime  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from saidex.langgraph import SaidexToolNode  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

EXECUTED: list[str] = []

# WHY this sentinel exists: a bare `ToolNode(...).ainvoke(state)` (stock or
# wrapped) called *outside* a compiled graph has no Pregel executor to inject
# a Runtime through `config`, and the installed langgraph release requires
# one unconditionally — it raises "Missing required config key" without it.
# Supply a no-op default via the public `langgraph.runtime.Runtime` API so
# standalone tests below can actually execute.
#
# The graph-embedded tests further down deliberately do NOT use this
# sentinel: Pregel injects a real Runtime through `config` when a node runs
# inside `graph.invoke()`/`graph.ainvoke()`, and `RunnableCallable.invoke`/
# `ainvoke` (langgraph/_internal/_runnable.py) only fall back to that
# config-based lookup when `runtime` is absent from `kwargs` — an explicit
# `runtime=` kwarg would short-circuit that path and silently paper over a
# regression where `_arun` stops forwarding `config` to the inner node.
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


def _agent_node(_state: MessagesState) -> dict[str, list[BaseMessage]]:
    """Canned agent step: always answers with one tool call to ``add``."""
    return {
        "messages": [_ai([_call("add", {"a": 3, "b": 4}, "graph-c1")], msg_id="graph-ai-1")],
    }


def _build_tool_call_graph(tools_node: Any) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Minimal ``StateGraph(MessagesState)``: agent emits a tool call, tools node runs it.

    Used to prove ``SaidexToolNode`` behaves as a true drop-in when Pregel
    (not the test) injects the runtime through ``config`` — the graph-embedded
    tests below invoke the compiled graph directly and never pass ``runtime=``.
    """
    graph: StateGraph[Any, Any, Any, Any] = StateGraph(MessagesState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", "tools")
    graph.add_edge("tools", END)
    return graph.compile()


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
async def test_no_ai_message_in_state_raises() -> None:
    with pytest.raises(ValueError, match="AIMessage"):
        await SaidexToolNode([add]).ainvoke({"messages": [HumanMessage(content="hi")]})


@pytest.mark.asyncio
async def test_ai_message_followed_by_other_messages_still_executes() -> None:
    # Stock ToolNode._parse_input scans backward for the last AIMessage anywhere
    # in the list, not just messages[-1]. A trailing non-AI message after the
    # tool-calling AIMessage must therefore still execute the tool calls.
    state = {
        "messages": [
            _ai([_call("add", {"a": 5, "b": 6}, "c1")]),
            HumanMessage(content="please continue"),
        ]
    }
    ours = await SaidexToolNode([add]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)
    stock = await ToolNode([add]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)
    ours_tm = _tool_messages(ours)
    stock_tm = _tool_messages(stock)
    assert len(ours_tm) == len(stock_tm) == 1
    assert ours_tm[0].content == stock_tm[0].content == "11"
    assert ours_tm[0].tool_call_id == "c1"


@pytest.mark.asyncio
async def test_unknown_tool_passes_through_to_stock_error() -> None:
    state = {"messages": [_ai([_call("nope", {"x": 1}, "c1")])]}
    result = await SaidexToolNode([add]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)
    tool_message = _tool_messages(result)[0]
    assert tool_message.tool_call_id == "c1"
    assert tool_message.status == "error"
    assert "not a valid tool" in str(tool_message.content)


# ---------------------------------------------------------------------------
# Graph-embedded execution (real Pregel-injected runtime, no runtime= kwarg)
# ---------------------------------------------------------------------------
#
# The tests above invoke SaidexToolNode standalone and pass runtime=
# explicitly. langgraph's RunnableCallable.invoke/ainvoke skip the
# config-based runtime lookup entirely whenever `runtime` is already present
# in kwargs (see langgraph/_internal/_runnable.py), so those tests would keep
# passing even if a future change broke config propagation from _arun to the
# inner ToolNode. The tests below close that gap: they run SaidexToolNode
# inside a compiled StateGraph, exactly as documented in the class docstring
# (`graph.add_node("tools", SaidexToolNode(tools))`), so Pregel is the one
# injecting the runtime through `config` — proving the drop-in claim for real.


@pytest.mark.asyncio
async def test_graph_embedded_async_execution() -> None:
    app = _build_tool_call_graph(SaidexToolNode([add]))
    result = await app.ainvoke({"messages": [HumanMessage(content="please add")]})
    tool_messages = _tool_messages(result)
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "7"
    assert tool_messages[0].tool_call_id == "graph-c1"
    assert tool_messages[0].name == "add"


def test_graph_embedded_sync_execution() -> None:
    app = _build_tool_call_graph(SaidexToolNode([add]))
    result = app.invoke({"messages": [HumanMessage(content="please add")]})
    tool_messages = _tool_messages(result)
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "7"
    assert tool_messages[0].tool_call_id == "graph-c1"
    assert tool_messages[0].name == "add"


@pytest.mark.asyncio
async def test_graph_embedded_matches_stock_toolnode() -> None:
    ours_app = _build_tool_call_graph(SaidexToolNode([add]))
    stock_app = _build_tool_call_graph(ToolNode([add]))

    ours_result = await ours_app.ainvoke({"messages": [HumanMessage(content="please add")]})
    stock_result = await stock_app.ainvoke({"messages": [HumanMessage(content="please add")]})

    ours_tm = _tool_messages(ours_result)
    stock_tm = _tool_messages(stock_result)
    assert len(ours_tm) == len(stock_tm) == 1
    assert ours_tm[0].content == stock_tm[0].content
    assert ours_tm[0].tool_call_id == stock_tm[0].tool_call_id
    assert ours_tm[0].name == stock_tm[0].name
