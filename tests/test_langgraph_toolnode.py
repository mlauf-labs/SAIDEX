"""Tests for saidex.langgraph.SaidexToolNode (no network, no API keys)."""

from __future__ import annotations

from typing import Annotated, Any

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.tools import InjectedToolCallId, tool  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from langgraph.graph.state import CompiledStateGraph  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402
from langgraph.runtime import Runtime  # noqa: E402
from langgraph.types import Command  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from saidex import on_extraction  # noqa: E402
from saidex.langgraph import SaidexToolNode, ToolCallValidationError  # noqa: E402
from tests._mock_llm import make_llm, make_response  # noqa: E402

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


@tool
def lookup_table(schema: str, table: str) -> str:
    """Look up a table by schema and table name (regression: field named 'schema')."""
    EXECUTED.append(f"lookup_table:{schema}.{table}")
    return f"{schema}.{table}"


@tool
def move(to: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command[None]:
    """Move to a location, returning a Command state update instead of a plain result."""
    EXECUTED.append(f"move:{to}")
    return Command(
        update={"messages": [ToolMessage(content=f"moved to {to}", tool_call_id=tool_call_id)]}
    )


@tool
def move_list_state(to: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command[None]:
    """Like ``move`` but returns a list-shaped Command.update for list-shape state."""
    EXECUTED.append(f"move_list_state:{to}")
    return Command(update=[ToolMessage(content=f"moved to {to}", tool_call_id=tool_call_id)])


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


@pytest.mark.asyncio
async def test_tool_arg_named_schema_executes_normally() -> None:
    """Regression: a tool arg named 'schema' must not collide with the schema
    parameter of create_instance_with_issues during pre-validation (see
    saidex.utils.create_instance_with_issues)."""
    state = {"messages": [_ai([_call("lookup_table", {"schema": "public", "table": "t"}, "c1")])]}
    ours = await SaidexToolNode([lookup_table]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)
    stock = await ToolNode([lookup_table]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)

    ours_tm = _tool_messages(ours)
    stock_tm = _tool_messages(stock)
    assert len(ours_tm) == 1
    assert ours_tm[0].status != "error"
    assert ours_tm[0].content == stock_tm[0].content == "public.t"
    assert EXECUTED == ["lookup_table:public.t", "lookup_table:public.t"]


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


# ---------------------------------------------------------------------------
# Recovery of invalid_tool_calls + validation feedback (Task 6)
# ---------------------------------------------------------------------------


def _invalid(name: str, raw_args: str, call_id: str) -> dict[str, Any]:
    return {
        "name": name,
        "args": raw_args,
        "id": call_id,
        "error": "Malformed args",
        "type": "invalid_tool_call",
    }


def _invalid_no_id(name: str, raw_args: str) -> dict[str, Any]:
    """An invalid_tool_calls entry with no id — LangChain types id as str | None."""
    return {
        "name": name,
        "args": raw_args,
        "id": None,
        "error": "Malformed args",
        "type": "invalid_tool_call",
    }


def _invalid_no_name(raw_args: str, call_id: str) -> dict[str, Any]:
    """An invalid_tool_calls entry with an id but no name — also str | None in LangChain."""
    return {
        "name": None,
        "args": raw_args,
        "id": call_id,
        "error": "Malformed args",
        "type": "invalid_tool_call",
    }


def _ai_of(result: Any) -> AIMessage | None:
    messages = result["messages"] if isinstance(result, dict) else result
    for m in messages:
        if isinstance(m, AIMessage):
            return m
    return None


@pytest.mark.asyncio
async def test_invalid_call_recovered_via_think_strip() -> None:
    raw = '<think>let me compute</think>{"a": 2, "b": 3}'
    state = {"messages": [_ai(invalid=[_invalid("add", raw, "c1")])]}
    result = await SaidexToolNode([add]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)

    tool_message = _tool_messages(result)[0]
    assert tool_message.content == "5"
    assert tool_message.tool_call_id == "c1"
    assert EXECUTED == ["add:2+3"]


@pytest.mark.asyncio
async def test_invalid_call_recovered_via_json_repair() -> None:
    state = {"messages": [_ai(invalid=[_invalid("add", '{"a": 2, "b": 3', "c1")])]}
    result = await SaidexToolNode([add]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)
    assert _tool_messages(result)[0].content == "5"


@pytest.mark.asyncio
async def test_repair_flags_can_be_disabled() -> None:
    state = {"messages": [_ai(invalid=[_invalid("add", '{"a": 2, "b": 3', "c1")])]}
    node = SaidexToolNode([add], json_repair=False)
    result = await node.ainvoke(state, runtime=_NO_GRAPH_RUNTIME)
    tool_message = _tool_messages(result)[0]
    assert tool_message.status == "error"
    assert EXECUTED == []


@pytest.mark.asyncio
async def test_strip_thinking_disabled_leaves_think_tags_unparseable() -> None:
    # With json_repair also disabled, disabling strip_thinking is the only
    # thing standing between a parseable and an unparseable payload here:
    # json.loads('<think>...</think>{...}') always fails, so the <think>
    # block must actually be stripped first for this to recover.
    raw = '<think>let me compute</think>{"a": 2, "b": 3}'
    state = {"messages": [_ai(invalid=[_invalid("add", raw, "c1")])]}
    node = SaidexToolNode([add], strip_thinking=False, json_repair=False)
    result = await node.ainvoke(state, runtime=_NO_GRAPH_RUNTIME)
    tool_message = _tool_messages(result)[0]
    assert tool_message.status == "error"
    assert "could not be parsed" in str(tool_message.content)
    assert EXECUTED == []


@pytest.mark.asyncio
async def test_strip_thinking_alone_recovers_think_tag_payload() -> None:
    # The missing control for test_strip_thinking_disabled_leaves_think_tags_
    # unparseable (review Finding 4): that test flips strip_thinking AND
    # json_repair together, so it never isolates which flag matters. Here
    # json_repair is disabled and strip_thinking is the only thing enabled —
    # if the payload still recovers, stripping (not repair) did the work.
    raw = '<think>let me compute</think>{"a": 2, "b": 3}'
    state = {"messages": [_ai(invalid=[_invalid("add", raw, "c1")])]}
    node = SaidexToolNode([add], strip_thinking=True, json_repair=False)
    result = await node.ainvoke(state, runtime=_NO_GRAPH_RUNTIME)
    tool_message = _tool_messages(result)[0]
    assert tool_message.content == "5"
    assert tool_message.tool_call_id == "c1"
    assert EXECUTED == ["add:2+3"]


@pytest.mark.asyncio
async def test_non_dict_json_args_fall_through_to_feedback() -> None:
    # Valid JSON that parses to a list (not a dict) must not be treated as an
    # args mapping — _repair_raw_args returns None and the call falls through
    # to feedback rather than being spread as **args. The feedback wording
    # must not claim the JSON itself failed to parse (misleading here: "[1, 2,
    # 3]" parses fine, it just isn't an object) — it must say arguments have
    # to be a single JSON object and that a list/scalar is not acceptable
    # (review Finding 3).
    state = {"messages": [_ai(invalid=[_invalid("add", "[1, 2, 3]", "c1")])]}
    result = await SaidexToolNode([add]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)
    tool_message = _tool_messages(result)[0]
    assert tool_message.status == "error"
    assert tool_message.tool_call_id == "c1"
    content = str(tool_message.content)
    assert "single JSON object" in content
    assert "list" in content.lower()
    assert EXECUTED == []


@pytest.mark.asyncio
async def test_validation_failure_yields_structured_feedback() -> None:
    state = {"messages": [_ai([_call("add", {"a": "notanint", "b": 2}, "c1")])]}
    result = await SaidexToolNode([add]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)

    tool_message = _tool_messages(result)[0]
    assert tool_message.status == "error"
    assert tool_message.tool_call_id == "c1"
    assert tool_message.name == "add"
    assert "TYPE ERRORS" in str(tool_message.content)
    assert "was NOT executed" in str(tool_message.content)
    assert EXECUTED == []


@pytest.mark.asyncio
async def test_unrecoverable_invalid_gets_feedback_with_schema_summary() -> None:
    state = {"messages": [_ai(invalid=[_invalid("add", "utter garbage no json here", "c1")])]}
    result = await SaidexToolNode([add]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)
    tool_message = _tool_messages(result)[0]
    assert tool_message.status == "error"
    assert tool_message.tool_call_id == "c1"
    assert "could not be parsed" in str(tool_message.content)
    assert "add" in str(tool_message.content)
    # Pin the actual schema_hint block, not just the "add" substring that the
    # surrounding sentences already contain regardless of the field list.
    assert "Expected arguments for 'add': a, b." in str(tool_message.content)


@pytest.mark.asyncio
async def test_every_id_answered_mixed_batch() -> None:
    state = {
        "messages": [
            _ai(
                [
                    _call("add", {"a": 1, "b": 2}, "ok-1"),
                    _call("add", {"a": "bad", "b": 2}, "bad-1"),
                    _call("nope", {"x": 1}, "unknown-1"),
                ],
                invalid=[_invalid("add", "garbage", "junk-1")],
            )
        ]
    }
    result = await SaidexToolNode([add]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)
    ids = [m.tool_call_id for m in _tool_messages(result)]
    assert ids == ["ok-1", "bad-1", "unknown-1", "junk-1"]
    assert EXECUTED == ["add:1+2"]


@pytest.mark.asyncio
async def test_sanitize_replaces_ai_message_after_repair() -> None:
    raw = '<think>hmm</think>{"a": 2, "b": 3}'
    state = {"messages": [_ai(invalid=[_invalid("add", raw, "c1")], msg_id="ai-9")]}
    result = await SaidexToolNode([add]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)

    updated = _ai_of(result)
    assert updated is not None
    assert updated.id == "ai-9"
    assert updated.invalid_tool_calls == []
    assert updated.tool_calls[0]["args"] == {"a": 2, "b": 3}


@pytest.mark.asyncio
async def test_sanitize_keeps_feedback_calls_untouched() -> None:
    state = {
        "messages": [
            _ai(
                [_call("add", {"a": "bad", "b": 2}, "c1")],
                invalid=[_invalid("add", '{"a": 5, "b": 5', "c2")],
                msg_id="ai-9",
            )
        ]
    }
    result = await SaidexToolNode([add]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)
    updated = _ai_of(result)
    assert updated is not None
    # The repaired call joined tool_calls; the feedback-path call stayed as-is.
    assert [c["id"] for c in updated.tool_calls] == ["c1", "c2"]
    assert updated.tool_calls[0]["args"] == {"a": "bad", "b": 2}
    assert updated.invalid_tool_calls == []


@pytest.mark.asyncio
async def test_no_sanitized_message_when_nothing_changed() -> None:
    state = {"messages": [_ai([_call("add", {"a": 1, "b": 2}, "c1")])]}
    result = await SaidexToolNode([add]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)
    assert _ai_of(result) is None


@pytest.mark.asyncio
async def test_sanitize_messages_false_returns_no_ai_message() -> None:
    raw = '{"a": 2, "b": 3'
    state = {"messages": [_ai(invalid=[_invalid("add", raw, "c1")])]}
    node = SaidexToolNode([add], sanitize_messages=False)
    result = await node.ainvoke(state, runtime=_NO_GRAPH_RUNTIME)
    assert _ai_of(result) is None


@pytest.mark.asyncio
async def test_sanitize_skipped_without_message_id() -> None:
    raw = '{"a": 2, "b": 3'
    state = {"messages": [_ai(invalid=[_invalid("add", raw, "c1")], msg_id=None)]}
    result = await SaidexToolNode([add]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)
    assert _ai_of(result) is None
    assert _tool_messages(result)[0].content == "5"


# ---------------------------------------------------------------------------
# id-less / name-less invalid_tool_calls entries (review Finding 2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_call_with_id_no_name_gets_feedback_without_fabricated_name() -> None:
    state = {"messages": [_ai(invalid=[_invalid_no_name("garbage", "c1")])]}
    result = await SaidexToolNode([add]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)

    tool_message = _tool_messages(result)[0]
    assert tool_message.tool_call_id == "c1"
    assert tool_message.status == "error"
    assert tool_message.name is None
    assert "no tool name" in str(tool_message.content).lower()
    assert EXECUTED == []


@pytest.mark.asyncio
async def test_sanitize_keeps_raw_entry_for_id_no_name_call() -> None:
    # Paired with a genuinely recovered call so sanitize actually rebuilds the
    # message (see Minor 8): the id-no-name entry's raw form must survive
    # unmodified in invalid_tool_calls so it keeps matching the feedback
    # ToolMessage that answered it.
    state = {
        "messages": [
            _ai(
                invalid=[
                    _invalid("add", '{"a": 2, "b": 3}', "c-good"),
                    _invalid_no_name("garbage", "c-no-name"),
                ],
                msg_id="ai-1",
            )
        ]
    }
    result = await SaidexToolNode([add]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)

    feedback = next(m for m in _tool_messages(result) if m.tool_call_id == "c-no-name")
    assert feedback.status == "error"
    assert feedback.name is None

    updated = _ai_of(result)
    assert updated is not None
    assert [c["id"] for c in updated.invalid_tool_calls] == ["c-no-name"]


@pytest.mark.asyncio
async def test_invalid_call_with_no_id_produces_no_orphan_tool_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No id => no way to answer it: no ToolMessage, no fabricated empty id."""
    caplog.set_level("WARNING", logger="saidex.langgraph")
    state = {"messages": [_ai(invalid=[_invalid_no_id("add", "garbage")], msg_id="ai-1")]}
    result = await SaidexToolNode([add]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)

    assert _tool_messages(result) == []
    assert EXECUTED == []
    assert any("no id" in record.message.lower() for record in caplog.records)


@pytest.mark.asyncio
async def test_sanitize_drops_no_id_entry_leaving_no_unanswered_call() -> None:
    """No orphan ToolMessage AND no leftover unanswerable entry in history."""
    state = {"messages": [_ai(invalid=[_invalid_no_id("add", "garbage")], msg_id="ai-1")]}
    result = await SaidexToolNode([add]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)

    assert _tool_messages(result) == []
    updated = _ai_of(result)
    assert updated is not None
    assert updated.invalid_tool_calls == []
    assert updated.tool_calls == []


# ---------------------------------------------------------------------------
# id-less entries on the tool_calls side, and id="" (second review Findings 1+2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_id_less_tool_calls_entry_dropped_not_executed_not_retained(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Blocker (Finding 1): LangChain types ``id`` as ``str | None`` on
    ``ToolCall`` too, not only on ``InvalidToolCall`` — so an id-less entry
    can arrive directly in ``tool_calls``.
    test_sanitize_drops_no_id_entry_leaving_no_unanswered_call above only
    covers the ``invalid_tool_calls`` side; before this fix, an id-less
    ``tool_calls`` entry was correctly never executed/answered but was still
    copied back into the sanitized AIMessage's ``tool_calls`` — a call the
    node itself guarantees nothing will ever answer, contradicting its own
    docstring and warning text."""
    caplog.set_level("WARNING", logger="saidex.langgraph")
    state = {
        "messages": [
            _ai(
                [{"name": "add", "args": {"a": 1, "b": 2}, "id": None, "type": "tool_call"}],
                msg_id="ai-1",
            )
        ]
    }
    result = await SaidexToolNode([add]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)

    assert _tool_messages(result) == []
    assert EXECUTED == []
    assert any("no id" in record.message.lower() for record in caplog.records)

    updated = _ai_of(result)
    assert updated is not None
    assert updated.tool_calls == []


@pytest.mark.asyncio
async def test_invalid_call_with_empty_string_id_treated_as_no_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Minor (Finding 2): the repair gate used truthiness (``if name and
    entry_id``) while the drop check used identity (``is None``), so
    ``id=""`` slipped past both — never repaired even though the payload
    below is trivially repairable, then reached ``_feedback_message`` as an
    orphan ``ToolMessage(tool_call_id="")`` answering no real call. Normalizing
    ``entry_id = entry.get("id") or None`` at construction makes "" agree with
    "no id" everywhere, so this call is now dropped like any other id-less
    entry: no repair attempt, no ToolMessage, no sanitized-history entry."""
    caplog.set_level("WARNING", logger="saidex.langgraph")
    raw = '<think>let me compute</think>{"a": 2, "b": 3}'
    state = {"messages": [_ai(invalid=[_invalid("add", raw, "")], msg_id="ai-1")]}
    result = await SaidexToolNode([add]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)

    assert _tool_messages(result) == []
    assert EXECUTED == []
    assert any("no id" in record.message.lower() for record in caplog.records)

    updated = _ai_of(result)
    assert updated is not None
    assert updated.tool_calls == []
    assert updated.invalid_tool_calls == []


# ---------------------------------------------------------------------------
# Graph-embedded exercise of the repair/sanitize path (review Finding 3)
# ---------------------------------------------------------------------------


def _agent_node_invalid_call(_state: MessagesState) -> dict[str, list[BaseMessage]]:
    """Agent step that emits a malformed (invalid) tool call needing repair."""
    raw = '<think>let me compute</think>{"a": 2, "b": 3}'
    return {
        "messages": [
            _ai(invalid=[_invalid("add", raw, "graph-inv-1")], msg_id="graph-ai-invalid"),
        ],
    }


@pytest.mark.asyncio
async def test_graph_embedded_repairs_and_replaces_ai_message_in_final_state() -> None:
    """The load-bearing new behavior (_substitute_message + sanitized AIMessage
    REPLACED via add_messages by id) is only real when exercised inside a
    compiled graph, where add_messages actually runs. Deliberately does NOT
    pass runtime= — see the module-level note on _NO_GRAPH_RUNTIME."""
    graph: StateGraph[Any, Any, Any, Any] = StateGraph(MessagesState)
    graph.add_node("agent", _agent_node_invalid_call)
    graph.add_node("tools", SaidexToolNode([add]))
    graph.add_edge(START, "agent")
    graph.add_edge("agent", "tools")
    graph.add_edge("tools", END)
    app = graph.compile()

    result = await app.ainvoke({"messages": [HumanMessage(content="please add")]})

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) == 1, "sanitized AIMessage must replace, not duplicate, the original"
    final_ai = ai_messages[0]
    assert final_ai.id == "graph-ai-invalid"
    assert final_ai.invalid_tool_calls == []
    assert len(final_ai.tool_calls) == 1
    assert final_ai.tool_calls[0]["args"] == {"a": 2, "b": 3}
    assert final_ai.tool_calls[0]["id"] == "graph-inv-1"

    tool_messages = _tool_messages(result["messages"])
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "graph-inv-1"
    assert tool_messages[0].content == "5"
    assert EXECUTED == ["add:2+3"]


# ---------------------------------------------------------------------------
# Untested sanitize branch: repaired into a dict that is still schema-invalid
# (review Finding 4 + Minor 8)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sanitize_keeps_raw_entry_when_repaired_but_still_schema_invalid() -> None:
    # '{"a": "not_an_int", "b": 3' repairs to a syntactically-valid dict
    # ({"a": "not_an_int", "b": 3}) that still fails the tool's schema (a must
    # be an int) — the genuine "repaired=True AND from_invalid and not
    # executable" branch, distinct from test_sanitize_keeps_feedback_calls_
    # untouched (whose repaired entry is schema-valid and gets promoted).
    state = {
        "messages": [
            _ai(
                invalid=[
                    _invalid("add", '{"a": 5, "b": 5', "c-recovered"),
                    _invalid("add", '{"a": "not_an_int", "b": 3', "c-bad-schema"),
                ],
                msg_id="ai-9",
            )
        ]
    }
    result = await SaidexToolNode([add]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)

    feedback = next(m for m in _tool_messages(result) if m.tool_call_id == "c-bad-schema")
    assert feedback.status == "error"
    assert "TYPE ERRORS" in str(feedback.content)

    recovered = next(m for m in _tool_messages(result) if m.tool_call_id == "c-recovered")
    assert recovered.content == "10"
    assert EXECUTED == ["add:5+5"]

    updated = _ai_of(result)
    assert updated is not None
    assert [c["id"] for c in updated.invalid_tool_calls] == ["c-bad-schema"]
    assert [c["id"] for c in updated.tool_calls] == ["c-recovered"]


@pytest.mark.asyncio
async def test_no_sanitized_message_when_only_change_is_still_invalid_after_repair() -> None:
    """Minor 8: repairing an entry into a dict that is still schema-invalid,
    with nothing else in the batch, must not emit an inert AIMessage copy
    (same tool_calls/invalid_tool_calls content as the original, new identity)."""
    state = {
        "messages": [
            _ai(invalid=[_invalid("add", '{"a": "not_an_int", "b": 3', "c1")], msg_id="ai-1")
        ]
    }
    result = await SaidexToolNode([add]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)

    assert _ai_of(result) is None
    feedback = _tool_messages(result)[0]
    assert feedback.tool_call_id == "c1"
    assert feedback.status == "error"
    assert EXECUTED == []


# ---------------------------------------------------------------------------
# Command-returning tool mixed with feedback messages (review Minor 7)
# ---------------------------------------------------------------------------
#
# Once a tool returns a Command, stock ToolNode._combine_tool_outputs stops
# returning a flat list of BaseMessage and instead returns a list mixing the
# Command with a {messages_key: [...]} (dict input) or [...] (list input)
# entry per non-Command output. _merge_output's non-message branch must keep
# ordering our own feedback/sanitize extras and must shape them the same way
# (bare list vs. {messages_key: [...]}) rather than always wrapping in a dict.


@pytest.mark.asyncio
async def test_command_tool_mixed_with_feedback_respects_dict_shape() -> None:
    state = {
        "messages": [
            _ai(
                [
                    _call("move", {"to": "x"}, "c1"),
                    _call("add", {"a": "bad", "b": 2}, "c2"),
                ]
            )
        ]
    }
    result = await SaidexToolNode([add, move]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)

    assert isinstance(result, list)
    commands = [e for e in result if isinstance(e, Command)]
    assert len(commands) == 1
    assert commands[0].update["messages"][0].tool_call_id == "c1"

    dict_updates = [e for e in result if isinstance(e, dict)]
    assert len(dict_updates) == 1
    feedback_messages = dict_updates[0]["messages"]
    assert len(feedback_messages) == 1
    assert feedback_messages[0].tool_call_id == "c2"
    assert feedback_messages[0].status == "error"
    assert EXECUTED == ["move:x"]


@pytest.mark.asyncio
async def test_command_tool_mixed_with_feedback_respects_list_shape() -> None:
    messages = [
        _ai(
            [
                _call("move_list_state", {"to": "x"}, "c1"),
                _call("add", {"a": "bad", "b": 2}, "c2"),
            ]
        )
    ]
    result = await SaidexToolNode([add, move_list_state]).ainvoke(
        messages, runtime=_NO_GRAPH_RUNTIME
    )

    assert isinstance(result, list)
    commands = [e for e in result if isinstance(e, Command)]
    assert len(commands) == 1
    # No dict entries at all: extras must be shaped as a bare list, matching
    # the list-shaped input, not wrapped in a {messages_key: [...]} dict.
    assert not any(isinstance(e, dict) for e in result)
    list_updates = [e for e in result if isinstance(e, list)]
    assert len(list_updates) == 1
    assert list_updates[0][0].tool_call_id == "c2"
    assert list_updates[0][0].status == "error"
    assert EXECUTED == ["move_list_state:x"]


# ---------------------------------------------------------------------------
# correct + raise policies (Task 7)
# ---------------------------------------------------------------------------


def _correction_llm(args_sequence: list[dict[str, Any] | None]) -> Any:
    """Mock correction model: each entry is a tool-call args dict or None (invalid)."""
    responses = []
    for args in args_sequence:
        if args is None:
            responses.append(make_response(invalid=True))
        else:
            responses.append(
                make_response(tool_calls=[{"args": args, "name": "add", "id": "corr"}])
            )
    return make_llm(responses)


@pytest.mark.asyncio
async def test_correct_policy_fixes_and_executes() -> None:
    state = {"messages": [_ai([_call("add", {"a": "five", "b": 2}, "c1")], msg_id="ai-9")]}
    node = SaidexToolNode(
        [add],
        on_invalid="correct",
        correction_model=_correction_llm([{"a": 5, "b": 2}]),
    )
    result = await node.ainvoke(state, runtime=_NO_GRAPH_RUNTIME)

    tool_message = _tool_messages(result)[0]
    assert tool_message.content == "7"
    assert tool_message.tool_call_id == "c1"
    assert EXECUTED == ["add:5+2"]
    updated = _ai_of(result)
    assert updated is not None
    assert updated.tool_calls[0]["args"] == {"a": 5, "b": 2}


@pytest.mark.asyncio
async def test_correct_policy_recovers_unparseable_invalid_call() -> None:
    state = {"messages": [_ai(invalid=[_invalid("add", "garbage text", "c1")], msg_id="ai-9")]}
    node = SaidexToolNode(
        [add],
        on_invalid="correct",
        correction_model=_correction_llm([{"a": 1, "b": 9}]),
    )
    result = await node.ainvoke(state, runtime=_NO_GRAPH_RUNTIME)
    assert _tool_messages(result)[0].content == "10"
    updated = _ai_of(result)
    assert updated is not None
    assert updated.invalid_tool_calls == []


@pytest.mark.asyncio
async def test_correct_policy_falls_back_to_feedback() -> None:
    state = {"messages": [_ai([_call("add", {"a": "five", "b": 2}, "c1")])]}
    node = SaidexToolNode(
        [add],
        on_invalid="correct",
        correction_model=_correction_llm([None, None]),
        max_correction_retries=2,
    )
    result = await node.ainvoke(state)
    tool_message = _tool_messages(result)[0]
    assert tool_message.status == "error"
    assert "was NOT executed" in str(tool_message.content)
    assert EXECUTED == []


@pytest.mark.asyncio
async def test_correct_policy_skips_valid_calls() -> None:
    correction = _correction_llm([{"a": 0, "b": 0}])
    state = {"messages": [_ai([_call("add", {"a": 3, "b": 4}, "c1")])]}
    node = SaidexToolNode([add], on_invalid="correct", correction_model=correction)
    result = await node.ainvoke(state, runtime=_NO_GRAPH_RUNTIME)
    assert _tool_messages(result)[0].content == "7"
    correction.bind_tools.assert_not_called()


@pytest.mark.asyncio
async def test_correct_policy_does_not_emit_extraction_event() -> None:
    """The correction cycle is a private sub-extraction: Task 8 reports it via
    ToolCallStats.correction_retries, not through the normal extraction-observer
    channel — no ExtractionEvent may escape it (extract_data is called with
    _notify_observers=False)."""
    events: list[Any] = []
    sub = on_extraction(events.append)
    try:
        state = {"messages": [_ai([_call("add", {"a": "five", "b": 2}, "c1")])]}
        node = SaidexToolNode(
            [add],
            on_invalid="correct",
            correction_model=_correction_llm([{"a": 5, "b": 2}]),
        )
        await node.ainvoke(state, runtime=_NO_GRAPH_RUNTIME)
    finally:
        sub.unsubscribe()
    assert events == []


@pytest.mark.asyncio
async def test_raise_policy_fails_fast_without_executing() -> None:
    state = {
        "messages": [
            _ai(
                [
                    _call("add", {"a": 1, "b": 2}, "ok-1"),
                    _call("add", {"a": "bad", "b": 2}, "bad-1"),
                ]
            )
        ]
    }
    node = SaidexToolNode([add], on_invalid="raise")
    with pytest.raises(ToolCallValidationError) as excinfo:
        await node.ainvoke(state)
    assert EXECUTED == []
    assert excinfo.value.failures[0][0] == "add"
    assert excinfo.value.failures[0][1] == "bad-1"


@pytest.mark.asyncio
async def test_raise_policy_passes_when_all_valid() -> None:
    state = {"messages": [_ai([_call("add", {"a": 1, "b": 2}, "c1")])]}
    result = await SaidexToolNode([add], on_invalid="raise").ainvoke(
        state, runtime=_NO_GRAPH_RUNTIME
    )
    assert _tool_messages(result)[0].content == "3"


# ---------------------------------------------------------------------------
# Task 6 review leftover (Finding a): id="" must be dropped on the tool_calls
# side too, not only invalid_tool_calls — the drop-check must be falsy-based
# so both sides agree on what "no id" means.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_string_id_tool_calls_entry_dropped_not_executed_not_retained(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Leftover from the Task 6 review: the drop-check compared ``id`` by
    identity (``is None``) while only the invalid_tool_calls-side construction
    normalized ``id=""`` to ``None`` — so a ``tool_calls`` entry with ``id=""``
    slipped past the drop-check as answerable and could reach
    _feedback_message as an orphan ``ToolMessage(tool_call_id="")`` answering
    no real call. The drop-check is now falsy-based so both sides agree."""
    caplog.set_level("WARNING", logger="saidex.langgraph")
    state = {
        "messages": [
            _ai(
                [{"name": "add", "args": {"a": "bad", "b": 2}, "id": "", "type": "tool_call"}],
                msg_id="ai-1",
            )
        ]
    }
    result = await SaidexToolNode([add]).ainvoke(state, runtime=_NO_GRAPH_RUNTIME)

    assert _tool_messages(result) == []
    assert EXECUTED == []
    assert any("no id" in record.message.lower() for record in caplog.records)

    updated = _ai_of(result)
    assert updated is not None
    assert updated.tool_calls == []
