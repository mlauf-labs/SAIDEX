"""Example 16 — SaidexToolNode: validated tool calling in a LangGraph agent.

``SaidexToolNode`` is a drop-in replacement for ``langgraph.prebuilt.ToolNode``
that repairs ``invalid_tool_calls``, validates arguments against each tool's
Pydantic schema *before* execution, and answers bad calls with SAIDEX's
structured, field-level feedback instead of letting an unanswered tool-call id
break the next model turn.

Requires: pip install "saidex[langgraph]" langchain-openai

Run:
    python examples/16_langgraph_toolnode.py
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import tools_condition

from saidex import collect_stats
from saidex.langgraph import SaidexToolNode

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def get_price(product_id: str, currency: str) -> str:
    """Look up the price of a product in the given ISO currency code."""
    return f"{product_id}: 42.00 {currency}"


TOOLS = [get_price]

# ---------------------------------------------------------------------------
# Graph: model -> (SaidexToolNode) -> model -> END
# ---------------------------------------------------------------------------


async def main() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(TOOLS)

    async def agent(state: MessagesState) -> dict[str, list]:
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    # Drop-in for langgraph.prebuilt.ToolNode: repairs invalid_tool_calls,
    # validates args against the tool schema, and answers bad calls with
    # structured feedback the model can act on — instead of silently
    # dropping them and leaving the tool-call id unanswered.
    builder.add_node("tools", SaidexToolNode(TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    graph = builder.compile()

    async with collect_stats() as sink:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="What does product A-7 cost in euros?")]}
        )

    print(result["messages"][-1].content)
    for stats in sink:
        print(
            f"tool node: executed={getattr(stats, 'executed_count', '-')} "
            f"feedback={getattr(stats, 'feedback_count', '-')}"
        )


if __name__ == "__main__":
    asyncio.run(main())
