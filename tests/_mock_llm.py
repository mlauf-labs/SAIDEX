"""Shared mock-LLM builders for the test suite.

One canonical MagicMock/AsyncMock wiring for chat-model doubles, so every test
file mocks LLM responses the same way and shape changes happen in one place.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock


def make_response(
    tool_calls: list[dict[str, Any]] | None = None,
    content: str = "",
    invalid: bool = False,
) -> MagicMock:
    """Build a chat-model response double as LangChain returns them."""
    r = MagicMock()
    r.tool_calls = tool_calls or []
    r.invalid_tool_calls = (
        [{"name": "bad", "args": "{bad", "id": "x", "error": "parse error"}] if invalid else []
    )
    r.content = content
    return r


def make_llm(responses: list[MagicMock] | list[Any]) -> MagicMock:
    """LLM mock whose bind_tools() returns a bound model yielding *responses*.

    ``responses`` may mix response doubles with exception instances —
    ``AsyncMock(side_effect=...)`` raises the exceptions in sequence.
    """
    bound = MagicMock()
    bound.ainvoke = AsyncMock(side_effect=responses)
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=bound)
    llm.ainvoke = bound.ainvoke  # for JSON mode without tools
    return llm
