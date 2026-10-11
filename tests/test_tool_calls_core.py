"""Tests for saidex._tool_calls, the framework-agnostic tool-call core (no langgraph needed)."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from saidex._tool_calls import build_correction_prompt, render_conversation


def test_core_module_does_not_import_langgraph() -> None:
    code = "import sys, saidex._tool_calls; print('langgraph' in sys.modules)"
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "False"


def _history() -> list[Any]:
    return [
        SystemMessage(content="secret system prompt"),
        HumanMessage(content="What is 5 + 2?"),
        AIMessage(
            content="",
            tool_calls=[{"name": "add", "args": {"a": 1, "b": 1}, "id": "p1", "type": "tool_call"}],
        ),
        ToolMessage(content="2", name="add", tool_call_id="p1"),
        AIMessage(content="Let me add five and two."),
    ]


def test_render_conversation_selects_and_formats_messages() -> None:
    rendered = render_conversation(_history(), ai_index=4, limit=10)
    assert rendered is not None
    assert "secret system prompt" not in rendered
    assert rendered.splitlines() == [
        "[user] What is 5 + 2?",
        "[assistant] ",
        '  -> add({"a": 1, "b": 1})',
        "[tool result: add] 2",
        "[assistant] Let me add five and two.",
    ]


def test_render_conversation_keeps_only_the_last_n_messages() -> None:
    rendered = render_conversation(_history(), ai_index=4, limit=1)
    assert rendered == "[tool result: add] 2\n[assistant] Let me add five and two."


def test_render_conversation_disabled_with_zero() -> None:
    assert render_conversation(_history(), ai_index=4, limit=0) is None


def test_render_conversation_without_preceding_messages() -> None:
    assert render_conversation([AIMessage(content="")], ai_index=0, limit=10) is None


def test_render_conversation_truncates_long_messages() -> None:
    history = [HumanMessage(content="x" * 5000), AIMessage(content="")]
    rendered = render_conversation(history, ai_index=1, limit=10)
    assert rendered is not None
    assert len(rendered) == 2000
    assert rendered.endswith("…")


def test_render_conversation_uses_placeholders_for_non_text_blocks() -> None:
    history = [
        HumanMessage(content=[{"type": "text", "text": "Look:"}, {"type": "image_url"}]),
        AIMessage(content=""),
    ]
    assert render_conversation(history, ai_index=1, limit=10) == "[user] Look: [image_url]"


def test_build_correction_prompt_contains_every_part() -> None:
    prompt = build_correction_prompt(
        tool_name="add",
        description="Add two integers.",
        arguments='{"a": "five"}',
        error_text="a: not an int",
        conversation="[user] add five",
    )
    assert "invalid call to the tool 'add'" in prompt
    assert "Tool description:\nAdd two integers." in prompt
    assert 'Invalid arguments:\n{"a": "five"}' in prompt
    assert "Validation errors:\na: not an int" in prompt
    assert "Conversation so far (oldest first):\n[user] add five" in prompt
    assert "never invent missing values" in prompt


def test_build_correction_prompt_omits_empty_parts() -> None:
    prompt = build_correction_prompt(
        tool_name="add", description="", arguments="{}", error_text=None, conversation=None
    )
    assert "Tool description" not in prompt
    assert "Validation errors" not in prompt
    assert "Conversation so far" not in prompt
