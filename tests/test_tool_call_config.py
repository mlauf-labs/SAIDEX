"""Tests for ToolCallConfig — configurable bind_tools flags and auto-relax."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from saidex import (
    ExtractionMode,
    ToolCallConfig,
    extract_data,
    extract_data_from_text,
    extract_data_list,
    extract_data_sync,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleSchema(BaseModel):
    name: str
    value: int


VALID_ARGS = {"name": "Alice", "value": 42}


class _Response:
    """Chat-model response double carrying a single tool call."""

    def __init__(self, args: dict[str, Any] | None = None, content: str = "") -> None:
        self.tool_calls = (
            [{"args": args, "name": "SimpleSchema", "id": "call_1"}] if args is not None else []
        )
        self.invalid_tool_calls: list[dict[str, Any]] = []
        self.content = content


class _BadRequestError(Exception):
    """Stand-in for ``openai.BadRequestError`` — an HTTP 400, not retryable."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.status_code = 400


class _Bound:
    def __init__(self, parent: FakeChatModel, kwargs: dict[str, Any]) -> None:
        self._parent = parent
        self._kwargs = kwargs

    async def ainvoke(self, messages: Any, **_: Any) -> Any:
        self._parent.invoke_binds.append(self._kwargs)
        rejected = self._parent.rejected_flag(self._kwargs)
        if rejected is not None:
            raise _BadRequestError(f"Unsupported parameter: '{rejected}'")
        return self._parent.responses.pop(0)


class FakeChatModel:
    """Records every ``bind_tools`` kwarg set and can reject flags like a gateway.

    ``rejects`` names the kwargs the fake gateway refuses.  ``tool_choice`` is
    only refused when it is a *forced* named choice — ``"auto"`` is accepted,
    mirroring how real OpenAI-compatible gateways behave.  ``bind_rejects``
    names kwargs the model class does not accept at all, raising ``TypeError``
    at bind time (LangChain integrations without a ``strict`` parameter).
    """

    def __init__(
        self,
        responses: list[_Response] | None = None,
        *,
        rejects: tuple[str, ...] = (),
        bind_rejects: tuple[str, ...] = (),
    ) -> None:
        self.responses = list(responses or [])
        self.bind_calls: list[dict[str, Any]] = []
        self.invoke_binds: list[dict[str, Any]] = []
        self._rejects = set(rejects)
        self._bind_rejects = set(bind_rejects)

    def rejected_flag(self, kwargs: dict[str, Any]) -> str | None:
        for flag in sorted(self._rejects):
            if flag not in kwargs:
                continue
            if flag == "tool_choice" and kwargs[flag] == "auto":
                continue
            return flag
        return None

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> _Bound:
        self.bind_calls.append(kwargs)
        unsupported = self._bind_rejects & set(kwargs)
        if unsupported:
            raise TypeError(
                f"bind_tools() got an unexpected keyword argument {sorted(unsupported)[0]!r}"
            )
        return _Bound(self, kwargs)

    async def ainvoke(self, messages: Any, **_: Any) -> Any:
        """Used by JSON mode, which never binds tools."""
        return self.responses.pop(0)


def _messages() -> list[HumanMessage]:
    return [HumanMessage(content="Alice is 42.")]


# ---------------------------------------------------------------------------
# ToolCallConfig presets
# ---------------------------------------------------------------------------


def test_openai_preset_matches_defaults() -> None:
    assert ToolCallConfig() == ToolCallConfig.OPENAI
    assert ToolCallConfig.OPENAI.tool_choice == "forced"
    assert ToolCallConfig.OPENAI.strict is True
    assert ToolCallConfig.OPENAI.parallel_tool_calls is False
    assert ToolCallConfig.OPENAI.auto_relax is True


def test_compatible_preset_omits_openai_only_flags() -> None:
    assert ToolCallConfig.COMPATIBLE.tool_choice == "auto"
    assert ToolCallConfig.COMPATIBLE.strict is None
    assert ToolCallConfig.COMPATIBLE.parallel_tool_calls is None


# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_binding_is_unchanged() -> None:
    llm = FakeChatModel([_Response(VALID_ARGS)])

    result, stats = await extract_data(llm, SimpleSchema, _messages())

    assert stats.success
    assert result is not None
    assert llm.bind_calls == [
        {"tool_choice": "SimpleSchema", "parallel_tool_calls": False, "strict": True}
    ]


@pytest.mark.asyncio
async def test_compatible_config_omits_flags_entirely() -> None:
    llm = FakeChatModel([_Response(VALID_ARGS)])

    _, stats = await extract_data(
        llm, SimpleSchema, _messages(), tool_config=ToolCallConfig.COMPATIBLE
    )

    assert stats.success
    assert llm.bind_calls == [{"tool_choice": "auto"}]


@pytest.mark.asyncio
async def test_custom_config_passes_flags_verbatim() -> None:
    llm = FakeChatModel([_Response(VALID_ARGS)])

    _, stats = await extract_data(
        llm,
        SimpleSchema,
        _messages(),
        tool_config=ToolCallConfig(tool_choice="forced", strict=False, parallel_tool_calls=True),
    )

    assert stats.success
    assert llm.bind_calls == [
        {"tool_choice": "SimpleSchema", "strict": False, "parallel_tool_calls": True}
    ]


@pytest.mark.asyncio
async def test_tool_choice_none_omits_the_kwarg() -> None:
    llm = FakeChatModel([_Response(VALID_ARGS)])

    _, stats = await extract_data(
        llm,
        SimpleSchema,
        _messages(),
        tool_config=ToolCallConfig(tool_choice=None, strict=None, parallel_tool_calls=None),
    )

    assert stats.success
    assert llm.bind_calls == [{}]


# ---------------------------------------------------------------------------
# Auto-relax
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_relax_recovers_from_strict_rejection() -> None:
    llm = FakeChatModel([_Response(VALID_ARGS)], rejects=("strict",))

    result, stats = await extract_data(llm, SimpleSchema, _messages())

    assert stats.success
    assert result == SimpleSchema(**VALID_ARGS)
    # The relaxed rebind must not burn a validation retry.
    assert stats.primary_retries == 0
    assert llm.bind_calls == [
        {"tool_choice": "SimpleSchema", "parallel_tool_calls": False, "strict": True},
        {"tool_choice": "auto"},
    ]


@pytest.mark.asyncio
async def test_auto_relax_recovers_from_forced_tool_choice_rejection() -> None:
    llm = FakeChatModel([_Response(VALID_ARGS)], rejects=("tool_choice",))

    _, stats = await extract_data(llm, SimpleSchema, _messages())

    assert stats.success
    assert llm.bind_calls[-1] == {"tool_choice": "auto"}


@pytest.mark.asyncio
async def test_auto_relax_recovers_from_bind_time_type_error() -> None:
    llm = FakeChatModel([_Response(VALID_ARGS)], bind_rejects=("strict",))

    _, stats = await extract_data(llm, SimpleSchema, _messages())

    assert stats.success
    assert llm.bind_calls == [
        {"tool_choice": "SimpleSchema", "parallel_tool_calls": False, "strict": True},
        {"tool_choice": "auto"},
    ]


@pytest.mark.asyncio
async def test_auto_relax_happens_at_most_once() -> None:
    class AlwaysRejecting(FakeChatModel):
        """Even the relaxed binding is refused — nothing SAIDEX can send helps."""

        def rejected_flag(self, kwargs: dict[str, Any]) -> str | None:
            return "strict"

    llm = AlwaysRejecting([_Response(VALID_ARGS)])

    result, stats = await extract_data(llm, SimpleSchema, _messages())

    assert result is None
    assert stats.failure_reason == "llm_error"
    assert len(llm.bind_calls) == 2


@pytest.mark.asyncio
async def test_auto_relax_disabled_fails_immediately() -> None:
    llm = FakeChatModel([_Response(VALID_ARGS)], rejects=("strict",))

    result, stats = await extract_data(
        llm, SimpleSchema, _messages(), tool_config=ToolCallConfig(auto_relax=False)
    )

    assert result is None
    assert stats.failure_reason == "llm_error"
    assert len(llm.bind_calls) == 1


@pytest.mark.asyncio
async def test_unrelated_llm_error_does_not_trigger_relax() -> None:
    class Exploding(FakeChatModel):
        def rejected_flag(self, kwargs: dict[str, Any]) -> str | None:
            raise RuntimeError("upstream connection reset by peer")

    llm = Exploding([_Response(VALID_ARGS)])

    result, stats = await extract_data(llm, SimpleSchema, _messages())

    assert result is None
    assert stats.failure_reason == "llm_error"
    assert len(llm.bind_calls) == 1


@pytest.mark.asyncio
async def test_bind_time_type_error_propagates_when_auto_relax_disabled() -> None:
    llm = FakeChatModel([_Response(VALID_ARGS)], bind_rejects=("strict",))

    result, stats = await extract_data(
        llm, SimpleSchema, _messages(), tool_config=ToolCallConfig(auto_relax=False)
    )

    assert result is None
    assert stats.failure_reason == "llm_error"


# ---------------------------------------------------------------------------
# Mode / plumbing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_json_mode_never_binds_tools() -> None:
    llm = FakeChatModel([_Response(content='{"name": "Alice", "value": 42}')])

    _, stats = await extract_data(
        llm,
        SimpleSchema,
        _messages(),
        mode=ExtractionMode.JSON,
        tool_config=ToolCallConfig.COMPATIBLE,
    )

    assert stats.success
    assert llm.bind_calls == []


@pytest.mark.asyncio
async def test_extract_data_from_text_forwards_tool_config() -> None:
    llm = FakeChatModel([_Response(VALID_ARGS)])

    _, stats = await extract_data_from_text(
        llm, SimpleSchema, "Alice is 42.", tool_config=ToolCallConfig.COMPATIBLE
    )

    assert stats.success
    assert llm.bind_calls == [{"tool_choice": "auto"}]


@pytest.mark.asyncio
async def test_extract_data_list_forwards_tool_config() -> None:
    container_args = {"items": [VALID_ARGS, {"name": "Bob", "value": 7}]}
    llm = FakeChatModel([_Response(container_args)])

    results, stats = await extract_data_list(
        llm, SimpleSchema, _messages(), tool_config=ToolCallConfig.COMPATIBLE
    )

    assert stats.success
    assert len(results) == 2
    assert llm.bind_calls == [{"tool_choice": "auto"}]


def test_sync_wrapper_forwards_tool_config() -> None:
    llm = FakeChatModel([_Response(VALID_ARGS)])

    _, stats = extract_data_sync(
        llm, SimpleSchema, _messages(), tool_config=ToolCallConfig.COMPATIBLE
    )

    assert stats.success
    assert llm.bind_calls == [{"tool_choice": "auto"}]


@pytest.mark.asyncio
async def test_fallback_model_uses_the_same_tool_config() -> None:
    primary = FakeChatModel([_Response({"name": "Alice", "value": "not-an-int"})] * 3)
    fallback = FakeChatModel([_Response(VALID_ARGS)])

    result, stats = await extract_data(
        primary,
        SimpleSchema,
        _messages(),
        fallback_llm_model=fallback,
        tool_config=ToolCallConfig.COMPATIBLE,
    )

    assert stats.success
    assert stats.fallback_used
    assert result == SimpleSchema(**VALID_ARGS)
    assert fallback.bind_calls == [{"tool_choice": "auto"}]
