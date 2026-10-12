"""Integration tests for ``grounding_source`` (issue #91).

A common prompt puts reference data and the source document into one user
message. ``grounding_source`` limits ``Grounded()`` to the document, so a value
that only appears in the reference data is rejected. Mock LLMs only — no network.
"""

from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType
from typing import Annotated, Any
from unittest.mock import MagicMock

import pytest
from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from saidex import (
    ExtractionMode,
    Grounded,
    GroundedField,
    extract_data,
    extract_data_from_text,
    extract_data_from_text_sync,
    extract_data_list,
    extract_data_list_from_text,
    extract_data_list_from_text_sync,
    extract_data_list_sync,
    extract_data_sync,
    extract_data_with_tools,
    extract_data_with_tools_sync,
    run_extractor_agent,
    run_extractor_agent_sync,
)
from saidex.retry import RetryConfig
from tests._mock_llm import make_llm, make_response

NO_RETRY = RetryConfig(
    max_retries=0, retry_delays=[], retryable_exceptions=(), rate_limit_exceptions=()
)

CUSTOMER_DATA = (
    'Customer data (JSON): {"orders": [{"order_number": 32008}, {"order_number": 28870}]}'
)
CATALOG = "Catalog: UltraView 27 monitor, ProDesk keyboard"
MAIL = "Subject: Monitor broken\nMy UltraVlew 27 monitor from order 32008 has dead pixels."
PROMPT = f"{CUSTOMER_DATA}\n{CATALOG}\n\n<mail>\n{MAIL}\n</mail>"


class Concern(BaseModel):
    order_number: Annotated[str | None, Grounded()]


class RoutedConcern(BaseModel):
    order_number: Annotated[str | None, Grounded(source="mail")]
    product: Annotated[str | None, Grounded(source="catalog")]


def _answer(name: str, args: dict[str, Any], call_id: str = "c1") -> MagicMock:
    return make_response(tool_calls=[{"name": name, "args": args, "id": call_id}])


def _single(order: str, call_id: str) -> MagicMock:
    return _answer("Concern", {"order_number": order}, call_id)


def _batch(order: str, call_id: str) -> MagicMock:
    return _answer("ConcernList", {"items": [{"order_number": order}]}, call_id)


def _messages() -> list[HumanMessage]:
    return [HumanMessage(content=PROMPT)]


def _grounding_received(stats: Any) -> list[str | None]:
    return [i.received for i in stats.field_issues if i.category == "grounding"]


# ---------------------------------------------------------------------------
# extract_data
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_without_grounding_source_a_context_value_passes() -> None:
    llm = make_llm([_single("28870", "c1")])

    result, stats = await extract_data(llm, Concern, _messages(), retry_config=NO_RETRY)

    assert result is not None
    assert result.order_number == "28870"
    assert _grounding_received(stats) == []


@pytest.mark.asyncio
async def test_grounding_source_rejects_a_context_value() -> None:
    llm = make_llm([_single("28870", "c1"), _single("32008", "c2")])

    result, stats = await extract_data(
        llm, Concern, _messages(), grounding_source=MAIL, retry_config=NO_RETRY
    )

    assert result is not None
    assert result.order_number == "32008"
    assert stats.primary_retries == 1
    assert _grounding_received(stats) == ["28870"]


@pytest.mark.asyncio
async def test_grounding_source_in_json_mode() -> None:
    llm = make_llm(
        [
            make_response(content='{"order_number": "28870"}'),
            make_response(content='{"order_number": "32008"}'),
        ]
    )

    result, stats = await extract_data(
        llm,
        Concern,
        _messages(),
        mode=ExtractionMode.JSON,
        grounding_source=MAIL,
        retry_config=NO_RETRY,
    )

    assert result is not None
    assert result.order_number == "32008"
    assert _grounding_received(stats) == ["28870"]


@pytest.mark.asyncio
async def test_fallback_model_uses_the_same_grounding_source() -> None:
    primary = make_llm([_single("28870", "p1")])
    fallback = make_llm([_single("28870", "f1"), _single("32008", "f2")])

    result, stats = await extract_data(
        primary,
        Concern,
        _messages(),
        fallback_llm_model=fallback,
        max_primary_retries=1,
        grounding_source=MAIL,
        retry_config=NO_RETRY,
    )

    assert result is not None
    assert result.order_number == "32008"
    assert stats.fallback_used


@pytest.mark.asyncio
async def test_empty_grounding_source_only_accepts_none() -> None:
    llm = make_llm([_single("32008", "c1"), _answer("Concern", {"order_number": None}, "c2")])

    result, stats = await extract_data(
        llm, Concern, _messages(), grounding_source="", retry_config=NO_RETRY
    )

    assert result is not None
    assert result.order_number is None
    assert _grounding_received(stats) == ["32008"]


@pytest.mark.asyncio
async def test_image_only_messages_still_skip_grounding() -> None:
    image_only = [
        HumanMessage(
            content=[{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]
        )
    ]
    llm = make_llm([_single("28870", "c1")])

    result, stats = await extract_data(llm, Concern, image_only, retry_config=NO_RETRY)

    assert result is not None
    assert result.order_number == "28870"
    assert _grounding_received(stats) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("given", "expected"),
    [({"default": MAIL}, "32008"), ({"catalog": CATALOG}, "28870")],
    ids=["default-key", "no-default-key-uses-human-text"],
)
async def test_mapping_default_key_or_human_text(given: dict[str, str], expected: str) -> None:
    llm = make_llm([_single("28870", "c1"), _single("32008", "c2")])

    result, _ = await extract_data(
        llm, Concern, _messages(), grounding_source=given, retry_config=NO_RETRY
    )

    assert result is not None
    assert result.order_number == expected


@pytest.mark.asyncio
async def test_named_sources_route_each_field() -> None:
    llm = make_llm(
        [
            _answer("RoutedConcern", {"order_number": "28870", "product": "UltraView 27"}, "c1"),
            _answer("RoutedConcern", {"order_number": "32008", "product": "UltraView 27"}, "c2"),
        ]
    )

    result, stats = await extract_data(
        llm,
        RoutedConcern,
        _messages(),
        grounding_source={"mail": MAIL, "catalog": CATALOG},
        retry_config=NO_RETRY,
    )

    assert result is not None
    assert (result.order_number, result.product) == ("32008", "UltraView 27")
    assert _grounding_received(stats) == ["28870"]
    assert "the source text 'mail'" in stats.field_issues[0].message


@pytest.mark.asyncio
async def test_unknown_source_name_raises_before_any_model_call() -> None:
    llm = make_llm([])

    with pytest.raises(ValueError, match="'mail'"):
        await extract_data(
            llm,
            RoutedConcern,
            _messages(),
            grounding_source={"catalog": CATALOG},
            retry_config=NO_RETRY,
        )

    llm.bind_tools.assert_not_called()
    llm.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_wrong_grounding_source_type_raises_before_any_model_call() -> None:
    llm = make_llm([])

    with pytest.raises(TypeError, match="grounding_source"):
        await extract_data(
            llm, Concern, _messages(), grounding_source=["mail"], retry_config=NO_RETRY
        )

    llm.bind_tools.assert_not_called()


@pytest.mark.asyncio
async def test_flag_mode_against_explicit_source() -> None:
    class Soft(BaseModel):
        order_number: Annotated[str | None, Grounded(on_mismatch="flag")]

    llm = make_llm([_answer("Soft", {"order_number": "28870"})])

    result, stats = await extract_data(
        llm, Soft, _messages(), grounding_source=MAIL, retry_config=NO_RETRY
    )

    assert result is not None
    assert result.order_number == "28870"
    assert stats.primary_retries == 0
    assert _grounding_received(stats) == ["28870"]


@pytest.mark.asyncio
async def test_fuzzy_mode_against_explicit_source() -> None:
    class Product(BaseModel):
        product: Annotated[str, Grounded(mode="fuzzy", threshold=0.8)]

    llm = make_llm(
        [
            _answer("Product", {"product": "ProDesk keyboard"}, "c1"),  # only in the catalog
            _answer("Product", {"product": "UltraView 27"}, "c2"),  # OCR typo in the mail
        ]
    )

    result, stats = await extract_data(
        llm, Product, _messages(), grounding_source=MAIL, retry_config=NO_RETRY
    )

    assert result is not None
    assert result.product == "UltraView 27"
    assert _grounding_received(stats) == ["ProDesk keyboard"]


@pytest.mark.asyncio
async def test_locale_against_explicit_source() -> None:
    class Refund(BaseModel):
        amount: float = GroundedField(locale="de")

    mail = "Bitte erstatten Sie 1.234,50 EUR."
    messages = [HumanMessage(content=f"Kundenkonto: Guthaben 99,00 EUR\n<mail>{mail}</mail>")]
    llm = make_llm(
        [_answer("Refund", {"amount": 99.0}, "c1"), _answer("Refund", {"amount": 1234.5}, "c2")]
    )

    result, stats = await extract_data(
        llm, Refund, messages, grounding_source=mail, retry_config=NO_RETRY
    )

    assert result is not None
    assert result.amount == 1234.5
    assert _grounding_received(stats) == ["99.0"]


# ---------------------------------------------------------------------------
# run_extractor_agent (final answer)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grounding_source_in_run_extractor_agent() -> None:
    llm = make_llm([_single("28870", "c1"), _single("32008", "c2")])

    result, stats = await run_extractor_agent(
        llm, Concern, _messages(), tools=[], grounding_source=MAIL, retry_config=NO_RETRY
    )

    assert result is not None
    assert result.order_number == "32008"
    assert stats.validation_retries == 1
    assert _grounding_received(stats) == ["28870"]


@pytest.mark.asyncio
async def test_named_sources_in_run_extractor_agent() -> None:
    llm = make_llm([_answer("RoutedConcern", {"order_number": "32008", "product": "UltraView 27"})])

    result, stats = await run_extractor_agent(
        llm,
        RoutedConcern,
        _messages(),
        tools=[],
        grounding_source={"mail": MAIL, "catalog": CATALOG},
        retry_config=NO_RETRY,
    )

    assert result is not None
    assert stats.validation_retries == 0


@pytest.mark.asyncio
async def test_unknown_source_name_raises_in_agent_loop() -> None:
    llm = make_llm([])

    with pytest.raises(ValueError, match="'mail'"):
        await run_extractor_agent(
            llm, RoutedConcern, _messages(), tools=[], grounding_source=MAIL, retry_config=NO_RETRY
        )

    llm.bind_tools.assert_not_called()


# ---------------------------------------------------------------------------
# Every other entry point forwards grounding_source
# ---------------------------------------------------------------------------

Answer = Callable[[str, str], MagicMock]

ASYNC_ENTRY_POINTS = [
    pytest.param(
        lambda llm, **kw: extract_data_from_text(llm, Concern, PROMPT, **kw),
        _single,
        id="extract_data_from_text",
    ),
    pytest.param(
        lambda llm, **kw: extract_data_list(llm, Concern, _messages(), **kw),
        _batch,
        id="extract_data_list",
    ),
    pytest.param(
        lambda llm, **kw: extract_data_list_from_text(llm, Concern, PROMPT, **kw),
        _batch,
        id="extract_data_list_from_text",
    ),
    pytest.param(
        lambda llm, **kw: extract_data_with_tools(llm, Concern, PROMPT, tools=[], **kw),
        _single,
        id="extract_data_with_tools",
    ),
]

SYNC_ENTRY_POINTS = [
    pytest.param(
        lambda llm, **kw: extract_data_sync(llm, Concern, _messages(), **kw),
        _single,
        id="extract_data_sync",
    ),
    pytest.param(
        lambda llm, **kw: extract_data_from_text_sync(llm, Concern, PROMPT, **kw),
        _single,
        id="extract_data_from_text_sync",
    ),
    pytest.param(
        lambda llm, **kw: extract_data_list_sync(llm, Concern, _messages(), **kw),
        _batch,
        id="extract_data_list_sync",
    ),
    pytest.param(
        lambda llm, **kw: extract_data_list_from_text_sync(llm, Concern, PROMPT, **kw),
        _batch,
        id="extract_data_list_from_text_sync",
    ),
    pytest.param(
        lambda llm, **kw: extract_data_with_tools_sync(llm, Concern, PROMPT, tools=[], **kw),
        _single,
        id="extract_data_with_tools_sync",
    ),
    pytest.param(
        lambda llm, **kw: run_extractor_agent_sync(llm, Concern, _messages(), tools=[], **kw),
        _single,
        id="run_extractor_agent_sync",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("call", "answer"), ASYNC_ENTRY_POINTS)
async def test_async_entry_points_forward_grounding_source(
    call: Callable[..., Any], answer: Answer
) -> None:
    llm = make_llm([answer("28870", "c1"), answer("32008", "c2")])

    _, stats = await call(llm, grounding_source=MAIL, retry_config=NO_RETRY)

    assert stats.success
    assert _grounding_received(stats) == ["28870"]


@pytest.mark.parametrize(("call", "answer"), SYNC_ENTRY_POINTS)
def test_sync_wrappers_forward_grounding_source(call: Callable[..., Any], answer: Answer) -> None:
    llm = make_llm([answer("28870", "c1"), answer("32008", "c2")])

    _, stats = call(llm, grounding_source=MAIL, retry_config=NO_RETRY)

    assert stats.success
    assert _grounding_received(stats) == ["28870"]


@pytest.mark.asyncio
async def test_named_sources_in_batch_extraction() -> None:
    llm = make_llm(
        [
            _answer(
                "RoutedConcernList",
                {
                    "items": [
                        {"order_number": "32008", "product": "UltraView 27"},
                        {"order_number": "28870", "product": "ProDesk keyboard"},
                    ]
                },
                "c1",
            ),
            _answer(
                "RoutedConcernList",
                {"items": [{"order_number": "32008", "product": "UltraView 27"}]},
                "c2",
            ),
        ]
    )

    items, stats = await extract_data_list(
        llm,
        RoutedConcern,
        _messages(),
        grounding_source={"mail": MAIL, "catalog": CATALOG},
        retry_config=NO_RETRY,
    )

    assert items is not None
    assert [i.order_number for i in items] == ["32008"]
    assert [i.field_path for i in stats.field_issues if i.category == "grounding"] == [
        "items -> 1 -> order_number"
    ]


@pytest.mark.asyncio
async def test_unknown_source_name_raises_in_batch_extraction() -> None:
    llm = make_llm([])

    # The error names the caller's item schema, not the internal ``…List`` container.
    with pytest.raises(ValueError, match=r"'order_number' of RoutedConcern is .*'mail'"):
        await extract_data_list(
            llm,
            RoutedConcern,
            _messages(),
            grounding_source={"catalog": CATALOG},
            retry_config=NO_RETRY,
        )

    llm.bind_tools.assert_not_called()


BATCH_CALLS = [
    pytest.param(
        lambda llm, **kw: extract_data_list(llm, Concern, _messages(), **kw),
        id="extract_data_list",
    ),
    pytest.param(
        lambda llm, **kw: extract_data_list_from_text(llm, Concern, PROMPT, **kw),
        id="extract_data_list_from_text",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("call", BATCH_CALLS)
@pytest.mark.parametrize("bad", [["mail"], 123])
async def test_wrong_grounding_source_type_in_batch_with_captured_text(
    call: Callable[..., Any], bad: Any
) -> None:
    llm = make_llm([])

    with pytest.raises(TypeError, match="grounding_source must be a str"):
        await call(llm, grounding_source=bad, capture_source_text=True, retry_config=NO_RETRY)

    llm.bind_tools.assert_not_called()


# ---------------------------------------------------------------------------
# Trace inputs — grounding_source only behind capture_source_text
# ---------------------------------------------------------------------------


class _InputsRecorder(AsyncCallbackHandler):
    """Records the inputs of every chain-run start."""

    def __init__(self) -> None:
        self.inputs: list[dict[str, Any]] = []

    async def on_chain_start(self, serialized: Any, inputs: Any, **kw: Any) -> None:
        self.inputs.append(dict(inputs))


TRACED_CALLS = [
    pytest.param(
        lambda llm, **kw: extract_data(llm, Concern, _messages(), **kw),
        _single,
        id="extract_data",
    ),
    pytest.param(
        lambda llm, **kw: extract_data_list(llm, Concern, _messages(), **kw),
        _batch,
        id="extract_data_list",
    ),
    pytest.param(
        lambda llm, **kw: run_extractor_agent(llm, Concern, _messages(), tools=[], **kw),
        _single,
        id="run_extractor_agent",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("call", "answer"), TRACED_CALLS)
async def test_trace_inputs_carry_grounding_source_when_captured(
    call: Callable[..., Any], answer: Answer
) -> None:
    recorder = _InputsRecorder()
    llm = make_llm([answer("32008", "c1")])

    _, stats = await call(
        llm,
        callbacks=[recorder],
        grounding_source=MAIL,
        capture_source_text=True,
        retry_config=NO_RETRY,
    )

    assert recorder.inputs[0]["grounding_source"] == MAIL
    assert recorder.inputs[0]["text"] == PROMPT
    assert stats.source_text == PROMPT


@pytest.mark.asyncio
@pytest.mark.parametrize(("call", "answer"), TRACED_CALLS)
async def test_trace_inputs_omit_grounding_source_by_default(
    call: Callable[..., Any], answer: Answer
) -> None:
    recorder = _InputsRecorder()
    llm = make_llm([answer("32008", "c1")])

    _, stats = await call(llm, callbacks=[recorder], grounding_source=MAIL, retry_config=NO_RETRY)

    assert "grounding_source" not in recorder.inputs[0]
    assert "text" not in recorder.inputs[0]
    assert stats.source_text is None


@pytest.mark.asyncio
async def test_trace_inputs_have_no_grounding_key_without_a_source() -> None:
    recorder = _InputsRecorder()
    llm = make_llm([_single("28870", "c1")])

    await extract_data(
        llm,
        Concern,
        _messages(),
        callbacks=[recorder],
        capture_source_text=True,
        retry_config=NO_RETRY,
    )

    assert "grounding_source" not in recorder.inputs[0]
    assert recorder.inputs[0]["text"] == PROMPT


@pytest.mark.asyncio
async def test_trace_inputs_copy_a_mapping_source() -> None:
    recorder = _InputsRecorder()
    sources = {"mail": MAIL, "catalog": CATALOG}
    llm = make_llm([_answer("RoutedConcern", {"order_number": "32008", "product": "UltraView 27"})])

    await extract_data(
        llm,
        RoutedConcern,
        _messages(),
        callbacks=[recorder],
        grounding_source=MappingProxyType(sources),
        capture_source_text=True,
        retry_config=NO_RETRY,
    )

    traced = recorder.inputs[0]["grounding_source"]
    assert traced == sources
    assert type(traced) is dict
