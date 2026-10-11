"""Concurrency tests for the module-level observer registries and stats sinks.

Guards the shared state in ``saidex.observability`` against cross-thread
interference; most valuable on the free-threaded (no-GIL) Python build, but
the assertions must hold on every interpreter.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, create_model

from saidex import (
    ExtractionEvent,
    Tool,
    collect_stats,
    extract_data_sync,
    extract_data_with_tools_sync,
    observability,
    on_extraction,
)
from saidex.retry import RetryConfig

THREADS = 8
ROUNDS = 5

NO_RETRY = RetryConfig(
    max_retries=0,
    retry_delays=[],
    retryable_exceptions=(),
    rate_limit_exceptions=(),
)


class _ToolArgs(BaseModel):
    query: str


async def _noop_handler(**kwargs: Any) -> dict[str, Any]:
    return {"status": "ok", **kwargs}


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    yield
    observability._global_listeners.clear()


def _schema(index: int) -> type[BaseModel]:
    return create_model(f"ThreadSchema{index}", name=(str, ...), value=(int, ...))


def _llm(schema_name: str, count: int) -> MagicMock:
    responses = []
    for n in range(count):
        r = MagicMock()
        r.tool_calls = [{"args": {"name": "x", "value": n}, "name": schema_name, "id": f"c{n}"}]
        r.invalid_tool_calls = []
        r.content = ""
        responses.append(r)
    bound = MagicMock()
    bound.ainvoke = AsyncMock(side_effect=responses)
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=bound)
    return llm


def _run_threads(workers: list[Callable[[], None]]) -> list[BaseException]:
    errors: list[BaseException] = []
    barrier = threading.Barrier(len(workers))

    def wrap(fn: Callable[[], None]) -> Callable[[], None]:
        def run() -> None:
            try:
                barrier.wait()
                fn()
            except BaseException as exc:  # noqa: BLE001 — collected and asserted on
                errors.append(exc)

        return run

    threads = [threading.Thread(target=wrap(fn)) for fn in workers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def test_sinks_and_global_listener_under_concurrent_extractions() -> None:
    lock = threading.Lock()
    seen: list[str] = []

    def listener(event: ExtractionEvent) -> None:
        with lock:
            seen.append(event.stats.schema_name)

    subscription = on_extraction(listener)
    sinks: dict[int, list[str]] = {}

    def worker(index: int) -> Callable[[], None]:
        schema = _schema(index)

        def run() -> None:
            with collect_stats() as sink:
                for _ in range(ROUNDS):
                    extract_data_sync(
                        _llm(schema.__name__, 1),
                        schema,
                        [HumanMessage(content="x")],
                        retry_config=NO_RETRY,
                    )
                    extract_data_with_tools_sync(
                        _llm(schema.__name__, 1),
                        schema,
                        "x",
                        tools=[
                            Tool(
                                name="lookup",
                                description="t",
                                parameters=_ToolArgs,
                                handler=_noop_handler,
                            )
                        ],
                        retry_config=NO_RETRY,
                    )
            sinks[index] = [s.schema_name for s in sink.all()]

        return run

    errors = _run_threads([worker(i) for i in range(THREADS)])
    subscription.unsubscribe()

    assert errors == []
    for index in range(THREADS):
        assert sinks[index] == [f"ThreadSchema{index}"] * (2 * ROUNDS)
    assert len(seen) == THREADS * 2 * ROUNDS
    for index in range(THREADS):
        assert seen.count(f"ThreadSchema{index}") == 2 * ROUNDS


def test_concurrent_subscribe_and_unsubscribe_leaves_registry_empty() -> None:
    def churner() -> None:
        for _ in range(200):
            sub = on_extraction(lambda event: None)
            sub.unsubscribe()

    def extractor(index: int) -> Callable[[], None]:
        schema = _schema(index)

        def run() -> None:
            for _ in range(ROUNDS * 4):
                extract_data_sync(
                    _llm(schema.__name__, 1),
                    schema,
                    [HumanMessage(content="x")],
                    retry_config=NO_RETRY,
                )

        return run

    errors = _run_threads([churner for _ in range(4)] + [extractor(i) for i in range(4)])

    assert errors == []
    assert observability._global_listeners == []
