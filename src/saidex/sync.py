"""Synchronous wrappers around the async-first extraction API.

The public API is async-first.  These thin wrappers let callers in a purely
synchronous context — notebooks, simple scripts, CLI tools, sync web handlers —
drive an extraction without managing :func:`asyncio.run` themselves.

Each wrapper mirrors the signature of its async counterpart exactly and
delegates to it via :func:`asyncio.run`.  When called from inside an already
running event loop they raise a clear :class:`RuntimeError` instead of
deadlocking — in that situation the caller is already async and should simply
``await`` the coroutine version directly.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any, TypeVar

from langchain_core.messages.base import BaseMessage
from pydantic import BaseModel

from .extractor import extract_from_text, get_structured_data
from .models import ExtractionMode, StructuredOutputStats

if TYPE_CHECKING:
    from .retry import RetryConfig

MODEL_T = TypeVar("MODEL_T", bound=BaseModel)
_T = TypeVar("_T")


def _run_sync(coro: Coroutine[Any, Any, _T], *, sync_name: str, async_name: str) -> _T:
    """Drive *coro* to completion on a fresh event loop.

    Args:
        coro: The coroutine returned by an async API function.
        sync_name: Name of the calling sync wrapper, used in the error message.
        async_name: Name of the async counterpart to suggest in the error.

    Returns:
        Whatever *coro* resolves to.

    Raises:
        RuntimeError: If a running event loop is detected in the current
            thread.  Awaiting from within a loop via ``asyncio.run`` would
            deadlock, so this fails fast with an actionable message.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # A loop is already running — close the coroutine to avoid a
    # "coroutine was never awaited" warning before raising.
    coro.close()
    raise RuntimeError(
        f"{sync_name}() cannot be called from within a running event loop. "
        f"You are already in an async context (e.g. a Jupyter notebook, a "
        f"FastAPI/async handler, or another coroutine) — "
        f"use 'await {async_name}(...)' directly instead."
    )


def get_structured_data_sync(
    llm_model: Any,
    schema: type[MODEL_T],
    messages: list[BaseMessage],
    *,
    mode: ExtractionMode = ExtractionMode.TOOL_CALLING,
    callbacks: list[Any] | None = None,
    fallback_llm_model: Any = None,
    max_primary_retries: int = 3,
    max_fallback_retries: int = 3,
    retry_config: RetryConfig | None = None,
) -> tuple[MODEL_T | None, StructuredOutputStats]:
    """Synchronous wrapper around :func:`~saidex.get_structured_data`.

    Identical behaviour and return value; runs the coroutine to completion on a
    fresh event loop.  See :func:`~saidex.get_structured_data` for the full
    parameter documentation.

    Args:
        llm_model: Any LangChain-compatible chat model.
        schema: The Pydantic ``BaseModel`` subclass to populate.
        messages: Conversation history passed to the model.
        mode: Which extraction strategy to use.
        callbacks: Optional LangChain callback handlers.
        fallback_llm_model: Optional fallback model.
        max_primary_retries: Validation retries for the primary model.
        max_fallback_retries: Validation retries for the fallback model.
        retry_config: Network-level retry configuration.

    Returns:
        ``(model_instance, stats)`` — see :func:`~saidex.get_structured_data`.

    Raises:
        RuntimeError: If called from within a running event loop.
    """
    return _run_sync(
        get_structured_data(
            llm_model,
            schema,
            messages,
            mode=mode,
            callbacks=callbacks,
            fallback_llm_model=fallback_llm_model,
            max_primary_retries=max_primary_retries,
            max_fallback_retries=max_fallback_retries,
            retry_config=retry_config,
        ),
        sync_name="get_structured_data_sync",
        async_name="get_structured_data",
    )


def extract_from_text_sync(
    llm_model: Any,
    schema: type[MODEL_T],
    text: str,
    *,
    mode: ExtractionMode = ExtractionMode.TOOL_CALLING,
    system_prompt: str | None = None,
    callbacks: list[Any] | None = None,
    fallback_llm_model: Any = None,
    max_primary_retries: int = 3,
    max_fallback_retries: int = 3,
    retry_config: RetryConfig | None = None,
) -> tuple[MODEL_T | None, StructuredOutputStats]:
    """Synchronous wrapper around :func:`~saidex.extract_from_text`.

    Identical behaviour and return value; runs the coroutine to completion on a
    fresh event loop.  See :func:`~saidex.extract_from_text` for the full
    parameter documentation.

    Args:
        llm_model: Any LangChain-compatible chat model.
        schema: The Pydantic ``BaseModel`` subclass to populate.
        text: The text to analyse.
        mode: Which extraction strategy to use.
        system_prompt: Optional system instruction prepended to the messages.
        callbacks: Optional LangChain callback handlers.
        fallback_llm_model: Optional fallback model.
        max_primary_retries: Validation retries for the primary model.
        max_fallback_retries: Validation retries for the fallback model.
        retry_config: Network-level retry configuration.

    Returns:
        ``(model_instance, stats)`` — see :func:`~saidex.extract_from_text`.

    Raises:
        RuntimeError: If called from within a running event loop.
    """
    return _run_sync(
        extract_from_text(
            llm_model,
            schema,
            text,
            mode=mode,
            system_prompt=system_prompt,
            callbacks=callbacks,
            fallback_llm_model=fallback_llm_model,
            max_primary_retries=max_primary_retries,
            max_fallback_retries=max_fallback_retries,
            retry_config=retry_config,
        ),
        sync_name="extract_from_text_sync",
        async_name="extract_from_text",
    )
