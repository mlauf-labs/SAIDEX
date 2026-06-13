"""SAIDEX — Structured AI Data EXtraction.

Extract validated Pydantic models from LLM responses with automatic retry,
fallback models, and an agentic tool loop.

Public API
----------
.. code-block:: python

    from saidex import (
        # Single-shot extraction (no caller tools)
        get_structured_data,
        extract_from_text,
        # Agentic tool loop
        run_agent_loop,
        extract_with_tools,
        Tool,
        AgentRunStats,
        # Shared
        ExtractionMode,
        StructuredOutputStats,
        RetryConfig,
        create_instance_safe,
    )
"""

from .extractor import extract_from_text, extract_with_tools, get_structured_data, run_agent_loop
from .models import AgentRunStats, ExtractionMode, StructuredOutputStats
from .retry import DEFAULT_RETRY_CONFIG, RetryConfig, RetryResult, with_retry
from .tools import Tool
from .utils import create_instance_safe

__all__ = [
    # single-shot extraction
    "get_structured_data",
    "extract_from_text",
    # agentic tool loop
    "run_agent_loop",
    "extract_with_tools",
    "Tool",
    "AgentRunStats",
    # shared
    "ExtractionMode",
    "StructuredOutputStats",
    "RetryConfig",
    "RetryResult",
    "DEFAULT_RETRY_CONFIG",
    "with_retry",
    "create_instance_safe",
]

__version__ = "0.2.0.post1"
