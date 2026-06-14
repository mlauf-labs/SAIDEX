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
        # Synchronous wrappers (for non-async callers)
        get_structured_data_sync,
        extract_from_text_sync,
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
        # Reusable schema field types
        ISODateStr,
        IbanStr,
        VatIdStr,
        CountryCodeStr,
        CurrencyCodeStr,
        IsinStr,
        PhoneStr,
        LanguageCodeStr,
    )
"""

from .extractor import extract_from_text, extract_with_tools, get_structured_data, run_agent_loop
from .models import AgentRunStats, ExtractionMode, StructuredOutputStats
from .retry import DEFAULT_RETRY_CONFIG, RetryConfig, RetryResult, with_retry
from .sync import extract_from_text_sync, get_structured_data_sync
from .tools import Tool
from .utils import create_instance_safe
from .validators import (
    CountryCodeStr,
    CurrencyCodeStr,
    IbanStr,
    IsinStr,
    ISODateStr,
    LanguageCodeStr,
    PhoneStr,
    VatIdStr,
    validate_country_code,
    validate_currency_code,
    validate_iban,
    validate_isin,
    validate_iso_date,
    validate_language_code,
    validate_phone,
    validate_vat_id,
)

__all__ = [
    # single-shot extraction
    "get_structured_data",
    "extract_from_text",
    # synchronous wrappers
    "get_structured_data_sync",
    "extract_from_text_sync",
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
    # reusable schema field types
    "ISODateStr",
    "validate_iso_date",
    "IbanStr",
    "validate_iban",
    "VatIdStr",
    "validate_vat_id",
    "CountryCodeStr",
    "validate_country_code",
    "CurrencyCodeStr",
    "validate_currency_code",
    "IsinStr",
    "validate_isin",
    "PhoneStr",
    "validate_phone",
    "LanguageCodeStr",
    "validate_language_code",
]

__version__ = "0.2.0"
