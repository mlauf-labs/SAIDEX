"""SAIDEX — Structured AI Data EXtraction.

Extract validated Pydantic models from LLM responses with automatic retry,
fallback models, and an agentic tool loop.

Public API
----------
.. code-block:: python

    from saidex import (
        # Single-shot extraction (no caller tools)
        extract_data,
        extract_data_from_text,
        # Batch extraction returning list[ModelT]
        extract_data_list,
        extract_data_list_from_text,
        # Synchronous wrappers (for non-async callers)
        extract_data_sync,
        extract_data_from_text_sync,
        extract_data_list_sync,
        extract_data_list_from_text_sync,
        # Agentic tool loop
        run_extractor_agent,
        extract_data_with_tools,
        # Synchronous agent-loop wrappers
        run_extractor_agent_sync,
        extract_data_with_tools_sync,
        Tool,
        ExtractorRunStats,
        # Shared
        ExtractionMode,
        ExtractDataStats,
        FieldIssue,
        ExtractionEvent,
        OnComplete,
        Validator,
        collect_stats,
        StatsSink,
        on_extraction,
        Subscription,
        ExtractionListener,
        RetryConfig,
        create_instance_safe,
        # Source grounding (schema-declared, source-aware field checks)
        Grounded,
        GroundedField,
        GroundedStr,
        FieldCheck,
        ExtractionContext,
        field_check,
        # Cross-run field-issue analytics
        summarize_field_issues,
        render_field_issue_report,
        FieldIssueSummary,
        SchemaProblemSummary,
        FieldProblemStat,
        # Reusable schema field types
        IsoDateStr,
        IbanStr,
        VatIdStr,
        CountryCodeStr,
        CurrencyCodeStr,
        IsinStr,
        PhoneStr,
        LanguageCodeStr,
        RRuleStr,
    )
"""

from .analytics import (
    FieldIssueSummary,
    FieldProblemStat,
    SchemaProblemSummary,
    render_field_issue_report,
    summarize_field_issues,
)
from .extractor import (
    OnComplete,
    Validator,
    extract_data,
    extract_data_from_text,
    extract_data_list,
    extract_data_list_from_text,
    extract_data_with_tools,
    run_extractor_agent,
)
from .grounding import (
    ExtractionContext,
    FieldCheck,
    Grounded,
    GroundedField,
    GroundedStr,
    field_check,
)
from .models import (
    ExtractDataStats,
    ExtractionEvent,
    ExtractionMode,
    ExtractorRunStats,
    FieldIssue,
)
from .observability import (
    ExtractionListener,
    StatsSink,
    Subscription,
    collect_stats,
    on_extraction,
)
from .retry import DEFAULT_RETRY_CONFIG, RetryConfig, RetryResult, with_retry
from .sync import (
    extract_data_from_text_sync,
    extract_data_list_from_text_sync,
    extract_data_list_sync,
    extract_data_sync,
    extract_data_with_tools_sync,
    run_extractor_agent_sync,
)
from .tools import Tool
from .utils import create_instance_safe
from .validators import (
    CountryCodeStr,
    CurrencyCodeStr,
    IbanStr,
    IsinStr,
    IsoDateStr,
    LanguageCodeStr,
    PhoneStr,
    RRuleStr,
    VatIdStr,
    validate_country_code,
    validate_currency_code,
    validate_iban,
    validate_isin,
    validate_iso_date,
    validate_language_code,
    validate_phone,
    validate_rrule,
    validate_vat_id,
)

__all__ = [
    # single-shot extraction
    "extract_data",
    "extract_data_from_text",
    # batch extraction returning list[ModelT]
    "extract_data_list",
    "extract_data_list_from_text",
    # synchronous wrappers
    "extract_data_sync",
    "extract_data_from_text_sync",
    "extract_data_list_sync",
    "extract_data_list_from_text_sync",
    # agentic tool loop
    "run_extractor_agent",
    "extract_data_with_tools",
    # synchronous agent-loop wrappers
    "run_extractor_agent_sync",
    "extract_data_with_tools_sync",
    "Tool",
    "ExtractorRunStats",
    # shared
    "ExtractionMode",
    "ExtractDataStats",
    "FieldIssue",
    "ExtractionEvent",
    "OnComplete",
    "Validator",
    "collect_stats",
    "StatsSink",
    "on_extraction",
    "Subscription",
    "ExtractionListener",
    "RetryConfig",
    # cross-run field-issue analytics
    "summarize_field_issues",
    "render_field_issue_report",
    "FieldIssueSummary",
    "SchemaProblemSummary",
    "FieldProblemStat",
    "RetryResult",
    "DEFAULT_RETRY_CONFIG",
    "with_retry",
    "create_instance_safe",
    # source grounding / schema-declared field checks
    "Grounded",
    "GroundedField",
    "GroundedStr",
    "FieldCheck",
    "ExtractionContext",
    "field_check",
    # reusable schema field types
    "IsoDateStr",
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
    "RRuleStr",
    "validate_rrule",
]

__version__ = "0.6.0"
