# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


### Added

- Initial release (not published to PyPI)
- `get_structured_data()` — extract a Pydantic model from a LangChain message list
- `extract_from_text()` — convenience wrapper for single-text extraction
- Two-phase retry strategy: primary model → optional fallback model
- Automatic validation-error feedback loop: detailed error messages are sent back to the LLM
- `StructuredOutputStats` return type with `primary_retries`, `fallback_retries`, `fallback_used`, `total_retries`
- `create_instance_safe()` utility for safe Pydantic model instantiation with structured error reporting
- Configurable `RetryConfig` for network-level retries (rate limits, connection errors)
- Optional OpenAI rate-limit handling via `saidex[openai]` extra
- Full type annotations (PEP 561 `py.typed` marker)
- Async-first API

[Unreleased]: https://github.com/mlauf-labs/saidex/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/mlauf-labs/saidex/releases/tag/v0.2.0
[0.1.0]: https://github.com/mlauf-labs/saidex/releases/tag/v0.1.0

## v0.6.0 (2026-07-06)

### Feat

- **observability**: wrap extract_data and batch runs in chain runs
- **observability**: emit tool spans for agent-loop tool calls
- **observability**: wrap the agent loop in a chain run
- **tools**: surface handler exceptions via Tool._invoke
- **observability**: add private LangChain chain-run/tool-span helper

### Fix

- **observability**: close trace spans on cancellation and callback failures
- **tools**: reject non-mapping tool args instead of crashing the loop

## v0.5.0 (2026-07-04)

### Feat

- **grounding**: add fuzzy matching mode for grounded fields
- **grounding**: localise number-words and boolean yes/no in candidate rendering
- **observability**: add on_extraction global listener
- **observability**: add collect_stats scoped stats sink

### Fix

- **observability**: isolate failing observers and log with traceback
- **observability**: fire observers once per top-level extract_data_list

## v0.4.0 (2026-07-04)

### Feat

- **validators**: add RRuleStr for RFC 5545 recurrence rules
- **grounding**: add on_mismatch="flag" for advisory grounding
- **grounding**: verify extracted field values against the source text

## v0.3.0 (2026-06-15)

### BREAKING CHANGE

- the int/str/comparison shims on ExtractDataStats were
removed. Use stats.total_retries instead of int(stats), and stats.success
instead of comparing stats against an int.
- The public functions get_structured_data, extract_from_text,
run_agent_loop, extract_with_tools, their *_sync wrappers, the types
StructuredOutputStats and AgentRunStats, and the validator alias ISODateStr
have been renamed. The old names are removed with no compatibility aliases;
callers must update imports and call sites to the new names.

### Feat

- **extractor**: add external validator callable to all extraction methods
- **benchmarks**: collect and report field-level extraction stats across the sweep
- add on_complete hook with optional source-text capture
- render a markdown report from a FieldIssueSummary
- aggregate field issues across runs into a per-schema summary
- enrich extraction stats with success, failure reason, and field issues
- **extractor**: add batch extraction returning list[ModelT]
- unify public API names to extract_data*/run_extractor_agent scheme
- **api**: add sync wrappers for the agent loop
- **api**: add synchronous extraction wrappers
- **benchmarks**: add validators, make schema fields required, rename company fields

### Fix

- **docs**: replace uv pip install --system with pip install

## v0.2.0 (2026-06-13)

### Feat

- switch release target to production PyPI, reset version to 0.2.0
- initial public release of saidex 0.2.0

### Fix

- correct owner typo mlauff-labs -> mlauf-labs across all files
- restore canonical Apache-2.0 LICENSE and target TestPyPI
