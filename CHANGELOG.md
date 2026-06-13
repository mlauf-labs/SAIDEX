# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-06-12

First public release on PyPI.

### Changed

- **Project renamed to `saidex`** (Structured AI Data EXtraction) — formerly
  `llm-structured-output`. Install with `pip install saidex`, import with
  `from saidex import ...`.
- License simplified to plain Apache 2.0 — the additional attribution
  requirement in `NOTICE` has been removed.

### Added

- **Agentic tool loop**: `extract_with_tools()` and `run_agent_loop()` let the
  LLM call caller-supplied `Tool`s (lookups, side effects, …) in a loop before
  producing a validated final answer. Returns `AgentRunStats` with
  `iterations`, `tool_calls`, `validation_retries`, and `fallback_used`.
- `Tool` dataclass: wrap an async handler + Pydantic parameter schema; argument
  validation and error reporting back to the LLM are handled automatically.
- `ExtractionMode` enum with `TOOL_CALLING` (default) and `JSON` modes
- `mode` parameter on `get_structured_data()` and `extract_from_text()` to run
  extraction without tool calling — the schema's JSON Schema is injected into
  the prompt and the model's raw JSON reply is parsed and validated. Works with
  any chat model, including local models without tool-calling support.
- Tolerant JSON-mode parsing: strips markdown code fences, chain-of-thought
  `<think>` blocks (Qwen3, DeepSeek-R1, …), and stray text around the JSON
  object; repairs truncated/malformed JSON via `json-repair`.
- `examples/07_json_mode.py` and `docs/extraction-modes.md`

## [0.1.0] - 2026-06-03

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
