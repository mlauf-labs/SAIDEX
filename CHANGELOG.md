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

## v0.8.0 (2026-10-12)

### BREAKING CHANGE

- SaidexToolNode.invoke() now executes tools on the stock
ToolNode sync path instead of bridging the whole node to ainvoke(). Tools
that only implement a coroutine (e.g. many MCP adapter tools) raise
NotImplementedError under graph.invoke(), exactly as with langgraph's
ToolNode; run such graphs with ainvoke().
- Python 3.10 is no longer supported; the minimum version is now 3.11.

### Feat

- **grounding**: include grounding_source in trace inputs when capture_source_text is on
- **grounding**: forward grounding_source from the text, batch, tool-loop and sync entry points
- **grounding**: add grounding_source to choose what Grounded() checks against
- **grounding**: resolve grounding sources and reject unknown source names
- **grounding**: add Grounded(source=...) and ExtractionContext.sources
- **langgraph**: let interrupt() in tool validators reach LangGraph
- **langgraph**: sanitize tool-call content blocks and raw OpenAI tool calls
- **langgraph**: execute tools on the stock sync path in SaidexToolNode.invoke
- **langgraph**: give the correction model the tool description and conversation context
- **langgraph**: send calls missing required fields to the agent instead of the correction model
- **langgraph**: add per-tool validators with ToolCallValidatorContext
- **langgraph**: add an on_unresolved hook to salvage or answer invalid tool calls
- **langgraph**: enable tool-call correction via correction_model and add on_unresolved
- **agent-loop**: escalate reasoning effort across final-answer attempts
- **extractor**: escalate reasoning effort across retries
- **reasoning**: add reasoning-effort level resolution helpers
- **extractor**: record the provider-independent kind of llm_error in stats
- **retry**: follow langchain-core's retryable model errors in the default retries
- **retry**: add ModelErrorKind and classify_model_error
- declare Python 3.14 support
- **analytics**: count salvaged runs separately from clean successes
- **agent-loop**: support on_exhausted in the agent loop and all sync wrappers
- **batch**: on_exhausted for list extraction and issue_item_index helper
- **extractor**: add on_exhausted hook to salvage a result after all retries
- **observability**: surface last_candidate on events, flag it in traces, keep it out of stats sinks
- **agent-loop**: expose the last schema-valid candidate on failed agent runs
- **extractor**: expose the last schema-valid candidate on failed extractions
- **agent-loop**: provide ValidatorContext to context-aware validators
- **extractor**: pass an attempt context to validators wrapped with with_context
- **models**: add FailureReason StrEnum for stats.failure_reason
- **extractor**: accept lists of error strings from external validators
- **batch**: report external-validator findings per list item
- **extractor**: accept structured ValidationIssue results from external validators
- **models**: add ValidationIssue for structured external-validator findings

### Fix

- **grounding**: find named sources behind type aliases and TypedDicts
- **grounding**: validate grounding_source in batch extraction before tracing
- **langgraph**: store corrected tool-call args in JSON form
- **langgraph**: run SaidexToolNode.invoke inside a running event loop
- **extractor**: keep URLs and container-wrapped arrays intact when repairing JSON replies
- **extractor**: resolve top-level JSON arrays against the target schema
- **extractor**: scan only the JSON region for comments and try the plain parse first
- **extractor**: ask for a single object when a JSON reply contains several
- **extractor**: strip JSON comments before repairing malformed model output
- **extractor**: do not mistake a rejected reasoning_effort for a strict-schema rejection
- **extractor**: enforce strict object rules on nested models for older langchain-core
- **extractor**: relax rejected strict schemas and widen strict fallback
- **extractor**: actually apply strict tool schemas
- **extractor**: report a rejected reasoning_effort kwarg as invalid_request
- **deps**: raise dependency floors to tested minimums
- **extractor**: hand on_exhausted a copy and re-validate from field values
- **models**: carry salvaged through ExtractorRunStats.__add__
- **extractor**: strip computed fields at every depth and type-check list salvage items
- **models**: keep extraction stats hashable and out of repr with last_candidate
- **extractor**: make WithContext contravariant so shared context validators type-check
- **agent-loop**: stop once a final answer is rejected with no retries left
- **observability**: emit failure_reason as a plain string in chain-run outputs
- **extractor**: apply one primary/fallback failure-reason rule everywhere

### Refactor

- **langgraph**: tidy v2 docstrings, field order and edge guards
- **langgraph**: simplify sibling cancellation in concurrent corrections
- **langgraph**: move the tool-call pipeline into a framework-agnostic core
- **extractor**: label the relaxed-rebind llm_error_kind argument
- **agent-loop**: merge the duplicate fallback success branches
- **observability**: narrow StatsSink candidate drop with isinstance instead of getattr
- **extractor**: track which candidate-check issues reject the instance
- **extractor**: share one candidate-check pipeline across all final-answer paths
- apply ruff py311 modernisation

### Perf

- **langgraph**: correct invalid tool calls concurrently


- require Python 3.11 or newer

## v0.7.0 (2026-07-14)

### BREAKING CHANGE

- `create_instance_safe` and `create_instance_with_issues` now take `schema` as a
  **positional-only** parameter. Callers passing it by keyword —
  `create_instance_safe(schema=Foo, ...)` — must switch to positional:
  `create_instance_safe(Foo, ...)`. Passing it positionally (the documented usage)
  is unaffected.

  This is what fixes the underlying bug: previously *any* extraction target schema
  with a field literally named `schema` raised
  `TypeError: got multiple values for argument 'schema'`, which also broke
  `extract_data` for such schemas.

### Feat

- **langgraph**: emit ToolNodeEvent stats and verify tracing passthrough
- **langgraph**: add correct and raise policies for invalid tool calls
- **langgraph**: recover invalid tool calls and pre-validate args with feedback
- **langgraph**: add SaidexToolNode skeleton with stock delegation
- **observability**: collect ToolNodeStats in sinks and add on_tool_node listener
- **models**: add ToolCallStats, ToolNodeStats and ToolNodeEvent
- **extractor**: make tool-binding flags configurable via ToolCallConfig

### Fix

- **langgraph**: fail fast on unsupported object state before substitution
- **langgraph**: pass Send-API tool-call payloads through to the executor
- **langgraph**: reject empty or whitespace-only node name
- **observability**: restore typed Subscription callback
- **langgraph**: never let a correction failure crash SaidexToolNode
- **langgraph**: drop id-less tool_calls entries from sanitized history
- **langgraph**: answer every id-less/misordered tool call correctly
- **utils**: accept schemas with a field named 'schema'
- **langgraph**: scan backward for the last AIMessage like stock ToolNode
- **extractor**: keep Responses-API output_text blocks in JSON mode

### Refactor

- **models**: type ToolCallStats.outcome as a Literal
- **extractor**: accept a CallbackManager in callbacks parameters
- **observability**: share one dispatch path between extraction and tool-node events

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
