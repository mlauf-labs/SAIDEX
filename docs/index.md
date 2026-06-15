# SAIDEX — Documentation

**Structured AI Data EXtraction: validated Pydantic models from LLM responses — with automatic retry, fallback, and an agentic tool loop.**

---

## Documentation map

| Document | What it covers |
| --- | --- |
| **[Quickstart](quickstart.md)** | Installation, core API, first working example |
| **[Extraction](extraction.md)** | `extract_data_from_text` and `extract_data` in depth |
| **[Batch Extraction](batch-extraction.md)** | `extract_data_list` / `extract_data_list_from_text` — return `list[ModelT]` from one call |
| **[Extraction Modes](extraction-modes.md)** | Tool calling vs. raw JSON (no tool-calling required) |
| **[Agent Loop](agent-loop.md)** | `extract_data_with_tools` / `run_extractor_agent` — let the LLM call your tools before the final answer |
| **[Retry & Fallback](retry-and-fallback.md)** | Validation retries, network retries, fallback models, error handling |
| **[Schema Design](schema-design.md)** | Which types to use, field descriptions, types to avoid |
| **[Built-in Field Types & Validators](built-in-types.md)** | Reusable types the library ships (`IsoDateStr`, `IbanStr`, `IsinStr`, …) |
| **[Pydantic Validators](validators.md)** | Pre/post/model validators, computed fields, patterns, pitfalls |
| **[Image Extraction](images.md)** | Multimodal messages, OpenAI Vision, local vLLM |
| **[Observability](observability.md)** | Callbacks, Langfuse, LangSmith, logging, async/sync |

---

## Examples

All examples are runnable Python files in the [`examples/`](../examples/) directory.

| File | Demonstrates |
| --- | --- |
| [`01_extract_from_text.py`](../examples/01_extract_from_text.py) | Basic text → Pydantic model |
| [`02_chat_history.py`](../examples/02_chat_history.py) | Multi-turn LangChain messages → model |
| [`03_with_fallback.py`](../examples/03_with_fallback.py) | Primary + fallback model |
| [`04_custom_retry_config.py`](../examples/04_custom_retry_config.py) | Custom `RetryConfig` + `create_instance_safe` standalone |
| [`05_image_analysis.py`](../examples/05_image_analysis.py) | Image URL, local file, vLLM, multiple images |
| [`06_pydantic_validators.py`](../examples/06_pydantic_validators.py) | All four validator types + computed fields |
| [`07_json_mode.py`](../examples/07_json_mode.py) | Extraction without tool calling (raw JSON) |
| [`08_agent_tools.py`](../examples/08_agent_tools.py) | Agentic tool loop: `Tool`, `extract_data_with_tools`, `run_extractor_agent` |
| [`10_batch_extraction.py`](../examples/10_batch_extraction.py) | Batch extraction → `list[ModelT]`: `extract_data_list_from_text` |
| [`11_quality_analytics.py`](../examples/11_quality_analytics.py) | Aggregate field issues across runs: `summarize_field_issues` |

---

## Quick navigation

**I want to…**

- …run my first extraction → [Quickstart](quickstart.md)
- …use a model that does not support tool calling → [Extraction Modes — JSON](extraction-modes.md#extractionmodejson)
- …pass a chat history instead of plain text → [Extraction](extraction.md#chat-history)
- …extract many repeated records (line items, attendees) as a list → [Batch Extraction](batch-extraction.md)
- …let the LLM look things up with my own tools first → [Agent Loop](agent-loop.md)
- …make extraction more reliable with retries → [Retry & Fallback](retry-and-fallback.md)
- …understand which Pydantic types to use → [Schema Design](schema-design.md)
- …validate a date/IBAN/country code field with one annotation → [Built-in Types](built-in-types.md)
- …clean messy LLM output (currency symbols, date formats) → [Validators — `mode='before'`](validators.md#field_validatormodebefore--clean-raw-llm-output)
- …add cross-field rules (end > start, total = sum) → [Validators — `mode='after'`](validators.md#model_validatormodeafter--cross-field-consistency-rules)
- …extract data from images → [Image Extraction](images.md)
- …add tracing with Langfuse or LangSmith → [Observability](observability.md)
- …call the library from synchronous code → [Observability — Sync usage](observability.md#sync-usage)
