# Quickstart

→ [Documentation index](index.md)

---

## Installation

```bash
pip install saidex
```

With langchain-core ≥ 1.6, rate limits, timeouts and 5xx errors of any provider
integration that raises the standard model exceptions are handled automatically. For
older langchain-core versions, or to catch raw OpenAI SDK errors, add the optional
extra:

```bash
pip install "saidex[openai]"
```

**Runtime dependencies**

| Package | Required version | Purpose |
| --- | --- | --- |
| `pydantic` | `>= 2.7.4` | Schema validation |
| `langchain-core` | `>= 0.3.69` | Message types + tool calling |
| `json-repair` | `>= 0.60.1` | Repairs malformed JSON from the model |
| `openai` *(optional)* | `>= 1.0` | Rate-limit retry handling (older langchain-core / raw SDK errors) |
| `langgraph` *(optional)* | `>= 1.1.5` | `SaidexToolNode` for LangGraph agents |

Any LangChain-compatible chat model that supports tool calling works:
OpenAI, Anthropic, Google Gemini, Mistral, Azure OpenAI, and others.

---

## Development setup (with uv)

```bash
# Install uv (if not already installed)
pip install uv

git clone https://github.com/mlauf-labs/saidex
cd saidex
uv sync          # creates .venv and installs all dependencies
```

---

## Core API

The library exposes two async functions:

```python
from saidex import extract_data_from_text, extract_data
```

| Function | Input | Best for |
| --- | --- | --- |
| `extract_data_from_text(llm, schema, text, ...)` | Plain string | Single-shot text → model |
| `extract_data(llm, schema, messages, ...)` | `list[BaseMessage]` | Multi-turn chat → model |

Both return `tuple[ModelT | None, ExtractDataStats]`.

Both accept a `mode` parameter — tool calling (default) or raw JSON for models
without tool-calling support.  See [Extraction Modes](extraction-modes.md).

### `ExtractDataStats`

```python
result, stats = await extract_data_from_text(...)

stats.success           # bool — did the run produce a validated instance?
stats.failure_reason    # FailureReason | None — why it failed (None on success)
stats.llm_error_kind    # ModelErrorKind | None — why the model call failed (llm_error only)
stats.primary_retries   # int — retries on the primary model
stats.fallback_retries  # int — retries on the fallback model
stats.total_retries     # int — sum of both
stats.fallback_used     # bool — did the fallback model run?
stats.salvaged          # bool — result produced by an on_exhausted hook
stats.format_errors     # int — pure parse / tool-call failures
stats.problem_fields    # tuple[str, ...] — fields that ever failed validation
stats.field_issues      # tuple[FieldIssue, ...] — structured per-field problems
stats.last_candidate    # last schema-valid but rejected instance
                        # (None on a clean success; kept when salvaged)
stats.last_rejection    # tuple[FieldIssue, ...] — the issues that rejected it
```

Prefer `stats.success` over `result is not None` to branch on the outcome.
`field_issues` are recorded even on a successful run when an earlier attempt was
self-corrected, so you can see which fields the model struggled with.

> **Breaking change (was `int(stats)`):** the legacy integer shims were removed.
> Use `stats.total_retries` instead of `int(stats)`.

---

## Your first extraction

> **Runnable example:** [`examples/01_extract_from_text.py`](../examples/01_extract_from_text.py)

```python
import asyncio
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from saidex import extract_data_from_text

class PersonInfo(BaseModel):
    name:       str
    age:        int  = Field(ge=0, le=150)
    occupation: str  = Field(description="Current job title or profession")
    city:       str | None = Field(default=None, description="City of residence or null")

async def main() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    person, stats = await extract_data_from_text(
        llm,
        PersonInfo,
        "Alice Müller, 34, works as a software engineer in Munich.",
    )

    if person is None:
        print(f"Extraction failed after {stats.total_retries} retries.")
        return

    print(f"Name:       {person.name}")
    print(f"Age:        {person.age}")
    print(f"Occupation: {person.occupation}")
    print(f"City:       {person.city}")
    print(f"Retries:    {stats.total_retries}")

asyncio.run(main())
```

**Expected output:**
```
Name:       Alice Müller
Age:        34
Occupation: software engineer
City:       Munich
Retries:    0
```

---

## How it works

```
Your text / messages
        │
        ▼
  LLM via tool-calling
  (forced to fill in the schema as a JSON tool call)
        │
        ▼
  Pydantic validates the JSON
        │
        ├─ Valid → return (instance, stats) ✅
        │
        └─ Invalid → format a field-level error message
                │
                ▼
          Append as HumanMessage to the conversation
          "The output is not correct. Please correct …"
                │
                ▼
          LLM gets another attempt (up to max_primary_retries)
                │
                └─ Still failing after all retries?
                   → Try fallback_llm_model (if configured)
                   → Return (None, stats)
```

The retry loop means a single API function call can make multiple LLM
requests internally.  `ExtractDataStats` tells you exactly how many.

---

## Next steps

| Goal | Document |
| --- | --- |
| Pass a multi-turn chat history | [Extraction](extraction.md) |
| Use a fallback model / configure retries | [Retry & Fallback](retry-and-fallback.md) |
| Design a reliable Pydantic schema | [Schema Design](schema-design.md) |
| Clean messy LLM output with validators | [Pydantic Validators](validators.md) |
| Extract from images | [Image Extraction](images.md) |
