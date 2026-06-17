# SAIDEX — **S**tructured **AI** **D**ata **EX**traction

> **LangChain-native · async · Pydantic validated · tool-calling + JSON modes · agentic tool loop · auto-retry · fallback model · multimodal · small & local models**

[![CI](https://img.shields.io/github/actions/workflow/status/mlauf-labs/saidex/tests.yml?label=CI&logo=github)](https://github.com/mlauf-labs/saidex/actions)
[![PyPI](https://img.shields.io/pypi/v/saidex.svg)](https://pypi.org/project/saidex/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://mlauf-labs.github.io/SAIDEX/)

**[Docs](docs/index.md) · [Install](#installation) · [Quick start](#quick-start) · [Examples](examples/) · [Contributing](#contributing) · [Changelog](CHANGELOG.md)**

---

Extract validated [Pydantic](https://docs.pydantic.dev/) models from LLM responses — with automatic retry, field-level error feedback, an optional fallback model, and an agentic tool loop for extractions that need lookups or side effects first.

`saidex` turns any LangChain-compatible chat model into a reliable structured-data extractor. It drives the LLM via tool-calling (or raw JSON for models without tool-calling support), validates the output with Pydantic, and feeds detailed error messages back to the model for self-correction — all transparently.

**Works with the models you already have** — from GPT-4o down to small open-weight models running locally via [Ollama](https://ollama.com/) or [vLLM](https://docs.vllm.ai/). If the model doesn't support tool calling, `ExtractionMode.JSON` handles it without any code changes on your side.

---

## Why use this library?

### Key advantages

| Advantage | Detail |
| --- | --- |
| **LangChain-native** | Plugs directly into any LangChain chat model — no wrapper, no adapter, no glue code. If you already use LangChain, you're set. |
| **Small & local models supported** | Works with compact open-weight models (Llama, Mistral, Phi, Qwen, …) served locally via Ollama or vLLM — no cloud dependency required. |
| **No tool-calling requirement** | Models that don't support function/tool calling (most local models) work via `ExtractionMode.JSON` — same validation and retry logic, zero extra code. |
| **Cost-optimised** | Pair a cheap small model as primary with a powerful one as fallback — pay for the big model only when the small one can't deliver. |
| **Agentic tool loop** | Give the LLM your own tools (lookups, API calls, side effects) — it calls them in a loop and finishes with a schema-validated final answer. |

### Problems it solves

| Problem | What this library does |
| --- | --- |
| LLMs sometimes return malformed JSON | Detects invalid tool calls and retries automatically |
| LLM output passes JSON parsing but fails schema validation | Sends a detailed, field-level error report back to the LLM |
| One model isn't reliable enough | Configure a fallback model that takes over after the primary exhausts its retries |
| Network errors / rate limits mid-call | Built-in configurable retry with exponential back-off |
| Your model doesn't support tool calling | Switch to `ExtractionMode.JSON` — works with any chat model |
| Debugging is hard | Structured `ExtractDataStats` return value — know exactly how many retries each phase needed |

---

## Installation

```bash
pip install saidex
```

If you use OpenAI models and want automatic rate-limit handling:

```bash
pip install "saidex[openai]"
```

---

## Quick start

### Extract from a text string

```python
import asyncio
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from saidex import extract_data_from_text

class PersonInfo(BaseModel):
    name: str
    age: int
    occupation: str

async def main():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    person, stats = await extract_data_from_text(
        llm,
        PersonInfo,
        "Alice Müller, 34, works as a software engineer in Munich.",
    )

    print(person)                # name='Alice Müller' age=34 occupation='software engineer'
    print(stats.total_retries)   # 0 — first attempt was valid

asyncio.run(main())
```

### Extract from a chat history

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from saidex import extract_data

messages = [
    SystemMessage(content="Analyse the following support conversation."),
    HumanMessage(content="Here is the transcript:"),
    AIMessage(content="Customer: My order is late!\nAgent: We'll ship a replacement today."),
    HumanMessage(content="Provide the structured summary."),
]

summary, stats = await extract_data(llm, SupportTicketSummary, messages)
```

---

## Core concepts

### `extract_data` — the main function

```python
async def extract_data(
    llm_model: Any,
    schema: type[ModelT],
    messages: list[BaseMessage],
    *,
    mode: ExtractionMode = ExtractionMode.TOOL_CALLING,
    callbacks: list[Any] | None = None,
    fallback_llm_model: Any = None,
    max_primary_retries: int = 3,
    max_fallback_retries: int = 3,
    retry_config: RetryConfig | None = None,
) -> tuple[ModelT | None, ExtractDataStats]:
```

Takes a list of `BaseMessage` objects (the full conversation context) and returns a validated `(instance, stats)` tuple.

### `extract_data_from_text` — convenience wrapper

```python
async def extract_data_from_text(
    llm_model: Any,
    schema: type[ModelT],
    text: str,
    *,
    system_prompt: str | None = None,
    ...
) -> tuple[ModelT | None, ExtractDataStats]:
```

Converts a plain string into `[SystemMessage, HumanMessage]` and calls `extract_data`. You can supply a custom `system_prompt`; a sensible default is used otherwise.

### `extract_data_list` — batch extraction → `list[ModelT]`

When one document contains several repeated records (invoice line items, multiple
people in a transcript, products on a page), `extract_data_list` and
`extract_data_list_from_text` return a precisely typed `list[ModelT]` from a
**single** LLM call. Define the schema for *one* item:

```python
from saidex import extract_data_list_from_text

class InvoiceLine(BaseModel):
    description: str
    quantity: int
    unit_price: float

lines, stats = await extract_data_list_from_text(llm, InvoiceLine, document)
# lines: list[InvoiceLine] | None
print(f"Extracted {stats.item_count} items")
```

The per-item schema is wrapped in a one-field container internally, so both
extraction modes, per-item validation (errors point at `items -> 2 -> quantity`),
retries, and fallback all work unchanged. See
**[Batch Extraction](docs/batch-extraction.md)**.

### Extraction modes — with or without tool calling

By default the library forces the model to call a bound tool. If your model
does not support tool calling (e.g. many local models served by Ollama or
llama.cpp), switch to `ExtractionMode.JSON`: the schema's JSON Schema is
injected into the prompt and the model's raw JSON reply is parsed and
validated instead.

```python
from saidex import ExtractionMode, extract_data_from_text

# Default — tool calling
result, stats = await extract_data_from_text(llm, MySchema, text)

# No tool calling required — model replies with raw JSON
result, stats = await extract_data_from_text(
    llm, MySchema, text, mode=ExtractionMode.JSON
)
```

Both modes share the same validation, error-feedback, retry, and fallback
behaviour. JSON mode tolerates markdown code fences and stray text around the
JSON object. See **[Extraction Modes](docs/extraction-modes.md)** for details.

### Retry strategy

```
┌───────────────────────────────────────────────┐
│                 Primary model                  │
│  Attempt 1 → Attempt 2 → ... → Attempt N      │
│  (validation errors fed back as messages)      │
└──────────────────────┬────────────────────────┘
                       │ all attempts failed
                       ▼
┌───────────────────────────────────────────────┐
│            Fallback model (optional)           │
│  Attempt 1 → Attempt 2 → ... → Attempt M      │
│  (fresh context + hint about previous failure) │
└───────────────────────────────────────────────┘
```

When a validation attempt fails, the library appends a **detailed error message** to the conversation — covering missing fields, type errors, enum violations, and constraint failures — so the model knows exactly what to fix.

### `ExtractDataStats`

```python
result, stats = await extract_data(...)

stats.success           # bool — did the run produce a validated instance?
stats.failure_reason    # str | None — why it failed (None on success)
stats.primary_retries   # int — retries against the primary model
stats.fallback_retries  # int — retries against the fallback model
stats.total_retries     # int — sum of both
stats.fallback_used     # bool — was the fallback model invoked?
stats.item_count        # int — items returned by extract_data_list (else 0)
stats.format_errors     # int — pure parse / tool-call failures
stats.problem_fields    # tuple[str, ...] — fields that ever failed validation
stats.field_issues      # tuple[FieldIssue, ...] — structured per-field problems

if not stats.success:
    print(stats.failure_reason, stats.problem_fields)
```

> **Breaking change:** `int(stats)` and the comparison shims were removed —
> use `stats.total_retries`. `field_issues` are kept even on a successful run
> when an earlier attempt was self-corrected.

---

## Agentic tool loop

Sometimes an extraction needs more than the text in front of the model — a
database lookup, an API call, or a resource that must be created first.
`extract_data_with_tools` runs an agent loop: the LLM may call your tools any
number of times, then delivers a final answer validated against your schema.

```python
from pydantic import BaseModel, Field
from saidex import Tool, extract_data_with_tools

# 1. Describe the tool's arguments with a Pydantic model
class OrderStatusArgs(BaseModel):
    order_id: str = Field(description="Order id, e.g. ORD-1042")

# 2. Provide an async handler — its result is shown to the LLM
async def order_status_handler(order_id: str) -> dict:
    return {"order_id": order_id, "status": "shipped", "eta": "2026-06-15"}

order_status = Tool(
    name="get_order_status",
    description="Look up the current status and ETA of an order.",
    parameters=OrderStatusArgs,
    handler=order_status_handler,
)

# 3. Define the final-answer schema and run the loop
class TicketResolution(BaseModel):
    order_id: str
    current_status: str
    customer_reply: str = Field(description="Friendly reply for the customer")

resolution, stats = await extract_data_with_tools(
    llm,
    TicketResolution,
    "Customer asks: where is my order ORD-1042?",
    tools=[order_status],
)

print(resolution.customer_reply)
print(stats.iterations, stats.tool_calls)  # e.g. 2 1
```

How it works:

1. Your `Tool`s are bound to the model alongside a final-answer tool built from `schema`.
2. The LLM calls helper tools as needed; every result is appended to the conversation as a `ToolMessage`.
3. When the LLM submits the final answer, it is validated with Pydantic — on failure the field-level error report is fed back and the loop continues.
4. The loop ends with a validated instance, or `None` once `max_iterations` / `max_validation_retries` are exhausted (optionally retried on a fallback model first).

Good to know:

- Handler exceptions and invalid tool arguments don't crash the loop — they are returned to the LLM as tool output so it can self-correct.
- Models without tool-calling support can deliver the final answer as raw JSON via `final_answer_mode=ExtractionMode.JSON` (the helper tools themselves always require tool calling).
- Need full control over the conversation (multi-turn, prior context)? Use `run_extractor_agent(llm, schema, messages, tools=...)` — same behaviour, but takes a ready-built message list.

### `ExtractorRunStats`

```python
stats.iterations          # int  — LLM invocations in the loop
stats.tool_calls          # int  — helper-tool executions
stats.validation_retries  # int  — final answers that failed validation
stats.fallback_used       # bool — was the fallback model invoked?
```

---

## Examples

### With a fallback model

Use a cheaper primary model and fall back to a more capable one only when needed:

```python
from langchain_openai import ChatOpenAI
from saidex import extract_data_from_text

primary  = ChatOpenAI(model="gpt-4o-mini", temperature=0)
fallback = ChatOpenAI(model="gpt-4o",      temperature=0)

result, stats = await extract_data_from_text(
    primary,
    FinancialReport,
    report_text,
    fallback_llm_model=fallback,
    max_primary_retries=2,
    max_fallback_retries=3,
)

if stats.fallback_used:
    print(f"Fallback needed after {stats.primary_retries} primary retries.")
```

### Custom retry configuration

```python
from saidex import RetryConfig, extract_data_from_text

config = RetryConfig(
    max_retries=2,
    retry_delays=[2.0, 5.0],          # exponential-ish back-off
    rate_limit_retry_interval=30.0,
    rate_limit_max_duration_seconds=300.0,
)

result, stats = await extract_data_from_text(
    llm, MySchema, text, retry_config=config
)
```

### Using `create_instance_safe` standalone

Need safe Pydantic instantiation with rich error messages but without an LLM?

```python
from saidex import create_instance_safe

instance, error = create_instance_safe(MySchema, **data)
if instance is None:
    print(error)  # structured, human-readable validation report
```

### LangChain callbacks (Langfuse, LangSmith, …)

Any LangChain `BaseCallbackHandler` can be passed via `callbacks`:

```python
from langfuse.callback import CallbackHandler

result, stats = await extract_data(
    llm,
    MySchema,
    messages,
    callbacks=[CallbackHandler()],
)
```

---

## Multimodal: Extract from images

The library works with any vision-capable LLM — cloud models such as GPT-4o as
well as locally hosted models via **vLLM**. Images are passed as standard
LangChain multimodal messages.

### Helper functions

Two small helpers (shown below, also in [`examples/05_image_analysis.py`](examples/05_image_analysis.py))
build the right message format for you:

```python
def image_message_from_url(url: str, prompt: str = "...", *, detail: str = "high") -> HumanMessage:
    ...

def image_message_from_file(path: str | Path, prompt: str = "...", *, media_type: str | None = None) -> HumanMessage:
    ...
```

Both return a `HumanMessage` with a text block and an image block that any
vision model understands.

### Extract from an image URL (OpenAI GPT-4o)

```python
import asyncio
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from saidex import extract_data

class ReceiptItem(BaseModel):
    description: str
    total: float

class Receipt(BaseModel):
    vendor: str
    date: str | None = None
    items: list[ReceiptItem]
    total: float
    currency: str = "EUR"

def image_message_from_url(url: str, prompt: str) -> HumanMessage:
    return HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": url, "detail": "high"}},
    ])

async def main():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    messages = [
        SystemMessage(content="Extract all data from the receipt image precisely."),
        image_message_from_url(
            "https://example.com/receipt.jpg",
            prompt="Extract the receipt data.",
        ),
    ]

    receipt, stats = await extract_data(llm, Receipt, messages)
    print(receipt)

asyncio.run(main())
```

### Extract from a local image file (base64)

```python
import base64
from pathlib import Path

def image_message_from_file(path: str | Path, prompt: str) -> HumanMessage:
    path = Path(path)
    mime_map = {".jpg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    mime = mime_map.get(path.suffix.lower(), "image/jpeg")
    b64 = base64.b64encode(path.read_bytes()).decode()
    return HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
    ])

messages = [
    SystemMessage(content="Extract contact details from the business card."),
    image_message_from_file("card.jpg", prompt="Read this business card."),
]

result, stats = await extract_data(llm, BusinessCard, messages)
```

### Use a local vLLM vision model

[vLLM](https://docs.vllm.ai/) exposes an OpenAI-compatible API, so you simply
point `ChatOpenAI` at your local server.

**1. Start the vLLM server**

```bash
pip install vllm
vllm serve Qwen/Qwen2-VL-7B-Instruct --port 8000
```

Popular vision models supported by vLLM:

| Model | Strengths |
| --- | --- |
| `Qwen/Qwen2-VL-7B-Instruct` | Strong OCR, multilingual, recommended |
| `Qwen/Qwen2.5-VL-7B-Instruct` | Updated Qwen VL, better reasoning |
| `llava-hf/llava-1.5-7b-hf` | Lightweight, fast |
| `microsoft/Phi-3.5-vision-instruct` | Compact, efficient |
| `OpenGVLab/InternVL2-8B` | Good at charts and documents |

**2. Connect and extract**

```python
from langchain_openai import ChatOpenAI
from saidex import extract_data

vllm = ChatOpenAI(
    model="Qwen/Qwen2-VL-7B-Instruct",  # must match the loaded model
    base_url="http://localhost:8000/v1",  # vLLM default endpoint
    api_key="EMPTY",                      # vLLM needs no real API key
    temperature=0,
    max_tokens=1024,
)

messages = [
    SystemMessage(content="Extract the chart data precisely."),
    image_message_from_url(chart_url, prompt="Analyse this chart."),
]

result, stats = await extract_data(vllm, ChartData, messages)
```

### Multiple images in one request

```python
def multi_image_message(urls: list[str], prompt: str) -> HumanMessage:
    content = [{"type": "text", "text": prompt}]
    for url in urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    return HumanMessage(content=content)

messages = [
    SystemMessage(content="Compare the two product labels."),
    multi_image_message([url1, url2], prompt="Analyse both labels."),
]

result, stats = await extract_data(llm, ComparisonSchema, messages)
```

> See [`examples/05_image_analysis.py`](examples/05_image_analysis.py) for the
> complete runnable example including a fallback model and all helper functions.

---

## Schema design guide

A well-designed Pydantic schema is the single most important factor for
reliable extraction. Two rules cover most of the ground:

1. **Use the right types** — the schema becomes a JSON Schema that the LLM must
   fill in; unclear types produce unclear output.
2. **Describe every field** — the LLM reads `Field(description=...)` to
   understand *what* to put there.

### Supported field types

| Type | Example | Notes |
| --- | --- | --- |
| `str` | `name: str` | Plain text |
| `int` | `age: int` | Whole number |
| `float` | `price: float` | Decimal number |
| `bool` | `active: bool` | `true` / `false` |
| `str \| None` | `nickname: str \| None = None` | Nullable — use `None` as default |
| `list[str]` | `tags: list[str]` | Zero-or-more strings |
| `list[MyModel]` | `items: list[LineItem]` | Nested model list |
| `dict[str, str]` | `metadata: dict[str, str]` | Simple string map |
| `dict[str, list[str]]` | `groups: dict[str, list[str]]` | Map with list values |
| `StrEnum` / `str` Enum | `status: Status` | Constrains value to fixed set |
| `Literal["a", "b"]` | `unit: Literal["kg", "g"]` | Inline allowed-values list |
| Nested `BaseModel` | `address: Address` | Structured sub-object |
| `Field(ge=0, le=100)` | `score: int = Field(ge=0, le=100)` | Numeric constraints |
| `Field(min_length=1)` | `title: str = Field(min_length=1)` | String length constraints |

### ⚠️ Types to avoid

> **Never use `object`, bare `dict`, or `Any` as a field type.**

```python
# BAD — the LLM has no idea what structure to produce.
# Pydantic accepts anything, so validation never catches wrong output.
class Bad(BaseModel):
    data: object          # ❌
    extra: dict           # ❌
    value: Any            # ❌
    payload: dict[str, Any]  # ❌ — the Any part is the problem

# GOOD — replace with a typed model or a concrete dict type
class Good(BaseModel):
    data: MySubModel              # ✅ explicit structure
    extra: dict[str, str]         # ✅ both key and value typed
    count: int                    # ✅ concrete type
    payload: dict[str, list[str]] # ✅ fully typed
```

Avoid `datetime`, `date`, and `time` objects — many models struggle with
Python's ISO 8601 serialisation. Use `str` with a format hint in the
description instead:

```python
# BAD
from datetime import date
due: date           # ❌ — serialisation surprises

# GOOD
due: str | None = Field(default=None, description="Due date as YYYY-MM-DD, or null")  # ✅
```

### Field descriptions are essential

The `description` string is included in the JSON Schema that the LLM receives.
It is the primary way to tell the model *exactly* what to put in a field.

```python
# BAD — the LLM has to guess the meaning, unit, and format
class Invoice(BaseModel):
    amount: float       # ❌ — which currency? incl. tax? total or subtotal?
    date: str           # ❌ — which format? invoice date or due date?

# GOOD — the LLM knows exactly what is expected
class Invoice(BaseModel):
    amount: float = Field(
        description="Total invoice amount in EUR including VAT, e.g. 119.00"
    )
    date: str = Field(
        description="Invoice issue date in ISO 8601 format: YYYY-MM-DD"
    )
```

**Tips for writing good descriptions:**

- State the **unit** (`EUR`, `kg`, `%`, `seconds`)
- State the **format** for strings (`YYYY-MM-DD`, `HH:MM`, `E.164 phone`)
- Give a **concrete example** (`e.g. 42.50`, `e.g. "2026-01-31"`)
- Clarify **ambiguous names** (`"subtotal before tax"` vs `"total including VAT"`)
- For optional fields, state what `null` means (`"null if not mentioned"`)

### Complete well-designed schema example

```python
from enum import Enum
from pydantic import BaseModel, Field

class Priority(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"

class Subtask(BaseModel):
    title: str = Field(description="Short action item, max one sentence")
    done: bool = Field(default=False, description="true if already completed")

class Task(BaseModel):
    title: str = Field(description="Short task title, max 80 characters")
    description: str = Field(description="Full description of what needs to be done")
    priority: Priority = Field(description="Urgency level: low / medium / high / critical")
    due_date: str | None = Field(
        default=None,
        description="Deadline as YYYY-MM-DD, or null if no deadline is mentioned",
    )
    estimated_hours: float | None = Field(
        default=None,
        ge=0,
        description="Estimated effort in hours, or null if unknown",
    )
    subtasks: list[Subtask] = Field(
        default_factory=list,
        description="List of smaller steps; empty list if none mentioned",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Short lowercase labels, e.g. ['backend', 'urgent']",
    )
```

> For a complete reference including edge cases see [`docs/schema-design.md`](docs/schema-design.md).

---

## Pydantic validators — clean and verify LLM output

LLMs are inconsistent: a price might come back as `"€ 1.234,56"`, a date as
`"05.01.2026"`, a boolean as `"ja"`. Pydantic validators let you fix this
**transparently** — before your application code sees the data and without
needing extra retries.

| Validator | When it runs | Typical use |
| --- | --- | --- |
| `@field_validator(mode='before')` | Before type coercion | Strip `€`, normalise `"05.01.2026"` → `"2026-01-05"`, map `"ja"` → `True` |
| `@field_validator(mode='after')` | After type coercion | Lowercase emails, uppercase country codes, prepend `https://` |
| `@model_validator(mode='before')` | Before any field | Parse a flat string into a dict structure |
| `@model_validator(mode='after')` | After all fields | Cross-field rules: `end_date > start_date`, total matches line items |
| `@computed_field` | Read-only property | Derive `subtotal`, `price_gross` — never asked from the LLM |

### Example — clean messy LLM amounts and dates

```python
from typing import Any
import re
from pydantic import BaseModel, Field, field_validator

class Invoice(BaseModel):
    amount: float = Field(ge=0, description="Total in EUR")
    date: str      = Field(description="Issue date as YYYY-MM-DD")
    paid: bool

    @field_validator("amount", mode="before")
    @classmethod
    def clean_amount(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = re.sub(r"[€$£\s]", "", v).replace(",", ".")
        return v

    @field_validator("date", mode="before")
    @classmethod
    def normalise_date(cls, v: Any) -> Any:
        if isinstance(v, str):
            m = re.match(r"^(\d{1,2})[./](\d{1,2})[./](\d{4})$", v.strip())
            if m:
                return f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"
        return v

    @field_validator("paid", mode="before")
    @classmethod
    def normalise_bool(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.lower() in {"yes", "ja", "true", "1", "paid"}
        return v

# All these LLM outputs are accepted and normalised:
Invoice(amount="€ 1.190,00", date="05.01.2026", paid="ja")
Invoice(amount="$42.50",     date="31/12/2026", paid="yes")
Invoice(amount=99.0,         date="2026-03-15", paid=False)
```

### Example — cross-field consistency (`@model_validator`)

```python
from pydantic import BaseModel, model_validator, computed_field

class LineItem(BaseModel):
    quantity: int
    unit_price: float

    @computed_field
    @property
    def subtotal(self) -> float:
        return round(self.quantity * self.unit_price, 2)

class Order(BaseModel):
    order_date: str
    delivery_date: str | None = None
    items: list[LineItem]
    declared_total: float

    @model_validator(mode="after")
    def check_dates(self) -> "Order":
        if self.delivery_date and self.delivery_date < self.order_date:
            raise ValueError("delivery_date must not be before order_date")
        return self

    @model_validator(mode="after")
    def check_total(self) -> "Order":
        calculated = round(sum(i.subtotal for i in self.items), 2)
        if abs(self.declared_total - calculated) / max(calculated, 0.01) > 0.01:
            raise ValueError(
                f"declared_total {self.declared_total} differs from "
                f"calculated {calculated} by more than 1 %"
            )
        return self
```

When a validator raises `ValueError`, `create_instance_safe` catches it,
formats it into a structured error message, and — inside
`extract_data` — sends it back to the LLM so it can self-correct.

> Full example with all four validator types:
> [`examples/06_pydantic_validators.py`](examples/06_pydantic_validators.py)
> — [detailed docs](docs/validators.md)

For common cases the library ships **ready-made field types** so you don't have
to write a validator at all — e.g. `IsoDateStr` for `yyyy-mm-dd` dates, plus
`IbanStr`, `VatIdStr`, `CountryCodeStr`, `CurrencyCodeStr`, `IsinStr`,
`PhoneStr`, and `LanguageCodeStr` (and `RRuleStr` for RFC 5545 recurrence rules,
via the optional `saidex[rrule]` extra):

```python
from saidex import IsoDateStr, IbanStr, CountryCodeStr

class Payment(BaseModel):
    due_date: IsoDateStr | None = Field(None, description="Due date as yyyy-mm-dd")
    iban: IbanStr | None = Field(None, description="Payee IBAN")
    country: CountryCodeStr | None = Field(None, description="ISO 3166-1 alpha-2 code")
```

> Full list and a guide to building your own:
> [`docs/built-in-types.md`](docs/built-in-types.md)

For checks that span fields, hit a database, or apply a business rule, pass an
**external `validator`** callable (sync or async) to any extraction function. It
runs after Pydantic validation; raising — or returning an error string — feeds the
message back into the same retry loop so the model can self-correct:

```python
def validate_invoice(inv: Invoice) -> None:
    if inv.total != sum(line.amount for line in inv.lines):
        raise ValueError("Line items do not sum to the stated total")

invoice, stats = await extract_data_from_text(llm, Invoice, text, validator=validate_invoice)
```

> Cross-field example and full guide:
> [`examples/12_external_validator.py`](examples/12_external_validator.py)
> — [detailed docs](docs/external-validators.md)

---

## Source grounding — reject hallucinated values

To make sure the model only returns values that are actually in the document,
mark a field as **grounded**. After Pydantic validation, SAIDEX checks that the
value appears in the source text; if it doesn't, the same retry loop asks the
model to correct it. Both surfaces are visible right in the schema:

```python
from typing import Annotated
from saidex import Grounded, GroundedField

class Invoice(BaseModel):
    vendor: Annotated[str, Grounded()]                    # marker form
    invoice_no: str = GroundedField(description="No.")     # field-helper form
    country: str
    total: float = GroundedField(locale_field="country")   # locale-aware matching
```

Matching is case/whitespace/diacritics-insensitive and **locale-aware**: a
`total` of `1234.5` is matched against `1.234,50` (de) or `1,234.50` (en), with
the locale taken from a sibling field (`locale_field`) or a fixed `locale`.
It is built on a reusable `FieldCheck` pattern, so you can add your own
source-aware checks the same way.

> Full guide and the custom-check pattern:
> [`examples/13_source_grounding.py`](examples/13_source_grounding.py)
> — [detailed docs](docs/source-grounding.md)

---

## API reference

### `extract_data`

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `llm_model` | `Any` | — | LangChain chat model (`.bind_tools()` required for tool-calling mode) |
| `schema` | `type[ModelT]` | — | Pydantic `BaseModel` subclass |
| `messages` | `list[BaseMessage]` | — | Conversation context |
| `mode` | `ExtractionMode` | `TOOL_CALLING` | `TOOL_CALLING` or `JSON` (no tool calling) |
| `callbacks` | `list[Any] \| None` | `None` | LangChain callback handlers |
| `fallback_llm_model` | `Any \| None` | `None` | Second model tried on primary failure |
| `max_primary_retries` | `int` | `3` | Validation retries for primary model |
| `max_fallback_retries` | `int` | `3` | Validation retries for fallback model |
| `retry_config` | `RetryConfig \| None` | `DEFAULT_RETRY_CONFIG` | Network retry settings |
| `validator` | `Validator[ModelT] \| None` | `None` | External sync/async check run after Pydantic validation; rejecting re-enters the retry loop ([docs](docs/external-validators.md)) |

**Returns:** `tuple[ModelT | None, ExtractDataStats]`

---

### `extract_data_from_text`

All parameters of `extract_data` plus:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `text` | `str` | — | Text to analyse |
| `mode` | `ExtractionMode` | `TOOL_CALLING` | `TOOL_CALLING` or `JSON` (no tool calling) |
| `system_prompt` | `str \| None` | auto | System instruction (auto-generated if omitted) |

### `ExtractionMode`

| Value | Description |
| --- | --- |
| `ExtractionMode.TOOL_CALLING` | Bind the schema as a tool and force the model to call it. Requires a tool-calling-capable model. **Default.** |
| `ExtractionMode.JSON` | Inject the JSON Schema into the prompt and parse the model's raw JSON reply. Works with any chat model. |

---

### `extract_data_with_tools`

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `llm_model` | `Any` | — | LangChain chat model with `bind_tools` support |
| `schema` | `type[ModelT]` | — | Pydantic model for the final answer |
| `text` | `str` | — | Task description (becomes the `HumanMessage`) |
| `tools` | `list[Tool]` | — | Helper tools the LLM may call freely |
| `final_answer_mode` | `ExtractionMode` | `TOOL_CALLING` | How the final answer is collected |
| `system_prompt` | `str \| None` | auto | System instruction (auto-generated if omitted) |
| `callbacks` | `list[Any] \| None` | `None` | LangChain callback handlers |
| `fallback_llm_model` | `Any \| None` | `None` | Second model tried when the loop budget is exhausted |
| `max_iterations` | `int` | `12` | Max LLM invocations per model |
| `max_validation_retries` | `int` | `3` | Max final answers that may fail validation |
| `retry_config` | `RetryConfig \| None` | `DEFAULT_RETRY_CONFIG` | Network retry settings |
| `validator` | `Validator[ModelT] \| None` | `None` | External sync/async check run on the final answer; rejecting keeps the agent loop running ([docs](docs/external-validators.md)) |

**Returns:** `tuple[ModelT | None, ExtractorRunStats]`

`run_extractor_agent` accepts the same parameters but takes a full
`messages: list[BaseMessage]` instead of `text` / `system_prompt`.

---

### `Tool`

| Field | Type | Description |
| --- | --- | --- |
| `name` | `str` | Function name exposed to the LLM |
| `description` | `str` | What the tool does — shown to the LLM |
| `parameters` | `type[BaseModel]` | Pydantic schema describing the tool's arguments |
| `handler` | async callable | Receives validated arguments as kwargs; the result is serialised into a `ToolMessage` |

---

### `RetryConfig`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `max_retries` | `int` | `3` | Max transient-error retries |
| `retry_delays` | `list[float]` | `[1.0, 2.0, 4.0]` | Per-attempt delays in seconds |
| `retryable_exceptions` | `tuple[type[Exception], ...]` | httpx + openai errors | Exceptions that trigger a transient retry |
| `rate_limit_exceptions` | `tuple[type[Exception], ...]` | `openai.RateLimitError` | Exceptions that trigger rate-limit retry |
| `rate_limit_retry_interval` | `float` | `10.0` | Seconds between rate-limit retries |
| `rate_limit_max_duration_seconds` | `float` | `900.0` | Maximum total wait for rate-limit retries |

---

## Requirements

- Python 3.10+
- `pydantic >= 2.0`
- `langchain-core >= 0.2`
- Any LangChain chat model. Tool-calling mode (default) needs a tool/function-calling-capable model (OpenAI, Anthropic, Gemini, Mistral, …); `ExtractionMode.JSON` works with any chat model, including local models without tool-calling support.

---

## Project structure

```
saidex/
├── src/saidex/   # library source
│   ├── __init__.py              # public API
│   ├── extractor.py             # extract_data, extract_data_from_text + agent loop
│   ├── models.py                # ExtractDataStats, ExtractorRunStats, ExtractionMode
│   ├── retry.py                 # RetryConfig + with_retry
│   ├── tools.py                 # Tool dataclass for the agent loop
│   ├── utils.py                 # create_instance_safe
│   └── py.typed                 # PEP 561 marker (enables type checking by consumers)
├── examples/                    # runnable usage examples
├── tests/                       # pytest test suite
├── docs/                        # topic-specific documentation (see docs/index.md)
├── pyproject.toml               # project metadata + tool config
├── uv.lock                      # pinned dependency tree (committed to git)
└── .python-version              # Python version pin for uv (3.12)
```

---

## Development

This project uses [**uv**](https://docs.astral.sh/uv/) for dependency and virtual-environment management.
uv is a fast, modern replacement for `pip` + `venv` + `pip-tools`, written in Rust.

### First-time setup

```bash
# 1. Install uv (if not already installed)
pip install uv
# or on macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repository
git clone https://github.com/mlauf-labs/saidex
cd saidex

# 3. Create the virtual environment and install all dependencies
uv sync
```

`uv sync` reads `uv.lock` and installs the exact pinned versions into `.venv`.
The virtual environment is created automatically — no manual `python -m venv` needed.

### Daily commands

```bash
uv run pytest                          # run the full test suite
uv run pytest tests/test_utils.py -v   # run a single test file
uv run ruff check src/ tests/          # lint
uv run ruff check src/ tests/ --fix    # lint + auto-fix
uv run ruff format src/ tests/         # format code
uv run mypy src/                       # static type checking
```

### Managing dependencies

```bash
# Add a runtime dependency (updates pyproject.toml + uv.lock)
uv add langchain-openai

# Add a dev-only dependency
uv add --group dev ipython

# Add an optional extra (e.g. openai rate-limit support)
uv add --optional openai openai

# Remove a dependency
uv remove langchain-openai

# Upgrade all dependencies to their latest allowed versions
uv lock --upgrade
uv sync
```

### Build and publish

```bash
# Build source distribution + wheel into dist/
uv build

# Publish to PyPI (requires PYPI_TOKEN or interactive login)
uv publish
```

### About `uv.lock`

`uv.lock` is a machine-generated file that pins the **exact version** of every
direct and transitive dependency. It is committed to the repository so that
every developer and CI run installs the identical package tree.

To update all dependencies to their latest allowed versions:

```bash
uv lock --upgrade   # re-solves the dependency graph
uv sync             # applies the new lock file to .venv
```

---

## Contributing

Contributions are welcome! Please open an issue first to discuss significant changes.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, the
test/lint/type-check commands, and the pull-request checklist.

Planned work and feature ideas are tracked as [GitHub issues](https://github.com/mlauf-labs/saidex/issues).

> Have a use case SAIDEX doesn't cover yet? [Open an issue](https://github.com/mlauf-labs/saidex/issues/new) — we'd love to hear about it.

---

## Security

If you discover a security vulnerability, please **do not** open a public issue.
Report it privately via
[GitHub Security Advisories](https://github.com/mlauf-labs/saidex/security/advisories/new) —
see [SECURITY.md](SECURITY.md) for details.

---

## License

Copyright 2026 Martin Lauff.
Licensed under the [Apache License, Version 2.0](LICENSE).
