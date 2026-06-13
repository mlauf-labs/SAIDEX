# Extraction

→ [Documentation index](index.md)

This document covers both extraction entry points: `extract_from_text` for
simple text strings and `get_structured_data` for full LangChain message lists.

---

## `extract_from_text` — plain text → model

> **Runnable example:** [`examples/01_extract_from_text.py`](../examples/01_extract_from_text.py)

`extract_from_text` is the simplest entry point.  It takes a plain string,
wraps it in a `[SystemMessage, HumanMessage]` pair, and calls
`get_structured_data` internally.

```python
from langchain_openai import ChatOpenAI
from saidex import extract_from_text

result, stats = await extract_from_text(
    llm_model,           # any LangChain chat model
    MySchema,            # Pydantic BaseModel subclass
    "your text here",    # the text to analyse
)
```

### Full signature

```python
async def extract_from_text(
    llm_model:           Any,
    schema:              type[ModelT],
    text:                str,
    *,
    mode:                ExtractionMode = ExtractionMode.TOOL_CALLING,
    system_prompt:       str | None = None,
    callbacks:           list[Any] | None = None,
    fallback_llm_model:  Any = None,
    max_primary_retries: int = 3,
    max_fallback_retries: int = 3,
    retry_config:        RetryConfig | None = None,
) -> tuple[ModelT | None, StructuredOutputStats]:
```

> `mode` selects tool calling (default) or raw JSON output — see
> [Extraction Modes](extraction-modes.md).

### Custom system prompt

By default the function generates a generic instruction like:

> *"You are a precise data-extraction assistant. Extract the requested
> information from the user's text and populate the 'MySchema' schema
> exactly."*

Override it when the domain requires specific instructions:

```python
result, stats = await extract_from_text(
    llm,
    Address,
    "Bitte liefern an: Maria Schmidt, Hauptstraße 42, 80331 München",
    system_prompt=(
        "Du bist ein Parser für deutsche Postadressen. "
        "Extrahiere die Adressbestandteile exakt aus dem Text. "
        "Wenn kein Land angegeben ist, verwende 'DE'."
    ),
)
```

The example file [`01_extract_from_text.py`](../examples/01_extract_from_text.py)
shows a complete `PersonInfo` extraction with an optional `city` field and
demonstrates how `stats` is used to detect whether retries were needed.

---

## `get_structured_data` — chat history → model

> **Runnable example:** [`examples/02_chat_history.py`](../examples/02_chat_history.py)

Use `get_structured_data` when you already have a list of LangChain messages —
either because your application builds conversations naturally, or because you
want fine-grained control over the system instruction, context, and prompt.

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from saidex import get_structured_data

messages = [
    SystemMessage(content="You are a support analyst. Analyse the conversation."),
    HumanMessage(content="Here is the transcript:"),
    AIMessage(content="Customer: My order is late!\nAgent: We'll ship a replacement today."),
    HumanMessage(content="Please provide the structured summary now."),
]

result, stats = await get_structured_data(llm, SupportTicketSummary, messages)
```

### Full signature

```python
async def get_structured_data(
    llm_model:           Any,
    schema:              type[ModelT],
    messages:            list[BaseMessage],
    *,
    mode:                ExtractionMode = ExtractionMode.TOOL_CALLING,
    callbacks:           list[Any] | None = None,
    fallback_llm_model:  Any = None,
    max_primary_retries: int = 3,
    max_fallback_retries: int = 3,
    retry_config:        RetryConfig | None = None,
) -> tuple[ModelT | None, StructuredOutputStats]:
```

> `mode` selects tool calling (default) or raw JSON output — see
> [Extraction Modes](extraction-modes.md).

### Message list is never mutated

The library makes an internal copy of `messages` before the first attempt and
resets to the original when switching to the fallback model.  Your list is
always safe to reuse:

```python
base_messages = [SystemMessage(content="Analyse this."), HumanMessage(content=doc)]

result_a, _ = await get_structured_data(llm, SchemaA, base_messages)
result_b, _ = await get_structured_data(llm, SchemaB, base_messages)
# base_messages is unchanged after both calls
```

---

## Typical message patterns

### Pattern 1 — Single document analysis

```python
messages = [
    SystemMessage(content="Extract key facts from the document below."),
    HumanMessage(content=document_text),
]
```

### Pattern 2 — System + user + assistant transcript

Useful when the document you want to analyse was produced by a previous LLM
call (e.g. a summarisation step):

```python
messages = [
    SystemMessage(content="Classify the following support conversation."),
    HumanMessage(content="Conversation transcript:"),
    AIMessage(content=previous_llm_output),
    HumanMessage(content="Now fill in the classification schema."),
]
```

The example [`02_chat_history.py`](../examples/02_chat_history.py) uses exactly
this pattern to classify a customer-support conversation into a
`SupportTicketSummary` schema with `topic`, `sentiment`, `resolved`,
`action_items`, and `priority` fields.

### Pattern 3 — Few-shot examples

Providing one or two filled-in examples dramatically improves accuracy for
unusual or highly domain-specific schemas:

```python
import json

good_example = MySchema(field_a="value", field_b=42)

messages = [
    SystemMessage(content="Extract invoice data from the text."),
    HumanMessage(content="Invoice text: Muster GmbH, 100 EUR, 2026-01-15, paid"),
    AIMessage(content=good_example.model_dump_json()),   # show ideal output
    HumanMessage(content=f"Now extract from this invoice: {new_invoice_text}"),
]
```

### Pattern 4 — Existing conversation + extraction request

When you want to extract structured data from a real user conversation:

```python
# history built up during normal chatbot operation
history: list[BaseMessage] = [...]

# append an extraction request at the end
extraction_messages = history + [
    HumanMessage(
        content=(
            "Based on everything discussed above, please fill in the "
            "OrderSummary schema with the order details you identified."
        )
    )
]

order, stats = await get_structured_data(llm, OrderSummary, extraction_messages)
```

---

## Handling the result

Always check for `None` before using the result:

```python
result, stats = await extract_from_text(llm, MySchema, text)

if result is None:
    # All retries exhausted — log and handle gracefully
    logger.error(
        "Extraction failed after %d retries (fallback used: %s)",
        stats.total_retries,
        stats.fallback_used,
    )
    raise RuntimeError("Could not extract structured data")

# Safe to use here
process(result)
```

For a complete discussion of retry budgets and fallback models see
[Retry & Fallback](retry-and-fallback.md).

---

## Related

- [Retry & Fallback](retry-and-fallback.md) — controlling retries and adding a fallback model
- [Schema Design](schema-design.md) — writing schemas the LLM can fill reliably
- [Image Extraction](images.md) — passing images in messages
