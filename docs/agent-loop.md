# Agent Loop — let the LLM call your tools

→ [Documentation index](index.md)

`extract_with_tools` and `run_agent_loop` extend structured extraction with an
**agentic tool loop**: the LLM may call caller-supplied tools any number of
times — look up data, create resources, trigger side effects — and then
delivers a final answer that is validated against your Pydantic schema.

Use it when the answer cannot be produced from the prompt text alone:

- the model needs data that only your application can provide (database rows,
  API responses, file contents),
- a resource has to be created first and its id must appear in the final
  answer,
- the extraction requires a decision based on intermediate lookups.

If the model only needs to read the text you already have, use
[`extract_from_text` / `get_structured_data`](extraction.md) instead — they
are simpler and cheaper.

---

## Quick example

```python
import asyncio
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from saidex import Tool, extract_with_tools

# 1. Describe the tool's arguments with a Pydantic model
class OrderStatusArgs(BaseModel):
    order_id: str = Field(description="Order id, e.g. ORD-1042")

# 2. Provide an async handler — its return value is shown to the LLM
async def order_status_handler(order_id: str) -> dict:
    # ... real lookup ...
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

async def main() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    resolution, stats = await extract_with_tools(
        llm,
        TicketResolution,
        "Customer asks: where is my order ORD-1042?",
        tools=[order_status],
    )

    print(resolution)
    print(f"iterations={stats.iterations} tool_calls={stats.tool_calls}")

asyncio.run(main())
```

> Full runnable example: [`examples/08_agent_tools.py`](../examples/08_agent_tools.py)

---

## How the loop works

```text
┌────────────────────────────────────────────────────────┐
│  LLM  (your tools + a final-answer tool are bound)     │
└───────────────┬────────────────────────────────────────┘
                │
   ┌────────────┴─────────────┐
   │ calls a helper tool?      │──── yes ──► validate args → run handler
   │                           │            append result as ToolMessage
   │                           │            └──► back to the LLM
   │ calls the final-answer    │
   │ tool (or emits JSON)?     │──── yes ──► validate against schema
   └───────────────────────────┘             ├─ valid   → return (instance, stats)
                                             └─ invalid → feed error report back,
                                                          continue the loop
```

1. Every `Tool` is converted to an OpenAI-style function definition and bound
   to the model, together with a final-answer tool generated from `schema`.
2. When the LLM calls a helper tool, the arguments are validated against the
   tool's `parameters` model, the handler runs, and the result is appended to
   the conversation as a `ToolMessage`.
3. When the LLM submits the final answer, it is validated with Pydantic. On
   failure, the field-level error report is sent back and the loop continues.
4. The loop ends with a validated instance — or `None` once `max_iterations`
   or `max_validation_retries` is exhausted (after optionally retrying on a
   fallback model).

Two failure-safety properties worth knowing:

- **Handler exceptions never crash the loop.** They are caught and returned to
  the LLM as the tool result (`"Tool 'x' raised an error: …"`), so the model
  can retry or work around the failure.
- **Invalid tool arguments are reported, not raised.** The LLM receives a
  structured validation message and can correct its call.

---

## The `Tool` dataclass

```python
from saidex import Tool
```

| Field | Type | Description |
| --- | --- | --- |
| `name` | `str` | Function name exposed to the LLM (valid identifier) |
| `description` | `str` | What the tool does — the LLM decides from this when to call it |
| `parameters` | `type[BaseModel]` | Pydantic model whose fields define the arguments; the JSON Schema is generated automatically |
| `handler` | `Callable[..., Awaitable[Any]]` | Async callable receiving the validated arguments as keyword arguments |

The handler's return value becomes the `ToolMessage` content: strings are
passed through, everything else is serialised with `json.dumps` (falling back
to `str`). Return compact, informative results — the LLM has to read them.

```python
class SearchArgs(BaseModel):
    query: str = Field(description="Search term")
    limit: int = Field(default=5, ge=1, le=20, description="Max results")

async def search_handler(query: str, limit: int = 5) -> list[dict]:
    return await my_database.search(query, limit=limit)

search = Tool(
    name="search_products",
    description="Search the product catalogue by free-text query.",
    parameters=SearchArgs,
    handler=search_handler,
)
```

**Tips for reliable tools:**

- Write `description` and field descriptions as carefully as your extraction
  schema — they are the only things the LLM sees.
- Keep results small and structured; trim large payloads to the fields the
  model actually needs.
- Tools may be called multiple times and in any order — make handlers
  idempotent where possible, or say in the description when a tool must be
  called only once.

---

## `extract_with_tools` parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `llm_model` | `Any` | — | LangChain chat model with `bind_tools` support |
| `schema` | `type[ModelT]` | — | Pydantic model for the final answer |
| `text` | `str` | — | Task description (becomes the `HumanMessage`) |
| `tools` | `list[Tool]` | — | Helper tools the LLM may call freely |
| `final_answer_mode` | `ExtractionMode` | `TOOL_CALLING` | How the final answer is collected (see below) |
| `system_prompt` | `str \| None` | auto | System instruction (a generic one is used if omitted) |
| `callbacks` | `list[Any] \| None` | `None` | LangChain callback handlers |
| `fallback_llm_model` | `Any \| None` | `None` | Second model tried when the loop budget is exhausted |
| `max_iterations` | `int` | `12` | Max LLM invocations per model attempt |
| `max_validation_retries` | `int` | `3` | Max final answers that may fail validation |
| `retry_config` | `RetryConfig \| None` | `DEFAULT_RETRY_CONFIG` | Network retry settings |

**Returns:** `tuple[ModelT | None, AgentRunStats]`

---

## `run_agent_loop` — full message control

`extract_with_tools` builds `[SystemMessage, HumanMessage]` for you.
When you need precise control over the conversation — multi-turn history,
few-shot examples, previously gathered context — use `run_agent_loop` with the
same parameters but a ready-built message list:

```python
from langchain_core.messages import HumanMessage, SystemMessage
from saidex import run_agent_loop

messages = [
    SystemMessage(content="You are the filing assistant for project X."),
    HumanMessage(content="Earlier decision: contracts go to folder LEGAL-7."),
    HumanMessage(content="File this document: 'Service agreement with ACME...'"),
]

decision, stats = await run_agent_loop(
    llm, FilingDecision, messages, tools=[create_folder, list_folders]
)
```

---

## `final_answer_mode` — with or without tool calling

| Value | Behaviour |
| --- | --- |
| `ExtractionMode.TOOL_CALLING` (default) | The schema is bound as an additional "final answer" tool; the loop ends when the LLM calls it. Most reliable. |
| `ExtractionMode.JSON` | Only helper tools are bound. When the LLM stops calling tools and replies with plain text, the content is parsed as JSON (tolerant of code fences, `<think>` blocks, and minor JSON damage). |

Note: the **helper tools always require a tool-calling-capable model** — only
the final answer can be collected as raw JSON. For fully tool-free extraction
see [Extraction Modes](extraction-modes.md).

---

## Fallback model

Like `get_structured_data`, the loop accepts a `fallback_llm_model`. When the
primary model exhausts `max_iterations` (or `max_validation_retries`) without
a valid final answer, the fallback model starts over with a fresh conversation
built from the original messages plus a brief hint that a previous attempt
failed. The returned `AgentRunStats` are the **sum of both attempts** with
`fallback_used=True`.

```python
resolution, stats = await extract_with_tools(
    cheap_llm,
    TicketResolution,
    task_text,
    tools=[order_status],
    fallback_llm_model=strong_llm,
)
```

---

## `AgentRunStats`

| Field | Type | Meaning |
| --- | --- | --- |
| `iterations` | `int` | Total LLM invocations in the loop |
| `tool_calls` | `int` | Helper-tool executions (final-answer calls not counted) |
| `validation_retries` | `int` | Final answers that failed schema validation |
| `fallback_used` | `bool` | Whether the fallback model was invoked |

`AgentRunStats` instances support `+` to merge (used internally to combine
primary and fallback attempts).

---

## See also

- [Extraction](extraction.md) — single-shot extraction without tools
- [Extraction Modes](extraction-modes.md) — tool calling vs. raw JSON
- [Retry & Fallback](retry-and-fallback.md) — network retries and fallback details
- [Schema Design](schema-design.md) — applies to tool `parameters` models too
