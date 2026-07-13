# Extraction Modes

→ [Documentation index](index.md)

> **Runnable example:** [`examples/07_json_mode.py`](../examples/07_json_mode.py)

The library supports two strategies for getting structured output out of an
LLM.  Both are selected with the `mode` parameter on `extract_data_from_text` and
`extract_data`.

```python
from saidex import ExtractionMode, extract_data_from_text

# Tool calling (default)
result, stats = await extract_data_from_text(llm, MySchema, text)

# Raw JSON — no tool calling required
result, stats = await extract_data_from_text(
    llm, MySchema, text, mode=ExtractionMode.JSON
)
```

---

## `ExtractionMode.TOOL_CALLING` (default)

The schema is bound as an OpenAI-style tool and the model is **forced** to
call it (`bind_tools(..., tool_choice=...)`).  The arguments of that tool call
are the structured output.

| | |
| --- | --- |
| **Requires** | A model that supports tool / function calling |
| **Reliability** | Highest — the provider guarantees a JSON tool call |
| **Schema visibility** | Sent as a tool definition (not in the prompt text) |
| **Best for** | OpenAI, Anthropic, Gemini, Mistral, Azure OpenAI, … |

```python
result, stats = await extract_data(
    llm, MySchema, messages,
    mode=ExtractionMode.TOOL_CALLING,   # this is the default
)
```

### OpenAI-compatible gateways (`ToolCallConfig`)

By default the schema tool is bound with OpenAI's most reliable flag set: a
**forced** `tool_choice`, `strict=True` (structured outputs) and
`parallel_tool_calls=False`.  Several OpenAI-*compatible* gateways — endpoints
hosting Kimi/Moonshot, some Qwen or DeepSeek deployments — reject one or more of
those flags with an HTTP 400, even though the model itself calls tools perfectly
well.

`ToolCallConfig` makes each flag configurable:

```python
from saidex import ToolCallConfig, extract_data

result, stats = await extract_data(
    llm, MySchema, messages,
    tool_config=ToolCallConfig.COMPATIBLE,
)
```

| Preset | `tool_choice` | `strict` | `parallel_tool_calls` |
| --- | --- | --- | --- |
| `ToolCallConfig.OPENAI` (default) | forced (the tool's name) | `True` | `False` |
| `ToolCallConfig.COMPATIBLE` | `"auto"` | *not sent* | *not sent* |

`None` means **the keyword argument is left out of the request entirely** — which
is not the same as sending `False`.  Some gateways already fail on the mere
presence of `strict`; others accept the parameter but refuse the value `True`.
Both cases are expressible:

```python
# Gateway knows `strict` but only supports `false`
tool_config=ToolCallConfig(strict=False)

# Gateway 400s on the parameter itself — omit it, keep everything else
tool_config=ToolCallConfig(strict=None)
```

### Automatic recovery

You usually do not have to configure anything.  When a provider rejects the
bound flags — an HTTP 400 naming `strict` / `tool_choice` /
`parallel_tool_calls`, or a `TypeError` from a `bind_tools` implementation that
never accepted them — SAIDEX re-binds the schema tool **once** with
`ToolCallConfig.COMPATIBLE`, logs a warning, and carries on **without consuming a
validation retry**.  A second failure ends the attempt as `llm_error` as before.

Switch the behaviour off with `ToolCallConfig(auto_relax=False)` when you want a
hard failure instead of a silently degraded request.

`tool_config` is ignored in `ExtractionMode.JSON` mode, which never calls
`bind_tools`.

---

## `ExtractionMode.JSON`

The schema's JSON Schema is injected into the prompt and the model is asked to
reply with a single raw JSON object.  The library parses the response
**content** directly — `bind_tools` is never called.

| | |
| --- | --- |
| **Requires** | Any chat model with `.ainvoke()` — no tool calling needed |
| **Reliability** | High, but depends on the model following instructions |
| **Schema visibility** | Embedded as JSON Schema in an appended prompt message |
| **Best for** | Local models (Ollama, llama.cpp), older APIs, non-tool models |

```python
result, stats = await extract_data(
    llm, MySchema, messages,
    mode=ExtractionMode.JSON,
)
```

### What gets added to the prompt

In JSON mode the library appends one `HumanMessage` before invoking the model:

```
Respond with a SINGLE JSON object that strictly conforms to the JSON Schema
below. Output ONLY the raw JSON object — no markdown code fences, no comments,
and no explanatory text before or after it.

JSON Schema for 'MySchema':
{ ... full JSON Schema ... }
```

### Robust parsing

Real models do not always follow "raw JSON only" perfectly.  The JSON-mode
parser is tolerant of common deviations:

- **Markdown fences** — ```` ```json ... ``` ```` blocks are stripped.
- **Surrounding text** — if the content has stray text around the object, the
  outermost `{ ... }` block is isolated and parsed.
- **Chunked / multimodal content** — list-style content is concatenated from
  its text parts before parsing.  Which blocks count is decided by the `text`
  key, not by the block's `type` name, so provider-specific names all work:
  LangChain's `{"type": "text", …}` and the Responses API's
  `{"type": "output_text", …}` alike.
- **Reasoning blocks** — `reasoning` / `thinking` blocks are skipped, even when
  they carry their payload under a `text` key, so a model's private chain of
  thought never reaches the JSON parser.

If parsing still fails, or the JSON is not an object, the model receives a
correction message and the normal [retry loop](retry-and-fallback.md) applies —
exactly like a validation failure.

---

## When to use which mode

| Situation | Recommended mode |
| --- | --- |
| OpenAI / Anthropic / Gemini / Mistral | `TOOL_CALLING` (default) |
| OpenAI-compatible gateway (Kimi, Qwen, DeepSeek, …) | `TOOL_CALLING` with [`ToolCallConfig.COMPATIBLE`](#openai-compatible-gateways-toolcallconfig) |
| Local model via Ollama / llama.cpp | `JSON` |
| Model/endpoint without tool-calling support | `JSON` |
| Tool calling behaves inconsistently for a provider | `JSON` (as a workaround) |
| Maximum structural reliability | `TOOL_CALLING` |

Both modes share everything else: validation, the field-level error feedback
loop, validation retries, the optional [fallback model](retry-and-fallback.md),
network [`RetryConfig`](retry-and-fallback.md#network-retries-retryconfig),
callbacks, and statistics.  The `mode` applies to both the primary and fallback
models.

---

## Tips for JSON mode

### Use a low temperature

```python
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

Deterministic output makes the model far more likely to produce clean JSON.

### Keep schemas focused

Very large or deeply nested schemas produce a large JSON Schema in the prompt
and are harder for weaker models to satisfy.  See [Schema Design](schema-design.md)
for guidance — it matters even more in JSON mode.

### Combine with a fallback model

A small local model in JSON mode with a capable cloud model as fallback gives a
good cost/reliability balance:

```python
local    = ChatOpenAI(model="llama3.1", base_url="http://localhost:11434/v1", api_key="x", temperature=0)
fallback = ChatOpenAI(model="gpt-4o", temperature=0)

result, stats = await extract_data(
    local, MySchema, messages,
    mode=ExtractionMode.JSON,
    fallback_llm_model=fallback,
    max_primary_retries=2,
)
```

---

## Related

- [`examples/07_json_mode.py`](../examples/07_json_mode.py) — runnable JSON-mode example
- [`examples/15_openai_compatible_gateway.py`](../examples/15_openai_compatible_gateway.py) — `ToolCallConfig` against a compatible gateway
- [Extraction](extraction.md) — the two extraction functions
- [Retry & Fallback](retry-and-fallback.md) — the shared retry / fallback behaviour
- [Schema Design](schema-design.md) — writing schemas the model can satisfy
