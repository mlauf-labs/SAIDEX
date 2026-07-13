# LangGraph: SaidexToolNode

→ [Documentation index](index.md)

`SaidexToolNode` is a drop-in replacement for LangGraph's `ToolNode` that
repairs, validates and (optionally) corrects tool calls **before** executing
them. It brings SAIDEX's extraction-grade reliability to agentic tool calling.

```bash
pip install "saidex[langgraph]"
```

```python
from saidex.langgraph import SaidexToolNode

tool_node = SaidexToolNode(tools)          # instead of ToolNode(tools)
graph.add_node("tools", tool_node)
```

Core `saidex` never imports `langgraph` — this integration lives entirely in
`saidex.langgraph` and is only pulled in when you import it.

## Why

The stock `ToolNode` executes whatever it is given:

- **`invalid_tool_calls` are silently ignored.** A tool call whose argument
  JSON did not parse never gets a `ToolMessage` — and an unanswered tool-call
  id makes OpenAI-compatible providers reject the *next* model turn.
- **Validation feedback is generic.** A schema mismatch surfaces as
  `"Please fix the error and try again."` with a raw exception message.

`SaidexToolNode` guarantees **every tool-call id receives exactly one
`ToolMessage`**, and invalid calls get SAIDEX's structured, field-level
feedback (missing fields, type errors, allowed enum values, required changes).

## What it does per call

1. **Repair** — `invalid_tool_calls` entries are recovered deterministically:
   `<think>` blocks are stripped (`strip_thinking=True`) and malformed JSON is
   fixed with json-repair (`json_repair=True`).
2. **Validate** — arguments are checked against the tool's Pydantic schema
   (`tool.tool_call_schema`, which correctly excludes injected arguments such
   as `InjectedState` / `InjectedToolCallId`).
   Tools with non-Pydantic schemas and unknown tool names are passed through
   to the executor unchanged.
3. **Execute or apply the policy** — valid calls run on an internal stock
   `ToolNode` (parallel execution, `handle_tool_errors`, `Command` returns and
   `InjectedState` keep working). Invalid calls follow `on_invalid`.

## Policies (`on_invalid`)

| Policy | Behaviour |
| --- | --- |
| `"feedback"` (default) | The call is **not** executed; a `ToolMessage(status="error")` with structured correction guidance is returned so the agent fixes the call on its next turn. |
| `"correct"` | A bounded LLM correction cycle (`correction_model`, `max_correction_retries`, `correction_mode`) coerces the args into a schema-valid shape, then the call executes. If correction fails, falls back to feedback. |
| `"raise"` | `ToolCallValidationError` is raised **before any call executes** (fail-fast, no side effects). |

```python
from saidex import ExtractionMode
from saidex.langgraph import SaidexToolNode

tool_node = SaidexToolNode(
    tools,
    on_invalid="correct",
    correction_model=small_fast_llm,
    max_correction_retries=2,
    correction_mode=ExtractionMode.JSON,   # for models without tool calling
)
```

## Clean message history

Two guarantees keep your graph state token-lean:

- **Correction traffic never enters state.** The `correct` policy's LLM
  conversation is private to the node — you pay correction tokens once, not on
  every later turn.
- **`sanitize_messages=True`** (default) returns an updated copy of the
  `AIMessage` (same `id`) carrying the repaired/corrected arguments. The
  standard `add_messages` reducer (used by `MessagesState`) **replaces** the
  malformed message in place, so leaked chain-of-thought or broken JSON
  disappears from history. Calls answered with feedback stay untouched (their
  feedback message references the call id, so removing the call would orphan
  it). Set `sanitize_messages=False` if your messages channel uses a plain
  append reducer, where the update would **duplicate** the message instead of
  replacing it.

## Edge cases, documented honestly

- **A tool call with no usable `id` cannot be answered.** A `ToolMessage`
  requires a `tool_call_id`, so a call that carries none at all — well-formed
  or malformed — is dropped: not executed, not answered, and (with
  `sanitize_messages=True`) removed from the sanitized message. A
  `logging.Logger.warning` records the drop, and the call's `ToolCallStats`
  reports `outcome="dropped"`.
- **Correction LLM calls perform no network-level retries.** The bounded
  `max_correction_retries` budget governs *validation* attempts only. A
  transient provider error (e.g. an HTTP 429) on a correction call is not
  retried at the transport level — it degrades that one call straight to the
  `"feedback"` policy instead of crashing the node.

## Observability

Each invocation emits a `ToolNodeEvent` with per-call `ToolCallStats`
(outcome, deterministic repairs, correction retries, field issues):

```python
from saidex import collect_stats, on_tool_node

async with collect_stats() as sink:
    result = await graph.ainvoke(inputs)
for stats in sink:          # ToolNodeStats alongside extraction stats
    ...

sub = on_tool_node(lambda event: print(event.stats.feedback_count))
```

`on_extraction()` listeners do **not** receive tool-node events — register
`on_tool_node()` separately. No `ToolNodeEvent` is emitted on the `"raise"`
policy's fail-fast path, since it exits before any call executes.

LangChain callbacks (e.g. Langfuse) flow through unchanged: tool executions
appear exactly as with the stock node, and correction LLM calls nest under the
node's span as a `saidex.extract_data` chain run.

## Sync graphs

`invoke()` bridges to the async implementation on a fresh event loop. Inside
an already running loop (notebooks, async handlers) call `ainvoke()` — the
sync path raises a `RuntimeError` pointing there, matching the rest of the
SAIDEX sync API.

## Scope

`SaidexToolNode` fixes tool calls that are *present but wrong* on the last
`AIMessage`. Tool calls lost or garbled during model **streaming** happen one
layer earlier (the chat-model integration) and cannot be recovered by any tool
node.

**Send-API fan-out payloads are passed straight through.** When tool calls are
dispatched with LangGraph's `Send("tools", tool_call)` — a bare `list[ToolCall]`
or a `ToolCallWithContext` dict — there is no `AIMessage` in the payload to
repair, validate, correct or sanitize: each `Send` already carries one
extracted tool call. `SaidexToolNode` detects both shapes up front and
delegates the call unchanged to the internal stock `ToolNode`, so its output
matches the stock node exactly on this path. No `ToolNodeEvent` is emitted for
these invocations either, since no calls were actually processed by the
pipeline.

---

## Related

- [`examples/16_langgraph_toolnode.py`](../examples/16_langgraph_toolnode.py) — a complete runnable graph
- [Observability](observability.md) — `collect_stats`, `on_tool_node`, callbacks
- [Agent Loop](agent-loop.md) — SAIDEX's own tool loop, for extractions that need lookups before a final answer
