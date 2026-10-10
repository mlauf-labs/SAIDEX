# Failure reasons

When an extraction ends without a result, `stats.success` is `False` and
`stats.failure_reason` says why. The value is a `saidex.FailureReason`, a
`StrEnum`: it compares equal to its plain string, so both styles work.

```python
from saidex import FailureReason

result, stats = await extract_data(llm, Invoice, messages)
if stats.failure_reason is FailureReason.LLM_ERROR:
    schedule_retry_later()
elif stats.failure_reason == "validation_exhausted":   # same thing, as a string
    send_to_review_queue()
```

## The reasons

| Value | Meaning | Typical cause | Schema-valid candidate possible? | Retry later? |
| --- | --- | --- | --- | --- |
| `validation_exhausted` | The last attempt produced parseable output, but it failed schema validation, source grounding or the external validator, and no retries were left. | The model cannot satisfy a strict rule; the value is not in the source text; ambiguous field descriptions. | **Yes.** Grounding and external-validator rejections always follow a schema-valid instance. | Rarely with the same input and model. Improve the prompt or schema, use a stronger fallback model, or relax the validator. In the agent loop it is also reported when `max_iterations` ran out before a valid final answer (for example only helper-tool calls): raise `max_iterations`. |
| `parse_error` | The last attempt's output could not be parsed into arguments: malformed JSON or an invalid tool call. | Small or reasoning models, truncated output (`max_tokens`), provider quirks. | **Yes**, if an *earlier* attempt produced one. | Sometimes. Truncation is often transient; consider a larger `max_tokens` or JSON mode. |
| `llm_error` | The model call failed after all network retries, or the provider rejected the tool binding. | Outage, authentication, a rate limit beyond the retry budget, unsupported tool-calling flags. | **Yes**, if an *earlier* attempt produced one. | **Yes** for transient outages and rate limits. No if the tool binding was rejected even after automatic relaxation (or with `auto_relax=False`): fix `tool_config`. |
| `no_tool_call` | Agent loop with tool-calling final answers only: the model replied with plain text instead of calling a tool, and no retries were left. | The model ignores tools; the prompt does not ask for the final-answer tool. | **Yes**, if an *earlier* attempt produced one. | Rarely. Adjust the prompt, or use `final_answer_mode=ExtractionMode.JSON`. |

!!! note "A reason describes the last attempt only"
    The retry loop keeps going after a rejection, so the run's reason is the
    outcome of its **last** attempt. Attempt 1 may return a schema-valid
    instance that the external validator rejects, and attempt 2 may then return
    malformed JSON: the run ends with `parse_error` even though a schema-valid
    candidate existed.

## Primary and fallback model

When a fallback model is configured and the primary fails, the fallback gets
fresh attempts. If both fail, the combined reason is the **fallback's**. It is
the **primary's** only when the fallback reported none, e.g. because it was
configured with zero attempts. The same rule applies to `extract_data*`, the
batch variants and the agent loop.
