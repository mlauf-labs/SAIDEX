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
| `validation_exhausted` | The last attempt produced parseable output, but it failed schema validation, source grounding or the external validator, and no retries were left. | The model cannot satisfy a strict rule; the value is not in the source text; ambiguous field descriptions. | **Yes.** Grounding and external-validator rejections always follow a schema-valid instance. Check `stats.last_candidate`. | Rarely with the same input and model. Improve the prompt or schema, use a stronger fallback model, or relax the validator. In the agent loop it is also reported when `max_iterations` ran out before a valid final answer (for example only helper-tool calls): raise `max_iterations`. |
| `parse_error` | The last attempt's output could not be parsed into arguments: malformed JSON or an invalid tool call. | Small or reasoning models, truncated output (`max_tokens`), provider quirks. | **Yes**, if an *earlier* attempt produced one. Check `stats.last_candidate`. | Sometimes. Truncation is often transient; consider a larger `max_tokens` or JSON mode. |
| `llm_error` | The model call failed after all network retries, or the provider rejected the tool binding. | Outage, authentication, a rate limit beyond the retry budget, unsupported tool-calling flags. `stats.llm_error_kind` says why — see below. | **Yes**, if an *earlier* attempt produced one. Check `stats.last_candidate`. | **Yes** for transient outages and rate limits. No if the tool binding was rejected even after automatic relaxation (or with `auto_relax=False`): fix `tool_config`. |
| `no_tool_call` | Agent loop with tool-calling final answers only: the model replied with plain text instead of calling a tool, and no retries were left. | The model ignores tools; the prompt does not ask for the final-answer tool. | **Yes**, if an *earlier* attempt produced one. Check `stats.last_candidate`. | Rarely. Adjust the prompt, or use `final_answer_mode=ExtractionMode.JSON`. |

!!! note "A reason describes the last attempt only"
    The retry loop keeps going after a rejection, so the run's reason is the
    outcome of its **last** attempt. Attempt 1 may return a schema-valid
    instance that the external validator rejects, and attempt 2 may then return
    malformed JSON: the run ends with `parse_error` even though a schema-valid
    candidate existed.

`stats.last_candidate` is the authoritative check — see
[External Validators](external-validators.md#when-every-attempt-fails-last_candidate).

## Why did the model call fail? — `llm_error_kind`

When `failure_reason` is `llm_error`, `stats.llm_error_kind` is a
`saidex.ModelErrorKind`, independent of the provider:

| Kind | Typical cause | Retried by default? |
| --- | --- | --- |
| `rate_limit` | HTTP 429 | yes (rate-limit backoff) |
| `timeout` | request timed out | yes |
| `connection` | provider unreachable | yes |
| `server_error` | HTTP 5xx | yes |
| `authentication` | HTTP 401, bad API key | no |
| `permission_denied` | HTTP 403 | no |
| `invalid_request` | HTTP 400, or tool-binding flags the provider rejects | no |
| `not_found` | HTTP 404, unknown model name | no |
| `context_overflow` | input larger than the context window | no |
| `other` | anything else | no |

The kind is derived by `saidex.classify_model_error(exc)` from langchain-core's
standard model exceptions (langchain-core ≥ 1.6) and, for older versions, from the
`openai` / `httpx` exception types. A salvaged run (`on_exhausted`) has no kind.

## Primary and fallback model

When a fallback model is configured and the primary fails, the fallback gets
fresh attempts. If both fail, the combined reason is the **fallback's**. It is
the **primary's** only when the fallback reported none, e.g. because it was
configured with zero attempts. The same rule applies to `extract_data*`, the
batch variants and the agent loop.

## Salvaged runs

A run salvaged by `on_exhausted` reports `success=True`, `failure_reason=None` and `salvaged=True`; the original rejection stays in `last_rejection`.
