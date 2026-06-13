# Security Policy

## Supported Versions

Only the latest release receives security updates.

| Version | Supported |
| --- | --- |
| 0.2.x | ✅ |
| < 0.2 | ❌ |

## Reporting a Vulnerability

Please **do not open a public issue** for security problems.

Report vulnerabilities privately via
[GitHub Security Advisories](https://github.com/mlauff-labs/saidex/security/advisories/new)
("Report a vulnerability" on the repository's Security tab).
If you cannot use GitHub, email [martin-l22@web.de](mailto:martin-l22@web.de)
with `[saidex security]` in the subject line.

Please include:

- A description of the issue and its impact
- Steps or a minimal code sample to reproduce it
- The affected version(s) (`pip show saidex`)

You can expect an initial response within **7 days**. Once the issue is
confirmed, a fix is developed privately and released together with a security
advisory; you will be credited unless you prefer otherwise.

## Scope notes for an LLM library

saidex sends your prompts to whatever LLM you configure — it does not include
its own network services, credentials, or telemetry. Keep in mind:

- **In scope:** anything where saidex itself mishandles data — e.g. code
  execution through malicious LLM responses during JSON parsing/repair,
  validation bypasses in `create_instance_safe`, or unsafe handling of tool
  arguments in the agent loop.
- **Out of scope:** prompt injection against your own prompts, the behaviour
  of the LLM provider, and vulnerabilities in dependencies (report those
  upstream — but do tell us if saidex needs a version bump to pick up a fix).
- The agent loop executes **caller-supplied tools only**; saidex never defines
  tools with side effects itself. Treat tool handlers you write with the same
  care as any code that processes untrusted input, since the LLM chooses the
  arguments.
