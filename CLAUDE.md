# CLAUDE.md — AI Agent Instructions for SAIDEX

> This file is the authoritative guide for all AI coding agents (Claude Code, Codex CLI, Cursor, Aider, Devin, etc.) working in this repository.  
> See also: [AGENTS.md](AGENTS.md) — every agent tool that reads `AGENTS.md` is directed here.

---

## Project overview

**SAIDEX** (`saidex`) is a LangChain-native Python library that turns any chat model into a reliable structured-data extractor. It drives the LLM via tool-calling or raw JSON mode, validates output with Pydantic, and retries with field-level error feedback.

- **Language:** Python 3.10+  
- **Package manager:** [uv](https://docs.astral.sh/uv/) — never use `pip` or `pip install` directly  
- **Core dependencies:** `pydantic >= 2`, `langchain-core >= 0.2`, `json-repair`  
- **Public API:** `src/saidex/__init__.py` — everything exported there is stable and must stay fully typed  
- **Docs:** `docs/` (MkDocs + mike for versioned GitHub Pages)

---

## Repository layout

```text
src/saidex/       library source (extractor, models, retry, tools, utils, validators)
tests/            pytest suite — must run without API keys or network access
examples/         runnable end-to-end scripts
benchmarks/       benchmark scripts and results
docs/             MkDocs documentation
```

---

## Git workflow — Git Flow (mandatory)

This repository follows **Git Flow**. Every code change lives on a short-lived branch and enters the long-lived branches only via a reviewed pull request.

### Branch naming

| Type | Pattern | Base branch | Target |
| --- | --- | --- | --- |
| Feature | `feature/<short-name>` | `develop` | `develop` |
| Bug fix | `fix/<short-name>` | `develop` | `develop` |
| Documentation | `docs/<short-name>` | `develop` | `develop` |
| Refactor | `refactor/<short-name>` | `develop` | `develop` |
| Chore / tooling | `chore/<short-name>` | `develop` | `develop` |
| Release | `release/<version>` | `develop` | `main` + `develop` |
| Hot fix | `hotfix/<version>` | `main` | `main` + `develop` |

### Rules — follow these without exception

1. **Never commit directly to `main` or `develop`.** Both branches are protected; direct pushes are rejected.
2. **Create one branch per logical change.** A branch should do exactly one thing.
3. **Always branch from `develop`** (except `hotfix/*`, which branches from `main`).
4. **Open a PR against `develop`** when the work is done. `main` is only touched by `release/*` and `hotfix/*` PRs.
5. **Delete the branch after it is merged.**
6. **Keep branches short-lived.** Rebase on `develop` regularly to avoid large merge conflicts.
7. **Always `git fetch` + `git pull` on `develop` before creating a new branch** so you branch from the latest state.
8. **Never delete or discard uncommitted changes on your own.** No `git reset --hard`, `git checkout -- <file>`, `git stash drop`, `git clean`, or force-overwrite of a dirty working tree without explicit user approval. Preserving the user's work always takes priority.
9. **If switching to `develop` fails** (e.g. uncommitted changes, a dirty working tree, or a merge conflict), **stop and report it to the user.** Do not auto-resolve by throwing away changes — ask how to proceed (commit, stash, or keep them).

### Starting work — required sequence

```bash
# 1. Make sure develop is up to date
git switch develop
git pull

# 2. Create a new branch
git switch -c feature/my-feature   # or fix/..., docs/..., etc.

# 3. Work, commit using Conventional Commits (see below)

# 4. Push and open a PR targeting develop
git push -u origin feature/my-feature
gh pr create --base develop --fill
```

### AI agent workflow — required for every implementation

Follow this for **every** feature/fix/docs/refactor/chore task, without exception:

1. **Sync `develop` first.** Run `git fetch` and `git pull` on `develop` before doing anything else.
   - If you cannot switch to `develop` cleanly (uncommitted changes, dirty working tree, conflicts), **stop and tell the user.** Never delete or discard their changes to "unblock" the switch — ask whether to commit, stash, or keep them.
2. **Create a new branch from the freshly updated `develop`** (`feature/…`, `fix/…`, etc.) — one branch per logical change.
3. **Work and commit on that branch** using Conventional Commits.
4. **Run all tests and CI checks** (see [Development commands](#development-commands)) and make them pass.
5. **Check documentation consistency before opening the PR.** Verify that the docs under `docs/`, the runnable scripts under `examples/`, and the `README` files still match the change, and update them wherever they have drifted.
6. **Update the changelog for end-user-relevant changes.** Anything an end user should know about (new feature, behaviour change, fix, breaking change) must end up in `CHANGELOG.md`. This repo generates the changelog with Commitizen from the commit history, so the practical rule is: make sure such changes are captured by a correctly typed commit (`feat`/`fix`/…, with a `BREAKING CHANGE:` footer where relevant) so they appear in the generated changelog. Do not hand-edit `CHANGELOG.md`.
7. **Open a PR targeting `develop`** only once the work is complete, the tests are green, and docs/changelog are in sync.

> ⚠️ **Never delete a user's changes on your own initiative.** Preserving uncommitted work always takes priority over any git operation.

---

## Conventional Commits (mandatory)

Every commit message **must** follow the [Conventional Commits](https://www.conventionalcommits.org/) specification. The `commit-msg` hook at `.githooks/commit-msg` enforces this and rejects non-conforming messages.

### Format

```text
<type>(<optional scope>): <short summary>

[optional body — explain WHY, not WHAT]

[optional footer(s) — BREAKING CHANGE: ..., Fixes #123]
```

### Allowed types

| Type | When to use |
| --- | --- |
| `feat` | New user-visible feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `test` | Adding or fixing tests |
| `chore` | Build, tooling, dependency updates (no src change) |
| `ci` | CI/CD pipeline changes |
| `perf` | Performance improvement |
| `build` | Changes to build system or external dependencies |

### Breaking changes

Add `!` after the type or include a `BREAKING CHANGE:` footer:

```text
feat!: rename get_structured_data parameter llm to llm_model

BREAKING CHANGE: The `llm` parameter is now `llm_model` in all public functions.
```

### Good examples

```text
feat(extractor): add batch extraction returning list[ModelT]
fix(retry): handle httpx.ReadTimeout as a retryable exception
docs(schema-design): add section on nested model arrays
test(agent-loop): cover max_iterations=1 edge case
chore: upgrade pydantic to 2.8.0
```

### Bad examples (will be rejected by hook)

```text
WIP
fixed stuff
Update README
add feature
```

### Interactive commit helper

Use `uv run cz commit` for a guided prompt that builds a valid message.

---

## Development commands

Always use `uv run` — never activate the virtual environment manually.

```bash
# Install / sync all dependencies (including dev group)
uv sync

# Tests
uv run pytest                           # full suite
uv run pytest tests/test_extractor.py -v  # single file
uv run pytest --cov --cov-report=term-missing  # with coverage

# Lint (must pass before every commit)
uv run ruff check src/ tests/
uv run ruff check src/ tests/ --fix     # auto-fix safe issues
uv run ruff format src/ tests/          # format

# Type checking (strict — must pass)
uv run mypy src/

# Commit helper
uv run cz commit
```

All four CI checks must pass locally before pushing:

```bash
uv run ruff check src/ tests/ && \
uv run ruff format --check src/ tests/ && \
uv run mypy src/ && \
uv run pytest --cov --cov-report=term-missing
```

---

## Code guidelines

### Type hints

- **`mypy --strict` is mandatory.** Every function argument and return value must have a type annotation.  
- The library ships `py.typed` — the public API in `src/saidex/__init__.py` must be fully typed at all times.

### Async

- The public API is **async-first**. Do not introduce blocking I/O (`requests`, `open()` in tight loops, `time.sleep()`).

### Dependencies

- Runtime deps: only `pydantic`, `langchain-core`, `json-repair`. Provider-specific packages go into optional extras (`saidex[openai]`).
- Dev deps go in the `[dependency-groups.dev]` section of `pyproject.toml`.

### Tests

- The test suite **must run without API keys or network access**. Use the mock-LLM pattern (see `tests/test_extractor.py`).
- Every behaviour change or new feature requires a test. A PR without tests for new logic will not be merged.
- Aim for the test to fail before the fix and pass after.

### Formatting

- Line length: **100 characters** (configured in `pyproject.toml`).
- `ruff format` is authoritative — no manual style debates.

### Comments and docstrings

- Public functions and classes require a docstring with `Args:` and `Returns:` sections (Google style).
- Do not explain *what* the code does in inline comments. Only document non-obvious *why* (hidden constraint, workaround, subtle invariant).

### Schema changes

- Any change to a public Pydantic model must update `docs/` and the relevant example in `examples/`.

### Context7 index (`context7.json`)

`context7.json` registers this library with [Context7](https://context7.com), so AI coding tools always have access to up-to-date SAIDEX documentation.

Keep it current whenever:

- A new doc page is added or removed from `docs/` → check `excludeFiles` if it should be hidden from AI tools.
- A new release is published → add the version that was just *replaced as latest* to `previousVersions`. The new current is indexed automatically from `main`. Example: when releasing `v0.3.0`, add `{"tag": "v0.2.0", "title": "version 0.2.0"}` if not already listed.
- A new usage rule or important constraint for library users is identified → add it to `rules`.

---

## Pull request checklist

Before opening or marking a PR as ready:

- [ ] Branch name follows the naming convention above
- [ ] All commits are Conventional Commits
- [ ] `uv run ruff check src/ tests/` passes (no errors)
- [ ] `uv run ruff format --check src/ tests/` passes
- [ ] `uv run mypy src/` passes
- [ ] `uv run pytest --cov` passes
- [ ] New behaviour is covered by tests
- [ ] Documentation is consistent — `docs/`, `examples/`, and the `README` files were reviewed and updated wherever the change affects them
- [ ] Changelog reflects every end-user-relevant change (captured via correctly typed `feat`/`fix`/… commits so Commitizen generates the `CHANGELOG.md` entry; never hand-edited)
- [ ] PR targets `develop` (not `main`)
- [ ] PR title follows Conventional Commit format (`feat: …`, `fix: …`, etc.)
- [ ] PR description explains *what* and *why*; references a GitHub issue if one exists

---

## Release process (maintainers only)

Releases are handled by the `release/*` branch flow and automated with Commitizen. See [RELEASING.md](RELEASING.md) and [docs/repo-setup.md](docs/repo-setup.md) for the full checklist.

Never bump versions or edit `CHANGELOG.md` by hand — Commitizen derives both from the commit history.

---

## Security

Never open a public issue for security vulnerabilities. Use [GitHub Security Advisories](https://github.com/mlauf-labs/saidex/security/advisories/new) instead. See [SECURITY.md](SECURITY.md).
