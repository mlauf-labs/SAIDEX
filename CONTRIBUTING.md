# Contributing to saidex

Thanks for your interest in contributing! This document explains how to set up
a development environment, what checks must pass, and how to submit changes.

## Before you start

- **Bugs and small fixes:** open a pull request directly, or file an
  [issue](https://github.com/mlauf-labs/saidex/issues) if you can't fix it
  yourself.
- **New features or behaviour changes:** please open an issue first to discuss
  the idea — this avoids wasted work when a feature doesn't fit the project's
  scope (see the [roadmap](README.md#roadmap)).
- **Security issues:** never via public issues — see [SECURITY.md](SECURITY.md).

## Development setup

The project uses [uv](https://docs.astral.sh/uv/) for dependency and
virtual-environment management.

```bash
# 1. Install uv (once)
pip install uv
# or on macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Fork on GitHub, then clone your fork
git clone https://github.com/<your-username>/saidex
cd saidex

# 3. Create .venv and install everything (including the dev group)
uv sync
```

`uv sync` reads `uv.lock` and installs the exact pinned versions — no manual
`python -m venv` or `pip install` needed.

Then activate the commit-message hook once (enforces the commit format below):

```bash
git config core.hooksPath .githooks
```

## Commit messages

This project uses [Conventional Commits](https://www.conventionalcommits.org/).
The `commit-msg` hook (above) rejects messages that don't follow the format, and
the changelog and version numbers are derived from them automatically.

```
<type>: <short summary>      # e.g.  feat: add batch extraction
```

Common types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`, `perf`,
`build`. Add `!` (e.g. `feat!:`) or a `BREAKING CHANGE:` footer for breaking
changes. Prefer `uv run cz commit` if you'd like an interactive prompt.

## Daily commands

```bash
uv run pytest                          # run the full test suite
uv run pytest tests/test_utils.py -v   # run a single test file
uv run pytest --cov                    # tests with coverage report
uv run ruff check src/ tests/          # lint
uv run ruff format src/ tests/         # format code
uv run mypy src/                       # static type checking (strict)
```

## What CI checks

Every pull request runs the following on Python 3.10–3.13 — all of it must
pass:

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/
uv run pytest --cov --cov-report=term-missing
```

Run these locally before pushing and you won't be surprised by CI.

## Code guidelines

- **Type hints everywhere** — `mypy --strict` runs in CI; the library ships a
  `py.typed` marker, so the public API must stay fully typed.
- **Docstrings on public API** — every public function/class has a docstring
  with `Args:` and `Returns:` sections (Google style, like the existing code).
- **Async-first** — the public API is async; don't add blocking I/O.
- **Keep dependencies minimal** — the runtime depends only on `pydantic`,
  `langchain-core`, and `json-repair`. Anything provider-specific belongs in
  an optional extra (like `saidex[openai]`).
- **Formatting/linting is ruff's job** — no manual style debates; line length
  is 100.
- **Tests are mock-based** — the test suite must run without API keys or
  network access (see `tests/test_agent_loop.py` for the mock-LLM pattern).

## Pull request checklist

The project follows the **Git Flow** model: `feature/*` branches off `develop`,
and pull requests target **`develop`** (never `main` directly). See
[docs/repo-setup.md](docs/repo-setup.md) for the full branching model and the
branch-protection rules.

1. Create a feature branch off `develop` in your fork:

   ```bash
   git switch develop
   git switch -c feature/my-feature      # or fix/..., docs/...
   ```

2. Make your changes:
   - add or adapt **tests** for any behaviour change,
   - update **docs** (`README.md`, `docs/`) and **examples** if the public API
     changes,
   - write clear **Conventional Commit** messages — `CHANGELOG.md` is generated
     from them at release time, so there's no need to edit it by hand.

3. Verify everything passes (see [What CI checks](#what-ci-checks)).

4. Open the pull request **against `develop`** with a short description of
   *what* and *why*. Reference the related issue if there is one (`Fixes #123`).

Small, focused PRs are reviewed much faster than large ones — when in doubt,
split it up.

## Managing dependencies

```bash
uv add some-package              # runtime dependency
uv add --group dev some-tool     # dev-only dependency
uv add --optional openai openai  # optional extra
uv lock --upgrade && uv sync     # upgrade everything (separate PR, please)
```

`pyproject.toml` and `uv.lock` change together — commit both.

## Releases (maintainers)

Versioning, changelog, and tagging are automated with Commitizen. See
[RELEASING.md](RELEASING.md) for the full step-by-step checklist, and
[docs/repo-setup.md](docs/repo-setup.md) for the branch-protection and
PyPI-publish safeguards.
