# Release process

This project automates versioning with [Commitizen](https://commitizen-tools.github.io/commitizen/)
and publishes to PyPI through GitHub Actions. This document is the step-by-step
checklist for cutting a release.

## How it fits together

| Piece | Role |
| --- | --- |
| **Conventional Commits** | Every commit message states the kind of change (`feat:`, `fix:`, `docs:`, …). A `commit-msg` hook enforces this. |
| **Commitizen** (`cz bump`) | Reads the commits since the last tag, decides the next version, updates the version everywhere, rewrites the changelog, commits, and tags. |
| **Version source of truth** | `src/saidex/__init__.py` (`__version__`). Hatch reads it for the build; Commitizen keeps it in sync via `version_files`. |
| **`release.yml`** | Runs on a **published GitHub Release**. Builds the sdist + wheel and uploads to PyPI via Trusted Publishing (OIDC — no token). |

The version bump is **always local and reviewed** before anything is pushed —
nothing reaches PyPI until you publish a GitHub Release.

## Versioning rules (Conventional Commits → SemVer)

While the project is on `0.x` (`major_version_zero = true` in `pyproject.toml`):

| Commit type | Example | Version effect |
| --- | --- | --- |
| `fix:` | `fix: handle empty tool result` | patch — `0.2.0 → 0.2.1` |
| `feat:` | `feat: add batch extraction` | minor — `0.2.0 → 0.3.0` |
| `feat!:` / `BREAKING CHANGE:` | `feat!: rename get_structured_data` | minor while on 0.x — `0.2.0 → 0.3.0` |
| `docs:` / `chore:` / `refactor:` / `test:` / `ci:` | `docs: clarify JSON mode` | no release on its own |

Once the project hits `1.0.0`, set `major_version_zero = false`; from then on a
breaking change bumps the **major** version.

## One-time setup (per clone)

Activate the commit-message hook so non-conventional messages are rejected:

```bash
git config core.hooksPath .githooks
```

The hook runs `uv run cz check` on every commit. Bypass only in emergencies with
`git commit --no-verify`.

## Releasing — step by step

### 1. Start from a clean, up-to-date `main`

```bash
git switch main
git pull
git status            # must be clean
```

### 2. Run the full quality gate

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/
uv run pytest
```

All four must pass — the same checks run in CI on every push.

### 3. Preview the bump

```bash
uv run cz bump --dry-run
```

This prints the version Commitizen would choose and why, without changing
anything. If the result looks wrong, it almost always means a commit used the
wrong type — fix the history or override the increment in the next step.

### 4. Perform the bump

```bash
uv run cz bump
```

This will, in one step:

- compute the next version from the commits since the last tag,
- update `__version__` in `src/saidex/__init__.py` and `version` in `pyproject.toml`,
- prepend the new section to `CHANGELOG.md`,
- create a `bump: ...` commit,
- create the `vX.Y.Z` git tag.

To force a specific level instead of auto-detection:

```bash
uv run cz bump --increment patch   # or minor / major
```

### 5. Review

```bash
git show           # inspect the bump commit + tag
git diff HEAD~1    # check CHANGELOG.md and version files
```

If the generated `CHANGELOG.md` needs polishing, edit it now and amend:

```bash
git commit --amend --no-edit -- CHANGELOG.md
```

### 6. Push the commit and the tag

```bash
git push --follow-tags
```

### 7. Publish the GitHub Release (this triggers PyPI)

```bash
gh release create vX.Y.Z --title "vX.Y.Z" --notes-from-tag
```

…or create it from the GitHub web UI. Publishing the release fires
`release.yml`, which builds and uploads to PyPI automatically.

### 8. Verify

- Watch the **Release to PyPI** workflow:  `gh run watch`
- Confirm the new version on PyPI: <https://pypi.org/project/saidex/>
- `pip install --upgrade saidex` in a clean environment as a smoke test.

## Notes for the very first release (v0.2.0)

There are no tags yet, so the automatic increment has no baseline. For the
initial `0.2.0` release the files already contain the correct version — just
tag and publish it manually:

```bash
git tag v0.2.0
git push origin v0.2.0
gh release create v0.2.0 --title "v0.2.0" --notes-file CHANGELOG.md
```

From the **next** release onward, use `uv run cz bump` as described above.

## One-time PyPI Trusted Publishing setup

Already configured in `release.yml`, but for reference — on
<https://pypi.org/manage/account/publishing/> the pending publisher must be:

| Field | Value |
| --- | --- |
| PyPI project | `saidex` |
| Owner | `mlauff-labs` |
| Repository | `saidex` |
| Workflow | `release.yml` |
| Environment | `pypi` |
