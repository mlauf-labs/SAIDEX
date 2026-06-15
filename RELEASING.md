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

> **Git Flow:** `main` and `develop` are protected — you never bump or push on
> them directly. The version bump happens on a short-lived `release/X.Y.Z`
> branch and reaches `main` through a reviewed PR.

### 1. Make sure `develop` is green and up to date

```bash
git switch develop
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

> On Windows, `uv run mypy` / `uv run pytest` may fail with
> `uv trampoline failed to canonicalize script path`. Run them as
> `uv run python -m mypy src/` / `uv run python -m pytest` instead.

### 3. Create the release branch

```bash
git switch -c release/X.Y.Z      # use the version from the dry-run below
```

### 4. Preview the bump

```bash
uv run cz bump --dry-run
```

This prints the version Commitizen would choose and why, without changing
anything. If the result looks wrong, it almost always means a commit used the
wrong type — fix the history or override the increment in the next step. While
the project is on `0.x`, a breaking change is still only a **minor** bump.

### 5. Perform the bump

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

### 6. Review

```bash
git show           # inspect the bump commit + tag
git diff HEAD~1    # check CHANGELOG.md and version files
```

If the generated `CHANGELOG.md` needs polishing, edit it now and amend:

```bash
git commit --amend --no-edit -- CHANGELOG.md
```

### 7. Push the release branch and the tag

```bash
git push -u origin release/X.Y.Z
git push origin vX.Y.Z
```

> Commitizen creates a **lightweight** tag, so `git push --follow-tags` will
> *not* push it — push the tag explicitly as shown above.

### 8. Open the release PR into `main`

```bash
gh pr create --base main --head release/X.Y.Z --title "release: vX.Y.Z" --fill
```

Wait for CI to pass, then **merge with a merge commit — never squash.** A squash
merge would rewrite the bump commit and leave the `vX.Y.Z` tag pointing at a
commit that is not in `main`'s history.

### 9. Publish the GitHub Release (this triggers PyPI)

```bash
gh release create vX.Y.Z --title "vX.Y.Z" --notes-from-tag --verify-tag
```

…or create it from the GitHub web UI. Publishing the release fires `release.yml`
(builds and uploads to PyPI via Trusted Publishing) and `docs.yml` (deploys the
versioned docs and moves the `latest` alias). **This is the only irreversible
step** — a version published to PyPI cannot be replaced.

### 10. Verify

- Watch the **Release to PyPI** workflow:  `gh run watch`
- Confirm the new version on PyPI: <https://pypi.org/project/saidex/>
- `pip install --upgrade saidex` in a clean environment as a smoke test.

### 11. Back-merge `main` into `develop` and clean up

Bring the bump commit and changelog back onto `develop` so the next release has
the correct baseline, then delete the merged release branch:

```bash
git fetch origin
git switch -c chore/sync-main-to-develop origin/main
git push -u origin chore/sync-main-to-develop
gh pr create --base develop --head chore/sync-main-to-develop \
  --title "chore: back-merge vX.Y.Z into develop" --fill

# after the release PR is merged, GitHub usually deletes the branch automatically:
git push origin --delete release/X.Y.Z   # only if it still exists
```

## One-time PyPI Trusted Publishing setup

Already configured in `release.yml`, but for reference — on
<https://pypi.org/manage/account/publishing/> the pending publisher must be:

| Field | Value |
| --- | --- |
| PyPI project | `saidex` |
| Owner | `mlauf-labs` |
| Repository | `saidex` |
| Workflow | `release.yml` |
| Environment | `pypi` |
