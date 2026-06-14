# Repository setup: branching model & branch protection

This guide locks the repository down so that **nothing reaches `main` or
`develop` — and therefore the published pip package — without going through a
reviewed pull request that passed CI.** It also describes the Git Flow branching
model the project uses.

Do this once, right after pushing the repo to GitHub.

---

## 1. Branching model (Git Flow)

| Branch | Purpose | Receives merges from | Lives forever? |
| --- | --- | --- | --- |
| `main` | Released, production code. Every commit is a release, tagged `vX.Y.Z`. | `release/*`, `hotfix/*` | yes |
| `develop` | Integration branch — the next release in progress. | `feature/*`, `release/*`, `hotfix/*` | yes |
| `feature/<name>` | One feature or fix. Branched from `develop`. | — | no (deleted after merge) |
| `release/<version>` | Stabilise a release (version bump, changelog). Branched from `develop`. | — | no |
| `hotfix/<version>` | Urgent fix against a release. Branched from `main`. | — | no |

Flow in one line: **`feature/*` → PR → `develop` → (release) → PR → `main` → tag → PyPI.**

Both long-lived branches (`main`, `develop`) are protected; you only ever change
them through pull requests.

---

## 2. Push the two long-lived branches

```bash
# main already exists with the initial commit
git push -u origin main

# create develop from main and push it
git switch -c develop
git push -u origin develop
```

### Make `develop` the default branch

So new PRs target `develop` (not `main`) by default:

**GitHub → repo → Settings → General → Default branch → switch to `develop`.**

---

## 3. Make sure only you have write access

On a personal repository you are the only one who can push unless you add
collaborators.

- **Settings → Collaborators** — confirm the list contains only you. Add people
  only with the minimum role they need (`Read`/`Triage` for most).
- The branch ruleset in the next step additionally restricts who may push to the
  protected branches, even among collaborators.

---

## 4. Protect `main` and `develop` (Rulesets — recommended)

Rulesets are GitHub's current branch-protection mechanism and can target several
branches at once.

**Settings → Rules → Rulesets → New ruleset → New branch ruleset.**

1. **Ruleset name:** `protect-main-and-develop`
2. **Enforcement status:** `Active`
3. **Bypass list:** leave **empty** (so the rules apply to everyone, including
   you — this is what guarantees "only via PR"). Add yourself here only if you
   later need an emergency escape hatch.
4. **Target branches → Add target → Include by pattern**, add both:
   - `main`
   - `develop`
5. **Rules — enable:**
   - ✅ **Restrict deletions**
   - ✅ **Block force pushes**
   - ✅ **Require a pull request before merging**
     - Required approvals: **0** (see the solo-maintainer note below)
     - ✅ Dismiss stale approvals when new commits are pushed
     - ✅ Require conversation resolution before merging
   - ✅ **Require status checks to pass**
     - ✅ Require branches to be up to date before merging
     - Add the check named **`CI success`** (from `tests.yml`). If it doesn't
       appear yet, open one PR first so the check runs once, then add it.
   - ✅ **Restrict who can push** (the “Restrict creations/updates” / push
     restriction) — limit pushes to the protected branches to yourself only.
   - (Optional) ✅ **Require signed commits** — see section 7.

Save the ruleset.

> Direct `git push` to `main`/`develop` is now rejected for everyone. The only
> way to change them is a pull request whose `CI success` check is green.

### ⚠️ Solo-maintainer note on “Required approvals”

**GitHub does not let you approve your own pull request.** So if you set
*Required approvals* to `1` while you are the only maintainer, you could never
merge. Two valid choices:

- **Required approvals = 0** (recommended for a solo project): every change
  still goes through a PR, CI must pass, and *you read the full diff in the PR
  and click “Merge” yourself*. You review everything; you just aren’t forced to
  click an “Approve” button you’re not allowed to click.
- **Required approvals = 1** only once a second trusted maintainer exists.

Either way, the hard automated gate against bad/“shady” code is the **required
`CI success` status check** plus your own diff review before merging.

---

## 5. Protect the actual PyPI publish (strongest safeguard)

Branch protection controls what enters the code. This step controls what leaves
as a release — a manual gate right before anything is uploaded to PyPI.

**Settings → Environments → `pypi` (used by `release.yml`):**

- ✅ **Required reviewers** → add **yourself**. Now the `publish` job pauses and
  waits for your explicit approval in the Actions tab before it uploads to PyPI.
- (Optional) **Deployment branches and tags** → restrict to `main` only / tags
  matching `v*`, so a publish can never run off `develop` or a feature branch.

Even if something unwanted were merged, it cannot reach the pip package without
you approving the deployment here.

---

## 6. How you work day-to-day (Git Flow)

You can use plain Git or the `git flow` helper (`git-flow` / `gitflow-avh`
package). Plain Git equivalents:

```bash
# Start a feature off develop
git switch develop && git pull
git switch -c feature/batch-extraction

# ... commit using Conventional Commits (the commit-msg hook enforces it) ...
git push -u origin feature/batch-extraction

# Open a PR:  feature/batch-extraction  ->  develop
gh pr create --base develop --fill

# After CI is green and you've reviewed the diff, merge in the GitHub UI,
# then delete the feature branch.
```

Cutting a release (see [RELEASING.md](../RELEASING.md) for the version/tag part):

```bash
git switch develop && git pull
git switch -c release/0.3.0
uv run cz bump            # version bump + changelog on the release branch
git push -u origin release/0.3.0
gh pr create --base main --fill     # release/0.3.0 -> main
# merge -> the tag triggers release.yml -> you approve the `pypi` environment
# finally merge main back into develop (or PR release/* -> develop too)
```

With the `git flow` extension the same is `git flow feature start …`,
`git flow release start …`, etc. — configure it once with:

```bash
git flow init      # accept: production = main, development = develop
```

---

## 7. Optional hardening

- **Signed commits / tags:** enable “Require signed commits” in the ruleset and
  sign locally with GPG or SSH (`git config commit.gpgsign true`). Gives every
  protected-branch commit a verified author.
- **Linear history:** add the “Require linear history” rule and use *Squash* or
  *Rebase* merges only, for a clean, bisectable `main`.
- **Dependabot / secret scanning:** Settings → Code security — turn on Dependabot
  alerts, the dependency graph, and secret scanning + push protection so leaked
  credentials are blocked at push time.
- **Tag protection:** add a tag ruleset for `v*` so release tags can't be moved
  or deleted.

---

## Checklist

- [ ] `main` and `develop` pushed; `develop` is the default branch
- [ ] Only you have write access (Collaborators)
- [ ] Ruleset active on `main` + `develop`: PR required, force-push/deletion
      blocked, `CI success` required, pushes restricted, empty bypass list
- [ ] `pypi` environment requires your approval before publishing
- [ ] You’ve done one PR end-to-end to confirm the gates work
