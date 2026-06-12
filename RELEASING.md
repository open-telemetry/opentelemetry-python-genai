# Release process

Every package releases independently. The default path is a coordinated
**release-all** workflow; per-package workflows are for urgent or partial
releases.

Releases are **tag-from-`main`**: each publish creates a tag
(`<pkg>==<version>`) pointing at the release commit on `main`. Backport
branches (`package-release/<pkg>/v*`) are created lazily from an old tag only
when patching an older minor line — not for every release.

Releases are driven by GitHub Actions workflows. They handle version bumps,
changelog generation (via [towncrier](https://towncrier.readthedocs.io/)),
tagging, PyPI publishing, GitHub releases, and changelog updates on `main`.

## Release model

Unlike `opentelemetry-python-contrib`, we do not maintain a long-lived release
branch for every minor. Normal releases tag `main` directly; backport branches
are created on demand from an existing tag when patching an older minor.

| | opentelemetry-python-contrib | This repo |
|---|---|---|
| Normal release | Long-lived `package-release/<pkg>/v*` branch | Tag on `main` |
| Tag target | Commit on the release branch | Commit on `main` |
| Patch in current line | Commits + tags on the release branch | Tags on `main` |
| Backport to older minor | Same branch (already exists) | Branch from old tag (lazy) |
| Branch sprawl | One branch per package per minor | Branches only for backports |

## Prerequisites (one-time, repo-level)

These must be configured by maintainers before the workflows can run:

- **`OTELBOT_APP_ID`** (repo or org variable) and **`OTELBOT_PRIVATE_KEY`**
  (repo or org secret) — credentials for the `otelbot` GitHub App. The
  workflows commit and open PRs through this App so the resulting PRs
  trigger CI (a plain `GITHUB_TOKEN` doesn't).
- **PyPI Trusted Publishing** — each publishable package needs a trusted
  publisher on PyPI (see [PyPI publishing setup](#pypi-publishing-setup)).
  No long-lived PyPI API token is stored in GitHub secrets.
- **`pypi` GitHub environment** — release jobs deploy through this
  environment so publishing can be restricted (branch rules, required
  reviewers). See [GitHub environment setup](#github-environment-setup).
- **Branch protection** on `package-release/*/v*` branches so backport changes
  only land via reviewed PRs.

### PyPI publishing setup

For each publishable package, add a trusted publisher on PyPI
(*Manage* → *Publishing* → *Add a new pending publisher*):

| Field | Value |
|-------|-------|
| PyPI project name | e.g. `opentelemetry-util-genai` |
| Owner | `open-telemetry` |
| Repository name | `opentelemetry-python-genai` |
| Workflow name | `_release-package.yml` |
| Environment name | `pypi` |

All packages share the same workflow and environment. Register one publisher
entry per PyPI project. The first upload from CI activates the publisher.

Publishable packages (from `eachdist.ini`):

- `opentelemetry-instrumentation-genai-anthropic`
- `opentelemetry-instrumentation-genai-claude-agent-sdk`
- `opentelemetry-instrumentation-google-genai`
- `opentelemetry-instrumentation-genai-langchain`
- `opentelemetry-instrumentation-genai-openai`
- `opentelemetry-instrumentation-genai-openai-agents`
- `opentelemetry-instrumentation-genai-weaviate-client`
- `opentelemetry-util-genai`

### GitHub environment setup

Create a **`pypi`** environment under repository *Settings* → *Environments*.

Recommended restrictions:

- **Deployment branches**: limit to `main` and `package-release/**` (backport
  releases run from backport branches).
- **Required reviewers** (optional): gate PyPI uploads behind maintainer
  approval. Bulk `[All] Release` runs one matrix job per package; each job
  waits on the environment, so consider whether reviewers should approve once
  per run or per matrix leg.

Release jobs in `_release-package.yml` set `environment: pypi` and request
`id-token: write` so GitHub can mint a short-lived OIDC token for PyPI.

## Bulk release (default)

For releasing every package that has towncrier changelog fragments:

1. Run the
   [`[All] Prepare release`](./.github/workflows/release-all-prepare.yml)
   workflow against `main`.
   - Finds packages with fragments under `.changelog/`.
   - Opens one combined PR on `main` that drops `.dev` suffixes and runs
     `towncrier build` for each eligible package.
   - Labels the PR `release`.
2. Review and merge the prepare PR.
3. The
   [`[All] Release`](./.github/workflows/release-all.yml)
   workflow runs automatically when a labelled prepare PR merges (or trigger it
   manually against `main`).
   - Publishes each ready package to PyPI.
   - Creates a GitHub release tag (`<pkg>==<version>`) on `main` for each.
   - Opens a PR bumping released packages back to the next `.dev` version.

Packages without changelog fragments are skipped during prepare and logged in
the workflow output.

## Per-package release

Use when only one package needs to ship, or the rest of the workspace is not
ready for a bulk release.

1. Run
   [`[Package] Prepare release`](./.github/workflows/package-prepare-release.yml)
   against `main`. Select the package from the dropdown.
   - Opens a PR on `main` that drops the `.dev` suffix and runs
     `towncrier build`.
2. Review and merge the prepare PR.
3. Run
   [`[Package] Release`](./.github/workflows/package-release.yml)
   against `main`.
   - Builds the wheel, publishes to PyPI, creates the GitHub release tag, and
     opens PRs for any changelog date updates and the next `.dev` bump.

## Patch release (current minor line)

1. Land the fix on `main` as a normal PR (with a towncrier fragment).
2. Run
   [`[Package] Prepare patch release`](./.github/workflows/package-prepare-patch-release.yml)
   against `main`.
   - Same mechanics as prepare release: drops `.dev`, runs `towncrier build`.
3. Review and merge the prepare PR.
4. Run
   [`[Package] Release`](./.github/workflows/package-release.yml)
   against `main`.

## Backport patch (older minor line)

1. Create `package-release/<pkg>/v<X>.<Y>bx` from the `<pkg>==<X>.<Y>b<N>`
   tag if it does not exist yet.
2. Cherry-pick or develop the fix on the branch.
3. Run
   [`[Package] Prepare patch release`](./.github/workflows/package-prepare-patch-release.yml)
   against the backport branch.
   - Bumps the patch version and runs `towncrier build`.
4. Review and merge the prepare PR into the backport branch.
5. Run
   [`[Package] Release`](./.github/workflows/package-release.yml)
   against the backport branch.
   - Tags the backport branch and opens a PR copying changelog updates to
     `main`.

## Pre-existing static `## Unreleased` entries

Several packages carry CHANGELOG entries that pre-date towncrier (added
before the towncrier marker was inserted). `towncrier build` does **not**
fold them into the generated release section. Before the first towncrier
release of a given package, fold those entries by hand into the new
release section produced by `towncrier build` (or convert them into
fragments first). The do-not-edit comment in each `CHANGELOG.md` flags
this.

## Claiming a PyPI namespace for a new package

When a new package is introduced, create the PyPI project and register a
trusted publisher (see above) before the first CI release. Optionally upload
the current `.dev` version manually once to prevent name-squatting, shortly
after the introductory PR lands on `main`.

## Troubleshooting

### No packages found during `[All] Prepare release`

At least one publishable package needs a towncrier fragment under
`.changelog/` (any file other than `.gitkeep` / `.gitignore`).

### PyPI publish failed mid-workflow

Re-run the release workflow (`[Package] Release` or `[All] Release`). Trusted
Publishing only works from GitHub Actions — there is no repo-stored PyPI token
for manual `twine upload`.

If the wheel was built but upload failed, fix the underlying issue (PyPI
project missing, trusted publisher misconfigured, environment approval pending)
and re-run. The workflow uses `skip-existing`, so a partial upload is safe to
retry.

After a successful PyPI upload, re-running picks up remaining steps (GitHub
release tag + follow-up PRs) if those failed.

### Version still has a `.dev` suffix at release time

Merge the prepare PR first. Release workflows require a non-`.dev` version in
`version.py`.

## Out of scope

- A `backport` workflow (create backport branches manually from release tags
  when needed).
