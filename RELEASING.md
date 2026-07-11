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

## Bulk release (default)

For releasing every package that has towncrier changelog fragments:

1. Run the
   [`Prepare minor release`](./.github/workflows/prepare-minor.yml)
   workflow against `main`. Leave the `package` input empty for the bulk case.
   - Finds packages with fragments under `.changelog/`.
   - Opens one combined PR on `main` that drops `.dev` suffixes and runs
     `towncrier build` for each eligible package.
   - Labels the PR `release`.
2. Review and merge the prepare PR.
3. The
   [`Release all`](./.github/workflows/release-all.yml)
   workflow runs automatically when a labelled prepare PR merges (or trigger
   it manually against `main`).
   - Publishes each ready package to PyPI.
   - Creates a GitHub release tag (`<pkg>==<version>`) on `main` for each.
   - Opens a PR bumping released packages back to the next `.dev` version.

Packages without changelog fragments are skipped during prepare and logged in
the workflow output.

## Single-package minor release

Use when only one package needs to ship on the current minor line, or the
rest of the workspace is not ready for a bulk release.

1. Run
   [`Prepare minor release`](./.github/workflows/prepare-minor.yml)
   against `main` and set the `package` input to the target package. The
   workflow opens a PR that drops the `.dev` suffix and runs
   `towncrier build` for just that package (still labelled `release`).
2. Review and merge the prepare PR.
3. Either wait for `Release all` to fire on the merged prepare PR, or run
   [`Release package`](./.github/workflows/release-package.yml)
   against `main` for that one package.

## Patch release (current minor line)

> [!NOTE]
> This only works if the `major.minor` version on `main` is the same version you want to patch (i.e., before the post-release bump PR to the next minor version is merged). Once `main` is bumped to the next minor dev version, all patch releases for the older minor version must use the backport workflow instead.

1. Land the fix on `main` as a normal PR (with a towncrier fragment).
2. Run
   [`Prepare package patch release`](./.github/workflows/prepare-package-patch.yml)
   against `main`. Drops `.dev` and runs `towncrier build`.
3. Review and merge the prepare PR.
4. Run
   [`Release package`](./.github/workflows/release-package.yml)
   against `main`.

## Backport patch (older minor line)

1. Create `package-release/<pkg>/v<X>.<Y>bx` from the `<pkg>==<X>.<Y>b<N>`
   tag if it does not exist yet.
2. Cherry-pick or develop the fix on the branch.
3. Run
   [`Prepare package patch release`](./.github/workflows/prepare-package-patch.yml)
   against the backport branch. Bumps the patch version and runs
   `towncrier build`.
4. Review and merge the prepare PR into the backport branch.
5. Run
   [`Release package`](./.github/workflows/release-package.yml)
   against the backport branch.
   - Tags the backport branch and opens a PR copying changelog updates to
     `main`.

## Major release

Major bumps (e.g. `1.YbN` → `2.0b0`) are not automated. To release a major:

1. Open a PR against `main` that manually edits the target package's
   `version.py` from `X.YbN.dev` to `(X+1).0b0.dev`.
2. Merge that PR, add a changelog fragment describing the major change, then
   follow the standard bulk-release or single-package minor path above.
   `Prepare minor release` picks up the new version verbatim.

## Pre-existing static `## Unreleased` entries

Several packages carry CHANGELOG entries that pre-date towncrier (added
before the towncrier marker was inserted). `towncrier build` does **not**
fold them into the generated release section. Before the first towncrier
release of a given package, fold those entries by hand into the new
release section produced by `towncrier build` (or convert them into
fragments first). The do-not-edit comment in each `CHANGELOG.md` flags
this.

## Adding a new publishable package

When a new package is ready to ship:

1. Add its name to the `packages=` list under `[release_packages]` in
   `eachdist.ini`. Packages not listed here are skipped by the release
   workflows.
2. Add the package to the dropdown options in the workflow files that offer
   a package selector: `release-package.yml`, `prepare-minor.yml`, and
   `prepare-package-patch.yml`.
3. Create the PyPI project and register **two** trusted publishers (*Manage*
   → *Publishing* → *Add a new pending publisher*), one for each workflow
   that publishes. For detailed instructions, refer to PyPI's documentation on
   [Creating a PyPI project with a Trusted Publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
   or [Adding a Trusted Publisher to an existing PyPI project](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).
   Note that
   [creating a pending publisher does not reserve the project name on PyPI](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/):

| Field             | Entry 1                         | Entry 2                      |
|-------------------|---------------------------------|------------------------------|
| PyPI project name | e.g. `opentelemetry-util-genai` | same                         |
| Owner             | `open-telemetry`                | `open-telemetry`             |
| Repository name   | `opentelemetry-python-genai`    | `opentelemetry-python-genai` |
| Workflow name     | `release-package.yml`           | `release-all.yml`            |
| Environment name  | `pypi`                          | `pypi`                       |

4. Optionally reserve the package name to prevent name-squatting shortly after
   the introductory PR lands on `main` by navigating to
   <https://pypi.org/manage/organization/opentelemetry/projects/>, scrolling to
   the bottom (**Add project to organization**), and using the form.

All packages share the same environment. The first upload from CI activates
each publisher.

## Troubleshooting

### No packages found during `Prepare minor release`

At least one publishable package needs a towncrier fragment under
`.changelog/` (any file other than `.gitkeep` / `.gitignore`).

### PyPI publish failed mid-workflow

Re-run the release workflow (`Release package` or `Release all`). Trusted
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
- An automated major-bump workflow (see [Major release](#major-release)).
