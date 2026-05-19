# Release process

Every package in this repo releases independently. There is no coordinated
bulk release across packages.

Releases are driven by GitHub Actions workflows ported from
`opentelemetry-python-contrib`. They handle version bumps, changelog
generation (via [towncrier](https://towncrier.readthedocs.io/)), tagging,
PyPI publishing, GitHub releases, and back-merging the changelog to `main`.

## Prerequisites (one-time, repo-level)

These must be configured by maintainers before the workflows can run:

- **`OTELBOT_APP_ID`** (repo or org variable) and **`OTELBOT_PRIVATE_KEY`**
  (repo or org secret) — credentials for the `otelbot` GitHub App. The
  workflows commit and open PRs through this App so the resulting PRs
  trigger CI (a plain `GITHUB_TOKEN` doesn't).
- **`pypi_password`** (repo secret) — a PyPI API token with publish rights
  on the relevant packages. Tracked in #15: migrating to PyPI trusted
  publishing once package names are decided.
- **Branch protection** on `package-release/*/v*` branches so changes only
  land via reviewed PRs.

## Minor/major release flow

For releasing `<pkg>` (e.g. `opentelemetry-instrumentation-anthropic`):

1. Run the
   [`[Package] Prepare release`](./.github/workflows/package-prepare-release.yml)
   workflow against `main`. Select the package from the dropdown.
   - Creates a long-term release branch
     `package-release/<pkg>/v<X>.<Y>.x` (or `v<X>.<Y>bx` for unstable).
   - Opens **two PRs**:
     - PR against the release branch: drops the `.dev` suffix from
       `version.py`, builds the changelog via `towncrier build`.
     - PR against `main`: bumps `version.py` to the next `.dev` version,
       builds the changelog via `towncrier build` (so fragments don't
       carry over to the next release cycle).
2. Review and merge **both** PRs.
3. Run the
   [`[Package] Release`](./.github/workflows/package-release.yml)
   workflow against the `package-release/<pkg>/v*` branch.
   - Verifies the changelog PR was merged to `main`.
   - Builds the wheel via `scripts/build_a_package.sh` and publishes to
     PyPI via `twine`.
   - Creates a GitHub release tagged `<pkg>==<version>`.
   - Opens a back-merge PR against `main` copying the resolved changelog
     section (in case any edits landed on the release branch).
4. Review and merge the back-merge PR if one was created.

## Patch release flow

1. Check out the package's existing release branch
   `package-release/<pkg>/v<X>.<Y>.x`.
2. Land any patch PRs against this branch (cherry-pick or direct).
3. Run the
   [`[Package] Prepare patch release`](./.github/workflows/package-prepare-patch-release.yml)
   workflow with the release branch selected.
   - Opens a PR against the release branch bumping the patch version and
     running `towncrier build` against the patch fragments.
4. Review and merge the PR.
5. Run the
   [`[Package] Release`](./.github/workflows/package-release.yml)
   workflow against the release branch. Same effect as for a
   minor/major release.

## Pre-existing static `## Unreleased` entries

Several packages carry CHANGELOG entries that pre-date towncrier (added
before the towncrier marker was inserted). `towncrier build` does **not**
fold them into the generated release section. Before the first towncrier
release of a given package, fold those entries by hand into the new
release section produced by `towncrier build` (or convert them into
fragments first). The do-not-edit comment in each `CHANGELOG.md` flags
this.

## Claiming a PyPI namespace for a new package

When a new package is introduced, release the current `.dev` version under
the `opentelemetry` PyPI org to prevent name-squatting. Do this shortly
after the introductory PR lands on `main`.

## Troubleshooting

### PyPI publish failed mid-workflow

Switch to the release branch locally and re-run the publish step manually:

```sh
git checkout package-release/<pkg>/v<X>.<Y>.x
./scripts/build_a_package.sh
twine upload --skip-existing --verbose dist/*
```

Then re-run the `[Package] Release` workflow to pick up the remaining
steps (GitHub release + back-merge PR).

## Out of scope

- A `backport` workflow (none yet — add when there's a real long-term
  release branch to backport into).
- Coordinated cross-package releases (every package here is independent;
  `eachdist.ini` lists all publishable packages under `[exclude_release]`).
