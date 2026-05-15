# Release process

This repo follows an **independent per-package release model**: every
publishable package owns its own version, `CHANGELOG.md`, and release cadence.
There is no coordinated bulk release across packages. The packages listed in
`[exclude_release]` of [`eachdist.ini`](eachdist.ini) are excluded from any
bulk-release tooling carried over from `opentelemetry-python-contrib`.

> [!IMPORTANT]
> The CI workflows for fully automated releases (`prepare-release`,
> `package-release`, etc.) have **not yet been ported** from
> `opentelemetry-python-contrib`. Until they are, releases are performed
> manually using the steps below. Tracked in #15.

## Publishable packages

| Package | PyPI name |
| --- | --- |
| `opentelemetry-instrumentation-anthropic`        | _TBD_ |
| `opentelemetry-instrumentation-claude-agent-sdk` | _TBD_ |
| `opentelemetry-instrumentation-google-genai`     | _TBD_ |
| `opentelemetry-instrumentation-langchain`        | _TBD_ |
| `opentelemetry-instrumentation-openai-agents-v2` | _TBD_ |
| `opentelemetry-instrumentation-openai-v2`        | _TBD_ |
| `opentelemetry-instrumentation-weaviate`         | _TBD_ |
| `opentelemetry-util-genai`                       | _TBD_ |

> [!NOTE]
> The PyPI names will differ from the names used in
> `opentelemetry-python-contrib`. Until each name is chosen and reserved on
> PyPI, packages cannot be published from this repo. Tracked in #15.

## Releasing a package

The steps below cover releasing a single package end-to-end. Repeat per
package as needed.

### 1. Prepare the changelog

Changelog entries are managed with [towncrier](https://towncrier.readthedocs.io/).
Each PR adds a fragment under `<package>/.changelog/<PR_NUMBER>.<type>`; at
release time those fragments are compiled into the package's `CHANGELOG.md`.

From the package directory, build the new release section into `CHANGELOG.md`:

```sh
cd <path/to/package>
uv run towncrier build --version <new-version>
```

This consumes every fragment under `./.changelog/` and inserts a new
`## Version <new-version> (<date>)` block. Commit the resulting changes.

> [!NOTE]
> Several `CHANGELOG.md` files have pre-existing entries under the static
> `## Unreleased` section that pre-date towncrier. These entries are **not**
> picked up by `towncrier build`. Fold them by hand into the new release
> block before committing.

### 2. Bump the version

Each package's version lives in
`<package>/src/.../version.py` (path depends on the package). Bump the
version (e.g. drop the `.dev` suffix for the release, then bump it back
afterwards for the next development cycle).

### 3. Tag and build

Tags follow the format `<pypi-name>==<version>` (matches the contrib
convention, used by [`scripts/build_a_package.sh`](scripts/build_a_package.sh)).

```sh
git tag <pypi-name>==<version>
git push --tags

PACKAGE_NAME=<pypi-name> VERSION=<version> ./scripts/build_a_package.sh
```

`build_a_package.sh` writes the `.tar.gz` and `.whl` into `dist/`.

### 4. Publish to PyPI

```sh
twine upload --skip-existing dist/*
```

> [!IMPORTANT]
> Publishing requires PyPI maintainer rights on the package. The token / API
> credential setup is not yet automated. Maintainers should publish from a
> machine that already has `~/.pypirc` configured, or via a dedicated CI
> workflow once one exists.

### 5. Create a GitHub release

```sh
gh release create <pypi-name>==<version> \
  --title "<pypi-name> <version>" \
  --notes-file <(awk '/^## Version <version>/,/^## /{print}' <package>/CHANGELOG.md | sed '$d')
```

Or use the GitHub UI: navigate to the new tag and click "Draft release",
pasting in the relevant section from the package's `CHANGELOG.md`.

### 6. Bump to the next development version

Open a follow-up PR bumping the package's `version.py` to the next
`X.Yb<N>.dev` (or `X.Y.<N>.dev` for stable components), so subsequent
fragments accumulate against the next release.

## Releasing a dev version to claim the PyPI namespace

When introducing a new package, release the current development version
under the `opentelemetry` PyPI org to prevent name-squatting. Do this
shortly after the introductory PR lands on `main`.

## Version numbering

- **Unstable components** (everything currently in this repo): versions look
  like `0.Yb0.dev` on `main`, `0.Yb0` at release time, then bump to
  `0.{Y+1}b0.dev`.
- **Stable components**: versions look like `X.Y.0.dev` on `main`, `X.Y.0`
  at release time, then bump to `X.{Y+1}.0.dev`. None of the current
  packages are stable.

## What's still missing

Tracked in #15:

- New PyPI names for each package
- `prepare-release` workflow (cuts the release PR and bumps versions)
- `package-release` workflow (tags, builds, publishes, opens GitHub release)
- `backport` workflow for patch releases on long-term release branches
- Patch release tooling once a package has a long-term release branch
