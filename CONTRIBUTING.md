# Contributing

Welcome to the OpenTelemetry Python GenAI Instrumentations repository!

New to OpenTelemetry? Read the
[New Contributor Guide](https://github.com/open-telemetry/community/blob/main/guides/contributor/README.md)
first — it covers the CLA, Code of Conduct, and other prerequisites.

If you are using AI agents to assist with contributions, please also read
[AGENTS.md](AGENTS.md) for guidance on how to use them responsibly in this
project.

## Prerequisites

- [Python](https://www.python.org/downloads/) — see [`tox.ini`](tox.ini) for
  the supported versions.
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) — used to
  manage the workspace and to back the `tox` test environments.

## Code structure

```
├── instrumentation/
│   └── opentelemetry-instrumentation-<name>/  # one package per GenAI library
│       ├── src/opentelemetry/instrumentation/<name>/
│       ├── tests/
│       └── pyproject.toml
└── util/
    └── opentelemetry-util-genai/              # shared GenAI utilities
        ├── src/opentelemetry/util/genai/
        ├── tests/
        └── pyproject.toml
```

The monorepo uses `uv` workspaces; each package owns its own `pyproject.toml`,
version, and entry points. `tox.ini` defines the test matrix.

## Making a change

### 1. Set up the environment

Install all packages and dev tools into a single workspace virtual environment:

```sh
uv sync --frozen --all-packages
```

### 2. Lint

```sh
uv run pre-commit run ruff --all-files
```

### 3. Test

Run the test environment for the package you changed (append `-oldest` or
`-latest` for the version variants defined in `tests/requirements.{oldest,latest}.txt`):

```sh
uv run tox -e py312-test-instrumentation-openai-v2-latest
```

Run type checking across the workspace:

```sh
uv run tox -e typecheck
```

### 4. Update the changelog

This repo uses [towncrier](https://towncrier.readthedocs.io/) to manage
changelogs. Each PR with user-visible impact must add a changelog fragment
under the affected package's `.changelog/` directory rather than editing
`CHANGELOG.md` directly.

**Fragment path:** `<package>/.changelog/<PR_NUMBER>.<TYPE>`

**Types:** `added`, `changed`, `deprecated`, `removed`, `fixed`.

The file contains a one-line description. For example,
`instrumentation/opentelemetry-instrumentation-anthropic/.changelog/123.fixed`:

```
fix request hook not being called when stream=True
```

Don't include the PR number in the body — towncrier appends it from the
filename.

Preview the rendered changelogs locally:

```sh
uv run tox -e changelog-preview
```

If your change doesn't need an entry (pure docs/tooling), add the
`Skip Changelog` label to the PR.

## Keep PRs small

One logical change per PR. Don't bundle unrelated fixes, refactors, or
features — split them so each can be reviewed and reverted independently.
Small, focused PRs are much easier to review, and therefore much more
likely to land quickly.

If a PR review surfaces contentious or difficult points, consider splitting
those into follow-up PRs so the uncontroversial parts can land and each of
the harder points gets its own focused discussion and review.

## Asking questions

Post in
[#otel-genai-instrumentation](https://cloud-native.slack.com/archives/C06KR7ARS3X)
on [CNCF Slack](https://slack.cncf.io/) or join the next
[GenAI SIG](https://github.com/open-telemetry/community#sig-genai-instrumentation)
meeting and add your topic to the
[meeting agenda](https://docs.google.com/document/d/1EKIeDgBGXQPGehUigIRLwAUpRGa7-1kXB736EaYuJ2M).
See the [community repo](https://github.com/open-telemetry/community#sig-genai-instrumentation)
for current meeting times.

## Approvers and Maintainers

### Maintainers

- [Trask Stalnaker](https://github.com/trask), Microsoft
- [Liudmila Molkova](https://github.com/lmolkova)
- [Aaron Abbott](https://github.com/aabmass), Google

For more information about the maintainer role, see the [community repository](https://github.com/open-telemetry/community/blob/main/guides/contributor/membership.md#maintainer).

### Approvers

- [Dylan Russell](https://github.com/DylanRussell), Google
- [Mike Goldsmith](https://github.com/MikeGoldsmith), Honeycomb
- [Keith Decker](https://github.com/keith-decker), Cisco
- [Leighton Chen](https://github.com/lzchen), Microsoft

For more information about the approver role, see the [community repository](https://github.com/open-telemetry/community/blob/main/guides/contributor/membership.md#approver).
