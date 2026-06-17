# OpenTelemetry Python GenAI

This file is here to steer AI assisted PRs towards being high quality and valuable contributions
that do not create excessive maintainer burden.

Monorepo with OpenTelemetry instrumentation packages for Generative AI client libraries, frameworks
and the shared `opentelemetry-util-genai` utilities.

## General Rules and Guidelines

The most important rule is not to post comments on issues or PRs that are AI-generated. Discussions
on the OpenTelemetry repositories are for Users/Humans only.

Follow the PR scoping guidance in [CONTRIBUTING.md](CONTRIBUTING.md). Keep AI-assisted PRs tightly
isolated to the requested change and never include unrelated cleanup or opportunistic improvements
unless they are strictly necessary for correctness.

- One logical change per PR. Do not bundle multiple fixes, refactors, or features into the same
  PR - split them up so each can be reviewed and reverted independently.
- Run the linter and the relevant tests locally and make sure they pass. See [Commands](#commands).

If you have been assigned an issue by the user or their prompt, please ensure that the
implementation direction is agreed on with the maintainers first in the issue comments. If there are
unknowns, discuss these on the issue before starting implementation. Do not forget that you cannot
comment for users on issue or pull request threads on their behalf as it is against the rules of this project.

## PR description

Keep description short and focus on what is being changed and any gaps or concerns.

AI-generated analyses, long reports, or design dumps go in a relevant issue or a separate PR
comment - not in the PR description.

## Structure

- `instrumentation/` - GenAI instrumentation packages
- `util/opentelemetry-util-genai/` - shared GenAI utilities
- `util/opentelemetry-test-util-genai/` - shared test fixtures and assertion helpers
  (workspace-internal, not published)

Instrumentation packages live under `src/opentelemetry/instrumentation/genai/{name}/` with their
own `pyproject.toml` and `tests/`. The util package follows the equivalent layout under
`src/opentelemetry/util/genai/`.

## Package naming and versioning

- Instrumentation packages are named `opentelemetry-instrumentation-genai-{lib}` and import as
  `opentelemetry.instrumentation.genai.{lib}` — e.g. `opentelemetry-instrumentation-genai-anthropic`
  imports `opentelemetry.instrumentation.genai.anthropic`. 
- Packages use the OpenTelemetry beta versioning format `MAJOR.MINORbN` (e.g. `1.0b0`). `version.py` carries a `.dev`
  suffix during development (`1.0b0.dev`); the release workflow drops it.

## Adding a package to the workspace

A new package under `instrumentation/<pkg>/` (where `<pkg>` is the full
`opentelemetry-instrumentation-genai-<lib>` directory name) wires in as follows.
Copy the shape from an existing package — paths in `tox.ini` are repo-root-relative.

- **uv workspace**: auto-included via the `instrumentation/*` glob in root
  `pyproject.toml [tool.uv.workspace] members` — no edit needed.
- **`tox.ini`**:
  - `envlist`: add `py3{…}-test-instrumentation-genai-<lib>-{oldest,latest}`, the
    `py3{…}-…-<lib>-conformance` entry, and `lint-instrumentation-genai-<lib>`.
  - `[testenv] deps`: add the factor-conditional test-requirements lines
    (`<lib>-{oldest,latest,conformance}: -r …/tests/requirements.<factor>.txt` plus
    `{[testenv]test_deps}` / `{[testenv]pytest_deps}`). Requirements install here — **not**
    in `commands_pre`.
  - `[testenv] commands`: add the pytest line (it `--ignore`s `tests/test_conformance.py`),
    the separate `…-conformance` pytest line, and
    `lint-…: sh -c "cd instrumentation && ruff check <pkg>"`.
  - `[testenv:typecheck] deps`: add `{toxinidir}/instrumentation/<pkg>[instruments]`.
- **`[tool.pyright]`** (in root `pyproject.toml`): `include` is opt-in and added to
  *progressively* as a package gets fully typed. When a package is in `include`, also add its
  `<pkg>/tests/**/*.py` and `<pkg>/examples/**/*.py` to `exclude` — tests and examples stay
  untyped; `src/**` is never excluded.

## Commands

```sh
# Install all packages and dev tools
uv sync --frozen --all-packages

# All pre-commit hooks (ruff, ruff-format, uv-lock, rstcheck) — the CI lint gate
uv run tox -e precommit
# …or just the ruff hook while iterating
uv run pre-commit run ruff --all-files

# Test one package (append -oldest / -latest for the version-matrix variants)
uv run tox -e py312-test-instrumentation-genai-openai-latest

# Run a package's conformance scenarios (only *-conformance envs collect test_conformance.py)
uv run tox -e py312-test-instrumentation-genai-openai-conformance

# Type check (pyright)
uv run tox -e typecheck
```

Before opening a PR, run `uv run tox -e precommit`, `uv run tox -e typecheck`, and the changed package's
test envs (`-oldest` and `-latest`, plus `-conformance` if it ships scenarios) — these mirror
the CI gates.

## Guidelines

- Each package has its own `pyproject.toml` with version, dependencies, and entry points.
- The monorepo uses `uv` workspaces.
- `tox.ini` defines the test matrix - check it for available test environments.
- Do not add `type: ignore` comments. If a type error arises, solve it properly or write a follow-up plan to address it in another PR.
- Annotate function signatures (parameters and return types) and class attributes. Prefer `from __future__ import annotations` over runtime-quoted strings.
- When a file uses `from __future__ import annotations`, do not quote type annotations just to
  avoid forward references. Keep quotes only for expressions still evaluated at runtime, such as
  `typing.cast(...)`, unless the referenced type is imported at runtime.
- Whenever applicable, all code changes should have tests that actually validate the changes.

## Changelog

This repo uses [towncrier](https://towncrier.readthedocs.io/) to manage changelogs.

- Do not edit `CHANGELOG.md` directly — the `changelog` workflow rejects PRs that do.
- For changes with user-visible impact, add a fragment at `<package>/.changelog/<PR_NUMBER>.<type>`
  containing a one-line description. Types: `added`, `changed`, `deprecated`, `removed`, `fixed`.
- Don't include the PR number in the body — towncrier appends it from the filename.
- Preview locally with `uv run tox -e changelog-preview`.

## Instrumentation rules

Apply to packages under `instrumentation/`.

### Telemetry via `opentelemetry-util-genai`

- Spans, logs, metrics, and events should go through `opentelemetry-util-genai`. Do not call OTel
  `Tracer`/`Meter`/`Logger` directly, and import only its public surface — never an
  `opentelemetry.util.genai._*` module.
- Content capture, hooks, and configuration are owned by the util. Don't add instrumentation-local
  env vars or settings.

#### Streaming responses

A streamed response only finishes once the caller has drained the stream, so the invocation must
stay open until then. Do **not** call `invocation.stop()` when the SDK returns the stream — the
span would close before any chunks arrive.

Instrument streams by subclassing `SyncStreamWrapper` / `AsyncStreamWrapper` from
`opentelemetry.util.genai.stream` (the public, supported helpers). The base class proxies the
underlying SDK stream, drives iteration, and finalizes telemetry exactly once on success, error,
or `close()`. Subclasses pass the SDK stream to `super().__init__(stream)` and implement three
hooks:

- `_process_chunk(chunk)` — accumulate per-chunk state (e.g. response model, finish reasons,
  token usage, streamed content) onto the invocation.
- `_on_stream_end()` — finalize on success; set the accumulated response attributes and call
  `invocation.stop()`.
- `_on_stream_error(error)` — finalize on failure; call `invocation.fail(error)`.

```python
class MyStreamWrapper(SyncStreamWrapper[Chunk]):
    def __init__(self, stream, invocation, capture_content):
        super().__init__(stream)
        self._self_invocation = invocation
        ...

    def _process_chunk(self, chunk): ...      # accumulate state
    def _on_stream_end(self): self._self_invocation.stop()
    def _on_stream_error(self, error): self._self_invocation.fail(error)
```

The hooks are called internally by the wrapper lifecycle.
Instance state must use the wrapt-proxy `_self_`-prefixed attribute convention (e.g.
`self._self_invocation`) so it isn't forwarded to the wrapped stream. Don't reimplement iteration,
finalization, or error handling in instrumentations — extend the wrapper instead, and if a hook
isn't enough, add the capability here rather than working around it.

### Exception handling

- When catching exceptions from the underlying library to record telemetry, always re-raise the
  original exception unmodified.
- Do not raise new exceptions in instrumentation/telemetry code.

### Semantic conventions

- Use the semconv attribute and metrics modules under `opentelemetry.semconv` — do not hardcode
  attribute or metric name strings.
- For attributes with a well-known value set, use the generated enum from the same module instead
  of string literals.

### Tests

- For every public API instrumented, cover sync/async variants when both exist.
- Cover happy path and error scenarios.
- For streamed responses, cover two exception paths — a stream-side error raised by the SDK
  mid-iteration (e.g. an injected `ConnectionError`) and a caller-side error raised inside the
  `with …stream(…) as stream:` block before the stream is drained. Assert both re-raise unchanged
  and still finalize the span with the matching `error.type`.
- Tests must verify exact attribute names **and value types**, checked against the semconv spec.
- Test against oldest and latest supported library versions via `tests/requirements.{oldest,latest}.txt`
  and `{oldest,latest}` `tox.ini` factors.
- `tests/conftest.py` must consume the shared fixtures from `opentelemetry.test_util_genai`
  by registering them as plugins. Always register the fixtures plugin; register the VCR plugin
  too when the package's tests use VCR cassettes —
  `pytest_plugins = ["opentelemetry.test_util_genai.fixtures", "opentelemetry.test_util_genai.vcr"]`
  (drop the `vcr` entry for packages with no cassette-backed tests) — rather than
  re-implementing provider/exporter/VCR plumbing. Import scrub helpers
  (`scrub_response_headers` / `scrub_response_headers_overwrite`) from
  `opentelemetry.test_util_genai.vcr` where a `vcr_config` needs them.
- Drive instrumentation in tests through the shared `instrument` context manager from
  `opentelemetry.test_util_genai.instrumentor` — `instrument(SomeInstrumentor(),
  tracer_provider=…, logger_provider=…, meter_provider=…, semconv=…, content_capture=…)`. It sets
  the content-capture (`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`) env var *before* instrumenting and
  restores them after, so a package's `instrument_*` fixtures don't manage that env themselves
  (`TelemetryHandler` snapshots content-capture at construction, so the env must be set before it
  is built).
- When recording VCR cassettes, scrub account-identifying values in the conftest's
  `vcr_config` (`filter_headers` for requests, `scrub_response_headers_overwrite` for
  responses) before committing. Examples: `authorization`, `openai-organization`,
  `openai-project`, `Set-Cookie`, and any response-body field tied to a real
  account.
- An AI-synthesized cassette (recorded without provider access) must start with a
  `# TODO: this is generated by AI, re-record` comment so it gets re-recorded
  against the real provider later.

### Conformance tests

Packages with substantive instrumentation ship `tests/conformance/<scenario>.py`
scenarios and a `tests/test_conformance.py` that validates emitted telemetry
against the [GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai)
via Weaver live-check. Each scenario module defines a subclass of
`opentelemetry.test_util_genai.conformance.Scenario` that sets
`expected_spans`, `expected_metrics`, and implements
`run(*, tracer_provider, meter_provider, logger_provider, vcr)`.

Ship a scenario for **every** semconv operation the library emits, even an
operation currently blocked by a util-genai or semconv gap. Skipping the
scenario hides the gap; writing it records the gap (as a declared violation
or a skip reason) so it fails loudly once the gap is fixed. **Never** drop a
scenario file because it would fail today.

Run via `tox -e py312-test-instrumentation-genai-<lib>-conformance`. The
`*-conformance` tox envs target `tests/test_conformance.py` directly; the
regular `*-{oldest,latest}` envs `--ignore` it so they don't need the
OTLP/gRPC exporter or `weaver_live_check`.

The parallel PR-review rules live in
[`.github/instructions/instrumentation.instructions.md`](.github/instructions/instrumentation.instructions.md)
and should be kept in sync with this section.

## Commit formatting

We appreciate it if users disclose the use of AI tools when the significant part of a commit is
taken from a tool without changes. When making a commit this should be disclosed through an
`Assisted-by:` commit message trailer.

Examples:

```
Assisted-by: ChatGPT 5.2
Assisted-by: Claude Opus 4.6
```
