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

Keep it short - prose under 200 words. Answer, in order:

- What does this change do? 2 sentences max: fixes issue XYZ, adds feature ABC, adds instrumentation for library Foo, etc.
- Why? Skip this for bug fixes.
- Any known gaps or concerns.

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

## Dependency versioning and compatibility

- Use compatible release specifiers (e.g., `~= x.y` or `>= x.y.z, < (x+1)`) that pin the major version and allow minor/patch updates.
- Avoid pinning versions to exact patch ranges (like `== x.y.z` or `~= x.y.z`) in `pyproject.toml` unless strictly necessary.
- Keep version requirements for shared OpenTelemetry packages (e.g., `opentelemetry-api`, `opentelemetry-instrumentation`, and `opentelemetry-semantic-conventions`) consistent across all packages.
- For OpenTelemetry-owned beta/pre-release packages (e.g., `opentelemetry-instrumentation`, `opentelemetry-semantic-conventions`, `opentelemetry-util-genai`), use `>=` specifiers and pin the upper boundary to the next major version (e.g., `>= 0.64b0, <1` for `0.x` packages, or `>= 1.0b0, <2` for `1.x` packages) rather than using `~=`.


## Before adding a new instrumentation

Before scaffolding a new instrumentation package, check whether the target library already
provides its own GenAI telemetry, and get the user's acknowledgement before proceeding. Follow the
[When to add an instrumentation here](CONTRIBUTING.md#when-to-add-an-instrumentation-here) policy.

- **Check the library's dependencies.** A runtime dependency on `opentelemetry-api` (or other
  `opentelemetry-*` packages) in the library's metadata (`pyproject.toml` / `setup.py` /
  `setup.cfg` / lockfile) can indicate it ships native instrumentation or a first-party
  OpenTelemetry integration — research it before writing any code. (A dev/test-only dependency,
  e.g. for examples, doesn't count.)
- **Check the library's public docs.** Look for an OpenTelemetry / observability / tracing
  integration that ships as a first-party plugin, even in a separate package that isn't a hard
  dependency.
- **Report findings and stop for acknowledgement.** Summarize whether the library is natively
  instrumented or has a first-party plugin, whether it's based on the OTel API, and how closely it
  follows the GenAI semantic conventions; map that onto the policy's decision tree and present a
  recommendation. **Do not scaffold or write instrumentation code until the user explicitly
  acknowledges that instrumentation here is needed.**


## Adding a package to the workspace

A new package under `instrumentation/<pkg>/` (where `<pkg>` is the full
`opentelemetry-instrumentation-genai-<lib>` directory name) wires in as follows.
Copy the shape from an existing package — paths in `tox.ini` are repo-root-relative.

- **uv workspace**: auto-included via the `instrumentation/*` glob in root
  `pyproject.toml [tool.uv.workspace] members` — no edit needed.
- **`CHANGELOG.md`**: copy from an existing package (e.g. `instrumentation/opentelemetry-instrumentation-genai-anthropic/CHANGELOG.md`).
- **`tox.ini`**:
  - `envlist`: add `py3{11,15}-test-instrumentation-genai-<lib>-latest`,
    `py311-test-instrumentation-genai-<lib>-oldest`, and the
    `py314-test-instrumentation-genai-<lib>-conformance` entry.
  - `[testenv] deps`: add the factor-conditional test-requirements lines
    (`<lib>-{oldest,latest,conformance}: -r …/tests/requirements.<factor>.txt` plus
    `{[testenv]test_deps}` / `{[testenv]pytest_deps}`). Requirements install here - **not**
    in `commands_pre`.
  - `[testenv] commands`: add the pytest line (it `--ignore`s `tests/test_conformance.py`) and
    the separate `…-conformance` pytest line.
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
uv run tox -e py314-test-instrumentation-genai-openai-conformance

# Type check (pyright)
uv run tox -e typecheck

# Check util-genai for public API breaks against main
uv run tox -e check-util-api
```

Before opening a PR, run `uv run tox -e precommit`, `uv run tox -e typecheck`, and the changed package's
test envs (`-oldest` and `-latest`, plus `-conformance` if it ships scenarios) — these mirror
the CI gates.

If you change `util/opentelemetry-util-genai`, also run `uv run tox -e check-util-api`. Released
instrumentations accept any util version below 2, so removing or changing a public object can
break them on import. If the break is intended, say in the PR which released instrumentations you
checked. A maintainer then adds the `Approve Public API check` label to skip the CI job.

tox reuses cached envs and won't re-resolve dependencies on its own, so pass `--recreate` (`-r`)
after editing a `tests/requirements.*.txt` or a `pyproject.toml` dependency bound — otherwise the
run silently uses the previously installed versions.

## Guidelines

- Each package has its own `pyproject.toml` with version, dependencies, and entry points.
- The monorepo uses `uv` workspaces.
- `tox.ini` defines the test matrix - check it for available test environments.
- Do not add `type: ignore` comments. If a type error arises, solve it properly or write a follow-up plan to address it in another PR.
- Annotate function signatures (parameters and return types) and class attributes. Prefer `from __future__ import annotations` over runtime-quoted strings.
- Avoid `Any` in the public API of `opentelemetry-util-genai` (anything not under a `_`-prefixed
  module). Use the narrowest type that fits: the semconv model's type where one exists, union, `AttributeValues`.
- When a file uses `from __future__ import annotations`, do not quote type annotations just to
  avoid forward references. Keep quotes only for expressions still evaluated at runtime, such as
  `typing.cast(...)`, unless the referenced type is imported at runtime.
- Whenever applicable, all code changes should have tests that actually validate the changes.

## Code comments

Write for someone reading the code as it is now, with no knowledge of the change that produced it.

- Comment only gotchas that can't be inferred from the code (protocol quirks, ordering
  requirements, upstream bugs worked around). 1-2 sentences max.
- Never restate what the code does, describe what it used to do, or justify a change - the last one
  belongs in the PR description or a PR comment.

## Self-review before opening a PR

Once the change implementation is complete and tests pass, review the diff in a **new session**. Look for:

- Test coverage gaps - sync/async variants, error paths, streaming, edge cases. Add the missing
  tests.
- Instrumentation of layers not owned by the target library (LLM spans reported by agentic framework).
- Functional issues

Fix the findings, keeping the PR scoped to the original change; for anything outside that scope,
suggest an issue or a separate PR. Do not open a PR with unresolved major issues or test gaps.

## Changelog

This repo uses [towncrier](https://towncrier.readthedocs.io/) to manage changelogs.

- Do not edit `CHANGELOG.md` directly — the `changelog` workflow rejects PRs that do.
- For changes with user-visible impact, add a fragment at `<package>/.changelog/<PR_NUMBER>.<type>`
  containing a one-line description. Types: `added`, `changed`, `deprecated`, `removed`, `fixed`.
- Don't include the PR number in the body — towncrier appends it from the filename.
- Preview locally with `uv run tox -e changelog-preview`.

## Instrumentation and Util rules

Detailed subsystem rules are maintained in their respective directories so they load on demand when working in those areas:

- For instrumentation package rules (telemetry handling, completion hooks, iterator capture, SDK contract preservation, README requirements, unit and conformance testing), see [instrumentation/AGENTS.md](instrumentation/AGENTS.md).
- For shared utility rules (semconv compliance, invocation lifecycle patterns, stream wrapper protocols), see [util/opentelemetry-util-genai/AGENTS.md](util/opentelemetry-util-genai/AGENTS.md).

## Commit formatting

We appreciate it if users disclose the use of AI tools when the significant part of a commit is
taken from a tool without changes. When making a commit this should be disclosed through an
`Assisted-by:` commit message trailer.

Examples:

```
Assisted-by: ChatGPT 5.2
Assisted-by: Claude Opus 4.6
```
