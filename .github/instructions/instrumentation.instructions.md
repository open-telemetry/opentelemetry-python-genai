---
applyTo: "instrumentation/**"
---

Review rules for PRs touching `instrumentation/**`. Flag violations with a link to the rule.

## 0. Reviewer mindset

Review as long-term maintainer.

For new instrumentations, consult upstream library docs and judge:

- Does the library already emit its own telemetry, making this instrumentation redundant?
- Is the library used widely enough to warrant a package in this repo?
- Does it avoid unbounded in-memory accumulation or other side-effects?

For changes to existing instrumentations: prefer back-compat. Break users only for a real reason;
prefer opt-in or additive. Breaking changes need explicit justification in the PR.

## 1. Maintenance commitment

- New instrumentations require contributor commitment to long-term maintenance. See
  [Expectations from contributors](../../CONTRIBUTING.md#expectations-from-contributors), the
  general [instrumentation checklist](../../CONTRIBUTING.md#guideline-for-instrumentations), and
  the GenAI-specific expectations in
  [`CONTRIBUTING.md#guideline-for-genai-instrumentations`](../../CONTRIBUTING.md#guideline-for-genai-instrumentations).

## 2. Telemetry and configuration via `opentelemetry-util-genai`

- Spans, logs, metrics, and events must go through `opentelemetry-util-genai`. Direct use of
  `Tracer`, `Meter`, `Logger`, or event APIs is not allowed.
- Content capture, hooks, and other cross-cutting configuration are owned by the util.
  Instrumentations must not introduce their own env vars, settings, or hook interfaces.
- Message content, prompts, and tool call arguments must only be set through the util's content
  capture path — never as unconditional span/log attributes.
- Adding attributes to invocations produced by the util is fine.
- If a capability is missing in `opentelemetry-util-genai`, land it in the util first.

## 3. Semantic conventions

- Attributes, spans, events, and metrics must match the
  [GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai/tree/main/docs/gen-ai).
- Attribute names must come from the semconv attribute modules, not hardcoded strings. Use the
  module matching the namespace under `opentelemetry.semconv` (e.g. `server_attributes`,
  `error_attributes`, `http_attributes`, `db_attributes`, …). `gen_ai.*` attribute names must
  come from `opentelemetry.semconv._incubating.attributes.gen_ai_attributes`.
- For attributes with a well-known value set in semconv, use the generated enum from the same
  module (e.g. `GenAiOutputTypeValues` for `gen_ai.output.type`) instead of string literals.
- If a signal is not in semconv, wait until semconv lands.

## 4. Exception handling

- When catching exceptions from the underlying library to record telemetry, always re-raise the
  original exception unmodified.
- Do not raise **new** exceptions in instrumentation/telemetry code.

## 5. Tests

- For every public API instrumented, cover sync/async variants when both exist.
- Cover streaming and non-streaming variants when both exist.
- Cover happy path and error scenarios. For error scenarios, at minimum include: provider error /
  endpoint unavailable, stream interrupted by network, stream closed early by the caller.
- Use recorded VCR cassettes for provider calls. No live-key-only tests; skipping on missing key
  is not acceptable.
- Tests must verify exact attribute names **and value types**, checked against the semconv spec.
- Test against oldest and latest supported library versions via `tests/requirements.{oldest,latest}.txt`
  and `{oldest,latest}` `tox.ini` factors.
- `tests/conftest.py` must consume the shared fixtures from `opentelemetry.test_util_genai`
  (`from opentelemetry.test_util_genai.fixtures import *` and
  `from opentelemetry.test_util_genai.vcr import fixture_vcr, scrub_response_headers`). Do not
  re-implement in-memory provider/exporter setup or the VCR pretty-print serializer locally.
- When recording VCR cassettes, scrub account-identifying values in the conftest's
  `vcr_config` (`filter_headers` for requests, `scrub_response_headers_overwrite` for
  responses) before committing. Examples: `authorization`, `openai-organization`,
  `openai-project`, `Set-Cookie`, and any response-body field tied to a real
  account.
- Conformance: packages ship `tests/conformance/<scenario>.py` modules (each
  defining a subclass of
  `opentelemetry.test_util_genai.conformance.Scenario` that sets
  `expected_spans`, `expected_metrics`, and implements `run(...)`) and a
  `tests/test_conformance.py` that runs them via
  `opentelemetry.test_util_genai.conformance.run_conformance`.

## 6. Examples

New instrumentations must ship a minimal example under the package's `examples/`, with both a
`manual/` and a `zero-code/` (auto-instrumentation) variant.

## 7. PR description

- Cover which part of the GenAI semconv the change implements or follows (when applicable) and
  how instrumentations should consume it.

## 8. Package naming and versioning

- Instrumentation packages must be named `opentelemetry-instrumentation-genai-{lib}` and import
  as `opentelemetry.instrumentation.genai.{lib}` (`opentelemetry-instrumentation-google-genai`
  is a pre-existing exception that keeps its historical name).
- Versions use the OpenTelemetry beta versioning format `MAJOR.MINORbN` (e.g. `1.0b0`);
  `version.py` carries a `.dev` suffix during development.

See also [AGENTS.md](../../AGENTS.md) for general repo rules.
