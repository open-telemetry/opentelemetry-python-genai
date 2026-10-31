# Instrumentation Guidelines

Guidelines for building and maintaining packages under `instrumentation/`.

## Decide what the library supports

An instrumentation emits telemetry only for operations the instrumented library itself performs. If
library A delegates an operation to another (also instrumentable) library B, do **not** emit that
operation from A's instrumentation — it belongs to B, and the two correlate through standard context
propagation.

**Fact-check against the library's real API reference.** Do not guess what a library does or doesn't
support from its name or reputation. Before deciding an operation is in or out of scope, confirm it
against the library's official API docs — many SDKs cover more than their obvious use case (e.g.
`openai` has retrieval via vector stores / file search and conversation state, not just inference).
Describe each operation in terms of the concrete API that backs it.

Run each litmus test to decide which operations apply.

### Model call (inference / embeddings)

Before emitting an inference (`chat`) or `embeddings` span, ask:

> Does this library delegate the actual model call to another instrumentable library
> (e.g. `openai`, `litellm`, `google-genai`)?

- **Yes** → do **not** emit inference or embeddings. It belongs to the underlying LLM library. Emit
  only framework-owned operations (`invoke_agent`, `invoke_workflow`, `execute_tool`, `retrieval`).
- **No** (the library is itself the model-call boundary — it calls the model directly, or owns it
  across a process boundary no in-process instrumentable library can observe) → emit it, and
  correlate via standard context propagation.

### Tool execution

> Does the library execute the tool itself — i.e. does it provide a tool-execution helper
> (automatic function calling, an agent/tool runner) that invokes the tool and feeds the result
> back?

- **Yes** (e.g. `google-genai` automatic function calling, agent frameworks that run tools) →
  `execute_tool` is instrumentable here; emit it.
- **No** (the client only returns tool calls and the tool runs in application code the library never
  sees) → `execute_tool` is **not instrumentable** by this library. Do not emit it from a
  model-client scenario; a span around the app's own function is not something generic
  instrumentation of the client could produce.

### Client invoke / create agent

> Does this library create and invoke remote AI agents (e.g. AWS Bedrock Agent or OpenAI
> Assistants)?

- **Yes** → emit client spans for the create-agent and invoke-agent operations.
- **No** → do not emit client agent spans. If the library implements agentic scenarios in-process,
  check the internal invoke-agent test below instead.

### Internal invoke agent

> Does the library have a concept of an AI agent?

- **Yes** → emit an `invoke_agent` span only for scenarios that use the library's agent API.
- **No** (no formal agent concept, or the AI scenario doesn't involve agents) → check the workflow
  test instead.

### Workflow

> Does the library have a concept of an AI workflow or graph — an operation that combines arbitrary
> AI-related operations such as agent invocations, chained LLM calls, and other auxiliary
> operations?

- **Yes** → emit an `invoke_workflow` span around the library API that runs the workflow.
- **No** → don't emit a workflow span. A single agent run or a plain chain of calls is not an AI
  workflow.

## Telemetry via `opentelemetry-util-genai`

- Spans, logs, metrics, and events should go through `opentelemetry-util-genai`. Do not call OTel
  `Tracer`/`Meter`/`Logger` directly, and import only its public surface — never an
  `opentelemetry.util.genai._*` module.
- Construct the `TelemetryHandler` with the dotted instrumentation package path as
  `instrumentation_scope_name` and the package's `__version__` as
  `instrumentation_scope_version`. Use `__package__` when it resolves to that exact package path;
  do not use a submodule's `__name__`. Import `__version__` from the package's `version` module.
  Subclass `opentelemetry.test_util_genai.scope.TelemetryHandlerScopeTest` in each package to verify
  its exact name and version on spans, metrics, and logs.
- Content capture, hooks, and configuration are owned by the util. Don't add instrumentation-local
  env vars or settings.
- Models describing complex attributes are owned by `opentelemetry.util.genai.types`. Land new type
  in utils when missing instead of defining it in instrumentation. If need to add instrumentation-specific properties,
  extend it. Don't workaround with dicts or type check suppressions.

### Completion hook

The `CompletionHook` (`opentelemetry.util.genai.completion_hook`) lets users forward captured
prompt/completion content to external storage (e.g. object stores) instead of, or in addition to,
recording it inline. Wiring is owned by the util — instrumentations just pass the hook through to
the `TelemetryHandler`. Follow the OpenAI package
([`OpenAIInstrumentor`](opentelemetry-instrumentation-genai-openai/src/opentelemetry/instrumentation/genai/openai/__init__.py))
as the reference:

- In `_instrument(**kwargs)`, resolve the hook as
  `kwargs.get("completion_hook") or load_completion_hook()` and pass it to the handler
  (`TelemetryHandler(..., completion_hook=...)`). `load_completion_hook()` returns the hook
  named by `OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK` (e.g. `upload`) via its entry point, or a
  no-op. An explicit `instrument(completion_hook=…)` argument takes precedence over the env var.
- Don't define your own hook interface, call `on_completion` yourself, or wrap it in `try/except` —
  the util calls the hook and swallows hook exceptions internally.
- Document the capability in the package `README.rst` (both the
  `OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK=upload` env var with
  `OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH`, and the programmatic
  `instrument(completion_hook=…)` override) and ship a `custom_hook.py` example mirroring the
  OpenAI package.

### Capturing properties that may contain iterators

- Use `opentelemetry.util.genai.iterator_capture.IteratorCapture` for captured request properties
  that may contain iterators, including nested values. Wrap those properties, not all `kwargs`.
  Don't drain or replay them for telemetry.
- Use one instance per named property: `IteratorCapture.from_property(kwargs, "messages", enabled=capture_content)`.
  It replaces that property in `kwargs`. Use `capture.extract(extractor)`
  after SDK consumption, including failures and lazy stream entry.
  The extractor receives a snapshot with `()` for unread iterators and `None`
  for disabled or missing properties; `capture.snapshot()` exposes the raw snapshot.
- Preserve file objects. Extend the util for missing support; don't write local wrappers.
- Test nested generators, partial reads, errors, and files across supported sync/async and
  streaming variants.

```python
from opentelemetry.util.genai.iterator_capture import IteratorCapture

capture = IteratorCapture.from_property(
    kwargs, "messages", enabled=capture_content
)
try:
    response = sdk_call(**kwargs)
finally:
    captured_messages = capture.extract(get_input_messages)
```

### Preserve the SDK's return contract

Instrumentation observes; it should not change what a call returns or when its work happens.

- **No new side effects.** Don't do work the SDK didn't — building telemetry must never consume,
  materialize, or otherwise trigger the result early. Stay as lazy as the original.
- **Don't change the returned type.** `isinstance` and `__class__` must still resolve to the
  original type. A transparent proxy (e.g. `wrapt.ObjectProxy`) satisfies this; returning a
  different or already-parsed type does not.
- **Keep wrappers transparent.** A wrapper should be indistinguishable from what it wraps —
  attributes and behavior forward unchanged, only telemetry is added.
- **Decorate replacement functions with `@functools.wraps(original)`.** Whenever a function or bound
  method is swapped for a stand-in (`obj.close = _close`, wrapt patches), so introspection and
  `help()` still see the original. Not needed for proxy-class methods, which shadow rather than
  replace.

Full transparency isn't always reachable — prefer the least intrusive option that works.

## Exception handling

- When catching exceptions from an underlying library after starting telemetry,
  catch `BaseException`, not `Exception`, so cancellation and interrupts
  finalize the invocation before the original exception is re-raised.
- Always re-raise the original exception unmodified.
- Do not raise new exceptions in instrumentation/telemetry code.

## Semantic conventions

- Use the semconv attribute and metrics modules under `opentelemetry.semconv` — do not hardcode
  attribute or metric name strings.
- For attributes with a well-known value set, use the generated enum from the same module instead
  of string literals.
- Attributes whose value is a structured document (messages, tool definitions, system
  instructions, retrieval documents, …) have a **model** defined in
  [`models.py`](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/non-normative/models.py) and a JSON schema under `model/gen-ai/gen-ai-*.json`. `opentelemetry-util-genai` has models
  for them defined in `opentelemetry.util.genai.types`. Use these models, don't invent new types
  or hand-roll dicts.
- Don't use the deprecated `GenericPart`. Map content to a specific part type or drop it.

## Package documentation (README.rst)

- Each package's `README.rst` is published as its PyPI long description. When a change introduces
  user-visible changes to the public API, configuration (env vars, `instrument()` keyword
  arguments), supported operations/span types, or examples, update the package `README.rst` in the
  same PR so its claims stay accurate.
- `README.rst` must render on PyPI. Do not use Sphinx-only roles (e.g. `:class:`, `:mod:`); they
  fail the PyPI renderer. Run `uv run tox -e readme` to validate.

## Tests

- For every public API instrumented, cover sync/async variants when both exist.
- Cover happy path and error scenarios.
- For streamed responses, cover two exception paths — a stream-side error raised by the SDK
  mid-iteration (e.g. an injected `ConnectionError`) and a caller-side error raised inside the
  `with …stream(…) as stream:` block before the stream is drained. Assert both re-raise unchanged
  and still finalize the span with the matching `error.type`.
- Tests must verify exact attribute names **and value types**, checked against the semconv spec.
- Test against oldest and latest supported library versions via `tests/requirements.{oldest,latest}.txt`
  and `{oldest,latest}` `tox.ini` factors.
- The `oldest` env must install exactly the lower bounds declared in `pyproject.toml`
  (`dependencies` and the `instruments` extra) — the declared and tested versions must not drift.
  `UV_RESOLUTION=lowest-direct` (set on the `oldest` factor) derives them from `pyproject.toml`, so
  it stays the single source of truth; only pin test-only deps with no `pyproject.toml` bound in
  `tests/requirements.oldest.txt`.
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

Run via `uv run tox -e py314-test-instrumentation-genai-<lib>-conformance`. The
`*-conformance` tox envs target `tests/test_conformance.py` directly; the
regular `*-{oldest,latest}` envs `--ignore` it so they don't need the
OTLP/gRPC exporter or `weaver_live_check`.

The parallel PR-review rules live in
[`.github/instructions/instrumentation.instructions.md`](../.github/instructions/instrumentation.instructions.md)
and should be kept in sync with this section.
