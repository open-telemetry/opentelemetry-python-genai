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

### One span per operation

- When one wrapped entry point calls another wrapped one for the same logical operation (e.g.
  `run` calls `run_async`), emit only the outer span. Distinct nested operations - tools run by
  an agent, nested workflows, individual items of a batch - still get their own spans. Test nested
  and threaded topologies for duplicate spans.
- If the library also emits its own telemetry, coexist with it. Don't disable or alter it by
  default.

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
- Provider-agnostic helpers (base64 and data-URL handling, provider-name maps) belong in the
  util. Reuse the util's helper, or add it there, instead of writing
  a package-local copy.

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

- Use `opentelemetry.util.genai.iterator_capture.IteratorCapture` (requires declaring
  `opentelemetry-util-genai[wrapt]` in `pyproject.toml`) for captured request properties
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
  materialize, or otherwise trigger the result early. Stay as lazy as the original. For example,
  don't call `raw_response.parse()` (it runs user validators), and don't read request bodies or
  iterate inputs before the SDK does - record what the SDK reads through a proxy instead.
- **Don't change what the library receives.** Don't inject or alter values in kwargs, request
  objects, or results in ways the library or other callbacks can observe. Transparent wrappers that
  pass the same values through (e.g. `IteratorCapture`) are fine.
- **Don't change the returned type.** `isinstance` and `__class__` must still resolve to the
  original type. A transparent proxy (e.g. `wrapt.ObjectProxy`) satisfies this; returning a
  different or already-parsed type does not. A method that returned a coroutine must still return
  one.
- **Keep wrappers transparent.** A wrapper should be indistinguishable from what it wraps —
  attributes and behavior forward unchanged, only telemetry is added.
- **Decorate replacement functions with `@functools.wraps(original)`.** Whenever a function or bound
  method is swapped for a stand-in (`obj.close = _close`, wrapt patches), so introspection and
  `help()` still see the original. Not needed for proxy-class methods, which shadow rather than
  replace.

Full transparency isn't always reachable — prefer the least intrusive option that works.

### Streaming responses

Follow [Streaming responses](../util/opentelemetry-util-genai/AGENTS.md#streaming-responses): declare
`opentelemetry-util-genai[wrapt]` in `pyproject.toml`, subclass
`SyncStreamWrapper` / `AsyncStreamWrapper`, create the invocation with `_attach_to_context=False`,
and don't call `invocation.stop()` when the SDK returns the stream.

### Context propagation

- When the instrumented call returns, its span must not be current in the caller's context, on
  any path: streams (including tool streams), raw or unparsed responses, early returns. Create
  invocations with `_attach_to_context=False` and scope work with `with invocation:` or
  `invocation.activate()`; don't call the deprecated `invocation.suspend()`.
- For async APIs, start the invocation inside the awaited coroutine, not when the coroutine is
  created. Otherwise concurrent calls nest under each other.
- Don't suppress context detach errors. They signal mismatched attach/detach; fix the pairing.

## Library versions

- If a symbol you patch is optional or only exists in some supported versions, skip that patch
  when it's missing - and skip it on uninstrument too. One missing symbol must not break
  `instrument()` for the rest of the library.
- Patch the class or function where it's defined, not a module that re-exports it, so the patch
  doesn't depend on import order.
- Avoid patching private library internals. When you must, check them against the oldest and
  latest supported versions.
- Every version-specific branch needs a test that runs in the `oldest` or `latest` env. A test
  that is skipped in both is not coverage.

## Performance

Instrumentation runs on every call of the instrumented library.

- Resolve imports, function signatures, and feature checks once at instrument or import time.
  Binding each call's arguments to the signature still happens per call.
- Keep per-chunk stream processing O(1): append chunks and join or parse once at the end.
- Don't block the event loop or move work onto it from async code paths.

## Common bugs

- Use `is not None` for optional values. `or` and truthiness checks drop valid `0`, `0.0`, and
  `False`.
- Don't invent `'unknown'`, `''`, or `0` for missing optional attributes - leave them unset,
  unless semconv defines a fallback value (e.g. `unknown` for `gen_ai.provider.name`). Omit
  optional list attributes when empty; follow the attribute's semconv requirement level otherwise.
- Every code path that starts an invocation must stop or fail it, including early returns and
  results the caller never reads, parses, or awaits.
- A stream abandoned before completion is finalized as a failure.
- When fixing a bug, check affected sync/async counterparts and callers of changed shared
  helpers. Keep unrelated pre-existing bugs in sibling instrumentations as follow-up suggestions;
  don't expand the PR to fix them or require a tracking issue.

## Private APIs

- Don't use private (`_`-prefixed) modules, functions, or attributes of other packages (util,
  other instrumentations). Keep your own helpers private. For the instrumented library, prefer
  its public API and use internals only when there is no good public way to do it.

## Exception handling

- When catching exceptions from an underlying library after starting telemetry,
  catch `BaseException`, not `Exception`, so cancellation and interrupts
  finalize the invocation before the original exception is re-raised.
- Always re-raise the original exception unmodified.
- Do not raise new exceptions in instrumentation/telemetry code.
- Telemetry code must never fail the user's call or import. Make extraction unable to raise
  (typed access, `isinstance` checks) instead of wrapping wrappers in broad `try`/`except`. Use
  `try`/`except` only around parsing untyped data (base64, JSON, user-provided dicts), catch the
  specific exception, and drop only the affected attribute.

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
- Don't use any other deprecated util API either (e.g. `LLMInvocation`, deprecated part aliases
  or fields) - use the replacement named in its docstring.
- `gen_ai.provider.name`: map to semconv well-known values. Fall back to `unknown`, never the
  framework or module name.
- `gen_ai.agent.id`: only a stable identifier assigned by the provider (remote agents) - not
  `id(obj)` or the agent name.
- Finish reasons go in `gen_ai.response.finish_reasons`, one per generation in response order
  (`OutputMessage.finish_reason` is deprecated). Map provider values to the util `FinishReason`
  values where one applies (e.g. `tool_calls`/`tool_use` -> `tool_call`). Don't infer them from
  library internals or default to `stop`.
- Emit one span and one metric point per logical operation. Retries and calls delegated to
  another instrumented library don't count.
- Record operation-specific metrics with exactly the attribute set semconv lists for them. Keep
  ids and other high-cardinality values off metrics.
- Pass every sampling-relevant value the library exposes (agent name, conversation id, ...) to
  the `handler.*()` factory, so nested spans inherit context attributes. Don't set them again
  afterwards.

### Messages and content parts

- Map every content block type the library supports to a specific util part type. If none fits,
  extend the util. Don't silently drop known blocks, stringify them into `TextPart`, or fall
  back to `GenericPart`.
- Pass structured values (dicts, lists) to util fields as-is. Don't serialize them to JSON
  strings.
- Map provider token usage to semconv (input tokens include cached tokens) and record every usage
  field the provider reports, in streaming too.
- When content capture is disabled, don't build or accumulate message parts, tool definitions, or
  streamed text at all. Read the capture mode from the invocation at call time, not from env vars
  or at patch time.

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
- Tests that accompany a fix or behavior change should fail without it. Tests that only add
  coverage are fine. Assert exact values when the value is the contract; use presence, absence,
  length, or substring checks when those are the behavior under test.
- For every content-bearing attribute, test content capture both enabled and disabled (assert the
  content is absent). Don't drop existing negative assertions when editing tests.
- Drive the instrumented library's public API and assert exported telemetry. Don't test private
  helpers or hand-built mocks of SDK types - they hide field mismatches with the real SDK.
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
- Never set `os.environ` directly in tests. Use fixtures so values are restored. To run a test in
  each content-capture mode, use the shared parametrized `content_capture` fixture.

  ```python
  # Env vars read when the instrumentor is built: pass them to `instrument()` in a fixture.
  @pytest.fixture
  def instrument_with_hook(tracer_provider, logger_provider, meter_provider):
      with instrument(
          MyInstrumentor(),
          tracer_provider=tracer_provider,
          logger_provider=logger_provider,
          meter_provider=meter_provider,
          content_capture="SPAN_ONLY",
          extra_env={"OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK": "upload"},
      ) as instrumentor:
          yield instrumentor


  # A one-off value read at call time: monkeypatch.
  def test_invalid_capture_mode(monkeypatch):
      monkeypatch.setenv(
          "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "bogus"
      )
      assert get_content_capturing_mode() == ContentCapturingMode.NO_CONTENT
  ```
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
