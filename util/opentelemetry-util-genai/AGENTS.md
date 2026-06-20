# GenAI Utils — Agent and Contributor Guidelines

This package provides shared telemetry utilities for OpenTelemetry GenAI instrumentation.

## 1. Semantic Convention Compliance

No new telemetry without semconv. If a signal, attribute, or operation is not in the
[OpenTelemetry GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/), land the semconv change first.

All attributes, operation names, and span names must match semconv.

Use the semconv attribute modules — do not hardcode attribute name strings:

- `gen_ai.*` attributes: `opentelemetry.semconv._incubating.attributes.gen_ai_attributes`
- `server.*` attributes: `opentelemetry.semconv.attributes.server_attributes`
- `error.*` attributes: `opentelemetry.semconv.attributes.error_attributes`
- Other namespaces: use the corresponding module from `opentelemetry.semconv`

Shared attributes should behave consistently across invocation types (same conditions, same
defaults). If an attribute applies to more than one invocation per semconv, set it on all
applicable ones.

## 2. Invocation Lifecycle Pattern

Every new operation type must follow this pattern:

```python
invocation = handler.inference(provider, request_model, server_address=..., server_port=...)
invocation.temperature = ...
try:
    response = client.call(...)
    invocation.response_model_name = response.model
    invocation.finish_reasons = response.finish_reasons
    invocation.stop()
except Exception as exc:
    invocation.fail(exc)
    raise
```

Factory methods on `TelemetryHandler` (`handler.py`):

- `inference(provider, request_model, *, server_address, server_port)` → `InferenceInvocation`
- `embedding(provider, request_model, *, server_address, server_port)` → `EmbeddingInvocation`
- `retrieval(*, data_source_id, provider, request_model, server_address, server_port)` → `RetrievalInvocation`
- `tool(name, *, arguments, tool_call_id, tool_type, tool_description)` → `ToolInvocation`
- `workflow(name)` → `WorkflowInvocation`

The returned object can also be used as a context manager (`with ... as invocation:`) when the span lifetime maps cleanly to a `with` block.

The above factories must map 1:1 to distinct semconv operation types (inference, embeddings,
retrieval, tool execution, agent invocation, workflow invocation). Names must match the operation
unambiguously — for example, `create_agent` and `invoke_agent` are different operations, so a
single `agent()` would be ambiguous and is not acceptable. Add a new factory per operation type
instead.

Factory names are Python-style singular verbs (`inference`, `embedding`, `retrieval`, `tool`, `workflow`); the op names
they map to follow semconv operations.

Factory methods must accept all attributes that semconv marks as important for sampling
decisions as parameters, so they are on the span at creation time. Attributes that are also
marked required by semconv must be required parameters (no default value). Operation name
is usually hardcoded in specific invocation and does not need to be passed.

### Streaming responses

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

### Anti-patterns

**Never construct invocation types directly** (`InferenceInvocation(...)`, `ToolInvocation(...)`,
etc.) in instrumentation or production code — direct construction skips span creation and context
propagation, so all telemetry calls become no-ops. Always use `handler.*()`.

## 3. Exception Handling

- When catching exceptions from the underlying library to record telemetry, always re-raise
  the original exception unmodified.
- Do not raise new exceptions in telemetry code.

## 4. Performance

Keep the hot path tight:

- Avoid per-invocation allocations; do not accumulate state unboundedly.
- Skip content capture when content capture is disabled.
- Skip setting span-only attributes when the span is not recording.
- Still record attributes that feed metrics — metric recording is independent of span sampling.

## 5. DRY

Do not copy-paste logic across invocation types. Extract shared helpers.

## 6. Documentation

- Docstrings for invocation types and span/event helpers must include a link to the
  corresponding operation in the semconv spec.
- When adding or changing attributes, update the docstring to describe what is set and under
  what conditions (e.g., "set only when `server_address` is provided").

## 7. Tests

- Every new operation type or attribute change must have tests verifying the exact attribute
  names **and value types**, checked against the semconv spec.
- Cover all paths: success (`invocation.stop()`), failure (`invocation.fail(exc)`), and any
  conditional attribute logic (e.g., attributes set only when optional fields are populated).
- Tests live in `tests/` — follow existing patterns there.
- Don't call internal API in tests when the public API is available.

## 8. Python API Conventions

- Mark private modules with an underscore. Objects inside a private module should be prefixed
  with an underscore if they are not used outside that module.
- When adding fields or methods on invocation types (or anywhere in the public surface), push
  back hard: does this need to be public? If instrumentations don't need it, keep it internal
  (`_`-prefixed). Every public addition becomes a back-compat commitment.
- Before removing or renaming an object exposed publicly, deprecate it first with a note in the
  docstring pointing to the replacement.
