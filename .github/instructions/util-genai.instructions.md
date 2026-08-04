---
applyTo: "util/opentelemetry-util-genai/**"
---

Review rules for PRs touching `util/opentelemetry-util-genai/**`. Flag violations with a link to
the rule.

## 0. Reviewer mindset

Review as long-term maintainer. Every GenAI instrumentation in the repo depends on this package;
churn breaks them. Default to back-compat changes. Every public addition is a long-term
commitment — limit public API.

## 1. Semconv first

No code without semconv. If a signal, attribute, or operation is not in the
[GenAI semconv](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/),
land the semconv change first.

## 2. Semantic conventions

- Names (attributes, operations, spans) must match semconv exactly.
- Use matching `opentelemetry.semconv` modules per namespace (`gen_ai`, `server`, `error`, …).
  Do not hardcode name strings if a constant exists.
- Shared attributes must behave consistently across invocation types (same conditions, same
  defaults). If semconv marks an attribute for multiple invocations, set it on all.
- Attributes whose value is a structured document (messages, tool definitions, system
  instructions, retrieval documents, …) are modeled in
  [`models.py`](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/non-normative/models.py),
  with a generated JSON schema under `model/gen-ai/gen-ai-*.json`. Check any `types.py` change
  against the ref pinned in `versions.env` (`SEMCONV_GENAI_REF`), not `main`, and flag:
  - a field whose name, optionality, or value type diverges from the model — e.g. required where
    the model says optional, or a field the model does not declare;
  - a type added or edited without citing the model it mirrors;

## 3. API backwards compatibility

- Do not remove or rename public objects. Deprecate first via a docstring note pointing to the
  replacement (not `@deprecated` — unreliable).
- Private modules and module-private objects start with `_`.
- Default to internal (`_`-prefixed) unless instrumentations need it public.
- Flag every `Any` in the public API (anything not under a `_`-prefixed module) — parameters,
  return types, and dataclass fields alike. Require the narrowest type that fits: the semconv
  model's type where one exists, `AttributeValue`, or a union/`TypeAlias` of the shapes actually accepted.

## 4. Invocation shape

- `start_*()` factories must accept all sampling-relevant semconv attributes as parameters.
  Attributes also marked required by semconv must be required parameters (no default value).
- `start_*()` factories must map 1:1 to distinct semconv operation types (inference, embeddings,
  tool execution, agent invocation, workflow invocation). Names must match the operation
  unambiguously — e.g., `create_agent` vs `invoke_agent` are distinct ops; `start_agent()` alone
  is ambiguous.
- Each operation exposes both a factory (`start_inference(...)`) and a context-manager
  (`inference(...)`) form.
- Never construct invocation types directly (`InferenceInvocation(...)`) — skips span creation,
  silent no-ops. Always use `handler.start_*()` or the context manager.

## 5. Exception handling

- When catching exceptions from the underlying library to record telemetry, always re-raise the
  original exception unmodified.
- Do not raise **new** exceptions in utils code.

## 6. Performance

Keep the hot path tight:

- Avoid per-invocation allocations; do not accumulate state unboundedly.
- Skip content capture when content capture is disabled.
- Skip setting span attributes when the span is not recording.
- Still record attributes that feed metrics — metric recording is independent of span sampling.

## 7. DRY

Do not copy-paste logic across invocation types. Extract shared helpers.

## 8. Tests

- Every new operation type or attribute change needs tests asserting exact attribute names **and
  value types**, checked against semconv.
- Cover success (`invocation.stop()`), failure (`invocation.fail(exc)`), and conditional
  attribute logic.
- Prefer public API in tests.

## 9. Documentation

- Docstrings for invocation types and span/event helpers must link to the corresponding semconv
  op.
- When adding/changing attributes, update the docstring with what is set and when.

## 10. PR description

- Cover which part of the GenAI semconv the change implements or follows (when applicable) and
  how instrumentations should consume it.

See also [AGENTS.md](../../AGENTS.md) for general repo rules.
