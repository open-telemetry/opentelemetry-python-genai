# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
Do *NOT* add changelog entries here!

This changelog is managed by towncrier and is compiled at release time.

See https://github.com/open-telemetry/opentelemetry-python-genai/blob/main/CONTRIBUTING.md#changelog for details.
-->

<!-- changelog start -->

## Version 1.1b0 (2026-08-20)

### Added

- Add ``TelemetryHandler.fetch_response()`` and ``FetchResponseInvocation`` for
  the ``fetch_response`` operation, which fetches a previously generated
  response by id without performing inference.
  ([#184](https://github.com/open-telemetry/opentelemetry-python-genai/pull/184))
- Add streaming timing metrics (`gen_ai.client.operation.time_to_first_chunk`,
  `gen_ai.client.operation.time_per_output_chunk`), the
  `gen_ai.response.time_to_first_chunk` span attribute, and the
  `gen_ai.request.stream` span attribute, set by the shared stream wrappers.
  ([#269](https://github.com/open-telemetry/opentelemetry-python-genai/pull/269))
- Add ``CompactionPart`` message part type, mirroring the semconv model
  (``type``, ``id``, ``content``) for representing server-side context
  compaction events.
  ([#289](https://github.com/open-telemetry/opentelemetry-python-genai/pull/289))
- Add an optional ``error_type_resolver`` callback to
  ``TelemetryHandler.inference()`` and ``InferenceInvocation`` so instrumentors
  can derive the ``error.type`` attribute from the raw exception.
  ([#304](https://github.com/open-telemetry/opentelemetry-python-genai/pull/304))
- Add `SyncStreamManagerWrapper` / `AsyncStreamManagerWrapper` bases and
  `finalize_on_close` / `finalize_on_aclose` helpers to
  `opentelemetry.util.genai.stream`
  ([#390](https://github.com/open-telemetry/opentelemetry-python-genai/pull/390))

### Changed

- Restricted start-time sampling attributes to invocation construction time so
  they are not settable after start, and prevented duplicating them on
  finishing.
  ([#150](https://github.com/open-telemetry/opentelemetry-python-genai/pull/150))
- `Error.type` is now the `error.type` attribute value (a string) instead of
  the exception class; the originating exception moved to the new
  `Error.exception` field. When derived from an exception, `error.type` is its
  canonical, fully qualified name.
  ([#282](https://github.com/open-telemetry/opentelemetry-python-genai/pull/282))
- `gen_ai.usage.output_tokens` is now taken directly from
  `invocation.output_tokens`; `thinking_tokens` is no longer auto-added into
  it.
  Instrumentations must include reasoning/thinking tokens in `output_tokens`
  when their underlying library API does not do it.
  ([#283](https://github.com/open-telemetry/opentelemetry-python-genai/pull/283))
- Align some dataclasses with the GenAI semconv schemas: `GenericPart.type` is
  now a required free-form string carrying the provider's own type name instead
  of the fixed literal `generic`, `Modality` gains `document`, and
  `FinishReason` gains `compaction`.
  ([#284](https://github.com/open-telemetry/opentelemetry-python-genai/pull/284))
- Finalize stream telemetry for streams that expose `aclose` instead of `close`
  ([#390](https://github.com/open-telemetry/opentelemetry-python-genai/pull/390))

### Fixed

- Raise the minimum ``wrapt`` to 1.14.0, the first release with the async
  ``ObjectProxy`` support the stream wrappers require.
  ([#269](https://github.com/open-telemetry/opentelemetry-python-genai/pull/269))
- Emit ``execute_tool`` spans with ``INTERNAL`` span kind instead of
  ``CLIENT``.
  ([#274](https://github.com/open-telemetry/opentelemetry-python-genai/pull/274))
- Set the `gen_ai.workflow.name` span attribute on workflow invocations when
  the workflow name is known.
  ([#275](https://github.com/open-telemetry/opentelemetry-python-genai/pull/275))
- Make invocation `stop()` / `fail()` idempotent: finishing an invocation twice
  no longer re-records
  finish telemetry or ends the span a second time.
  ([#278](https://github.com/open-telemetry/opentelemetry-python-genai/pull/278))

## Version 1.0b0 (2026-07-09)

### Added

- Add `tool_result` as a parameter to `ToolInvocation`, add `tool_result` and
  `arguments` as span attributes to the `execute_tool` span if `ContentCapture`
  flag is set.
  ([#17](https://github.com/open-telemetry/opentelemetry-python-genai/pull/17))
- Add `RetrievalInvocation` type with `start_retrieval` / `retrieval` span
  lifecycle, supporting `gen_ai.operation.name=retrieval` spans per the GenAI
  semantic conventions.
  ([#36](https://github.com/open-telemetry/opentelemetry-python-genai/pull/36))
- Add shared sync and async stream wrapper base classes for GenAI
  instrumentations.
  ([#4500](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4500))

### Changed

- Bumped the version to `1.0b0` to align with the OpenTelemetry GenAI packages.
  ([#60](https://github.com/open-telemetry/opentelemetry-python-genai/pull/60))
- Use `AttributeValue` instead of `Any` for span and metric attribute types in
  `GenAIInvocation` and its subclasses.
  ([#153](https://github.com/open-telemetry/opentelemetry-python-genai/pull/153))
- Apply attribute for sampling on instantiation of all invocation types.
  ([#4553](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4553))
- Change `InferenceInvocation` init params to only accept base params
- Minor code cleanup and changes in preparation of moving google's GenAI
  instrumentation library to use this util library
  ([#4556](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4556))
- Pass in `attributes` on invocation `_start` so samplers have access to
  attributes.
  ([#4538](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4538))

### Deprecated

- Deprecate all `start_` factories; update all `invocation` factories to return
  objects that can be used as context managers.
  ([#17](https://github.com/open-telemetry/opentelemetry-python-genai/pull/17))

### Removed

- Remove remaining usages of and references to deprecated functions
  ([#47](https://github.com/open-telemetry/opentelemetry-python-genai/pull/47))
- Removes all code referencing the "OTEL_SEMCONV_STABILITY_OPT_IN" flag, as we
  are removing all code that sits behind the "default" value of that flag, and
  having the code that sits behind "experimental" be the default.
  ([#117](https://github.com/open-telemetry/opentelemetry-python-genai/pull/117))
- Removed the `provider` parameter from the internal agent invocation APIs and
  stopped emitting `gen_ai.provider.name` on internal agent spans and metrics.
  ([#132](https://github.com/open-telemetry/opentelemetry-python-genai/pull/132))

## Version 0.4b0 (2026-05-01)

- Add `AgentInvocation` type with `invoke_agent` span lifecycle
  ([#4274](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4274))
- Add metrics support for EmbeddingInvocation
  ([#4377](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4377))
- Add support for workflow in genAI utils handler.
  ([#4366](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4366))
- Enrich ToolCall type, breaking change: usage of ToolCall class renamed to ToolCallRequest
  ([#4218](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4218))
- Add EmbeddingInvocation span lifecycle support
  ([#4219](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4219))
- Populate schema_url on metrics
  ([#4320](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4320))
- Add workflow invocation type to genAI utils
  ([#4310](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4310))
- Check if upload works at startup in initializer of the `UploadCompletionHook`, instead
of repeatedly failing on every upload ([#4390](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4390)).
- Refactor public API: add factory methods (`start_inference`, `start_embedding`, `start_tool`, `start_workflow`) and invocation-owned lifecycle (`invocation.stop()` / `invocation.fail(exc)`); rename `LLMInvocation` → `InferenceInvocation` and `ToolCall` → `ToolInvocation`. Existing usages remain fully functional via deprecated aliases.
  ([#4391](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4391))
- `TelemetryHandler` now accepts a `completion_hook` parameter and calls it after each LLM invocation, passing inputs, outputs, the active span, and the log record. Content capture is enabled automatically when a real hook is configured.
  ([#4315](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4315))
- Add metrics to ToolInvocations ([#4443](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4443))
- Wrap completion hooks loaded via `load_completion_hook` so exceptions raised by
  `on_completion` are logged and swallowed instead of propagating to instrumentation
  call sites.

## Version 0.3b0 (2026-02-20)

- Add `gen_ai.tool_definitions` to completion hook ([#4181](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4181))
- Add support for emitting inference events and enrich message types. ([#3994](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3994))
- Add support for `server.address`, `server.port` on all signals and additional metric-only attributes
  ([#4069](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4069))
- Log error when `fsspec` fails to be imported instead of silently failing ([#4037](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4037)).
- Minor change to check LRU cache in Completion Hook before acquiring semaphore/thread ([#3907](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3907)).
- Add environment variable for genai upload hook queue size
  ([#3943](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3943))
- Add more Semconv attributes to LLMInvocation spans.
  ([#3862](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3862))
- Limit the upload hook thread pool to 64 workers
  ([#3944](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3944))
- Add metrics to LLMInvocation traces
  ([#3891](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3891))
- Add parent class genAI invocation
  ([#3889](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3889))

## Version 0.2b0 (2025-10-14)

- Add jsonlines support to fsspec uploader
  ([#3791](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3791))
- Rename "fsspec_upload" entry point and classes to more generic "upload"
  ([#3798](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3798))
- Record content-type and use canonical paths in fsspec genai uploader
  ([#3795](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3795))
- Make inputs / outputs / system instructions optional params to `on_completion`,
  ([#3802](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3802)).
- Use a SHA256 hash of the system instructions as it's upload filename, and check
  if the file exists before re-uploading it, ([#3814](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3814)).

## Version 0.1b0 (2025-09-25)

- Add completion hook to genai utils to implement semconv v1.37.

  Includes a hook implementation using
  [`fsspec`](https://filesystem-spec.readthedocs.io/en/latest/) to support uploading to various
  pluggable backends.

  ([#3780](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3780))
  ([#3752](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3752))
  ([#3759](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3759))
  ([#3763](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3763))

- Add a utility to parse the `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` environment variable.
  Add `gen_ai_latest_experimental` as a new value to the Sem Conv stability flag ([#3716](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3716)).

### Added

- Generate Spans for LLM invocations
- Helper functions for starting and finishing LLM invocations
