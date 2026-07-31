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

## Version 1.0b0 (2026-07-09)

### Added

- instrument async Anthropic messages.create calls
  ([#96](https://github.com/open-telemetry/opentelemetry-python-genai/pull/96))
- Add Anthropic instrumentation for Messages.parse, AsyncMessages.parse, and
  AsyncMessages.stream.
  ([#191](https://github.com/open-telemetry/opentelemetry-python-genai/pull/191))
- Add async Anthropic message stream wrappers and manager wrappers, with
  wrapper tests
  ([#4346](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4346))
    - `AsyncMessagesStreamWrapper` for async message stream telemetry
    - `AsyncMessagesStreamManagerWrapper` for async `Messages.stream()`
  telemetry
- Add instrumentation for Anthropic `Messages.stream()` helper method
  ([#4499](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4499))
- Add sync streaming support for `Messages.create(stream=True)` and
  `Messages.stream()`
  ([#4155](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4155))
    - `StreamWrapper` for handling `Messages.create(stream=True)` telemetry
    - `MessageStreamManagerWrapper` for handling `Messages.stream()` telemetry
    - `MessageWrapper` for non-streaming response telemetry extraction
- Implement sync `Messages.create` instrumentation with GenAI semantic
  convention attributes
  ([#4034](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4034))
    - Captures request attributes: `gen_ai.request.model`,
  `gen_ai.request.max_tokens`, `gen_ai.request.temperature`,
  `gen_ai.request.top_p`, `gen_ai.request.top_k`,
  `gen_ai.request.stop_sequences`
    - Captures response attributes: `gen_ai.response.id`,
  `gen_ai.response.model`, `gen_ai.response.finish_reasons`,
  `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`
    - Error handling with `error.type` attribute
    - Minimum supported anthropic version is 0.16.0 (SDK uses modern
  `anthropic.resources.messages` module structure introduced in this version)
- Initial implementation of Anthropic instrumentation
  ([#3978](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3978))

### Changed

- Renamed package to `opentelemetry-instrumentation-genai-anthropic` (imports
  `opentelemetry.instrumentation.genai.anthropic`).
  ([#60](https://github.com/open-telemetry/opentelemetry-python-genai/pull/60))
- Use shared GenAI stream wrappers for Messages API streams.
  ([#92](https://github.com/open-telemetry/opentelemetry-python-genai/pull/92))
- Update `opentelemetry-util-genai` dependency range to `>= 0.4b0.dev, <0.5b0`
  ([#4520](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4520))

### Fixed

- Emit gen_ai.provider.name instead of deprecated gen_ai.system for Anthropic
  spans
  ([#111](https://github.com/open-telemetry/opentelemetry-python-genai/pull/111))
- Fix compatibility with wrapt 2.x by using positional arguments in
  `wrap_function_wrapper()` calls
  ([#4445](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4445))
