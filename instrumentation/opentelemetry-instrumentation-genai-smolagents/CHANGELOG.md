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

## Version 1.2b0 (2026-09-24)

### Added

- Add ``invoke_agent`` spans for streaming and non-streaming
  ``MultiStepAgent.run()`` calls.
  ([#403](https://github.com/open-telemetry/opentelemetry-python-genai/pull/403))

### Changed

- Bump the minimum `opentelemetry-util-genai` version to 1.2b0.
  ([#365](https://github.com/open-telemetry/opentelemetry-python-genai/pull/365))
- Tool definitions are not emitted when content capture is disabled
  ([#377](https://github.com/open-telemetry/opentelemetry-python-genai/pull/377))
- Record metric `gen_ai.invoke_agent.duration` instead of
  `gen_ai.client.operation.duration`.
  ([#616](https://github.com/open-telemetry/opentelemetry-python-genai/pull/616))

### Fixed

- Emit telemetry under the `opentelemetry.instrumentation.genai.smolagents`
  instrumentation scope instead of `opentelemetry.util.genai.handler`
  ([#632](https://github.com/open-telemetry/opentelemetry-python-genai/pull/632))
- Coerce top_k request parameter to integer.
  ([#707](https://github.com/open-telemetry/opentelemetry-python-genai/pull/707))

## Version 1.1b0 (2026-08-20)

### Added

- Add skeleton and boilerplate for smolagents instrumentation package
  (``opentelemetry-instrumentation-genai-smolagents``).
  ([#349](https://github.com/open-telemetry/opentelemetry-python-genai/pull/349))
- Add ``chat`` instrumentation for the in-process smolagents model classes
  (``TransformersModel``, ``VLLMModel``, ``MLXModel``).
  ([#352](https://github.com/open-telemetry/opentelemetry-python-genai/pull/352))
