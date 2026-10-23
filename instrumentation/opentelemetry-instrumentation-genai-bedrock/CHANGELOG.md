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

## Version 1.3b0 (2026-10-23)

### Added

- Add support for Bedrock Agent Runtime (retrieve_and_generate and
  retrieve_and_generate_stream) in synchronous and asynchronous calls.
  ([#733](https://github.com/open-telemetry/opentelemetry-python-genai/pull/733))
- Add instrumentation for ``bedrock-agent-runtime.invoke_inline_agent``.
  ([#736](https://github.com/open-telemetry/opentelemetry-python-genai/pull/736))
- Add support for Bedrock Embeddings (Amazon Titan, Cohere, etc.) and Bedrock
  Agent Runtime (invoke_agent and retrieve) in synchronous and asynchronous
  calls.
  ([#776](https://github.com/open-telemetry/opentelemetry-python-genai/pull/776))

### Changed

- Drop unmapped content blocks and non-text system instruction parts instead of
  recording them as type-only `GenericPart`.
  ([#936](https://github.com/open-telemetry/opentelemetry-python-genai/pull/936))

### Fixed

- Stop request extraction from raising on a Converse `document` block whose
  `format` is not a string.
  ([#866](https://github.com/open-telemetry/opentelemetry-python-genai/pull/866))
- Capture Bedrock Converse request messages provided as tuples.
  ([#870](https://github.com/open-telemetry/opentelemetry-python-genai/pull/870))
- Stop recording an empty reasoning part for redacted Converse reasoning
  content.
  ([#932](https://github.com/open-telemetry/opentelemetry-python-genai/pull/932))

## Version 1.2b0 (2026-09-24)

### Added

- Add the initial Amazon Bedrock instrumentation package setup.
  ([#359](https://github.com/open-telemetry/opentelemetry-python-genai/pull/359))
- Add instrumentation for ``Converse`` and ``ConverseStream``.
  ([#488](https://github.com/open-telemetry/opentelemetry-python-genai/pull/488))
- Add instrumentation for ``InvokeModel`` and ``InvokeModelWithResponseStream``
  ([#514](https://github.com/open-telemetry/opentelemetry-python-genai/pull/514))
- Add async client support for Amazon Bedrock (aiobotocore / aioboto3).
  ([#721](https://github.com/open-telemetry/opentelemetry-python-genai/pull/721))

### Changed

- Tool definitions are not emitted when content capture is disabled
  ([#377](https://github.com/open-telemetry/opentelemetry-python-genai/pull/377))

### Fixed

- Emit telemetry under the `opentelemetry.instrumentation.genai.bedrock`
  instrumentation scope instead of `opentelemetry.util.genai.handler`
  ([#632](https://github.com/open-telemetry/opentelemetry-python-genai/pull/632))
- Record failed Bedrock invocations when cancellation raises a `BaseException`.
  ([#656](https://github.com/open-telemetry/opentelemetry-python-genai/pull/656))
- Coerce top_k request parameter to integer.
  ([#707](https://github.com/open-telemetry/opentelemetry-python-genai/pull/707))
