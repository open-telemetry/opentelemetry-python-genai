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

## Version 1.2b0 (2026-09-23)

### Added

- Add Portkey AI instrumentation
  ([#465](https://github.com/open-telemetry/opentelemetry-python-genai/pull/465))
- Add inference and streaming span instrumentation for Portkey
  ([#466](https://github.com/open-telemetry/opentelemetry-python-genai/pull/466))
- Record cache-write, cache-read, reasoning, and explicit text, image, and
  audio token usage for chat and prompt completions, including streamed
  responses.
  ([#751](https://github.com/open-telemetry/opentelemetry-python-genai/pull/751))

### Changed

- Tool definitions are not emitted when content capture is disabled
  ([#377](https://github.com/open-telemetry/opentelemetry-python-genai/pull/377))
- Bump the minimum `opentelemetry-util-genai` version to 1.2b0.
  ([#575](https://github.com/open-telemetry/opentelemetry-python-genai/pull/575))

### Fixed

- Emit telemetry under the `opentelemetry.instrumentation.genai.portkey`
  instrumentation scope instead of `opentelemetry.util.genai.handler`
  ([#632](https://github.com/open-telemetry/opentelemetry-python-genai/pull/632))
- Record failed Portkey invocations when cancellation raises a `BaseException`.
  ([#656](https://github.com/open-telemetry/opentelemetry-python-genai/pull/656))
- Coerce top_k request parameter to integer.
  ([#707](https://github.com/open-telemetry/opentelemetry-python-genai/pull/707))
