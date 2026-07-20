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

- Add LangChain workflow and agent span support
  ([#25](https://github.com/open-telemetry/opentelemetry-python-genai/pull/25))
- Added tool spans and captured tool definitions on inference spans.
  ([#37](https://github.com/open-telemetry/opentelemetry-python-genai/pull/37))
- Added log and metrics provider to langchain genai utils handler
  ([#4214](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4214))
- Added span support for GenAI LangChain LLM invocation.
  ([#3665](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3665))
- Added support to call genai utils handler for langchain LLM invocations.
  ([#3889](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3889))

### Changed

- Renamed package to `opentelemetry-instrumentation-genai-langchain` (imports
  `opentelemetry.instrumentation.genai.langchain`).
  ([#60](https://github.com/open-telemetry/opentelemetry-python-genai/pull/60))
- Update langchain instrumentation to use latest semantic conventions
  ([#129](https://github.com/open-telemetry/opentelemetry-python-genai/pull/129))

### Removed

- Stopped setting `gen_ai.provider.name` on internal agent spans.
  ([#132](https://github.com/open-telemetry/opentelemetry-python-genai/pull/132))
- Removed the unused span_manager.py from the langchain instrumentation
  ([#190](https://github.com/open-telemetry/opentelemetry-python-genai/pull/190))

### Fixed

- Fix compatibility with wrapt 2.x by using positional arguments in
  `wrap_function_wrapper()` calls
  ([#4445](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4445))
