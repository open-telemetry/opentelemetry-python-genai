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

- Added retrieval span support.
  ([#124](https://github.com/open-telemetry/opentelemetry-python-genai/pull/124))
- Add ChatAnthropic tool-calling test coverage and fix finish_reason extraction
  for
  Anthropic responses in LangChain instrumentation.
  ([#188](https://github.com/open-telemetry/opentelemetry-python-genai/pull/188))
- (Openinference Migration: Langchain) - Add support for cache and reasoning
  token counts
  ([#272](https://github.com/open-telemetry/opentelemetry-python-genai/pull/272))
- Capture legacy OpenAI ``function_call`` responses
  (``additional_kwargs['function_call']``) as tool-call requests in input and
  output messages, matching the modern ``tool_calls`` path.
  ([#281](https://github.com/open-telemetry/opentelemetry-python-genai/pull/281))
- Forward the configured ``CompletionHook``
  (``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK`` or the
  ``instrument(completion_hook=...)`` argument) to the telemetry handler.
  ([#302](https://github.com/open-telemetry/opentelemetry-python-genai/pull/302))
- Populate gen_ai.response.model from responses API when the response body
  includes the served model header
  ([#305](https://github.com/open-telemetry/opentelemetry-python-genai/pull/305))
- Surface legacy OpenAI function calls in gen_ai.tool.definitions
  ([#334](https://github.com/open-telemetry/opentelemetry-python-genai/pull/334))

### Changed

- Support raw LangChain message inputs (`("role", content)` tuples, dicts, and
  strings) in input messages, so the prompt is recorded and duplicate
  `invoke_agent` spans are avoided.
  ([#261](https://github.com/open-telemetry/opentelemetry-python-genai/pull/261))

### Fixed

- Strip ``models/`` prefix from the request model attribute name when populated
  by certain providers.
  ([#254](https://github.com/open-telemetry/opentelemetry-python-genai/pull/254))
- Add guard for the response header value to avoid passing in empty model name
  ([#355](https://github.com/open-telemetry/opentelemetry-python-genai/pull/355))

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
