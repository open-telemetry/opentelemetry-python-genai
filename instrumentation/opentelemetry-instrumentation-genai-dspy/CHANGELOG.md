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

- Add instrumentation for ``dspy.LM``.
  ([#683](https://github.com/open-telemetry/opentelemetry-python-genai/pull/683))
- Extend DSPy retrieval instrumentation to cover `dspy.ColBERTv2`,
  `dspy.Embeddings`, `EmbeddingsWithScores`, and `dspy.Retrieve` subclasses
  overriding `forward()`.
  ([#780](https://github.com/open-telemetry/opentelemetry-python-genai/pull/780))

### Changed

- Record non-string tool call results as JSON strings, and convert objects
  through `model_dump` or `__dict__`. Previously primitives were recorded as-is
  and objects through `str()`, e.g. a result of `7` was recorded as `7`, now
  `"7"`.
  ([#821](https://github.com/open-telemetry/opentelemetry-python-genai/pull/821))

### Fixed

- Record `gen_ai.response.finish_reasons` on `dspy.ReAct` and `dspy.ReActV2`
  agent spans regardless of content capture.
  ([#885](https://github.com/open-telemetry/opentelemetry-python-genai/pull/885))

## Version 1.2b0 (2026-09-24)

### Added

- Add initial DSPy instrumentation package setup.
  ([#439](https://github.com/open-telemetry/opentelemetry-python-genai/pull/439))
- Add instrumentation for DSPy Tool execution and ReAct agent invocations.
  ([#529](https://github.com/open-telemetry/opentelemetry-python-genai/pull/529))
- Support copy and deepcopy on wrapped DSPy methods to ensure compatibility
  with DSPy compilation and optimizers.
  ([#591](https://github.com/open-telemetry/opentelemetry-python-genai/pull/591))
- Instrument dspy.Retrieve to emit GenAI retrieval spans.
  ([#594](https://github.com/open-telemetry/opentelemetry-python-genai/pull/594))

### Changed

- Record metrics `gen_ai.invoke_agent.duration` and
  `gen_ai.execute_tool.duration` instead of `gen_ai.client.operation.duration`.
  ([#616](https://github.com/open-telemetry/opentelemetry-python-genai/pull/616))
- Use the shared RetrievalDocument model for retrieval results, with null IDs
  and scores for text-only passages, instead of capturing passage text.
  ([#775](https://github.com/open-telemetry/opentelemetry-python-genai/pull/775))

### Fixed

- Emit telemetry under the `opentelemetry.instrumentation.genai.dspy`
  instrumentation scope instead of `opentelemetry.util.genai.handler`
  ([#632](https://github.com/open-telemetry/opentelemetry-python-genai/pull/632))
