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

### Fixed

- Emit telemetry under the `opentelemetry.instrumentation.genai.dspy`
  instrumentation scope instead of `opentelemetry.util.genai.handler`
  ([#632](https://github.com/open-telemetry/opentelemetry-python-genai/pull/632))
