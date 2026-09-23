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

- Add instrumentation for Agno Team and Workflow run methods.
  ([#350](https://github.com/open-telemetry/opentelemetry-python-genai/pull/350))
- Add streaming support for Agno Agent, Team, and Workflow executions.
  ([#590](https://github.com/open-telemetry/opentelemetry-python-genai/pull/590))

### Changed

- Bump the minimum `opentelemetry-util-genai` version to 1.2b0.
  ([#365](https://github.com/open-telemetry/opentelemetry-python-genai/pull/365))
- Tool definitions are not emitted when content capture is disabled
  ([#377](https://github.com/open-telemetry/opentelemetry-python-genai/pull/377))
- Record metrics `gen_ai.invoke_agent.duration` and
  `gen_ai.execute_tool.duration` instead of `gen_ai.client.operation.duration`.
  ([#616](https://github.com/open-telemetry/opentelemetry-python-genai/pull/616))

### Fixed

- Mark failed Agno tool executions as errors.
  ([#587](https://github.com/open-telemetry/opentelemetry-python-genai/pull/587))
- Emit telemetry under the `opentelemetry.instrumentation.genai.agno`
  instrumentation scope instead of `opentelemetry.util.genai.handler`
  ([#632](https://github.com/open-telemetry/opentelemetry-python-genai/pull/632))

## Version 1.1b0 (2026-08-20)

### Added

- Add skeleton and boilerplate for Agno instrumentation package
  (``opentelemetry-instrumentation-genai-agno``).
  ([#301](https://github.com/open-telemetry/opentelemetry-python-genai/pull/301))
- Instrument agent.run as a client side invoke agent span. Instrument agno
  function calls an execute tool span.
  ([#328](https://github.com/open-telemetry/opentelemetry-python-genai/pull/328))
