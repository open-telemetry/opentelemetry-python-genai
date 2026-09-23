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

### Changed

- Bump the minimum `opentelemetry-util-genai` version to 1.2b0.
  ([#365](https://github.com/open-telemetry/opentelemetry-python-genai/pull/365))
- Record metrics `gen_ai.invoke_agent.duration` and
  `gen_ai.execute_tool.duration` instead of `gen_ai.client.operation.duration`.
  ([#616](https://github.com/open-telemetry/opentelemetry-python-genai/pull/616))

### Fixed

- Emit telemetry under the `opentelemetry.instrumentation.genai.qwen_agent`
  instrumentation scope instead of `opentelemetry.util.genai.handler`
  ([#632](https://github.com/open-telemetry/opentelemetry-python-genai/pull/632))
- Record failed Qwen Agent tool invocations when cancellation raises a
  `BaseException`.
  ([#656](https://github.com/open-telemetry/opentelemetry-python-genai/pull/656))

## Version 1.1b0 (2026-08-20)

### Added

- Initial Qwen-Agent instrumentation with `invoke_agent` and `execute_tool`
  spans
  ([#310](https://github.com/open-telemetry/opentelemetry-python-genai/pull/310))
- Support strict typing: the package is now covered by the repository pyright
  configuration.
  ([#366](https://github.com/open-telemetry/opentelemetry-python-genai/pull/366))
