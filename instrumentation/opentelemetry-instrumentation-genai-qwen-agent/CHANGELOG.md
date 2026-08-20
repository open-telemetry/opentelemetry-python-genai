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

- Initial Qwen-Agent instrumentation with `invoke_agent` and `execute_tool`
  spans
  ([#310](https://github.com/open-telemetry/opentelemetry-python-genai/pull/310))
- Support strict typing: the package is now covered by the repository pyright
  configuration.
  ([#366](https://github.com/open-telemetry/opentelemetry-python-genai/pull/366))
