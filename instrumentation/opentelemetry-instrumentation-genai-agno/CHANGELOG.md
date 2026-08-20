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

- Add skeleton and boilerplate for Agno instrumentation package
  (``opentelemetry-instrumentation-genai-agno``).
  ([#301](https://github.com/open-telemetry/opentelemetry-python-genai/pull/301))
- Instrument agent.run as a client side invoke agent span. Instrument agno
  function calls an execute tool span.
  ([#328](https://github.com/open-telemetry/opentelemetry-python-genai/pull/328))
