# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
Do *NOT* add changelog entries here!

This changelog is managed by towncrier and is compiled at release time.

The static "## Unreleased" section below pre-dates towncrier; its entries
must be folded into the first towncrier-generated release manually.

See https://github.com/open-telemetry/opentelemetry-python-genai/blob/main/CONTRIBUTING.md#changelog for details.
-->

<!-- changelog start -->

## Unreleased

- Fix compatibility with wrapt 2.x by using positional arguments in `wrap_function_wrapper()` calls
  ([#4445](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4445))
- Added span support for genAI langchain llm invocation.
  ([#3665](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3665))
- Added support to call genai utils handler for langchain LLM invocations.
  ([#3889](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3889))
- Added log and metrics provider to langchain genai utils handler
  ([#4214](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4214))
