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

## Version 1.2b0 (2026-09-24)

### Added

- Add skeleton and boilerplate for LlamaIndex instrumentation package
  (``opentelemetry-instrumentation-genai-llama-index``).
  ([#309](https://github.com/open-telemetry/opentelemetry-python-genai/pull/309))
- Add agent invocation and tool execution spans for LlamaIndex agents.
  ([#494](https://github.com/open-telemetry/opentelemetry-python-genai/pull/494))
- Add tracing for LlamaIndex AgentWorkflow runs and member agent executions.
  ([#668](https://github.com/open-telemetry/opentelemetry-python-genai/pull/668))
- Add retrieval span instrumentation for LlamaIndex ``BaseRetriever``
  operations.
  ([#698](https://github.com/open-telemetry/opentelemetry-python-genai/pull/698))

### Changed

- Tool definitions are not emitted when content capture is disabled
  ([#377](https://github.com/open-telemetry/opentelemetry-python-genai/pull/377))
- Record metrics `gen_ai.invoke_agent.duration` and
  `gen_ai.execute_tool.duration` instead of `gen_ai.client.operation.duration`.
  ([#616](https://github.com/open-telemetry/opentelemetry-python-genai/pull/616))
- Simplify tool context propagation using invocation context.
  ([#704](https://github.com/open-telemetry/opentelemetry-python-genai/pull/704))
- Use the shared RetrievalDocument model to capture only document IDs and
  scores, with null for unavailable scores, instead of node text.
  ([#775](https://github.com/open-telemetry/opentelemetry-python-genai/pull/775))

### Fixed

- Emit telemetry under the `opentelemetry.instrumentation.genai.llama_index`
  instrumentation scope instead of `opentelemetry.util.genai.handler`
  ([#632](https://github.com/open-telemetry/opentelemetry-python-genai/pull/632))
- Keep one invoke_agent span open across an AgentWorkflow member's tool loop,
  parent execute_tool spans to the requesting agent, and keep a handing-off
  agent's span open until every tool call of that turn ends.
  ([#668](https://github.com/open-telemetry/opentelemetry-python-genai/pull/668))
- Exclude broken llama-index-workflows 2.24.0 and declare companion library
  bounds.
  ([#739](https://github.com/open-telemetry/opentelemetry-python-genai/pull/739))
