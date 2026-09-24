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

### Changed

- Bump the minimum `opentelemetry-util-genai` version to 1.1b0, where
  `Error.type` became the `error.type` string value.
  ([#485](https://github.com/open-telemetry/opentelemetry-python-genai/pull/485))
- Bump the minimum `opentelemetry-util-genai` version to 1.2b0.
  ([#593](https://github.com/open-telemetry/opentelemetry-python-genai/pull/593))
- Record metrics `gen_ai.invoke_agent.duration` and
  `gen_ai.execute_tool.duration` instead of `gen_ai.client.operation.duration`.
  ([#616](https://github.com/open-telemetry/opentelemetry-python-genai/pull/616))

### Fixed

- Mark tool and agent span errors recorded on the agents-library Span.error
  ([#485](https://github.com/open-telemetry/opentelemetry-python-genai/pull/485))
- Record ``gen_ai.tool.call.arguments`` on ``execute_tool`` spans.
  ([#588](https://github.com/open-telemetry/opentelemetry-python-genai/pull/588))
- Emit telemetry under the `opentelemetry.instrumentation.genai.openai_agents`
  instrumentation scope instead of `opentelemetry.util.genai.handler`
  ([#632](https://github.com/open-telemetry/opentelemetry-python-genai/pull/632))
- Do not record `gen_ai.provider.name` on tool execution metric attributes.
  ([#710](https://github.com/open-telemetry/opentelemetry-python-genai/pull/710))
- Record ``gen_ai.conversation.id`` from ``RunConfig.group_id`` on
  ``invoke_workflow`` and ``invoke_agent`` spans, and propagate it to the
  ``chat`` span emitted by the model SDK instrumentation.
  ([#726](https://github.com/open-telemetry/opentelemetry-python-genai/pull/726))

## Version 1.1b0 (2026-08-20)

### Fixed

- Honor ``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK`` by falling back to
  ``load_completion_hook()``
  when no ``completion_hook`` is passed to ``instrument()``.
  ([#302](https://github.com/open-telemetry/opentelemetry-python-genai/pull/302))

## Version 1.0b0 (2026-07-09)

### Added

- Document official package metadata and README for the OpenAI Agents
  instrumentation.
  ([#3859](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3859))
- Populate instructions and tool definitions from Response obj.
  ([#4196](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4196))

### Changed

- Renamed package from `opentelemetry-instrumentation-openai-agents-v2` to
  `opentelemetry-instrumentation-genai-openai-agents` (imports
  `opentelemetry.instrumentation.genai.openai_agents`); the version line
  restarts at `1.0b0`.
  ([#60](https://github.com/open-telemetry/opentelemetry-python-genai/pull/60))
- Switch instrumentation to use util-genai instead of hand-rolled signals.
  Stop capturing chat, embeddings, speech, and transcription spans — those are
  covered by the OpenAI instrumentation.
  Remove handoff and guardrail spans (not yet defined in semantic convention
  and not implemented
  by genai-util).
  ([#90](https://github.com/open-telemetry/opentelemetry-python-genai/pull/90))
- Align AgentSpanData test stubs and span processor with real OpenAI Agents
  SDK; remove non-existent `operation`, `description`, `agent_id`, and `model`
  fields.
  ([#4229](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4229))

## Version 0.1.0 (2025-10-15)

- Initial barebones package skeleton: minimal instrumentor stub, version module,
  and packaging metadata/entry point.
  ([#3805](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3805))
- Implement OpenAI Agents span processing aligned with GenAI semantic conventions.
  ([#3817](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3817))
- Input and output according to GenAI spec.
  ([#3824](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3824))
