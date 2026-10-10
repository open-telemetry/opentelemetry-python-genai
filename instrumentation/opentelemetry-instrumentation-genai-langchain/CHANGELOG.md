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

- (OpenInference Migration: LangChain) - Capture multimodal image content
  (OpenAI ``image_url`` and Responses API ``input_image``, Anthropic ``image``,
  and LangChain standard ``image`` blocks) as
  ``BlobPart``/``UriPart``/``FilePart`` message parts.
  ([#296](https://github.com/open-telemetry/opentelemetry-python-genai/pull/296))
- Capture document relevance score on retrieval spans per OpenTelemetry GenAI
  semantic conventions.
  ([#670](https://github.com/open-telemetry/opentelemetry-python-genai/pull/670))
- Record `cache_write_input_tokens` and modality token breakdown attributes
  (`text`, `image`, `audio`) from LangChain usage metadata on
  `InferenceInvocation`.
  ([#671](https://github.com/open-telemetry/opentelemetry-python-genai/pull/671))
- Capture request model from ls_model_name metadata on chat
  ([#682](https://github.com/open-telemetry/opentelemetry-python-genai/pull/682))
- Capture top_k - `gen_ai.request.top_k` and choice count -
  `gen_ai.request.choice.count` on chat
  ([#684](https://github.com/open-telemetry/opentelemetry-python-genai/pull/684))
- Record finish_reasons from chat generation metadata.
  ([#705](https://github.com/open-telemetry/opentelemetry-python-genai/pull/705))
- Classify agent chains by name on on_chain_start
  ([#766](https://github.com/open-telemetry/opentelemetry-python-genai/pull/766))

### Changed

- Bump the minimum `opentelemetry-util-genai` version to 1.2b0.
  ([#365](https://github.com/open-telemetry/opentelemetry-python-genai/pull/365))
- Tool definitions are not emitted when content capture is disabled
  ([#377](https://github.com/open-telemetry/opentelemetry-python-genai/pull/377))
- Raised the opentelemetry-util-genai dependency floor to 1.2b0 for
  ``GenAIInvocation.record_stream_chunk``
  ([#482](https://github.com/open-telemetry/opentelemetry-python-genai/pull/482))
- Populate name attribute on captured input and output messages when available.
  ([#611](https://github.com/open-telemetry/opentelemetry-python-genai/pull/611))
- Record metrics `gen_ai.invoke_agent.duration` and
  `gen_ai.execute_tool.duration` instead of `gen_ai.client.operation.duration`,
  and stop setting span attribute `gen_ai.agent.id` on internal agent spans.
  ([#616](https://github.com/open-telemetry/opentelemetry-python-genai/pull/616))
- Record modality token usage through the shared `InferenceInvocation` setters;
  `extract_token_details` no longer returns modality keys.
  ([#674](https://github.com/open-telemetry/opentelemetry-python-genai/pull/674))
- Use the shared RetrievalDocument model to capture only document IDs and
  scores, with null for unavailable fields, instead of document text.
  ([#775](https://github.com/open-telemetry/opentelemetry-python-genai/pull/775))

### Fixed

- Emit invoke_agent spans for LangChain create_agent graph roots, including
  nested agents, which get their own spans.
  ([#391](https://github.com/open-telemetry/opentelemetry-python-genai/pull/391))
- Populate ``gen_ai.conversation.id`` on model call and workflow spans,
  resolved from the ``thread_id``, ``session_id`` or ``conversation_id``
  metadata keys and inherited by nested runs.
  ([#474](https://github.com/open-telemetry/opentelemetry-python-genai/pull/474))
- Mark streamed calls with `gen_ai.request.stream` and record
  `gen_ai.response.time_to_first_chunk` plus the streaming timing metrics,
  which were never emitted for LangChain.
  ([#482](https://github.com/open-telemetry/opentelemetry-python-genai/pull/482))
- Record gen_ai.response.model on streamed spans
  ([#505](https://github.com/open-telemetry/opentelemetry-python-genai/pull/505))
- Resolve message roles by class so streaming chunk messages no longer report
  their class name (``AIMessageChunk``) as the role
  ([#506](https://github.com/open-telemetry/opentelemetry-python-genai/pull/506))
- Record `gen_ai.tool.call.id` on failed `execute_tool` spans
  ([#510](https://github.com/open-telemetry/opentelemetry-python-genai/pull/510))
- Set `gen_ai.agent.name` on child `execute_tool` spans when running under an
  enclosing `invoke_agent`
  ([#512](https://github.com/open-telemetry/opentelemetry-python-genai/pull/512))
- Emit telemetry under the `opentelemetry.instrumentation.genai.langchain`
  instrumentation scope instead of `opentelemetry.util.genai.handler`
  ([#632](https://github.com/open-telemetry/opentelemetry-python-genai/pull/632))
- Propagate parent context to nested agent, tool, retrieval, and workflow spans
  in async execution paths.
  ([#758](https://github.com/open-telemetry/opentelemetry-python-genai/pull/758))

## Version 1.1b1 (2026-08-21)

### Changed

- Raised `opentelemetry-util-genai` dependency floor to 1.1b0.
  ([#433](https://github.com/open-telemetry/opentelemetry-python-genai/pull/433))

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
