# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from opentelemetry._logs import Logger
from opentelemetry.metrics import Meter
from opentelemetry.trace import Tracer
from opentelemetry.util.genai._attribute import _Attribute
from opentelemetry.util.genai._invocation import GenAIInvocation
from opentelemetry.util.genai.completion_hook import CompletionHook
from opentelemetry.util.genai.semconv.gen_ai import GenAiOperationName
from opentelemetry.util.genai.semconv.gen_ai._generated import (
    FetchResponseClientOperation,
)
from opentelemetry.util.genai.types import (
    ErrorTypeResolver,
    MessagePart,
    SystemInstructionPart,
)
from opentelemetry.util.genai.utils import (
    ContentCapturingMode,
    get_content_capturing_mode,
)


class FetchResponseInvocation(GenAIInvocation, FetchResponseClientOperation):
    """Represents a single fetch of a previously generated model response.

    Use handler.fetch_response() rather than constructing this directly.

    Reference: https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md#fetch-response

    The operation performs no inference and consumes no tokens: it returns a
    response produced by an earlier operation. Any token counts carried on the
    fetched response describe that original generation and MUST NOT be reported
    here, so this invocation deliberately exposes no token usage fields.

    Semantic convention attributes for fetch response spans:
    - gen_ai.operation.name: "fetch_response" (Required)
    - gen_ai.provider.name: Provider name (Required)
    - gen_ai.response.id: Identifier of the response being fetched (Required)
    - error.type: Error type when the fetch itself failed (Conditionally Required)
    - gen_ai.request.stream_cursor: Set from ``stream_cursor`` when the fetch
      resumes a streamed response from a prior position (Conditionally Required)
    - server.port: Set only when ``server_port`` is provided (Conditionally Required)
    - gen_ai.response.finish_reasons: Outcome of the original generation
      (Recommended)
    - gen_ai.response.model: Set from ``response_model_name`` (Recommended)
    - gen_ai.response.status: Lifecycle status of the fetched response
      (Recommended)
    - server.address: Set only when ``server_address`` is provided (Recommended)
    - gen_ai.output.messages, gen_ai.system_instructions,
      gen_ai.tool.definitions: content carried on the fetched response,
      recorded on the span only when content capturing is enabled (Opt-In).
      A fetched response does not carry the original input messages, so
      gen_ai.input.messages is never set.

    A fetched response whose *original* generation failed is not a failure of
    the fetch: report it through ``response_status`` and ``finish_reasons`` and
    still call ``stop()``. Only call ``fail()`` when the fetch call itself
    failed.
    """

    _provider = _Attribute[str]("provider_name")
    response_model_name = _Attribute[str | None]("response_model")
    finish_reasons = _Attribute[list[str] | None]("response_finish_reasons")
    stream_cursor = _Attribute[str | None]("request_stream_cursor")
    system_instruction = _Attribute[
        list[SystemInstructionPart] | list[MessagePart] | None
    ]("system_instructions")

    def __init__(
        self,
        tracer: Tracer,
        meter: Meter,
        logger: Logger,
        completion_hook: CompletionHook,
        provider: str,
        *,
        response_id: str,
        server_address: str | None = None,
        server_port: int | None = None,
        error_type_resolver: ErrorTypeResolver | None = None,
        content_capturing_mode: ContentCapturingMode | None = None,
    ) -> None:
        """Use handler.fetch_response() rather than calling this directly."""
        operation_name = GenAiOperationName.FETCH_RESPONSE.value
        mode = (
            get_content_capturing_mode()
            if content_capturing_mode is None
            else content_capturing_mode
        )
        FetchResponseClientOperation.__init__(
            self,
            tracer,
            meter,
            logger,
            completion_hook=completion_hook,
            error_type_resolver=error_type_resolver,
            operation_name=operation_name,
            provider_name=provider,
            response_id=response_id,
            server_address=server_address,
            server_port=server_port,
            content_capturing_mode=mode,
        )
        GenAIInvocation.__init__(self)
        self._stream_last_chunk_at: float | None = None
        self.start()

    def _on_stream_chunk(self, chunk_at: float) -> None:
        is_first_chunk = self._stream_last_chunk_at is None
        last_chunk_at = (
            self._stream_last_chunk_at
            if self._stream_last_chunk_at is not None
            else self._monotonic_start_s
        )
        self._stream_last_chunk_at = chunk_at
        delta = max(chunk_at - last_chunk_at, 0.0)

        if is_first_chunk:
            self.record_time_to_first_chunk(delta)
            return

        self.record_time_per_output_chunk(delta)
