# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0


from timeit import default_timer
from typing import Any, Optional

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    openai_attributes as OpenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    server_attributes as ServerAttributes,
)
from opentelemetry.trace import Span, SpanKind, Tracer
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import InferenceInvocation
from opentelemetry.util.genai.types import (
    Error,
)

from .chat_wrappers import AsyncChatStreamWrapper, ChatStreamWrapper
from .instruments import Instruments
from .utils import (
    _prepare_output_messages,
    create_chat_invocation,
    get_llm_request_attributes,
    handle_span_exception,
    is_streaming,
    set_span_attribute,
)


def chat_completions_create_v_new(
    handler: TelemetryHandler,
):
    """Wrap the `create` method of the `ChatCompletion` class to trace it."""
    capture_content = handler.should_capture_content()

    def traced_method(wrapped, instance, args, kwargs):
        chat_invocation = create_chat_invocation(
            handler, kwargs, instance, capture_content=capture_content
        )

        try:
            result = wrapped(*args, **kwargs)
            if hasattr(result, "parse"):
                # result is of type LegacyAPIResponse, call parse to get the actual response
                parsed_result = result.parse()
            else:
                parsed_result = result
            if is_streaming(kwargs):
                return ChatStreamWrapper(
                    parsed_result, chat_invocation, capture_content
                )

            _set_response_properties(
                chat_invocation, parsed_result, capture_content
            )
            chat_invocation.stop()
            return result
        except Exception as error:
            chat_invocation.fail(Error(type=type(error), message=str(error)))
            raise

    return traced_method


def async_chat_completions_create_v_new(
    handler: TelemetryHandler,
):
    """Wrap the `create` method of the `AsyncChatCompletion` class to trace it."""
    capture_content = handler.should_capture_content()

    async def traced_method(wrapped, instance, args, kwargs):
        chat_invocation = create_chat_invocation(
            handler, kwargs, instance, capture_content=capture_content
        )

        try:
            result = await wrapped(*args, **kwargs)
            if hasattr(result, "parse"):
                # result is of type LegacyAPIResponse, calling parse to get the actual response
                parsed_result = result.parse()
            else:
                parsed_result = result
            if is_streaming(kwargs):
                return AsyncChatStreamWrapper(
                    parsed_result, chat_invocation, capture_content
                )

            _set_response_properties(
                chat_invocation, parsed_result, capture_content
            )
            chat_invocation.stop()
            return result

        except Exception as error:
            chat_invocation.fail(Error(type=type(error), message=str(error)))
            raise

    return traced_method


def embeddings_create(
    tracer: Tracer,
    instruments: Instruments,
    latest_experimental_enabled: bool,
):
    """Wrap the `create` method of the `Embeddings` class to trace it."""

    def traced_method(wrapped, instance, args, kwargs):
        span_attributes = get_llm_request_attributes(
            kwargs,
            instance,
            latest_experimental_enabled,
            GenAIAttributes.GenAiOperationNameValues.EMBEDDINGS.value,
        )
        span_name = _get_embeddings_span_name(span_attributes)

        with tracer.start_as_current_span(
            name=span_name,
            kind=SpanKind.CLIENT,
            attributes=span_attributes,
            end_on_exit=True,
        ) as span:
            start = default_timer()
            result = None
            error_type = None

            try:
                result = wrapped(*args, **kwargs)

                if span.is_recording():
                    _set_embeddings_response_attributes(span, result)

                return result

            except Exception as error:
                error_type = type(error).__qualname__
                handle_span_exception(span, error)
                raise

            finally:
                duration = max((default_timer() - start), 0)
                _record_metrics(
                    instruments,
                    duration,
                    result,
                    span_attributes,
                    error_type,
                    GenAIAttributes.GenAiOperationNameValues.EMBEDDINGS.value,
                )

    return traced_method


def async_embeddings_create(
    tracer: Tracer,
    instruments: Instruments,
    latest_experimental_enabled: bool,
):
    """Wrap the `create` method of the `AsyncEmbeddings` class to trace it."""

    async def traced_method(wrapped, instance, args, kwargs):
        span_attributes = get_llm_request_attributes(
            kwargs,
            instance,
            latest_experimental_enabled,
            GenAIAttributes.GenAiOperationNameValues.EMBEDDINGS.value,
        )
        span_name = _get_embeddings_span_name(span_attributes)

        with tracer.start_as_current_span(
            name=span_name,
            kind=SpanKind.CLIENT,
            attributes=span_attributes,
            end_on_exit=True,
        ) as span:
            start = default_timer()
            result = None
            error_type = None

            try:
                result = await wrapped(*args, **kwargs)

                if span.is_recording():
                    _set_embeddings_response_attributes(span, result)

                return result

            except Exception as error:
                error_type = type(error).__qualname__
                handle_span_exception(span, error)
                raise

            finally:
                duration = max((default_timer() - start), 0)
                _record_metrics(
                    instruments,
                    duration,
                    result,
                    span_attributes,
                    error_type,
                    GenAIAttributes.GenAiOperationNameValues.EMBEDDINGS.value,
                )

    return traced_method


def _get_embeddings_span_name(span_attributes):
    """Get span name for embeddings operations."""
    operation_name = span_attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
    model = span_attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL)
    return f"{operation_name} {model}" if model else operation_name


def _record_metrics(
    instruments: Instruments,
    duration: float,
    result,
    request_attributes: dict,
    error_type: Optional[str],
    operation_name: str,
):
    common_attributes = {
        GenAIAttributes.GEN_AI_OPERATION_NAME: operation_name,
        GenAIAttributes.GEN_AI_SYSTEM: GenAIAttributes.GenAiSystemValues.OPENAI.value,
        GenAIAttributes.GEN_AI_REQUEST_MODEL: request_attributes[
            GenAIAttributes.GEN_AI_REQUEST_MODEL
        ],
    }

    if "gen_ai.embeddings.dimension.count" in request_attributes:
        common_attributes["gen_ai.embeddings.dimension.count"] = (
            request_attributes["gen_ai.embeddings.dimension.count"]
        )

    if error_type:
        common_attributes["error.type"] = error_type

    if result and getattr(result, "model", None):
        common_attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] = result.model

    if result and getattr(result, "service_tier", None):
        common_attributes[
            GenAIAttributes.GEN_AI_OPENAI_RESPONSE_SERVICE_TIER
        ] = result.service_tier

    if result and getattr(result, "system_fingerprint", None):
        common_attributes[
            GenAIAttributes.GEN_AI_OPENAI_RESPONSE_SYSTEM_FINGERPRINT
        ] = result.system_fingerprint

    if ServerAttributes.SERVER_ADDRESS in request_attributes:
        common_attributes[ServerAttributes.SERVER_ADDRESS] = (
            request_attributes[ServerAttributes.SERVER_ADDRESS]
        )

    if ServerAttributes.SERVER_PORT in request_attributes:
        common_attributes[ServerAttributes.SERVER_PORT] = request_attributes[
            ServerAttributes.SERVER_PORT
        ]

    instruments.operation_duration_histogram.record(
        duration,
        attributes=common_attributes,
    )

    if result and getattr(result, "usage", None):
        # Always record input tokens
        input_attributes = {
            **common_attributes,
            GenAIAttributes.GEN_AI_TOKEN_TYPE: GenAIAttributes.GenAiTokenTypeValues.INPUT.value,
        }
        instruments.token_usage_histogram.record(
            result.usage.prompt_tokens,
            attributes=input_attributes,
        )

        # For embeddings, don't record output tokens as all tokens are input tokens
        if (
            operation_name
            != GenAIAttributes.GenAiOperationNameValues.EMBEDDINGS.value
        ):
            output_attributes = {
                **common_attributes,
                GenAIAttributes.GEN_AI_TOKEN_TYPE: GenAIAttributes.GenAiTokenTypeValues.COMPLETION.value,
            }
            instruments.token_usage_histogram.record(
                result.usage.completion_tokens, attributes=output_attributes
            )


def _set_response_properties(
    chat_invocation: InferenceInvocation, result, capture_content: bool
) -> InferenceInvocation:
    if getattr(result, "model", None):
        chat_invocation.response_model_name = result.model

    if getattr(result, "choices", None):
        finish_reasons = []
        for choice in result.choices:
            finish_reasons.append(choice.finish_reason or "error")

        chat_invocation.finish_reasons = finish_reasons

        if capture_content:  # optimization
            chat_invocation.output_messages = _prepare_output_messages(
                result.choices
            )

    if getattr(result, "id", None):
        chat_invocation.response_id = result.id

    if getattr(result, "service_tier", None):
        chat_invocation.attributes.update(
            {
                OpenAIAttributes.OPENAI_RESPONSE_SERVICE_TIER: result.service_tier
            },
        )
        chat_invocation.metric_attributes.update(
            {
                OpenAIAttributes.OPENAI_RESPONSE_SERVICE_TIER: result.service_tier
            },
        )

    if getattr(result, "usage", None):
        chat_invocation.input_tokens = result.usage.prompt_tokens
        chat_invocation.output_tokens = result.usage.completion_tokens

    if getattr(result, "system_fingerprint", None):
        chat_invocation.attributes.update(
            {
                OpenAIAttributes.OPENAI_RESPONSE_SYSTEM_FINGERPRINT: result.system_fingerprint
            },
        )
        chat_invocation.metric_attributes.update(
            {
                OpenAIAttributes.OPENAI_RESPONSE_SYSTEM_FINGERPRINT: result.system_fingerprint
            },
        )

    return chat_invocation


def _set_embeddings_response_attributes(
    span: Span,
    result: Any,
):
    set_span_attribute(
        span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, result.model
    )

    # Set embeddings dimensions if we can determine it from the response
    if getattr(result, "data", None) and len(result.data) > 0:
        first_embedding = result.data[0]
        if getattr(first_embedding, "embedding", None):
            set_span_attribute(
                span,
                "gen_ai.embeddings.dimension.count",
                len(first_embedding.embedding),
            )

    # Get the usage
    if getattr(result, "usage", None):
        set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
            result.usage.prompt_tokens,
        )
        # Don't set output tokens for embeddings as all tokens are input tokens
