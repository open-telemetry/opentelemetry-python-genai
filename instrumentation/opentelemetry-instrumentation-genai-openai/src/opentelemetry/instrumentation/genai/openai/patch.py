# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0


import logging

from openai.types import CreateEmbeddingResponse

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    openai_attributes as OpenAIAttributes,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    EmbeddingInvocation,
    InferenceInvocation,
)
from opentelemetry.util.genai.types import (
    Error,
)

from .chat_wrappers import AsyncChatStreamWrapper, ChatStreamWrapper
from .utils import (
    _prepare_output_messages,
    create_chat_invocation,
    get_server_address_and_port,
    get_value,
    is_streaming,
)

_logger = logging.getLogger(__name__)


def _create_embedding_invocation(
    handler: TelemetryHandler,
    kwargs,
    client_instance,
) -> EmbeddingInvocation:
    address, port = get_server_address_and_port(client_instance)
    invocation = handler.embedding(
        GenAIAttributes.GenAiProviderNameValues.OPENAI.value,
        request_model=get_value(kwargs.get("model")),
        server_address=address if address else None,
        server_port=port if port else None,
    )

    if (dimensions := get_value(kwargs.get("dimensions"))) is not None:
        invocation.dimension_count = dimensions
        invocation.metric_attributes[
            GenAIAttributes.GEN_AI_EMBEDDINGS_DIMENSION_COUNT
        ] = dimensions

    if (
        encoding_format := get_value(kwargs.get("encoding_format"))
    ) is not None:
        invocation.encoding_formats = [encoding_format]

    return invocation


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


def embeddings_create(handler: TelemetryHandler):
    """Wrap the `create` method of the `Embeddings` class to trace it."""

    def traced_method(wrapped, instance, args, kwargs):
        invocation = _create_embedding_invocation(handler, kwargs, instance)

        try:
            result = wrapped(*args, **kwargs)
        except Exception as error:
            invocation.fail(Error(type=type(error), message=str(error)))
            raise

        _safe_set_embeddings_response_properties(invocation, result)
        invocation.stop()
        return result

    return traced_method


def async_embeddings_create(handler: TelemetryHandler):
    """Wrap the `create` method of the `AsyncEmbeddings` class to trace it."""

    async def traced_method(wrapped, instance, args, kwargs):
        invocation = _create_embedding_invocation(handler, kwargs, instance)

        try:
            result = await wrapped(*args, **kwargs)
        except Exception as error:
            invocation.fail(Error(type=type(error), message=str(error)))
            raise

        _safe_set_embeddings_response_properties(invocation, result)
        invocation.stop()
        return result

    return traced_method


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


def _set_embeddings_response_properties(
    invocation: EmbeddingInvocation,
    result: CreateEmbeddingResponse,
) -> None:
    if getattr(result, "model", None):
        invocation.response_model_name = result.model

    # Set embeddings dimensions if we can determine it from the response
    if getattr(result, "data", None) and len(result.data) > 0:
        first_embedding = result.data[0]
        if getattr(first_embedding, "embedding", None):
            dimension_count = len(first_embedding.embedding)
            invocation.dimension_count = dimension_count
            # Mirror _create_embedding_invocation: EmbeddingInvocation does
            # not put dimension_count on metric attributes, so surface it
            # explicitly when we derive it from the response too.
            invocation.metric_attributes[
                GenAIAttributes.GEN_AI_EMBEDDINGS_DIMENSION_COUNT
            ] = dimension_count

    # Embeddings only have input tokens; output tokens are not applicable.
    if getattr(result, "usage", None):
        invocation.input_tokens = result.usage.prompt_tokens


def _safe_set_embeddings_response_properties(
    invocation: EmbeddingInvocation,
    result: CreateEmbeddingResponse,
) -> None:
    """Best-effort wrapper around ``_set_embeddings_response_properties``.

    Instrumentation must never break the wrapped library call, so extraction
    errors (e.g., from an unexpected SDK response shape) are caught and logged
    rather than propagated.
    """
    try:
        _set_embeddings_response_properties(invocation, result)
    except Exception:  # pylint: disable=broad-except
        _logger.debug(
            "Failed to extract embeddings response properties",
            exc_info=True,
        )
