# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from google.genai._api_client import BaseApiClient
from google.genai.models import AsyncModels, Models
from google.genai.types import EmbedContentResponse
from wrapt import wrap_function_wrapper

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    EmbeddingInvocation,
)

_RAW_RESPONSE_BODY: ContextVar[str | None] = ContextVar(
    "raw_response_body", default=None
)


class _EmbeddingMethodsSnapshot:
    def __init__(self) -> None:
        self._original_embed_content = Models.embed_content
        self._original_async_embed_content = AsyncModels.embed_content
        self._original_client_request = BaseApiClient.request
        self._original_client_async_request = BaseApiClient.async_request

    def restore(self) -> None:
        Models.embed_content = self._original_embed_content
        AsyncModels.embed_content = self._original_async_embed_content
        BaseApiClient.request = self._original_client_request
        BaseApiClient.async_request = self._original_client_async_request


def _apply_embedding_response_attributes(
    response: EmbedContentResponse,
    invocation: EmbeddingInvocation,
) -> None:
    if response.embeddings:
        first_embedding = response.embeddings[0]
        if first_embedding.values:
            invocation.dimension_count = len(first_embedding.values)
            invocation.metric_attributes[
                GenAIAttributes.GEN_AI_EMBEDDINGS_DIMENSION_COUNT
            ] = invocation.dimension_count

    # In the future we can get rid of this and the monkey patching of the
    # requests, and use the parsed SDK response instead. See:
    # https://github.com/googleapis/python-genai/issues/2658
    if raw_body := _RAW_RESPONSE_BODY.get():
        try:
            body_dict = json.loads(raw_body)
            usage_metadata = body_dict.get("usageMetadata")
            if isinstance(usage_metadata, dict):
                invocation.input_tokens = usage_metadata.get(
                    "promptTokenCount"
                )
        except Exception:
            pass


def _get_client_info(instance: Any) -> tuple[bool, str | None]:
    is_vertex = False
    server_address = None
    if hasattr(instance, "_api_client"):
        api_client = instance._api_client
        is_vertex = getattr(api_client, "vertexai", False)
        if hasattr(api_client, "_http_options"):
            server_address = getattr(
                api_client._http_options, "base_url", None
            )
    elif hasattr(instance, "_client"):
        client = instance._client
        is_vertex = getattr(client, "_is_vertex", False)
        server_address = getattr(client, "server", None)
    elif hasattr(instance, "sdk_configuration"):
        config = instance.sdk_configuration
        server_url = getattr(config, "server_url", "")
        if server_url:
            server_address = server_url
            if "aiplatform.googleapis.com" in server_url:
                is_vertex = True

    return is_vertex, server_address


def _create_instrumented_embed_content(
    telemetry_handler: TelemetryHandler,
) -> Callable[
    [
        Callable[..., EmbedContentResponse],
        Models,
        tuple[Any, ...],
        dict[str, Any],
    ],
    EmbedContentResponse,
]:
    def instrumented_embed_content(
        wrapped: Callable[..., EmbedContentResponse],
        instance: Models,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> EmbedContentResponse:
        is_vertex, server_address = _get_client_info(instance)
        with telemetry_handler.embedding(
            provider=(
                GenAIAttributes.GenAiSystemValues.VERTEX_AI.value
                if is_vertex
                else GenAIAttributes.GenAiSystemValues.GEMINI.value
            ),
            request_model=kwargs.get("model"),
            server_address=server_address,
        ) as invocation:
            response = wrapped(*args, **kwargs)
            _apply_embedding_response_attributes(response, invocation)
            _RAW_RESPONSE_BODY.set(None)
            return response

    return instrumented_embed_content


def _create_instrumented_async_embed_content(
    telemetry_handler: TelemetryHandler,
) -> Callable[
    [
        Callable[..., Any],
        AsyncModels,
        tuple[Any, ...],
        dict[str, Any],
    ],
    Any,
]:
    async def instrumented_embed_content(
        wrapped: Callable[..., Any],
        instance: AsyncModels,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> EmbedContentResponse:
        is_vertex, server_address = _get_client_info(instance)
        with telemetry_handler.embedding(
            provider=(
                GenAIAttributes.GenAiSystemValues.VERTEX_AI.value
                if is_vertex
                else GenAIAttributes.GenAiSystemValues.GEMINI.value
            ),
            request_model=kwargs.get("model"),
            server_address=server_address,
        ) as invocation:
            response = await wrapped(*args, **kwargs)
            _apply_embedding_response_attributes(response, invocation)
            _RAW_RESPONSE_BODY.set(None)
            return response

    return instrumented_embed_content


def uninstrument_embeddings(snapshot: object) -> None:
    assert isinstance(snapshot, _EmbeddingMethodsSnapshot)
    snapshot.restore()


def instrument_embeddings(
    telemetry_handler: TelemetryHandler,
) -> object:
    snapshot = _EmbeddingMethodsSnapshot()

    wrap_function_wrapper(
        "google.genai.models",
        "Models.embed_content",
        _create_instrumented_embed_content(telemetry_handler),
    )
    wrap_function_wrapper(
        "google.genai.models",
        "AsyncModels.embed_content",
        _create_instrumented_async_embed_content(telemetry_handler),
    )

    # Wrap BaseApiClient to capture raw responses
    def instrumented_request(wrapped, instance, args, kwargs):
        response = wrapped(*args, **kwargs)
        if response and getattr(response, "body", None):
            _RAW_RESPONSE_BODY.set(response.body)
        return response

    async def instrumented_async_request(wrapped, instance, args, kwargs):
        response = await wrapped(*args, **kwargs)
        if response and getattr(response, "body", None):
            _RAW_RESPONSE_BODY.set(response.body)
        return response

    wrap_function_wrapper(
        "google.genai._api_client",
        "BaseApiClient.request",
        instrumented_request,
    )
    wrap_function_wrapper(
        "google.genai._api_client",
        "BaseApiClient.async_request",
        instrumented_async_request,
    )

    return snapshot
