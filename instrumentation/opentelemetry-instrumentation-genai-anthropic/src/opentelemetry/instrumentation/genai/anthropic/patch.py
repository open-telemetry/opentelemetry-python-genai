# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Patching functions for Anthropic instrumentation."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast

from anthropic.types import Message as AnthropicMessage

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import InferenceInvocation

from ._raw_response import wrap_raw_response
from .messages_extractors import (
    extract_params,
    get_input_messages,
    get_llm_request_attributes,
    get_server_address_and_port,
    get_system_instruction,
)
from .utils import is_anthropic_async_stream, is_anthropic_stream
from .wrappers import (
    AsyncMessagesStreamManagerWrapper,
    AsyncMessagesStreamWrapper,
    MessagesStreamManagerWrapper,
    MessagesStreamWrapper,
    MessageWrapper,
)

if TYPE_CHECKING:
    from anthropic._streaming import AsyncStream as AnthropicAsyncStream
    from anthropic._streaming import Stream as AnthropicStream
    from anthropic.lib.streaming._messages import (  # pylint: disable=no-name-in-module
        AsyncMessageStreamManager,
        MessageStreamManager,
    )
    from anthropic.resources.messages import AsyncMessages, Messages
    from anthropic.types import RawMessageStreamEvent


_logger = logging.getLogger(__name__)
ANTHROPIC = "anthropic"


def _is_raw_response(result: object) -> bool:
    """Whether ``result`` is a raw-response object to route through the proxy.

    Matched on shape rather than on the SDK's private response classes
    (``LegacyAPIResponse``, ``APIResponse``, ``AsyncAPIResponse``). The proxy
    needs both attributes: ``parse()`` to route on the parsed value and
    ``http_response`` to finalize the span for callers that never parse.
    """
    return hasattr(result, "parse") and hasattr(result, "http_response")


def messages_create(
    handler: TelemetryHandler,
) -> Callable[
    ...,
    AnthropicMessage
    | AnthropicStream[RawMessageStreamEvent]
    | MessagesStreamWrapper[None],
]:
    """Wrap the `create` method of the `Messages` class to trace it."""
    capture_content = handler.should_capture_content()

    def traced_method(
        wrapped: Callable[
            ...,
            AnthropicMessage | AnthropicStream[RawMessageStreamEvent],
        ],
        instance: Messages,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> (
        AnthropicMessage
        | AnthropicStream[RawMessageStreamEvent]
        | MessagesStreamWrapper[None]
    ):
        invocation = _create_invocation(
            handler, instance, args, kwargs, capture_content
        )
        try:
            # ``Any``: through ``with_raw_response`` / ``with_streaming_response``
            # this returns a response object that ``create``'s annotations do
            # not describe, so the checks below cannot be narrowed away.
            result: Any = wrapped(*args, **kwargs)
        except Exception as exc:
            invocation.fail(exc)
            raise

        if is_anthropic_stream(result):
            return MessagesStreamWrapper(
                cast("AnthropicStream[RawMessageStreamEvent]", result),
                invocation,
                capture_content,
            )
        if _is_raw_response(result):
            return wrap_raw_response(result, invocation, capture_content)

        if isinstance(result, AnthropicMessage):
            MessageWrapper(result, capture_content).extract_into(invocation)
        invocation.stop()
        return result

    return cast(
        'Callable[..., "AnthropicMessage" | "AnthropicStream[RawMessageStreamEvent]" | MessagesStreamWrapper[None]]',
        traced_method,
    )


def async_messages_create(
    handler: TelemetryHandler,
) -> Callable[
    ...,
    AnthropicMessage
    | AnthropicAsyncStream[RawMessageStreamEvent]
    | AsyncMessagesStreamWrapper[None],
]:
    """Wrap the async `create` method of the `AsyncMessages` class."""
    capture_content = handler.should_capture_content()

    async def traced_method(
        wrapped: Callable[
            ...,
            Awaitable[
                AnthropicMessage | AnthropicAsyncStream[RawMessageStreamEvent]
            ],
        ],
        instance: AsyncMessages,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> (
        AnthropicMessage
        | AnthropicAsyncStream[RawMessageStreamEvent]
        | AsyncMessagesStreamWrapper[None]
    ):
        invocation = _create_invocation(
            handler, instance, args, kwargs, capture_content
        )
        try:
            # See ``messages_create`` on why this is Any.
            result: Any = await wrapped(*args, **kwargs)
        except Exception as exc:
            invocation.fail(exc)
            raise

        if is_anthropic_async_stream(result):
            return AsyncMessagesStreamWrapper(
                cast("AnthropicAsyncStream[RawMessageStreamEvent]", result),
                invocation,
                capture_content,
            )
        if _is_raw_response(result):
            return wrap_raw_response(result, invocation, capture_content)

        if isinstance(result, AnthropicMessage):
            MessageWrapper(result, capture_content).extract_into(invocation)
        invocation.stop()
        return result

    return cast(
        'Callable[..., "AnthropicMessage" | "AnthropicAsyncStream[RawMessageStreamEvent]" | AsyncMessagesStreamWrapper[None]]',
        traced_method,
    )


def _create_invocation(
    handler: TelemetryHandler,
    instance: Messages | AsyncMessages,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    capture_content: bool,
) -> InferenceInvocation:
    params = extract_params(*args, **kwargs)
    attributes = get_llm_request_attributes(params, instance)
    request_model_attribute = attributes.get(
        GenAIAttributes.GEN_AI_REQUEST_MODEL
    )
    request_model = (
        request_model_attribute
        if isinstance(request_model_attribute, str)
        else params.model
    )

    server_address, server_port = get_server_address_and_port(instance)
    invocation = handler.inference(
        provider=ANTHROPIC,
        request_model=request_model,
        server_address=server_address,
        server_port=server_port,
    )
    invocation.input_messages = (
        get_input_messages(params.messages) if capture_content else []
    )
    invocation.system_instruction = (
        get_system_instruction(params.system) if capture_content else []
    )
    invocation.attributes = attributes
    return invocation


def messages_stream(
    handler: TelemetryHandler,
) -> Callable[..., MessagesStreamManagerWrapper[Any]]:
    """Wrap the sync `stream` method of the `Messages` class."""
    capture_content = handler.should_capture_content()

    def traced_method(
        wrapped: Callable[..., MessageStreamManager],
        instance: Messages,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> MessagesStreamManagerWrapper[Any]:
        return MessagesStreamManagerWrapper(
            wrapped(*args, **kwargs),
            lambda: _create_invocation(
                handler, instance, args, kwargs, capture_content
            ),
            capture_content,
        )

    return cast(
        "Callable[..., MessagesStreamManagerWrapper[Any]]", traced_method
    )


def async_messages_stream(
    handler: TelemetryHandler,
) -> Callable[..., AsyncMessagesStreamManagerWrapper[Any]]:
    """Wrap the async `stream` method of the `AsyncMessages` class."""
    capture_content = handler.should_capture_content()

    def traced_method(
        wrapped: Callable[..., AsyncMessageStreamManager[Any]],
        instance: AsyncMessages,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> AsyncMessagesStreamManagerWrapper[Any]:
        return AsyncMessagesStreamManagerWrapper(
            wrapped(*args, **kwargs),
            lambda: _create_invocation(
                handler, instance, args, kwargs, capture_content
            ),
            capture_content,
        )

    return cast(
        "Callable[..., AsyncMessagesStreamManagerWrapper[Any]]",
        traced_method,
    )
