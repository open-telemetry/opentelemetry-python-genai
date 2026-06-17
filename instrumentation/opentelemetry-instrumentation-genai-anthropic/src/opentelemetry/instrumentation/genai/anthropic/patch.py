# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Patching functions for Anthropic instrumentation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Union, cast

from anthropic._streaming import Stream as AnthropicStream
from anthropic.types import Message as AnthropicMessage

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import InferenceInvocation

from .messages_extractors import (
    extract_params,
    get_input_messages,
    get_llm_request_attributes,
    get_server_address_and_port,
    get_system_instruction,
)
from .wrappers import (
    AsyncMessagesStreamWrapper,
    MessagesStreamManagerWrapper,
    MessagesStreamWrapper,
    MessageWrapper,
)

if TYPE_CHECKING:
    from anthropic._streaming import AsyncStream as AnthropicAsyncStream
    from anthropic.lib.streaming._messages import (  # pylint: disable=no-name-in-module
        MessageStreamManager,
    )
    from anthropic.resources.messages import AsyncMessages, Messages
    from anthropic.types import RawMessageStreamEvent

try:
    from anthropic._streaming import AsyncStream as _AnthropicAsyncStream
except ImportError:
    _AnthropicAsyncStream = None


_logger = logging.getLogger(__name__)
ANTHROPIC = "anthropic"


def messages_create(
    handler: TelemetryHandler,
) -> Callable[
    ...,
    Union[
        AnthropicMessage,
        AnthropicStream[RawMessageStreamEvent],
        MessagesStreamWrapper[None],
    ],
]:
    """Wrap the `create` method of the `Messages` class to trace it."""
    capture_content = handler.should_capture_content()

    def traced_method(
        wrapped: Callable[
            ...,
            Union[
                AnthropicMessage,
                AnthropicStream[RawMessageStreamEvent],
            ],
        ],
        instance: Messages,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Union[
        AnthropicMessage,
        AnthropicStream[RawMessageStreamEvent],
        MessagesStreamWrapper[None],
    ]:
        invocation = _create_invocation(
            handler, instance, args, kwargs, capture_content
        )
        try:
            result = wrapped(*args, **kwargs)
            if isinstance(result, AnthropicStream):
                return MessagesStreamWrapper(
                    result, invocation, capture_content
                )

            wrapper = MessageWrapper(result, capture_content)
            wrapper.extract_into(invocation)
            invocation.stop()
            return wrapper.message
        except Exception as exc:
            invocation.fail(exc)
            raise

    return cast(
        'Callable[..., Union["AnthropicMessage", "AnthropicStream[RawMessageStreamEvent]", MessagesStreamWrapper[None]]]',
        traced_method,
    )


def async_messages_create(
    handler: TelemetryHandler,
) -> Callable[
    ...,
    Union[
        AnthropicMessage,
        AnthropicAsyncStream[RawMessageStreamEvent],
        AsyncMessagesStreamWrapper[None],
    ],
]:
    """Wrap the async `create` method of the `AsyncMessages` class."""
    capture_content = handler.should_capture_content()

    async def traced_method(
        wrapped: Callable[
            ...,
            Awaitable[
                Union[
                    AnthropicMessage,
                    AnthropicAsyncStream[RawMessageStreamEvent],
                ]
            ],
        ],
        instance: AsyncMessages,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Union[
        AnthropicMessage,
        AnthropicAsyncStream[RawMessageStreamEvent],
        AsyncMessagesStreamWrapper[None],
    ]:
        invocation = _create_invocation(
            handler, instance, args, kwargs, capture_content
        )
        try:
            result: (
                AnthropicMessage | AnthropicAsyncStream[RawMessageStreamEvent]
            ) = await wrapped(*args, **kwargs)
            if _is_anthropic_async_stream(result):
                return AsyncMessagesStreamWrapper(
                    cast(
                        "AnthropicAsyncStream[RawMessageStreamEvent]", result
                    ),
                    invocation,
                    capture_content,
                )

            wrapper = MessageWrapper(
                cast("AnthropicMessage", result), capture_content
            )
            wrapper.extract_into(invocation)
            invocation.stop()
            return wrapper.message
        except Exception as exc:
            invocation.fail(exc)
            raise

    return cast(
        'Callable[..., Union["AnthropicMessage", "AnthropicAsyncStream[RawMessageStreamEvent]", AsyncMessagesStreamWrapper[None]]]',
        traced_method,
    )


def _is_anthropic_async_stream(result: object) -> bool:
    if _AnthropicAsyncStream is not None and isinstance(
        result, _AnthropicAsyncStream
    ):
        return True
    return (
        hasattr(result, "__anext__")
        and callable(getattr(result, "close", None))
        and hasattr(result, "response")
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
