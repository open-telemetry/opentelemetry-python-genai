# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0


import logging

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    InferenceInvocation,
)
from opentelemetry.util.genai.types import (
    Error,
)

from .chat_wrappers import AsyncChatStreamWrapper, ChatStreamWrapper
from .utils import (
    _prepare_output_messages,
    create_chat_invocation,
    is_streaming,
)

_logger = logging.getLogger(__name__)


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

    if getattr(result, "usage", None):
        chat_invocation.input_tokens = result.usage.prompt_tokens
        chat_invocation.output_tokens = result.usage.completion_tokens
    elif getattr(result, "x_groq", None):
        usage = getattr(result.x_groq, "usage", None)
        if usage:
            chat_invocation.input_tokens = getattr(
                usage, "prompt_tokens", None
            )
            chat_invocation.output_tokens = getattr(
                usage, "completion_tokens", None
            )

    return chat_invocation
