# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Patching functions for the Ollama instrumentation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.genai.ollama.stream import (
    AsyncChatStreamWrapper,
    ChatStreamWrapper,
)
from opentelemetry.instrumentation.genai.ollama.utils import (
    create_chat_invocation,
    create_embed_invocation,
    materialize_chat_inputs,
    set_chat_response,
    set_embed_response,
)
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.handler import TelemetryHandler

# ``ollama.chat`` and ``ollama.embed`` are methods bound to a default client
# when ``ollama`` is imported, so wrapping the class alone does not reach them.
_MODULE_FUNCTIONS = ("chat", "embed")

_original_module_functions: dict[str, Any] = {}


def patch_ollama(handler: TelemetryHandler) -> None:
    import ollama  # pylint: disable=import-outside-toplevel

    wrap_function_wrapper(ollama.Client, "chat", _chat(handler))
    wrap_function_wrapper(ollama.AsyncClient, "chat", _async_chat(handler))
    wrap_function_wrapper(ollama.Client, "embed", _embed(handler))
    wrap_function_wrapper(ollama.AsyncClient, "embed", _async_embed(handler))

    default_client = getattr(ollama, "_client", None)
    if isinstance(default_client, ollama.Client):
        for name in _MODULE_FUNCTIONS:
            _original_module_functions[name] = getattr(ollama, name)
            setattr(ollama, name, getattr(default_client, name))


def unpatch_ollama() -> None:
    import ollama  # pylint: disable=import-outside-toplevel

    unwrap(ollama.Client, "chat")
    unwrap(ollama.AsyncClient, "chat")
    unwrap(ollama.Client, "embed")
    unwrap(ollama.AsyncClient, "embed")

    for name, function in _original_module_functions.items():
        setattr(ollama, name, function)
    _original_module_functions.clear()


def _chat(handler: TelemetryHandler) -> Callable[..., Any]:
    def traced_method(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        capture_content = handler.should_capture_content()
        args, kwargs = materialize_chat_inputs(args, kwargs)
        invocation = create_chat_invocation(
            handler, instance, args, kwargs, capture_content
        )
        try:
            result = wrapped(*args, **kwargs)
        except BaseException as error:
            invocation.fail(error)
            raise
        if kwargs.get("stream"):
            return ChatStreamWrapper(result, invocation, capture_content)
        set_chat_response(invocation, result, capture_content)
        invocation.stop()
        return result

    return traced_method


def _async_chat(handler: TelemetryHandler) -> Callable[..., Any]:
    async def traced_method(
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        capture_content = handler.should_capture_content()
        args, kwargs = materialize_chat_inputs(args, kwargs)
        invocation = create_chat_invocation(
            handler, instance, args, kwargs, capture_content
        )
        try:
            result = await wrapped(*args, **kwargs)
        except BaseException as error:
            invocation.fail(error)
            raise
        if kwargs.get("stream"):
            return AsyncChatStreamWrapper(result, invocation, capture_content)
        set_chat_response(invocation, result, capture_content)
        invocation.stop()
        return result

    return traced_method


def _embed(handler: TelemetryHandler) -> Callable[..., Any]:
    def traced_method(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        invocation = create_embed_invocation(handler, instance, args, kwargs)
        try:
            result = wrapped(*args, **kwargs)
        except BaseException as error:
            invocation.fail(error)
            raise
        set_embed_response(invocation, result)
        invocation.stop()
        return result

    return traced_method


def _async_embed(handler: TelemetryHandler) -> Callable[..., Any]:
    async def traced_method(
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        invocation = create_embed_invocation(handler, instance, args, kwargs)
        try:
            result = await wrapped(*args, **kwargs)
        except BaseException as error:
            invocation.fail(error)
            raise
        set_embed_response(invocation, result)
        invocation.stop()
        return result

    return traced_method
