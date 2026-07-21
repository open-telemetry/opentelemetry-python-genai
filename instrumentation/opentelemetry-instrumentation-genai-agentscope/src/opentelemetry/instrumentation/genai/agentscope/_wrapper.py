# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Wrapper classes for AgentScope v1 instrumentation.

Stacked ``ChatModelBase`` / ``AgentBase`` implementations (e.g. proxies where
each layer subclasses the base and ``__call__`` forwards to an inner model or
agent) share one logical invocation. A ``contextvars`` depth counter ensures
only the outermost ``__call__`` emits LLM / ``invoke_agent`` spans; inner
layers call through without duplicating telemetry.
"""

from __future__ import annotations

import contextvars
import logging
from collections.abc import AsyncGenerator
from functools import wraps
from typing import Any

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import InferenceInvocation

from .utils import (
    convert_agent_response_to_output_messages,
    convert_chatresponse_to_output_messages,
    start_agent_invocation,
    start_embedding_invocation,
    start_llm_invocation,
)

logger = logging.getLogger(__name__)

# Per-async-task nesting for stacked __call__ (proxy / decorator chains).
_CHAT_MODEL_CALL_DEPTH = contextvars.ContextVar(
    "opentelemetry_agentscope_chat_model_call_depth",
    default=0,
)
_AGENT_CALL_DEPTH = contextvars.ContextVar(
    "opentelemetry_agentscope_agent_call_depth",
    default=0,
)


class AgentScopeChatModelWrapper:
    """Wrapper for ChatModelBase that hijacks __init__ to replace __call__."""

    _original_methods: dict[type, dict[str, Any]] = {}

    def __init__(self, handler: TelemetryHandler) -> None:
        self._handler = handler
        self._instrumented_classes: set[type] = set()

    @classmethod
    def restore_original_methods(cls) -> None:
        """Restore all replaced original methods."""
        for class_obj, methods in cls._original_methods.items():
            for method_name, original_method in methods.items():
                setattr(class_obj, method_name, original_method)
        cls._original_methods.clear()

    async def _wrap_streaming_response(
        self, generator: AsyncGenerator, invocation: InferenceInvocation
    ) -> AsyncGenerator:
        """Wrap streaming response to finalize invocation once drained."""
        try:
            last_chunk = None
            async for chunk in generator:
                last_chunk = chunk
                yield chunk

            if last_chunk:
                self._finish_llm(invocation, last_chunk)

            invocation.stop()
        except GeneratorExit:
            invocation.stop()
            raise
        except Exception as e:
            invocation.fail(e)
            raise

    def _finish_llm(
        self, invocation: InferenceInvocation, result: Any
    ) -> None:
        if self._handler.should_capture_content():
            invocation.output_messages = (
                convert_chatresponse_to_output_messages(result)
            )
        invocation.finish_reasons = ["stop"]
        invocation.response_model_name = invocation.request_model

        if hasattr(result, "usage") and result.usage:
            invocation.input_tokens = getattr(
                result.usage, "input_tokens", None
            )
            invocation.output_tokens = getattr(
                result.usage, "output_tokens", None
            )
        if hasattr(result, "id"):
            invocation.response_id = getattr(result, "id", None)

    def __call__(
        self, wrapped: Any, instance: Any, args: Any, kwargs: Any
    ) -> Any:
        """Hijack ChatModelBase.__init__ to replace the instance's __call__."""
        model_class = type(instance)

        if model_class in self._instrumented_classes:
            return wrapped(*args, **kwargs)

        result = wrapped(*args, **kwargs)

        if not hasattr(model_class, "__call__") or not callable(
            getattr(model_class, "__call__", None)
        ):
            return result

        original_call = model_class.__call__
        self._original_methods.setdefault(model_class, {})["__call__"] = (
            original_call
        )
        handler = self._handler

        @wraps(original_call)
        async def async_wrapped_call(
            call_self: Any, *call_args: Any, **call_kwargs: Any
        ) -> Any:
            parent_depth = _CHAT_MODEL_CALL_DEPTH.get()
            depth_token = _CHAT_MODEL_CALL_DEPTH.set(parent_depth + 1)
            try:
                if parent_depth > 0:
                    return await original_call(
                        call_self, *call_args, **call_kwargs
                    )

                invocation = start_llm_invocation(
                    handler, call_self, call_args, call_kwargs
                )

                try:
                    result = await original_call(
                        call_self, *call_args, **call_kwargs
                    )

                    if isinstance(result, AsyncGenerator):
                        return self._wrap_streaming_response(
                            result, invocation
                        )

                    self._finish_llm(invocation, result)
                    invocation.stop()
                    return result
                except Exception as e:
                    invocation.fail(e)
                    raise
            finally:
                _CHAT_MODEL_CALL_DEPTH.reset(depth_token)

        instance.__class__.__call__ = async_wrapped_call
        self._instrumented_classes.add(model_class)
        return result


class AgentScopeAgentWrapper:
    """Wrapper for AgentBase that hijacks __init__ to replace __call__."""

    _original_methods: dict[type, dict[str, Any]] = {}

    def __init__(self, handler: TelemetryHandler) -> None:
        self._handler = handler
        self._instrumented_classes: set[type] = set()

    @classmethod
    def restore_original_methods(cls) -> None:
        """Restore all replaced original methods."""
        for class_obj, methods in cls._original_methods.items():
            for method_name, original_method in methods.items():
                setattr(class_obj, method_name, original_method)
        cls._original_methods.clear()

    def __call__(
        self, wrapped: Any, instance: Any, args: Any, kwargs: Any
    ) -> Any:
        """Hijack AgentBase.__init__ to replace the instance's __call__."""
        agent_class = type(instance)

        if agent_class in self._instrumented_classes:
            return wrapped(*args, **kwargs)

        result = wrapped(*args, **kwargs)

        if not hasattr(agent_class, "__call__") or not callable(
            getattr(agent_class, "__call__", None)
        ):
            return result

        original_call = agent_class.__call__
        self._original_methods.setdefault(agent_class, {})["__call__"] = (
            original_call
        )
        handler = self._handler

        @wraps(original_call)
        async def async_wrapped_call(
            call_self: Any, *call_args: Any, **call_kwargs: Any
        ) -> Any:
            parent_depth = _AGENT_CALL_DEPTH.get()
            depth_token = _AGENT_CALL_DEPTH.set(parent_depth + 1)
            try:
                if parent_depth > 0:
                    return await original_call(
                        call_self, *call_args, **call_kwargs
                    )

                try:
                    invocation = start_agent_invocation(
                        handler, call_self, call_args, call_kwargs
                    )
                except Exception as e:
                    logger.exception(
                        "Error starting agent instrumentation: %s", e
                    )
                    return await original_call(
                        call_self, *call_args, **call_kwargs
                    )

                try:
                    agent_result = await original_call(
                        call_self, *call_args, **call_kwargs
                    )

                    if handler.should_capture_content():
                        invocation.output_messages = (
                            convert_agent_response_to_output_messages(
                                agent_result
                            )
                        )
                    invocation.stop()
                    return agent_result
                except Exception as e:
                    invocation.fail(e)
                    raise
            finally:
                _AGENT_CALL_DEPTH.reset(depth_token)

        instance.__class__.__call__ = async_wrapped_call
        self._instrumented_classes.add(agent_class)
        return result


class AgentScopeEmbeddingModelWrapper:
    """Wrapper for EmbeddingModelBase that hijacks __init__ to replace __call__."""

    _original_methods: dict[type, dict[str, Any]] = {}

    def __init__(self, handler: TelemetryHandler) -> None:
        self._handler = handler
        self._instrumented_classes: set[type] = set()

    @classmethod
    def restore_original_methods(cls) -> None:
        """Restore all replaced original methods."""
        for class_obj, methods in cls._original_methods.items():
            for method_name, original_method in methods.items():
                setattr(class_obj, method_name, original_method)
        cls._original_methods.clear()

    def __call__(
        self, wrapped: Any, instance: Any, args: Any, kwargs: Any
    ) -> Any:
        """Hijack EmbeddingModelBase.__init__ to replace __call__."""
        embedding_class = type(instance)

        if embedding_class in self._instrumented_classes:
            return wrapped(*args, **kwargs)

        result = wrapped(*args, **kwargs)

        if not hasattr(embedding_class, "__call__") or not callable(
            getattr(embedding_class, "__call__", None)
        ):
            return result

        original_call = embedding_class.__call__
        self._original_methods.setdefault(embedding_class, {})["__call__"] = (
            original_call
        )
        handler = self._handler

        @wraps(original_call)
        async def async_wrapped_call(
            call_self: Any, *call_args: Any, **call_kwargs: Any
        ) -> Any:
            invocation = start_embedding_invocation(
                handler, call_self, call_args, call_kwargs
            )

            try:
                embedding_result = await original_call(
                    call_self, *call_args, **call_kwargs
                )

                if (
                    hasattr(embedding_result, "embeddings")
                    and embedding_result.embeddings
                ):
                    invocation.dimension_count = len(
                        embedding_result.embeddings[0]
                    )
                if (
                    hasattr(embedding_result, "usage")
                    and embedding_result.usage
                ):
                    tokens = getattr(embedding_result.usage, "tokens", None)
                    if tokens is not None:
                        invocation.input_tokens = tokens

                invocation.response_model_name = invocation.request_model

                invocation.stop()
                return embedding_result
            except Exception as e:
                invocation.fail(e)
                raise

        instance.__class__.__call__ = async_wrapped_call
        self._instrumented_classes.add(embedding_class)
        return result
