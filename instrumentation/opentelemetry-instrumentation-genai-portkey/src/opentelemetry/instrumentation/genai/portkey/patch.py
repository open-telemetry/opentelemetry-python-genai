# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Patching functions for Portkey AI instrumentation."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, cast

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.genai.portkey.utils import (
    create_inference_invocation,
    is_streaming,
    set_response_properties,
)
from opentelemetry.instrumentation.genai.portkey.wrappers import (
    AsyncPortkeyStreamWrapper,
    PortkeyStreamWrapper,
)
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.handler import TelemetryHandler

logger = logging.getLogger(__name__)

_CHAT_COMPLETE_MODULE = "portkey_ai.api_resources.apis.chat_complete"
_GENERATION_MODULE = "portkey_ai.api_resources.apis.generation"
_COMPLETIONS_CLASS = "Completions"
_ASYNC_COMPLETIONS_CLASS = "AsyncCompletions"


def patch_portkey(handler: TelemetryHandler) -> None:
    """Apply patches to Portkey AI completion methods."""
    try:
        wrap_function_wrapper(
            _CHAT_COMPLETE_MODULE,
            f"{_COMPLETIONS_CLASS}.create",
            _sync_completions_create(handler, is_prompt=False),
        )
        wrap_function_wrapper(
            _CHAT_COMPLETE_MODULE,
            f"{_ASYNC_COMPLETIONS_CLASS}.create",
            _async_completions_create(handler, is_prompt=False),
        )
    except (ImportError, AttributeError) as exc:
        logger.debug("Failed to patch Portkey chat completions: %s", exc)

    try:
        wrap_function_wrapper(
            _GENERATION_MODULE,
            f"{_COMPLETIONS_CLASS}.create",
            _sync_completions_create(handler, is_prompt=True),
        )
        wrap_function_wrapper(
            _GENERATION_MODULE,
            f"{_ASYNC_COMPLETIONS_CLASS}.create",
            _async_completions_create(handler, is_prompt=True),
        )
    except (ImportError, AttributeError) as exc:
        logger.debug("Failed to patch Portkey prompt completions: %s", exc)


def unpatch_portkey() -> None:
    """Remove patches from Portkey AI completion methods."""
    try:
        from portkey_ai.api_resources.apis import (  # pylint: disable=import-outside-toplevel
            chat_complete,
        )

        unwrap(chat_complete.Completions, "create")
        unwrap(chat_complete.AsyncCompletions, "create")
    except (ImportError, AttributeError):
        pass

    try:
        from portkey_ai.api_resources.apis import (  # pylint: disable=import-outside-toplevel
            generation,
        )

        unwrap(generation.Completions, "create")
        unwrap(generation.AsyncCompletions, "create")
    except (ImportError, AttributeError):
        pass


def _sync_completions_create(
    handler: TelemetryHandler,
    *,
    is_prompt: bool = False,
) -> Callable[..., Any]:
    capture_content = handler.should_capture_content()

    def traced_method(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        invocation = create_inference_invocation(
            handler, instance, kwargs, capture_content, is_prompt=is_prompt
        )
        try:
            result = wrapped(*args, **kwargs)
            if is_streaming(kwargs):
                return PortkeyStreamWrapper(
                    result, invocation, capture_content
                )

            set_response_properties(invocation, result, capture_content)
            invocation.stop()
            return result
        except Exception as error:
            invocation.fail(error)
            raise

    return traced_method


def _async_completions_create(
    handler: TelemetryHandler,
    *,
    is_prompt: bool = False,
) -> Callable[..., Any]:
    capture_content = handler.should_capture_content()

    async def traced_method(
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        invocation = create_inference_invocation(
            handler, instance, kwargs, capture_content, is_prompt=is_prompt
        )
        try:
            result = await wrapped(*args, **kwargs)
            if is_streaming(kwargs):
                return AsyncPortkeyStreamWrapper(
                    result, invocation, capture_content
                )

            set_response_properties(invocation, result, capture_content)
            invocation.stop()
            return result
        except Exception as error:
            invocation.fail(error)
            raise

    return cast(Callable[..., Any], traced_method)
