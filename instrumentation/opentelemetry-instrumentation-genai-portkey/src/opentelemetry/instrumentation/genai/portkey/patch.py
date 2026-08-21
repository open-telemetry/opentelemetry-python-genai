# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Patching functions for Portkey AI instrumentation."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, cast

from wrapt import wrap_function_wrapper

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
            _chat_completions_create(handler),
        )
        wrap_function_wrapper(
            _CHAT_COMPLETE_MODULE,
            f"{_ASYNC_COMPLETIONS_CLASS}.create",
            _async_chat_completions_create(handler),
        )
    except (ImportError, AttributeError) as exc:
        logger.debug("Failed to patch Portkey chat completions: %s", exc)

    try:
        wrap_function_wrapper(
            _GENERATION_MODULE,
            f"{_COMPLETIONS_CLASS}.create",
            _prompts_completions_create(handler),
        )
        wrap_function_wrapper(
            _GENERATION_MODULE,
            f"{_ASYNC_COMPLETIONS_CLASS}.create",
            _async_prompts_completions_create(handler),
        )
    except (ImportError, AttributeError) as exc:
        logger.debug("Failed to patch Portkey prompt completions: %s", exc)


def unpatch_portkey() -> None:
    """Remove patches from Portkey AI completion methods."""
    try:
        from portkey_ai.api_resources.apis import (
            chat_complete,  # pylint: disable=import-outside-toplevel
        )

        unwrap(chat_complete.Completions, "create")
        unwrap(chat_complete.AsyncCompletions, "create")
    except (ImportError, AttributeError):
        pass

    try:
        from portkey_ai.api_resources.apis import (
            generation,  # pylint: disable=import-outside-toplevel
        )

        unwrap(generation.Completions, "create")
        unwrap(generation.AsyncCompletions, "create")
    except (ImportError, AttributeError):
        pass


def _chat_completions_create(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    def traced_method(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        # TODO: Implement full inference invocation telemetry mapping
        return wrapped(*args, **kwargs)

    return traced_method


def _async_chat_completions_create(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    async def traced_method(
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        # TODO: Implement full inference invocation telemetry mapping
        return await wrapped(*args, **kwargs)

    return cast(Callable[..., Any], traced_method)


def _prompts_completions_create(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    def traced_method(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        # TODO: Implement full prompt completion inference invocation mapping
        return wrapped(*args, **kwargs)

    return traced_method


def _async_prompts_completions_create(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    async def traced_method(
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        # TODO: Implement full prompt completion inference invocation mapping
        return await wrapped(*args, **kwargs)

    return cast(Callable[..., Any], traced_method)
