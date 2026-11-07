# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Link a chat model call to the prompt template that produced its input.

The run tree alone cannot always tell: a prompt formatted outside any chain
has no parent run to publish to, and two prompts formatted under one parent
reach their models in any order. The callback handler records the prompt value
each prompt run returns; these wrappers find that value in a chat model's
input and record its template under the run id chosen here for the call.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from functools import wraps
from typing import Any
from uuid import UUID

from langchain_core.prompt_values import PromptValue

from opentelemetry.instrumentation.genai.langchain._run_context import (
    _new_run_id,
    _signature,
)
from opentelemetry.instrumentation.genai.langchain.invocation_manager import (
    _InvocationManager,
)

__all__ = ["_wrap_prompt_input"]

_Call = tuple[UUID | None, tuple[Any, ...], dict[str, Any]]


def _wrap_prompt_input(
    original: Callable[..., Any],
    invocations: _InvocationManager,
    stream: bool,
    asynchronous: bool,
) -> Callable[..., Any]:
    parameters = _signature(original)

    def prepare(
        instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> _Call:
        bound = parameters.bind(instance, *args, **kwargs)
        value = bound.arguments.get("input")
        prompt_context = (
            invocations.get_prompt_value_context(value)
            if isinstance(value, PromptValue)
            else None
        )
        if prompt_context is None:
            return None, args, kwargs
        # The model starts its run with this id, so the handler finds the
        # template when the run starts.
        config = dict(bound.arguments.get("config") or {})
        run_id: UUID = config.get("run_id") or _new_run_id()
        config["run_id"] = run_id
        bound.arguments["config"] = config
        invocations.set_model_prompt_context(run_id, prompt_context)
        return run_id, bound.args[1:], bound.kwargs

    def forget(run_id: UUID | None) -> None:
        # The handler takes the template when the run starts; this drops it
        # when the run never started.
        if run_id is not None:
            invocations.pop_model_prompt_context(run_id)

    @wraps(original)
    def invoke(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        run_id, args, kwargs = prepare(instance, args, kwargs)
        try:
            return wrapped(*args, **kwargs)
        finally:
            forget(run_id)

    @wraps(original)
    async def ainvoke(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        run_id, args, kwargs = prepare(instance, args, kwargs)
        try:
            return await wrapped(*args, **kwargs)
        finally:
            forget(run_id)

    # The stream wrappers are generators, so the input is read on the first
    # advancement, like the model's own stream, and an unread stream records
    # nothing.
    @wraps(original)
    def sync_stream(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Iterator[Any]:
        run_id, args, kwargs = prepare(instance, args, kwargs)
        try:
            yield from wrapped(*args, **kwargs)
        finally:
            forget(run_id)

    @wraps(original)
    async def async_stream(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> AsyncIterator[Any]:
        run_id, args, kwargs = prepare(instance, args, kwargs)
        chunks = wrapped(*args, **kwargs)
        try:
            async for chunk in chunks:
                yield chunk
        finally:
            forget(run_id)
            close = getattr(chunks, "aclose", None)
            if close is not None:
                await close()

    if stream:
        return async_stream if asynchronous else sync_stream
    return ainvoke if asynchronous else invoke
