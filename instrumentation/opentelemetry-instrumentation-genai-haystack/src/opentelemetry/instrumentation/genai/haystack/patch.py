# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Patching functions for Haystack instrumentation.

Builds ``opentelemetry-util-genai`` invocations around:

- ``haystack.Pipeline.run`` / ``run_async`` -> ``WorkflowInvocation``
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import WorkflowInvocation


# ---------------------------------------------------------------------------
# Pipeline.run / Pipeline.run_async / Pipeline.run_async_generator -> WorkflowInvocation
# ---------------------------------------------------------------------------
#
# Pipeline.run_async (the true async entry point) internally drains
# run_async_generator() to completion in the same asyncio task. Wrapping
# both unconditionally would double-count a single logical pipeline
# execution, so `_inside_run_async` -- set for the duration of the outer
# call -- lets the run_async_generator wrapper tell "called directly by
# user code" (create a span) from "driven internally by run_async" (already
# tracing it).
import contextvars

_inside_run_async: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_inside_run_async", default=False
)


def pipeline_run(handler: TelemetryHandler) -> Callable[..., Any]:
    def traced_method(
        wrapped: Callable[..., Any],
        instance: Any,  # noqa: ARG001
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        # Haystack pings deepset's telemetry endpoint on first Pipeline.run() /
        # run_async_generator(); this resolves the module lazily to avoid an
        # import cycle with deepset's own telemetry module attempting to import
        # haystack.Pipeline.
        from haystack import Pipeline  # pylint: disable=import-outside-toplevel

        invocation = handler.workflow(
            name=kwargs.get("name") or "Pipeline"
        )
        try:
            result = wrapped(*args, **kwargs)
        except Exception as exc:
            invocation.fail(exc)
            raise
        invocation.stop()
        return result

    return traced_method


def pipeline_run_async(handler: TelemetryHandler) -> Callable[..., Any]:
    async def traced_method(
        wrapped: Callable[..., Any],
        instance: Any,  # noqa: ARG001
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        invocation = handler.workflow(
            name=kwargs.get("name") or "Pipeline"
        )
        token = _inside_run_async.set(True)
        try:
            result = await wrapped(*args, **kwargs)
        except Exception as exc:
            invocation.fail(exc)
            raise
        finally:
            _inside_run_async.reset(token)
        invocation.stop()
        return result

    return traced_method


def pipeline_run_async_generator(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    async def traced_method(
        wrapped: Callable[..., Any],
        instance: Any,  # noqa: ARG001
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        is_outer_call = not _inside_run_async.get()
        invocation = None
        if is_outer_call:
            invocation = handler.workflow(
                name=kwargs.get("name") or "Pipeline"
            )

        try:
            # Haystack's pipeline_run_async_generator returns an async generator;
            # we need to call it to *get* it, then yield from it.
            generator = wrapped(*args, **kwargs)
            async for item in generator:
                yield item
        except Exception as exc:
            if invocation:
                invocation.fail(exc)
            raise
        if invocation:
            invocation.stop()

    return traced_method
