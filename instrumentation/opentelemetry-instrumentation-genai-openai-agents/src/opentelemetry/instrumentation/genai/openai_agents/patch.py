# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Monkey patches for openai-agents Runner to capture conversation_id / session."""

from __future__ import annotations

import contextvars
from collections.abc import Callable
from typing import Any

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.utils import unwrap

_current_conversation_id: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("_current_conversation_id", default=None)
)


def _extract_conversation_id(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> str | None:
    # 1. Explicit conversation_id parameter
    conversation_id = kwargs.get("conversation_id")
    if conversation_id:
        return str(conversation_id)

    # 2. Resumed RunState passed as input (args[1] or kwargs["input"])
    input_arg = kwargs.get("input")
    if input_arg is None and len(args) > 1:
        input_arg = args[1]
    if input_arg is not None:
        state_conv_id = getattr(input_arg, "_conversation_id", None)
        if state_conv_id:
            return str(state_conv_id)

    # 3. Client-side SDK session (Session.session_id)
    session = kwargs.get("session")
    if session is not None:
        session_id = getattr(session, "session_id", None)
        if session_id:
            return str(session_id)

    return None


async def _traced_agent_runner_run(
    wrapped: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    conversation_id = _extract_conversation_id(args, kwargs)
    if conversation_id is not None:
        token = _current_conversation_id.set(conversation_id)
        try:
            return await wrapped(*args, **kwargs)
        finally:
            _current_conversation_id.reset(token)
    return await wrapped(*args, **kwargs)


def _traced_agent_runner_run_sync(
    wrapped: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    conversation_id = _extract_conversation_id(args, kwargs)
    if conversation_id is not None:
        token = _current_conversation_id.set(conversation_id)
        try:
            return wrapped(*args, **kwargs)
        finally:
            _current_conversation_id.reset(token)
    return wrapped(*args, **kwargs)


def _traced_agent_runner_run_streamed(
    wrapped: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    conversation_id = _extract_conversation_id(args, kwargs)
    if conversation_id is not None:
        token = _current_conversation_id.set(conversation_id)
        try:
            return wrapped(*args, **kwargs)
        finally:
            _current_conversation_id.reset(token)
    return wrapped(*args, **kwargs)


def patch() -> None:
    wrap_function_wrapper(
        "agents.run", "AgentRunner.run", _traced_agent_runner_run
    )
    wrap_function_wrapper(
        "agents.run", "AgentRunner.run_sync", _traced_agent_runner_run_sync
    )
    wrap_function_wrapper(
        "agents.run",
        "AgentRunner.run_streamed",
        _traced_agent_runner_run_streamed,
    )


def unpatch() -> None:
    unwrap("agents.run.AgentRunner", "run")
    unwrap("agents.run.AgentRunner", "run_sync")
    unwrap("agents.run.AgentRunner", "run_streamed")
