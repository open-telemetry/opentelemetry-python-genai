# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Context propagation helpers for openai-agents Runner conversation_id and session."""

from __future__ import annotations

import contextvars
from typing import Any

from agents.memory.session import Session

_current_conversation_id: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("_current_conversation_id", default=None)
)
_current_session: contextvars.ContextVar[Session | None] = (
    contextvars.ContextVar("_current_session", default=None)
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
    session: Session | None = kwargs.get("session")
    if session is not None:
        try:
            return session.session_id
        except (ValueError, AttributeError):
            return None

    return None


def _extract_session(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Session | None:
    session: Any = kwargs.get("session")
    if session is None and len(args) > 9:
        session = args[9]
    if isinstance(session, Session):
        return session
    return None


def _get_active_conversation_id() -> str | None:
    conv_id = _current_conversation_id.get()
    if conv_id:
        return conv_id

    session = _current_session.get()
    if session is not None:
        try:
            return session.session_id
        except (ValueError, AttributeError):
            return None

    return None


def patch() -> None:
    """No-op; runner wrapping is handled by OpenAIAgentsInstrumentor."""


def unpatch() -> None:
    """No-op; runner wrapping is handled by OpenAIAgentsInstrumentor."""
