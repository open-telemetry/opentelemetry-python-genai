# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Map ``AgentRunner.query_handler`` call data onto OTel GenAI types."""

from __future__ import annotations

from typing import Any, cast

from opentelemetry.util.genai.types import InputMessage, OutputMessage, Text


def non_empty_str(value: Any) -> str | None:
    """Return ``str(value)`` stripped, or ``None`` when empty/absent."""
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def parse_query_handler_call(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, Any]:
    """Return ``(msgs, request)`` from ``query_handler`` positional/kwargs."""
    msgs: Any = None
    request: Any = None
    if args:
        msgs = args[0]
        if len(args) > 1:
            request = args[1]
    if msgs is None and "msgs" in kwargs:
        msgs = kwargs["msgs"]
    if request is None:
        request = kwargs.get("request")
    return msgs, request


def _msg_text(msg: Any) -> str | None:
    """Return the AgentScope message's text content, or ``None``."""
    get_text: Any = getattr(msg, "get_text_content", None)
    if not callable(get_text):
        return None
    text: Any = get_text()
    return text if isinstance(text, str) and text else None


def input_messages_from_msgs(msgs: Any) -> list[InputMessage]:
    """Turn an AgentScope message (or list of them) into ``InputMessage`` entries."""
    if not msgs:
        return []
    items: list[Any]
    if isinstance(msgs, (list, tuple)):
        items = [*cast("tuple[Any, ...]", msgs)]
    else:
        items = [msgs]
    messages: list[InputMessage] = []
    for msg in items:
        text = _msg_text(msg)
        if text is None:
            continue
        role = non_empty_str(getattr(msg, "role", None)) or "user"
        messages.append(InputMessage(role=role, parts=[Text(content=text)]))
    return messages


def output_message_from_yield_item(item: Any) -> OutputMessage | None:
    """Map a ``(Msg, last)`` yield item with assistant text to an ``OutputMessage``."""
    if not isinstance(item, tuple) or not item:
        return None
    msg = cast("tuple[Any, ...]", item)[0]
    if msg is None:
        return None
    if getattr(msg, "role", None) != "assistant":
        return None
    text = _msg_text(msg)
    if text is None:
        return None
    return OutputMessage(
        role="assistant",
        parts=[Text(content=text)],
        finish_reason="stop",
    )
