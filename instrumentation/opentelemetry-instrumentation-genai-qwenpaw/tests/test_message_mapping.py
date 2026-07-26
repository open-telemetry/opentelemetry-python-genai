# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for argument parsing and message mapping helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.genai.qwenpaw.utils import (
    input_messages_from_msgs,
    non_empty_str,
    output_message_from_yield_item,
    parse_query_handler_call,
)


def test_parse_query_handler_call_positional():
    msgs = [1]
    request = SimpleNamespace(session_id="a")
    parsed_msgs, parsed_request = parse_query_handler_call(
        (msgs, request), {}
    )
    assert parsed_msgs is msgs
    assert parsed_request is request


def test_parse_query_handler_call_kwargs():
    msgs = [2]
    request = SimpleNamespace(session_id="b")
    parsed_msgs, parsed_request = parse_query_handler_call(
        tuple(), {"msgs": msgs, "request": request}
    )
    assert parsed_msgs is msgs
    assert parsed_request is request


def test_non_empty_str():
    assert non_empty_str(None) is None
    assert non_empty_str("  ") is None
    assert non_empty_str(" sess ") == "sess"
    assert non_empty_str(123) == "123"


def test_input_messages_from_agentscope_msg():
    pytest.importorskip("agentscope.message")
    from agentscope.message import Msg, TextBlock  # noqa: PLC0415

    msg = Msg(
        name="u", role="user", content=[TextBlock(type="text", text="hi")]
    )
    input_messages = input_messages_from_msgs([msg])
    assert len(input_messages) == 1
    assert input_messages[0].role == "user"
    assert len(input_messages[0].parts) == 1
    assert input_messages[0].parts[0].content == "hi"


def test_input_messages_skips_unmappable_entries():
    assert input_messages_from_msgs(None) == []
    assert input_messages_from_msgs([object()]) == []


def test_output_message_from_yield_item():
    pytest.importorskip("agentscope.message")
    from agentscope.message import Msg, TextBlock  # noqa: PLC0415

    assistant = Msg(
        name="Friday",
        role="assistant",
        content=[TextBlock(type="text", text="done")],
    )
    output = output_message_from_yield_item((assistant, True))
    assert output is not None
    assert output.role == "assistant"
    assert output.finish_reason == "stop"
    assert output.parts[0].content == "done"

    user = Msg(name="user", role="user", content="hi")
    assert output_message_from_yield_item((user, True)) is None
    assert output_message_from_yield_item("not-a-tuple") is None
    assert output_message_from_yield_item((None, True)) is None
