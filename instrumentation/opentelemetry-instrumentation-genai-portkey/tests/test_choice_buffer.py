# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ChoiceBuffer tool call accumulation."""

from __future__ import annotations

from types import SimpleNamespace

from opentelemetry.instrumentation.genai.portkey.wrappers import ChoiceBuffer


def _delta(call_id, name, arguments, index=None):
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _calls(buffer):
    return [
        (b.tool_call_id, b.function_name, "".join(b.arguments))
        for b in buffer.tool_calls_buffers
        if b is not None
    ]


def test_tool_calls_without_index_are_kept_apart():
    buffer = ChoiceBuffer(0)
    buffer.append_tool_call(_delta("call_1", "get_weather", '{"city":'))
    buffer.append_tool_call(_delta(None, None, ' "Paris"}'))
    buffer.append_tool_call(_delta("call_2", "get_time", '{"zone": "UTC"}'))

    assert _calls(buffer) == [
        ("call_1", "get_weather", '{"city": "Paris"}'),
        ("call_2", "get_time", '{"zone": "UTC"}'),
    ]


def test_tool_call_without_index_repeating_the_same_id_continues():
    buffer = ChoiceBuffer(0)
    buffer.append_tool_call(_delta("call_1", "get_weather", '{"city":'))
    buffer.append_tool_call(_delta("call_1", None, ' "Paris"}'))

    assert _calls(buffer) == [("call_1", "get_weather", '{"city": "Paris"}')]


def test_tool_calls_with_index_keep_their_slots():
    buffer = ChoiceBuffer(0)
    buffer.append_tool_call(_delta("call_1", "get_weather", "{", index=0))
    buffer.append_tool_call(_delta("call_2", "get_time", "{", index=1))
    buffer.append_tool_call(_delta(None, None, "}", index=0))
    buffer.append_tool_call(_delta(None, None, "}", index=1))

    assert _calls(buffer) == [
        ("call_1", "get_weather", "{}"),
        ("call_2", "get_time", "{}"),
    ]
