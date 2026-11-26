# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ChoiceBuffer tool call accumulation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

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


def test_new_id_with_repeated_index_opens_a_new_call():
    # Gemini via Portkey: index is the position inside each chunk.
    buffer = ChoiceBuffer(0)
    buffer.append_tool_call(_delta("call_1", "get_weather", '{"city":"A"}', 0))
    buffer.append_tool_call(_delta("call_2", "get_weather", '{"city":"B"}', 0))

    assert _calls(buffer) == [
        ("call_1", "get_weather", '{"city":"A"}'),
        ("call_2", "get_weather", '{"city":"B"}'),
    ]


def test_interleaved_repeated_ids_without_index_stay_two_calls():
    buffer = ChoiceBuffer(0)
    buffer.append_tool_call(_delta("call_1", "f", "{"))
    buffer.append_tool_call(_delta("call_2", "g", "{"))
    buffer.append_tool_call(_delta("call_1", None, "}"))
    buffer.append_tool_call(_delta("call_2", None, "}"))

    assert _calls(buffer) == [("call_1", "f", "{}"), ("call_2", "g", "{}")]


@pytest.mark.parametrize("index", [0, None])
def test_id_arriving_late_fills_the_open_call(index):
    buffer = ChoiceBuffer(0)
    buffer.append_tool_call(_delta(None, "f", "{", index))
    buffer.append_tool_call(_delta("call_1", None, "}", index))

    assert _calls(buffer) == [("call_1", "f", "{}")]


def test_continuation_follows_a_call_moved_off_its_repeated_index():
    buffer = ChoiceBuffer(0)
    buffer.append_tool_call(_delta("call_1", "f", "{}", 0))
    buffer.append_tool_call(_delta("call_2", "g", "{", 0))
    buffer.append_tool_call(_delta(None, None, "}", 0))

    assert _calls(buffer) == [("call_1", "f", "{}"), ("call_2", "g", "{}")]


def test_bare_continuation_follows_the_last_continued_call():
    buffer = ChoiceBuffer(0)
    buffer.append_tool_call(_delta("call_1", "f", "{"))
    buffer.append_tool_call(_delta("call_2", "g", "{}"))
    buffer.append_tool_call(_delta("call_1", None, '"a":'))
    buffer.append_tool_call(_delta(None, None, "1}"))

    assert _calls(buffer) == [
        ("call_1", "f", '{"a":1}'),
        ("call_2", "g", "{}"),
    ]
