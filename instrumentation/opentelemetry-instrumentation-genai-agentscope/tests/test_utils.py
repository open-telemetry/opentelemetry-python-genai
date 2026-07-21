# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for message conversion in the AgentScope utils module."""

from __future__ import annotations

from agentscope.message import Msg, ToolResultBlock

from opentelemetry.instrumentation.genai.agentscope.utils import (
    _convert_block_to_part,
    convert_agentscope_messages_to_genai_format,
)
from opentelemetry.util.genai.types import ToolCallResponse


class TestToolResultConversion:
    def test_convert_msg_with_tool_result(self):
        """A ``ToolResultBlock`` maps to a ``ToolCallResponse`` part."""
        tool_result_block = ToolResultBlock(
            type="tool_result",
            id="call_test_123",
            name="test_tool",
            output="Tool execution success",
        )
        # AgentScope enforces role in {'user', 'assistant', 'system'}.
        msg = Msg(name="tool", role="user", content=[tool_result_block])

        converted_messages = convert_agentscope_messages_to_genai_format([msg])

        assert len(converted_messages) == 1
        assert converted_messages[0].role == "user"
        assert len(converted_messages[0].parts) == 1

        part = converted_messages[0].parts[0]
        assert isinstance(part, ToolCallResponse)
        assert part.id == "call_test_123"
        assert part.response == "Tool execution success"

    def test_convert_local_result_key(self):
        """The local converter emits a ``result`` key that still converts."""
        block = {
            "type": "tool_result",
            "id": "id_local",
            "name": "tool_local",
            "output": "local output",
        }
        part = _convert_block_to_part(block)

        assert "result" in part
        assert part["result"] == "local output"

        converted = convert_agentscope_messages_to_genai_format(
            {"role": "tool", "parts": [part]}
        )
        assert len(converted) == 1
        part_obj = converted[0].parts[0]
        assert isinstance(part_obj, ToolCallResponse)
        assert part_obj.response == "local output"

    def test_convert_framework_response_key(self):
        """A part dict using ``response`` (framework converter) also converts."""
        block = {
            "type": "tool_call_response",
            "id": "id_framework",
            "response": "framework output",
        }
        converted = convert_agentscope_messages_to_genai_format(
            {"role": "tool", "parts": [block]}
        )

        assert len(converted) == 1
        part_obj = converted[0].parts[0]
        assert isinstance(part_obj, ToolCallResponse)
        assert part_obj.response == "framework output"
