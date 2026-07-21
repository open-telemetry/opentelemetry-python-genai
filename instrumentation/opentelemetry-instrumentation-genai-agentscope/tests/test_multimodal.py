# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Multimodal content-type conversion tests.

Verify the AgentScope block/message converters map onto the correct
``opentelemetry-util-genai`` message parts (``Text``/``Reasoning``/``Uri``/
``Blob``/``ToolCallRequest``).
"""

from __future__ import annotations

import base64

from agentscope.message import ImageBlock, Msg

from opentelemetry.instrumentation.genai.agentscope.utils import (
    _convert_block_to_part,
    _format_msg_to_parts,
    convert_agent_response_to_output_messages,
    convert_agentscope_messages_to_genai_format,
)
from opentelemetry.util.genai.types import Blob, Text, Uri


class TestBlockConversion:
    """Test individual block conversion to part dicts."""

    def test_text_block_to_part(self):
        block = {"type": "text", "text": "Hello world"}
        part = _convert_block_to_part(block)

        assert part is not None
        assert part["type"] == "text"
        assert part["content"] == "Hello world"

    def test_image_url_block_to_uri_part(self):
        block = {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/cat.jpg"},
        }
        part = _convert_block_to_part(block)

        assert part is not None
        assert part["type"] == "uri"
        assert part["uri"] == "https://example.com/cat.jpg"
        assert part["modality"] == "image"
        assert part["mime_type"] == "image/jpeg"

    def test_image_base64_block_to_blob_part(self):
        test_data = base64.b64encode(
            bytes.fromhex(
                "89504e470d0a1a0a0000000d49484452000000010000000108020000009"
                "0774c53000000014944415408d76360f8ff0f00020100018d9c7d000000"
                "0049454e44ae426082"
            )
        ).decode("utf-8")

        block = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": test_data,
            },
        }
        part = _convert_block_to_part(block)

        assert part is not None
        assert part["type"] == "blob"
        assert part["content"] == test_data
        assert part["media_type"] == "image/png"
        assert part["modality"] == "image"

    def test_audio_url_block_to_uri_part(self):
        block = {
            "type": "audio",
            "source": {"type": "url", "url": "https://example.com/sound.mp3"},
        }
        part = _convert_block_to_part(block)

        assert part is not None
        assert part["type"] == "uri"
        assert part["modality"] == "audio"
        assert part["mime_type"] == "audio/mpeg"

    def test_image_url_with_query_params(self):
        block = {
            "type": "image",
            "source": {
                "type": "url",
                "url": "https://oss.example.com/image.png?token=abc&expires=1",
            },
        }
        part = _convert_block_to_part(block)

        assert part is not None
        assert part["type"] == "uri"
        assert part["mime_type"] == "image/png"

    def test_video_base64_block_to_blob_part(self):
        block = {
            "type": "video",
            "source": {
                "type": "base64",
                "media_type": "video/mp4",
                "data": "dGVzdHZpZGVv",
            },
        }
        part = _convert_block_to_part(block)

        assert part is not None
        assert part["type"] == "blob"
        assert part["modality"] == "video"
        assert part["media_type"] == "video/mp4"

    def test_thinking_block_to_reasoning_part(self):
        block = {"type": "thinking", "thinking": "Let me think..."}
        part = _convert_block_to_part(block)

        assert part is not None
        assert part["type"] == "reasoning"
        assert part["content"] == "Let me think..."

    def test_tool_use_block_to_tool_call_part(self):
        block = {
            "type": "tool_use",
            "id": "call_123",
            "name": "search",
            "input": {"query": "test"},
        }
        part = _convert_block_to_part(block)

        assert part is not None
        assert part["type"] == "tool_call"
        assert part["id"] == "call_123"
        assert part["name"] == "search"
        assert part["arguments"] == {"query": "test"}


class TestMsgConversion:
    """Test ``Msg`` to part-dict conversion."""

    def test_simple_text_msg(self):
        msg = Msg(name="user", content="Hello", role="user")
        result = _format_msg_to_parts(msg)

        assert result["role"] == "user"
        assert len(result["parts"]) == 1
        assert result["parts"][0]["type"] == "text"
        assert result["parts"][0]["content"] == "Hello"

    def test_msg_with_image_url(self):
        msg = Msg(
            name="assistant",
            role="assistant",
            content=[
                ImageBlock(
                    type="image",
                    source={
                        "type": "url",
                        "url": "https://example.com/image.jpg",
                    },
                )
            ],
        )
        result = _format_msg_to_parts(msg)

        assert result["role"] == "assistant"
        assert len(result["parts"]) == 1
        assert result["parts"][0]["type"] == "uri"
        assert result["parts"][0]["modality"] == "image"

    def test_msg_with_image_base64(self):
        test_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"

        msg = Msg(
            name="assistant",
            role="assistant",
            content=[
                ImageBlock(
                    type="image",
                    source={
                        "type": "base64",
                        "media_type": "image/png",
                        "data": test_data,
                    },
                )
            ],
        )
        result = _format_msg_to_parts(msg)

        assert result["role"] == "assistant"
        assert len(result["parts"]) == 1
        assert result["parts"][0]["type"] == "blob"
        assert result["parts"][0]["modality"] == "image"
        assert result["parts"][0]["content"] == test_data

    def test_msg_with_mixed_content(self):
        msg = Msg(
            name="assistant",
            role="assistant",
            content=[
                {"type": "text", "text": "Here is the image:"},
                ImageBlock(
                    type="image",
                    source={
                        "type": "url",
                        "url": "https://example.com/cat.jpg",
                    },
                ),
            ],
        )
        result = _format_msg_to_parts(msg)

        assert result["role"] == "assistant"
        assert len(result["parts"]) == 2
        assert result["parts"][0]["type"] == "text"
        assert result["parts"][1]["type"] == "uri"


class TestOutputMessageConversion:
    """Test ``convert_agent_response_to_output_messages``."""

    def test_convert_text_response(self):
        msg = Msg(name="Bot", role="assistant", content="Hello!")
        output_messages = convert_agent_response_to_output_messages(msg)

        assert len(output_messages) == 1
        assert output_messages[0].role == "assistant"
        assert len(output_messages[0].parts) == 1
        assert isinstance(output_messages[0].parts[0], Text)

    def test_convert_image_url_response(self):
        msg = Msg(
            name="Bot",
            role="assistant",
            content=[
                ImageBlock(
                    type="image",
                    source={
                        "type": "url",
                        "url": "https://example.com/image.jpg",
                    },
                )
            ],
        )
        output_messages = convert_agent_response_to_output_messages(msg)

        assert len(output_messages) == 1
        part = output_messages[0].parts[0]
        assert isinstance(part, Uri)
        assert part.modality == "image"

    def test_convert_image_base64_response(self):
        raw = b"fake-image-bytes"
        test_data = base64.b64encode(raw).decode("utf-8")

        msg = Msg(
            name="Bot",
            role="assistant",
            content=[
                ImageBlock(
                    type="image",
                    source={
                        "type": "base64",
                        "media_type": "image/png",
                        "data": test_data,
                    },
                )
            ],
        )
        output_messages = convert_agent_response_to_output_messages(msg)

        assert len(output_messages) == 1
        part = output_messages[0].parts[0]
        assert isinstance(part, Blob)
        assert part.modality == "image"
        assert part.content == raw


class TestInputMessageConversion:
    """Test ``convert_agentscope_messages_to_genai_format``."""

    def test_convert_simple_messages(self):
        messages = [
            Msg(name="user", content="Hello", role="user"),
            Msg(name="assistant", content="Hi there!", role="assistant"),
        ]
        input_messages = convert_agentscope_messages_to_genai_format(messages)

        assert len(input_messages) == 2
        assert input_messages[0].role == "user"
        assert input_messages[1].role == "assistant"

    def test_convert_messages_with_image(self):
        messages = [
            Msg(
                name="assistant",
                role="assistant",
                content=[
                    {"type": "text", "text": "Generated image:"},
                    ImageBlock(
                        type="image",
                        source={
                            "type": "url",
                            "url": "https://example.com/img.png",
                        },
                    ),
                ],
            )
        ]
        input_messages = convert_agentscope_messages_to_genai_format(messages)

        assert len(input_messages) == 1
        assert len(input_messages[0].parts) == 2
        uri_part = input_messages[0].parts[1]
        assert isinstance(uri_part, Uri)
        assert uri_part.type == "uri"

    def test_convert_messages_with_blob(self):
        raw = b"test"
        test_data = base64.b64encode(raw).decode("utf-8")
        messages = [
            {
                "role": "user",
                "parts": [
                    {
                        "type": "blob",
                        "content": test_data,
                        "media_type": "image/png",
                        "modality": "image",
                    }
                ],
            }
        ]

        input_messages = convert_agentscope_messages_to_genai_format(messages)

        assert len(input_messages) == 1
        assert len(input_messages[0].parts) == 1
        blob_part = input_messages[0].parts[0]
        assert isinstance(blob_part, Blob)
        assert blob_part.content == raw
        assert blob_part.mime_type == "image/png"
        assert blob_part.modality == "image"


class TestDefaultMediaType:
    """Test default media-type handling for blob parts."""

    def test_image_blob_default_media_type(self):
        block = {
            "type": "image",
            "source": {"type": "base64", "data": "dGVzdA=="},
        }
        part = _convert_block_to_part(block)

        assert part is not None
        assert part["type"] == "blob"
        assert part.get("media_type") == "image/jpeg"

    def test_audio_blob_default_media_type(self):
        block = {
            "type": "audio",
            "source": {"type": "base64", "data": "dGVzdA=="},
        }
        part = _convert_block_to_part(block)

        assert part is not None
        assert part.get("media_type") == "audio/wav"

    def test_video_blob_default_media_type(self):
        block = {
            "type": "video",
            "source": {"type": "base64", "data": "dGVzdA=="},
        }
        part = _convert_block_to_part(block)

        assert part is not None
        assert part.get("media_type") == "video/mp4"
