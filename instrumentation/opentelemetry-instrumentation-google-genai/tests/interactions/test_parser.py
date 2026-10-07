# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest
import unittest.mock

from opentelemetry.instrumentation.google_genai.interactions import (
    _HAS_INTERACTIONS,
    Interaction,
    _interactions_input_to_messages,
    _interactions_response_to_messages,
)
from opentelemetry.util.genai.types import (
    GenericPart,
    ServerToolCallPart,
    ServerToolCallResponsePart,
    TextPart,
    ToolCallRequestPart,
    ToolCallResponsePart,
    UriPart,
)


@unittest.skipIf(
    not _HAS_INTERACTIONS,
    "Interactions are not supported in this version of google-genai",
)
class TestInteractionsParser(unittest.TestCase):
    def test_input_to_messages_none(self) -> None:
        self.assertEqual(_interactions_input_to_messages(None), [])

    def test_input_to_messages_str(self) -> None:
        messages = _interactions_input_to_messages("Hello world")
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(len(messages[0].parts), 1)
        self.assertIsInstance(messages[0].parts[0], TextPart)
        self.assertEqual(messages[0].parts[0].content, "Hello world")

    def test_input_to_messages_list_of_strings(self) -> None:
        messages = _interactions_input_to_messages(["Hello", "world"])
        self.assertEqual(len(messages[0].parts), 2)
        self.assertIsInstance(messages[0].parts[0], TextPart)
        self.assertEqual(messages[0].parts[0].content, "Hello")
        self.assertIsInstance(messages[0].parts[1], TextPart)
        self.assertEqual(messages[0].parts[1].content, "world")

    def test_input_to_messages_text_step(self) -> None:
        steps = [{"type": "text", "text": "Hello text step"}]
        messages = _interactions_input_to_messages(steps)
        self.assertEqual(len(messages[0].parts), 1)
        self.assertIsInstance(messages[0].parts[0], TextPart)
        self.assertEqual(messages[0].parts[0].content, "Hello text step")

    def test_input_to_messages_document_step(self) -> None:
        steps = [
            {
                "type": "document",
                "mime_type": "application/pdf",
                "uri": "https://example.com/doc.pdf",
            }
        ]
        messages = _interactions_input_to_messages(steps)
        self.assertEqual(len(messages[0].parts), 1)
        self.assertIsInstance(messages[0].parts[0], UriPart)
        self.assertEqual(messages[0].parts[0].mime_type, "application/pdf")
        self.assertEqual(messages[0].parts[0].modality, "document")
        self.assertEqual(
            messages[0].parts[0].uri, "https://example.com/doc.pdf"
        )

    def test_input_to_messages_tool_call_step(self) -> None:
        steps = [
            {
                "type": "function_call",
                "id": "call-123",
                "name": "calc",
                "arguments": {"x": 5},
            }
        ]
        messages = _interactions_input_to_messages(steps)
        self.assertEqual(len(messages[0].parts), 1)
        self.assertIsInstance(messages[0].parts[0], ToolCallRequestPart)
        self.assertEqual(messages[0].parts[0].id, "call-123")
        self.assertEqual(messages[0].parts[0].name, "calc")
        self.assertEqual(messages[0].parts[0].arguments, {"x": 5})

    def test_input_to_messages_tool_result_step(self) -> None:
        steps = [
            {
                "type": "function_result",
                "call_id": "call-123",
                "result": {"val": 10},
            }
        ]
        messages = _interactions_input_to_messages(steps)
        self.assertEqual(len(messages[0].parts), 1)
        self.assertIsInstance(messages[0].parts[0], ToolCallResponsePart)
        self.assertEqual(messages[0].parts[0].id, "call-123")
        self.assertEqual(messages[0].parts[0].response, {"val": 10})

    def test_input_to_messages_server_tool_steps(self) -> None:
        steps = [
            {
                "type": "google_search_call",
                "id": "search-1",
                "arguments": {"query": "OpenTelemetry"},
            },
            {
                "type": "google_search_result",
                "call_id": "search-1",
                "result": [],
            },
        ]

        parts = _interactions_input_to_messages(steps)[0].parts

        self.assertIsInstance(parts[0], ServerToolCallPart)
        self.assertEqual(parts[0].id, "search-1")
        self.assertEqual(parts[0].name, "google_search")
        self.assertEqual(
            parts[0].server_tool_call,
            {
                "arguments": {"query": "OpenTelemetry"},
                "type": "google_search",
            },
        )
        self.assertIsInstance(parts[1], ServerToolCallResponsePart)
        self.assertEqual(parts[1].id, "search-1")
        self.assertEqual(
            parts[1].server_tool_call_response,
            {"result": [], "type": "google_search"},
        )

    def test_input_to_messages_generic_fallback(self) -> None:
        steps = [{"type": "some_unsupported_type"}]
        messages = _interactions_input_to_messages(steps)
        self.assertEqual(len(messages[0].parts), 1)
        self.assertIsInstance(messages[0].parts[0], GenericPart)
        self.assertEqual(messages[0].parts[0].type, "some_unsupported_type")

    def test_input_to_messages_single_non_sequence_step(self) -> None:
        step = {"type": "text", "text": "Hello single step"}
        messages = _interactions_input_to_messages(step)
        self.assertEqual(len(messages[0].parts), 1)
        self.assertIsInstance(messages[0].parts[0], TextPart)
        self.assertEqual(messages[0].parts[0].content, "Hello single step")

    def test_input_to_messages_none_type_fall_through(self) -> None:
        step = {"other_field": "no type specified"}
        messages = _interactions_input_to_messages(step)
        self.assertEqual(len(messages[0].parts), 0)

    def test_response_to_messages(self) -> None:
        mock_interaction = unittest.mock.MagicMock()
        mock_interaction.output_text = "Model response text"

        messages = _interactions_response_to_messages(mock_interaction)

        self.assertEqual(messages[0].role, "assistant")
        self.assertEqual(messages[0].finish_reason, "stop")
        self.assertEqual(len(messages[0].parts), 1)
        self.assertIsInstance(messages[0].parts[0], TextPart)
        self.assertEqual(messages[0].parts[0].content, "Model response text")

    def test_response_to_messages_includes_tool_steps(self) -> None:
        interaction = Interaction.model_validate(
            {
                "id": "interaction-1",
                "created": "2026-06-24T18:51:26Z",
                "updated": "2026-06-24T18:51:26Z",
                "status": "completed",
                "output_text": "Search complete",
                "steps": [
                    {
                        "type": "mcp_server_tool_call",
                        "id": "mcp-1",
                        "name": "search",
                        "server_name": "docs",
                        "arguments": {"query": "OpenTelemetry"},
                    },
                    {
                        "type": "mcp_server_tool_result",
                        "call_id": "mcp-1",
                        "name": "search",
                        "server_name": "docs",
                        "result": {"items": []},
                    },
                    {
                        "type": "model_output",
                        "content": [
                            {"type": "text", "text": "Search complete"}
                        ],
                    },
                ],
            }
        )

        parts = _interactions_response_to_messages(interaction)[0].parts

        self.assertIsInstance(parts[0], ServerToolCallPart)
        self.assertEqual(parts[0].id, "mcp-1")
        self.assertEqual(parts[0].name, "search")
        self.assertEqual(
            parts[0].server_tool_call,
            {
                "server_name": "docs",
                "arguments": {"query": "OpenTelemetry"},
                "type": "mcp",
            },
        )
        self.assertIsInstance(parts[1], ServerToolCallResponsePart)
        self.assertEqual(parts[1].id, "mcp-1")
        self.assertEqual(
            parts[1].server_tool_call_response,
            {
                "server_name": "docs",
                "result": {"items": []},
                "type": "mcp",
            },
        )
        self.assertEqual(parts[2], TextPart(content="Search complete"))
