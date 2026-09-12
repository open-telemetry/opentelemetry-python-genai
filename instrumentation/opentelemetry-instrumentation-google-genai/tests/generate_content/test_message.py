# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import unittest

from google.genai import types as genai_types

from opentelemetry.instrumentation.google_genai.message import (
    _to_part,
    to_system_instructions,
)
from opentelemetry.util.genai.types import (
    GenericPart,
    ServerToolCallPart,
    ServerToolCallResponsePart,
    TextPart,
)


class TestGoogleGenAiMessage(unittest.TestCase):
    def test_to_system_instructions_text(self):
        content = genai_types.Content(
            parts=[genai_types.Part.from_text(text="Be concise")]
        )
        instructions = to_system_instructions(content=content)
        self.assertEqual(instructions, [TextPart(content="Be concise")])

    def test_to_system_instructions_non_text_parts(self):
        content = genai_types.Content(
            parts=[
                genai_types.Part.from_text(text="Be concise"),
                genai_types.Part.from_bytes(
                    data=b"image_bytes", mime_type="image/png"
                ),
                genai_types.Part.from_uri(
                    file_uri="gs://bucket/doc.pdf",
                    mime_type="application/pdf",
                ),
            ]
        )
        instructions = to_system_instructions(content=content)
        self.assertEqual(
            instructions,
            [
                TextPart(content="Be concise"),
                GenericPart(type="blob"),
                GenericPart(type="uri"),
            ],
        )

    def test_to_system_instructions_empty_or_unknown_part(self):
        content = genai_types.Content(parts=[genai_types.Part()])
        instructions = to_system_instructions(content=content)
        self.assertEqual(instructions, [])

    def test_to_part_maps_code_execution(self):
        call_id = (
            {"id": "code-1"}
            if "id" in genai_types.ExecutableCode.model_fields
            else {}
        )
        response_id = (
            {"id": "code-1"}
            if "id" in genai_types.CodeExecutionResult.model_fields
            else {}
        )
        call = _to_part(
            genai_types.Part(
                executable_code=genai_types.ExecutableCode(
                    code="print(1)",
                    language=genai_types.Language.PYTHON,
                    **call_id,
                )
            ),
            0,
        )
        response = _to_part(
            genai_types.Part(
                code_execution_result=genai_types.CodeExecutionResult(
                    outcome=genai_types.Outcome.OUTCOME_OK,
                    output="1",
                    **response_id,
                )
            ),
            1,
        )

        self.assertIsInstance(call, ServerToolCallPart)
        self.assertEqual(call.id, call_id.get("id"))
        self.assertEqual(call.name, "code_execution")
        self.assertEqual(
            call.server_tool_call,
            {
                "type": "code_execution",
                "code": "print(1)",
                "language": "PYTHON",
            },
        )
        self.assertIsInstance(response, ServerToolCallResponsePart)
        self.assertEqual(response.id, response_id.get("id"))
        self.assertEqual(
            response.server_tool_call_response,
            {
                "type": "code_execution",
                "outcome": "OUTCOME_OK",
                "output": "1",
            },
        )

    @unittest.skipUnless(
        "tool_call" in genai_types.Part.model_fields,
        "server tool parts are unavailable in this SDK",
    )
    def test_to_part_maps_server_tool_call_and_response(self):
        call = _to_part(
            genai_types.Part(
                tool_call=genai_types.ToolCall(
                    id="tool-1",
                    tool_type=genai_types.ToolType.GOOGLE_SEARCH_WEB,
                    args={"query": "OpenTelemetry"},
                )
            ),
            0,
        )
        response = _to_part(
            genai_types.Part(
                tool_response=genai_types.ToolResponse(
                    id="tool-1",
                    tool_type=genai_types.ToolType.GOOGLE_SEARCH_WEB,
                    response={"results": []},
                )
            ),
            1,
        )

        self.assertIsInstance(call, ServerToolCallPart)
        self.assertEqual(call.id, "tool-1")
        self.assertEqual(call.name, "google_search_web")
        self.assertEqual(
            call.server_tool_call,
            {
                "type": "google_search_web",
                "arguments": {"query": "OpenTelemetry"},
            },
        )
        self.assertIsInstance(response, ServerToolCallResponsePart)
        self.assertEqual(response.id, "tool-1")
        self.assertEqual(
            response.server_tool_call_response,
            {
                "type": "google_search_web",
                "response": {"results": []},
            },
        )
