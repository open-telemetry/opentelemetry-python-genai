# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("genai_conformance")

from opentelemetry.conformance import ConformanceSession, scenarios
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.common_conformance import (
    assert_message_part,
    messages,
)

pytest_plugins = ["opentelemetry.test_util_genai.common_conformance"]

DIRECTORY = Path(__file__).parent / "conformance"


@pytest.mark.parametrize("scenario_name", scenarios(DIRECTORY))
def test_scenario(
    scenario_name: str, common_session: ConformanceSession
) -> None:
    result = common_session.run(scenario_name)
    assert not result.failures, "\n".join(result.failures)
    assert not result.violations, "\n".join(result.violations)
    if scenario_name == "server_tool_calling":
        assert result.report is not None
        _validate_server_tool_calling(result.report)
    if scenario_name == "multimodal":
        input_messages = list(messages(result, "gen_ai.input.messages"))
        assert len(input_messages) == 1
        assert input_messages[0]["role"] == "user"
        parts = input_messages[0]["parts"]
        assert len(parts) == 10
        assert parts[0] == {
            "type": "text",
            "content": "Describe these images.",
        }
        assert parts[1] == {
            "type": "blob",
            "mime_type": "image/png",
            "modality": "image",
            "content": "QUJD",
        }
        assert parts[2] == {
            "type": "uri",
            "mime_type": None,
            "modality": "image",
            "uri": "https://example.com/image.png",
        }
        assert parts[3] == {
            "type": "blob",
            "mime_type": "application/pdf",
            "modality": "document",
            "content": "QUJD",
        }
        assert parts[4] == {
            "type": "uri",
            "mime_type": None,
            "modality": "document",
            "uri": "https://example.com/document.pdf",
        }
        assert parts[5] == {
            "type": "blob",
            "mime_type": "text/plain",
            "modality": "document",
            "content": "RG9jdW1lbnQgdGV4dA==",
        }
        assert parts[6] == {"type": "text", "content": "Nested text"}
        assert parts[7] == {
            "type": "uri",
            "mime_type": None,
            "modality": "image",
            "uri": "https://example.com/nested.png",
        }
        assert parts[8] == {
            "type": "file",
            "mime_type": None,
            "modality": "image",
            "file_id": "file-image",
        }
        assert parts[9] == {
            "type": "file",
            "mime_type": None,
            "modality": "document",
            "file_id": "file-document",
        }
    if scenario_name == "tool_calling":
        assert_message_part(result, "gen_ai.output.messages", "tool_call")
        assert_message_part(
            result, "gen_ai.input.messages", "tool_call_response"
        )


def _validate_server_tool_calling(report: LiveCheckReport) -> None:
    output_messages = [
        json.loads(attribute["value"])
        for entry in report["samples"]
        if "span" in entry
        for attribute in entry["span"]["attributes"]
        if attribute["name"] == "gen_ai.output.messages"
    ]
    assert len(output_messages) == 1
    assert output_messages[0][0]["parts"][:2] == [
        {
            "name": "web_search",
            "server_tool_call": {
                "input": {"query": "OpenTelemetry"},
                "type": "web_search",
            },
            "id": "srvtoolu_01",
            "type": "server_tool_call",
        },
        {
            "server_tool_call_response": {
                "content": [
                    {
                        "type": "web_search_result",
                        "title": "OpenTelemetry",
                        "url": "https://opentelemetry.io/",
                        "encrypted_content": "mock-encrypted-content",
                    }
                ],
                "type": "web_search",
            },
            "id": "srvtoolu_01",
            "type": "server_tool_call_response",
        },
    ]
