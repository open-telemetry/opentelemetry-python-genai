# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("genai_conformance")

from opentelemetry.conformance import ConformanceSession, scenarios
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.common_conformance import (
    assert_message_part,
)

from .interactions.util import create_request_parameters

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
    if scenario_name == "interactions_server_tool_calling":
        assert result.report is not None
        _validate_interactions_server_tool_calling(result.report)
    if scenario_name == "fetch_response":
        assert result.report is not None
        _validate_fetch_response(result.report)
    if scenario_name == "inference_parameters":
        assert result.report is not None
        _validate_inference_parameters(result.report)
    if scenario_name == "multimodal":
        assert_message_part(result, "gen_ai.input.messages", "blob", "image")
    if scenario_name == "tool_calling":
        assert_message_part(result, "gen_ai.output.messages", "tool_call")
        assert_message_part(
            result, "gen_ai.input.messages", "tool_call_response"
        )


def _output_parts(report: LiveCheckReport) -> list[dict[str, Any]]:
    output_messages = [
        json.loads(attribute["value"])
        for entry in report["samples"]
        if "span" in entry
        for attribute in entry["span"]["attributes"]
        if attribute["name"] == "gen_ai.output.messages"
    ]
    assert len(output_messages) == 1
    return output_messages[0][0]["parts"]


def _attr(span: dict[str, Any], name: str) -> Any:
    for attr in span["attributes"]:
        if attr["name"] == name:
            return attr["value"]
    return None


def _part_types(messages_json: str | None) -> list[str]:
    messages = json.loads(messages_json) if messages_json else []
    return [part["type"] for message in messages for part in message["parts"]]


def _validate_server_tool_calling(report: LiveCheckReport) -> None:
    assert _output_parts(report)[:2] == [
        {
            "name": "code_execution",
            "server_tool_call": {
                "code": "print(1)",
                "language": "PYTHON",
                "type": "code_execution",
            },
            "id": "code-1",
            "type": "server_tool_call",
        },
        {
            "server_tool_call_response": {
                "outcome": "OUTCOME_OK",
                "output": "1",
                "type": "code_execution",
            },
            "id": "code-1",
            "type": "server_tool_call_response",
        },
    ]


def _validate_interactions_server_tool_calling(
    report: LiveCheckReport,
) -> None:
    assert _output_parts(report)[:2] == [
        {
            "name": "google_search",
            "server_tool_call": {
                "arguments": {"queries": ["OpenTelemetry"]},
                "search_type": "web_search",
                "type": "google_search",
            },
            "id": "search-1",
            "type": "server_tool_call",
        },
        {
            "server_tool_call_response": {
                "is_error": False,
                "result": [],
                "type": "google_search",
            },
            "id": "search-1",
            "type": "server_tool_call_response",
        },
    ]


def _validate_fetch_response(report: LiveCheckReport) -> None:
    spans = [
        entry["span"]
        for entry in report["samples"]
        if "span" in entry
        and _attr(entry["span"], "gen_ai.operation.name") == "fetch_response"
    ]
    assert len(spans) == 1, "expected exactly one fetch_response span"
    span = spans[0]
    assert _attr(span, "gen_ai.response.id") == "interaction-fetch-1"
    assert _attr(span, "gen_ai.response.status") == "completed"
    assert _part_types(_attr(span, "gen_ai.output.messages")) == ["text"]
    for name in ("gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens"):
        assert _attr(span, name) is None, f"{name} must not be recorded"
    assert _attr(span, "gen_ai.input.messages") is None


def _validate_inference_parameters(report: LiveCheckReport) -> None:
    response_ids = [
        attr["value"]
        for entry in report["samples"]
        if "span" in entry
        for attr in entry["span"]["attributes"]
        if attr["name"] == "gen_ai.response.id"
    ]
    assert len(response_ids) == 1 and response_ids[0]
    attributes = {
        attr["name"]: attr["value"]
        for entry in report["samples"]
        if "span" in entry
        for attr in entry["span"]["attributes"]
    }
    _, expected_attributes = create_request_parameters()
    for name, expected in expected_attributes.items():
        actual = attributes[name]
        assert actual == expected, name
        assert type(actual) is type(expected), name
