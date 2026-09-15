# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest
from crewai.tools import BaseTool

from opentelemetry.instrumentation.utils import suppress_instrumentation
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.trace.status import StatusCode


class AddTool(BaseTool):
    name: str = "add"
    description: str = "Add two integers"

    def _run(self, left: int, right: int = 1) -> int:
        return left + right


class FailingTool(BaseTool):
    name: str = "fail"
    description: str = "Always fail"

    def _run(self) -> None:
        raise LookupError("tool failed")


def test_direct_tool_span(
    instrument_crewai,
    span_exporter,
) -> None:
    assert AddTool().run(2, right=3) == 5

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "execute_tool add"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == "execute_tool"
    )
    assert span.attributes[GenAIAttributes.GEN_AI_TOOL_NAME] == "add"
    assert span.attributes[GenAIAttributes.GEN_AI_TOOL_TYPE] == "function"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_TOOL_DESCRIPTION]
        == "Add two integers"
    )
    assert GenAIAttributes.GEN_AI_AGENT_NAME not in span.attributes
    assert GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS not in span.attributes


def test_tool_content_capture(
    instrument_crewai_with_content,
    span_exporter,
) -> None:
    assert AddTool().run(2, right=4) == 6

    attributes = span_exporter.get_finished_spans()[0].attributes
    assert json.loads(
        attributes[GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS]
    ) == {"left": 2, "right": 4}
    assert attributes[GenAIAttributes.GEN_AI_TOOL_CALL_RESULT] == 6


def test_structured_tool_span_is_not_duplicated(
    instrument_crewai_with_content,
    span_exporter,
) -> None:
    structured_tool = AddTool().to_structured_tool()
    assert structured_tool.invoke({"left": 4, "right": 5}) == 9

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "execute_tool add"
    assert json.loads(
        spans[0].attributes[GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS]
    ) == {"left": 4, "right": 5}


def test_tool_error_is_reraised_and_recorded(
    instrument_crewai,
    span_exporter,
) -> None:
    with pytest.raises(LookupError, match="tool failed"):
        FailingTool().run()

    span = span_exporter.get_finished_spans()[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["error.type"] == "LookupError"


@pytest.mark.parametrize("structured", [False, True])
def test_tool_instrumentation_suppression(
    instrument_crewai,
    span_exporter,
    structured: bool,
) -> None:
    tool = AddTool()
    with suppress_instrumentation():
        result = (
            tool.to_structured_tool().invoke({"left": 2, "right": 4})
            if structured
            else tool.run(2, right=4)
        )

    assert result == 6
    assert span_exporter.get_finished_spans() == ()
