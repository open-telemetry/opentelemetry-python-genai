# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest
from google.genai import Client, types

from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.test_util_genai.instrumentor import instrument

_JSON_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string", "enum": ["Paris", "London"]}},
    "required": ["city"],
    "additionalProperties": False,
}
_PARAMETERS = {
    "type": "OBJECT",
    "properties": {"city": {"type": "STRING"}},
    "required": ["city"],
}


@pytest.mark.vcr
@pytest.mark.parametrize(
    "parameter_fields,expected_parameters",
    [
        pytest.param(
            {"parameters_json_schema": _JSON_SCHEMA},
            _JSON_SCHEMA,
            id="json-schema",
        ),
        pytest.param(
            {"parameters_json_schema": {}}, {}, id="empty-json-schema"
        ),
        pytest.param(
            {
                "parameters": _PARAMETERS,
                "parameters_json_schema": _JSON_SCHEMA,
            },
            _PARAMETERS,
            id="parameters-wins",
        ),
    ],
)
def test_tool_definition_parameters(
    tracer_provider: TracerProvider,
    meter_provider: MeterProvider,
    logger_provider: LoggerProvider,
    span_exporter: InMemorySpanExporter,
    parameter_fields: dict[str, object],
    expected_parameters: dict[str, object],
) -> None:
    declaration = types.FunctionDeclaration(
        name="get_weather",
        description="Get the weather for a city.",
        **parameter_fields,
    )
    config = types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=[declaration])]
    )
    with instrument(
        GoogleGenAiSdkInstrumentor(),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
        content_capture="SPAN_ONLY",
    ):
        client = Client(
            api_key="test-key",
            vertexai=False,
        )
        result = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="What is the weather in Paris?",
            config=config,
        )

    assert result.text == "Sunny."
    (span,) = span_exporter.get_finished_spans()
    assert span.attributes is not None
    assert (
        span.attributes[gen_ai_attributes.GEN_AI_RESPONSE_MODEL]
        == "gemini-2.5-flash"
    )
    definitions = span.attributes[gen_ai_attributes.GEN_AI_TOOL_DEFINITIONS]
    assert isinstance(definitions, str)
    assert json.loads(definitions) == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": expected_parameters,
        }
    ]
