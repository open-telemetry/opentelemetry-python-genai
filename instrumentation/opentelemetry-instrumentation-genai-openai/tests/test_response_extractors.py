# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest import mock

import pytest

from opentelemetry.instrumentation.genai.openai import response_extractors
from opentelemetry.instrumentation.genai.openai.utils import get_served_model
from opentelemetry.semconv._incubating.attributes import (
    openai_attributes as OpenAIAttributes,
)
from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    GenericToolDefinition,
    LLMInvocation,
)

try:
    # Responses types are not available in the oldest supported OpenAI SDK.
    # pylint: disable-next=no-name-in-module
    from openai.types.responses.response import Response

    HAS_RESPONSES_TYPES = True
except ImportError:
    Response = None
    HAS_RESPONSES_TYPES = False

pytestmark = pytest.mark.skipif(
    not HAS_RESPONSES_TYPES,
    reason="Responses SDK types require a newer openai SDK",
)


@pytest.fixture(scope="module", name="loaded_module")
def _loaded_module_fixture():
    return response_extractors


def _make_response(output=None, **overrides):
    payload = {
        "id": "resp_123",
        "created_at": 0.0,
        "model": "gpt-4.1",
        "object": "response",
        "output": output or [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "temperature": 1.0,
        "top_p": 1.0,
        "usage": {
            "input_tokens": 11,
            "input_tokens_details": {
                "cached_tokens": 0,
                "cache_write_tokens": 0,
            },
            "output_tokens": 7,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 18,
        },
    }
    payload.update(overrides)
    return Response.model_validate(payload)


class _RawResponse:
    def __init__(self, parsed_response):
        self.headers = {"x-ms-served-model": "served-gpt-4.1"}
        self.parsed_response = parsed_response
        self.parse_count = 0

    def parse(self):
        self.parse_count += 1
        return self.parsed_response


class _RawResponseWithHeaders(_RawResponse):
    def __init__(self, parsed_response, headers):
        super().__init__(parsed_response)
        self.headers = headers


def test_extract_system_instruction_returns_text_for_string(loaded_module):
    params = loaded_module.extract_params(instructions="Be concise")
    instructions = loaded_module.get_system_instruction(params.instructions)

    assert [part.content for part in instructions] == ["Be concise"]


def test_extract_input_messages_supports_string_and_mixed_message_content(
    loaded_module,
):
    from_string = loaded_module.get_input_messages("Hello")
    from_list = loaded_module.get_input_messages(
        [
            {"role": "user", "content": "First"},
            SimpleNamespace(
                role="assistant",
                content=[
                    {"text": "Second"},
                    SimpleNamespace(text="Third"),
                    {"type": "input_image", "image_url": "ignored"},
                ],
            ),
            {"role": None, "content": "ignored"},
        ]
    )

    assert [
        (msg.role, [part.content for part in msg.parts]) for msg in from_string
    ] == [("user", ["Hello"])]
    assert [
        (msg.role, [part.content for part in msg.parts]) for msg in from_list
    ] == [
        ("user", ["First"]),
        ("assistant", ["Second", "Third"]),
    ]


def test_extract_output_messages_maps_parts_and_finish_reasons(loaded_module):
    response = _make_response(
        output=[
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": "Done", "annotations": []},
                    {"type": "refusal", "refusal": "Cannot comply"},
                ],
            },
            {
                "id": "msg_2",
                "type": "message",
                "role": "assistant",
                "status": "incomplete",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Partial",
                        "annotations": [],
                    }
                ],
            },
            {
                "id": "msg_3",
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Pending",
                        "annotations": [],
                    }
                ],
            },
            {
                "id": "fc_1",
                "type": "function_call",
                "status": "completed",
                "name": "get_weather",
                "call_id": "call_123",
                "arguments": '{"city":"SF"}',
            },
            {
                "id": "rs_1",
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": "Thought step"}],
                "content": None,
            },
        ]
    )

    messages = loaded_module.get_output_messages_from_response(response)

    assert [(msg.role, msg.finish_reason) for msg in messages] == [
        ("assistant", "stop"),
        ("assistant", "incomplete"),
        ("assistant", "tool_call"),
        ("assistant", "stop"),
    ]
    assert [part.content for part in messages[0].parts] == [
        "Done",
        "Cannot comply",
    ]
    assert [part.content for part in messages[1].parts] == ["Partial"]
    assert messages[2].parts[0].type == "tool_call"
    assert messages[2].parts[0].name == "get_weather"
    assert messages[2].parts[0].arguments == {"city": "SF"}
    assert messages[3].parts[0].type == "reasoning"
    assert messages[3].parts[0].content == "Thought step"


def test_extract_finish_reasons_maps_terminal_message_and_tool_items(
    loaded_module,
):
    response = _make_response(
        output=[
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [],
            },
            {
                "id": "msg_2",
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            },
            {
                "id": "fc_1",
                "type": "function_call",
                "status": "incomplete",
                "name": "tool",
                "call_id": "call_1",
                "arguments": "{}",
            },
        ]
    )

    assert loaded_module.extract_finish_reasons(response) == [
        "stop",
        "tool_calls",
    ]


def test_get_response_error_returns_error_for_failed_response(loaded_module):
    response = _make_response(
        status="failed",
        error={"code": "server_error", "message": "boom"},
    )

    error = loaded_module.get_response_error(response)

    assert error is not None
    assert error.type == "server_error"
    assert error.message == "boom"


def test_get_response_error_parses_raw_response(loaded_module):
    response = _make_response(
        status="failed",
        error={"code": "server_error", "message": "boom"},
    )
    raw_response = SimpleNamespace(parse=mock.Mock(return_value=response))

    error = loaded_module.get_response_error(raw_response)

    assert error is not None
    assert error.type == "server_error"
    raw_response.parse.assert_called_once_with()


def test_get_response_error_keeps_streaming_raw_response_lazy(loaded_module):
    raw_response = SimpleNamespace(parse=mock.Mock())

    error = loaded_module.get_response_error(
        raw_response,
        {
            "extra_headers": {
                "X-Stainless-Raw-Response": "stream",
            }
        },
    )

    assert error is None
    raw_response.parse.assert_not_called()


def test_get_response_error_none_for_incomplete_response(loaded_module):
    # Incomplete is a finish reason, not an error.
    response = _make_response(
        status="incomplete",
        incomplete_details={"reason": "max_output_tokens"},
    )

    assert loaded_module.get_response_error(response) is None


def test_get_response_error_none_for_successful_response(loaded_module):
    assert loaded_module.get_response_error(_make_response()) is None


def test_extract_output_type_handles_text_format_mapping(loaded_module):
    assert (
        loaded_module.extract_params(
            text={"format": {"type": "json_schema"}}
        ).output_type
        == "json"
    )
    assert (
        loaded_module.extract_params(
            text={"format": {"type": "text"}}
        ).output_type
        == "text"
    )
    assert (
        loaded_module.extract_params(
            text=SimpleNamespace(format=SimpleNamespace(type="json_schema"))
        ).output_type
        == "json"
    )
    assert (
        loaded_module.extract_params(
            text=SimpleNamespace(format=SimpleNamespace(type="text"))
        ).output_type
        == "text"
    )
    assert (
        loaded_module.extract_params(text={"format": "plain"}).output_type
        is None
    )
    assert loaded_module.extract_params(text="plain").output_type is None


def test_extractors_handle_missing_genai_types_import(loaded_module):
    with (
        mock.patch.object(loaded_module, "TextPart", None),
        mock.patch.object(loaded_module, "InputMessage", None),
        mock.patch.object(loaded_module, "OutputMessage", None),
    ):
        assert loaded_module.get_system_instruction("hi") == []
        assert loaded_module.get_input_messages("hi") == []
        assert (
            loaded_module.get_output_messages_from_response(
                _make_response(
                    output=[
                        {
                            "id": "msg_1",
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [],
                        }
                    ]
                )
            )
            == []
        )


def test_set_invocation_response_attributes_populates_usage_and_metadata(
    loaded_module,
):
    invocation = LLMInvocation(request_model="gpt-4o-mini")
    result = _make_response(
        service_tier="scale",
        usage={
            "input_tokens": 11,
            "input_tokens_details": {
                "cached_tokens": 3,
                "cache_creation_input_tokens": 5,
                "cache_write_tokens": 0,
            },
            "output_tokens": 7,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 18,
        },
    )

    loaded_module.set_invocation_response_attributes(
        invocation, result, capture_content=False
    )

    assert invocation.response_model_name == "gpt-4.1"
    assert invocation.response_id == "resp_123"
    assert invocation.input_tokens == 11
    assert invocation.output_tokens == 7
    assert invocation.cache_read_input_tokens == 3
    assert invocation.cache_creation_input_tokens == 5
    assert invocation.attributes == {
        OpenAIAttributes.OPENAI_RESPONSE_SERVICE_TIER: "scale",
    }


def test_set_invocation_response_attributes_prefers_raw_served_model_header(
    loaded_module,
):
    invocation = LLMInvocation(request_model="gpt-4o-mini")
    raw_response = _RawResponse(_make_response(model="body-gpt-4.1"))

    loaded_module.set_invocation_response_attributes(
        invocation, raw_response, capture_content=False
    )

    assert raw_response.parse_count == 1
    assert invocation.response_model_name == "served-gpt-4.1"


def test_set_invocation_response_attributes_falls_back_to_body_model_when_header_empty(
    loaded_module,
):
    invocation = LLMInvocation(request_model="gpt-4o-mini")
    raw_response = _RawResponseWithHeaders(
        _make_response(model="deployment-gpt-4.1"),
        headers={"x-ms-served-model": "  "},
    )

    loaded_module.set_invocation_response_attributes(
        invocation, raw_response, capture_content=False
    )

    assert raw_response.parse_count == 1
    assert invocation.response_model_name == "deployment-gpt-4.1"


def test_set_invocation_response_attributes_falls_back_to_body_model_when_header_absent(
    loaded_module,
):
    invocation = LLMInvocation(request_model="gpt-4o-mini")
    raw_response = _RawResponseWithHeaders(
        _make_response(model="deployment-gpt-4.1"),
        headers={"content-type": "application/json"},
    )

    loaded_module.set_invocation_response_attributes(
        invocation, raw_response, capture_content=False
    )

    assert raw_response.parse_count == 1
    assert invocation.response_model_name == "deployment-gpt-4.1"


def test_set_invocation_response_attributes_populates_output_messages(
    loaded_module,
):
    invocation = LLMInvocation(request_model="gpt-4o-mini")
    result = _make_response(
        output=[
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": "Done", "annotations": []}
                ],
            }
        ]
    )

    loaded_module.set_invocation_response_attributes(
        invocation, result, capture_content=True
    )

    assert invocation.finish_reasons == ["stop"]
    assert [
        (message.role, message.finish_reason)
        for message in invocation.output_messages
    ] == [("assistant", "stop")]
    assert [
        [part.content for part in message.parts]
        for message in invocation.output_messages
    ] == [["Done"]]


def test_extractors_ignore_invalid_request_shapes_without_validation(
    loaded_module,
):
    params = loaded_module.extract_params(
        instructions=["not-a-string"], input=42, text={"format": {"type": 42}}
    )
    assert loaded_module.get_system_instruction(params.instructions) == []
    assert loaded_module.get_input_messages(params.input) == []
    assert params.output_type is None


def test_response_extractors_ignore_invalid_shapes_without_validation(
    loaded_module,
):
    invocation = LLMInvocation(request_model="gpt-4o-mini")
    invalid_result = SimpleNamespace(output=42, usage=42)

    assert (
        loaded_module.get_output_messages_from_response(invalid_result) == []
    )
    assert loaded_module.extract_finish_reasons(invalid_result) == []

    loaded_module.set_invocation_response_attributes(
        invocation, invalid_result, capture_content=True
    )

    assert invocation.response_model_name is None
    assert invocation.response_id is None
    assert invocation.input_tokens is None
    assert invocation.output_tokens is None
    assert invocation.finish_reasons is None
    assert not invocation.output_messages
    assert not invocation.attributes


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"status": "completed"}, ["stop"]),
        ({"status": "failed"}, ["error"]),
        ({"status": "cancelled"}, ["error"]),
        (
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
            },
            ["length"],
        ),
        (
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "content_filter"},
            },
            ["content_filter"],
        ),
        ({"status": "incomplete"}, ["incomplete"]),
        # Non-terminal and unknown statuses: generation is not known to have
        # stopped, so there is no finish reason to report. gen_ai.response.status
        # conveys the lifecycle state instead.
        ({"status": "queued"}, []),
        ({"status": "in_progress"}, []),
        ({}, []),
    ],
)
def test_extract_finish_reasons_maps_response_status(
    loaded_module, overrides, expected
):
    response = _make_response(**overrides)

    assert loaded_module.extract_finish_reasons(response) == expected


def test_set_fetch_response_attributes_tolerates_missing_service_tier():
    """`service_tier` is absent from the Response model on older SDKs.

    Deleting the field from the instance makes attribute access raise
    ``AttributeError``, exactly as it does on an SDK whose ``Response`` model
    never declared it (for example openai 1.70).
    """
    response = _make_response(status="completed")
    del response.__dict__["service_tier"]
    with pytest.raises(AttributeError):
        response.service_tier  # pylint: disable=pointless-statement

    invocation = SimpleNamespace(
        response_model_name=None,
        response_status=None,
        finish_reasons=None,
        output_messages=[],
        system_instruction=[],
        attributes={},
    )

    response_extractors.set_fetch_response_attributes(
        invocation, response, capture_content=False
    )

    assert invocation.response_status == "completed"
    assert invocation.finish_reasons == ["stop"]
    assert (
        OpenAIAttributes.OPENAI_RESPONSE_SERVICE_TIER
        not in invocation.attributes
    )


def test_get_tool_definitions_from_response_maps_flat_responses_tools(
    loaded_module,
):
    """Responses API tools are flat, unlike the nested Chat Completions shape."""
    response = _make_response(
        tools=[
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
                "strict": True,
            }
        ]
    )

    definitions = loaded_module.get_tool_definitions_from_response(response)

    (definition,) = definitions
    assert isinstance(definition, FunctionToolDefinition)
    assert definition.type == "function"
    assert definition.name == "get_weather"
    assert definition.description == "Get the weather"
    assert definition.parameters == {
        "type": "object",
        "properties": {"city": {"type": "string"}},
    }


def test_get_tool_definitions_from_response_maps_builtin_tools_by_type(
    loaded_module,
):
    """Built-in tools carry no name, so their type identifies them."""
    response = _make_response(
        tools=[{"type": "web_search_preview"}],
    )

    definitions = loaded_module.get_tool_definitions_from_response(response)

    (definition,) = definitions
    assert isinstance(definition, GenericToolDefinition)
    assert definition.type == "web_search_preview"
    assert definition.name == "web_search_preview"


def test_get_tool_definitions_from_response_returns_none_without_tools(
    loaded_module,
):
    assert loaded_module.get_tool_definitions_from_response(None) is None
    assert (
        loaded_module.get_tool_definitions_from_response(_make_response())
        is None
    )


def test_set_fetch_response_attributes_captures_tool_definitions(
    loaded_module,
):
    """Tool definitions are captured only when content capture is enabled."""
    response = _make_response(
        status="completed",
        tools=[
            {
                "type": "function",
                "name": "get_weather",
                "description": None,
                "parameters": {"type": "object"},
                "strict": True,
            }
        ],
    )

    def _make_invocation():
        return SimpleNamespace(
            response_model_name=None,
            response_status=None,
            finish_reasons=None,
            output_messages=[],
            system_instruction=[],
            tool_definitions=None,
            attributes={},
        )

    captured = _make_invocation()
    loaded_module.set_fetch_response_attributes(
        captured, response, capture_content=True
    )
    (definition,) = captured.tool_definitions
    assert isinstance(definition, FunctionToolDefinition)
    assert definition.name == "get_weather"

    not_captured = _make_invocation()
    loaded_module.set_fetch_response_attributes(
        not_captured, response, capture_content=False
    )
    assert not_captured.tool_definitions is None


def test_set_fetch_response_attributes_prefers_raw_served_model_header(
    loaded_module,
):
    invocation = SimpleNamespace(
        response_model_name=None,
        response_status=None,
        finish_reasons=None,
        output_messages=[],
        system_instruction=[],
        attributes={},
    )
    raw_response = _RawResponse(_make_response(model="body-gpt-4.1"))

    loaded_module.set_fetch_response_attributes(
        invocation, raw_response, capture_content=False
    )

    assert raw_response.parse_count == 1
    assert invocation.response_model_name == "served-gpt-4.1"


def test_get_served_model_returns_value_when_present():
    headers = {"x-ms-served-model": "gpt-4o-2024-08-06"}
    assert get_served_model(headers) == "gpt-4o-2024-08-06"


def test_get_served_model_case_insensitive_name():
    headers = {"X-MS-Served-Model": "gpt-4o-2024-08-06"}
    assert get_served_model(headers) == "gpt-4o-2024-08-06"


def test_get_served_model_empty_value_returns_none():
    # An empty header value must not overwrite a real model name.
    headers = {"x-ms-served-model": " "}
    assert get_served_model(headers) is None


def test_get_served_model_missing_header_returns_none():
    headers = {"content-type": "application/json"}
    assert get_served_model(headers) is None


def test_get_served_model_none_returns_none():
    # Parsed models / streaming chunks carry no HTTP headers.
    assert get_served_model(None) is None


def test_get_served_model_non_mapping_returns_none():
    assert get_served_model(object()) is None


def test_get_served_model_picks_served_model_among_others():
    headers = {
        "content-type": "application/json",
        "x-ms-served-model": "gpt-4o-2024-08-06",
        "x-request-id": "abc123",
    }
    assert get_served_model(headers) == "gpt-4o-2024-08-06"


def test_get_served_model_non_string_name_ignored():
    headers = {123: "not-a-header", "x-ms-served-model": "gpt-4o"}
    assert get_served_model(headers) == "gpt-4o"


def test_get_served_model_empty_string():
    headers = {123: "not-a-header", "x-ms-served-model": "  "}
    assert get_served_model(headers) is None


@pytest.mark.parametrize("value", ["", None])
def test_get_served_model_falsy_values_return_none(value):
    headers = {"x-ms-served-model": value}
    assert get_served_model(headers) is None
