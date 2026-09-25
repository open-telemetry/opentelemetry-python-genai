# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import functools
import json
import unittest
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from unittest.mock import patch

import pytest
from google.genai import types as genai_types

from opentelemetry._logs import get_logger_provider
from opentelemetry.instrumentation.google_genai import tool_call_wrapper
from opentelemetry.metrics import get_meter_provider
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import error_attributes as Error
from opentelemetry.trace import StatusCode, get_tracer_provider
from opentelemetry.util.genai.handler import TelemetryHandler

from ..common import otel_mocker


class TestCase(unittest.TestCase):
    def setUp(self):
        self._otel = otel_mocker.OTelMocker()
        self._otel.install()

    @property
    def otel(self):
        return self._otel

    @property
    def otel_wrapper(self):
        return TelemetryHandler(
            tracer_provider=get_tracer_provider(),
            logger_provider=get_logger_provider(),
            meter_provider=get_meter_provider(),
        )

    def wrap(self, tool_or_tools):
        return tool_call_wrapper.wrapped_tool(tool_or_tools, self.otel_wrapper)

    def test_wraps_none(self):
        result = self.wrap(None)
        self.assertIsNone(result)

    def test_wraps_multiple_tool_functions_as_list(self):
        def somefunction():
            pass

        def otherfunction():
            pass

        wrapped_functions = self.wrap([somefunction, otherfunction])
        wrapped_somefunction = wrapped_functions[0]
        wrapped_otherfunction = wrapped_functions[1]
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        self.otel.assert_does_not_have_span_named("execute_tool otherfunction")
        somefunction()
        otherfunction()
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        self.otel.assert_does_not_have_span_named("execute_tool otherfunction")
        wrapped_somefunction()
        self.otel.assert_has_span_named("execute_tool somefunction")
        self.otel.assert_does_not_have_span_named("execute_tool otherfunction")
        wrapped_otherfunction()
        self.otel.assert_has_span_named("execute_tool otherfunction")

    def test_wraps_multiple_tool_functions_as_dict(self):
        def somefunction():
            pass

        def otherfunction():
            pass

        wrapped_functions = self.wrap(
            {"somefunction": somefunction, "otherfunction": otherfunction}
        )
        wrapped_somefunction = wrapped_functions["somefunction"]
        wrapped_otherfunction = wrapped_functions["otherfunction"]
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        self.otel.assert_does_not_have_span_named("execute_tool otherfunction")
        somefunction()
        otherfunction()
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        self.otel.assert_does_not_have_span_named("execute_tool otherfunction")
        wrapped_somefunction()
        self.otel.assert_has_span_named("execute_tool somefunction")
        self.otel.assert_does_not_have_span_named("execute_tool otherfunction")
        wrapped_otherfunction()
        self.otel.assert_has_span_named("execute_tool otherfunction")

    def test_wraps_async_tool_function(self):
        async def somefunction():
            pass

        wrapped_somefunction = self.wrap(somefunction)
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        asyncio.run(somefunction())
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        asyncio.run(wrapped_somefunction())
        self.otel.assert_has_span_named("execute_tool somefunction")

    def test_preserves_tool_dict(self):
        tool_dict = genai_types.ToolDict()
        wrapped_tool_dict = self.wrap(tool_dict)
        self.assertEqual(tool_dict, wrapped_tool_dict)

    def test_does_not_have_description_if_no_doc_string(self):
        def somefunction():
            pass

        wrapped_somefunction = self.wrap(somefunction)
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        somefunction()
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        wrapped_somefunction()
        self.otel.assert_has_span_named("execute_tool somefunction")
        span = self.otel.get_span_named("execute_tool somefunction")
        self.assertNotIn("gen_ai.tool.description", span.attributes)

    def test_has_description_if_doc_string_present(self):
        def somefunction():
            """An example tool call function."""

        wrapped_somefunction = self.wrap(somefunction)
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        somefunction()
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        wrapped_somefunction()
        self.otel.assert_has_span_named("execute_tool somefunction")
        span = self.otel.get_span_named("execute_tool somefunction")
        self.assertEqual(
            span.attributes["gen_ai.tool.description"],
            "An example tool call function.",
        )

    # Capture content must be enabled to get arguments
    @patch.dict(
        "os.environ",
        {
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "SPAN_AND_EVENT",
        },
    )
    def test_handles_various_arg_types(self):
        def somefunction(
            primitive_int=None,
            dict_arg=None,
            list_arg=None,
            heterogenous_list_arg=None,
        ):
            pass

        wrapped_somefunction = self.wrap(somefunction)
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        somefunction(12345)
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        wrapped_somefunction(12345, {"key": "value"}, [1, 2, 3], [123, "abc"])
        self.otel.assert_has_span_named("execute_tool somefunction")
        span = self.otel.get_span_named("execute_tool somefunction")
        arguments = json.loads(span.attributes["gen_ai.tool.call.arguments"])
        self.assertEqual(span.attributes["gen_ai.tool.name"], "somefunction")
        self.assertEqual(
            arguments,
            {
                "primitive_int": 12345,
                "dict_arg": {"key": "value"},
                "list_arg": [1, 2, 3],
                "heterogenous_list_arg": [123, "abc"],
            },
        )

    @patch.dict(
        "os.environ",
        {
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "NO_CONTENT",
        },
    )
    def test_with_capture_content_disabled(self):
        def somefunction(arg=None):
            return arg

        wrapped_somefunction = self.wrap(somefunction)
        wrapped_somefunction("a string value")
        span = self.otel.get_span_named("execute_tool somefunction")

        self.assertNotIn(
            "gen_ai.tool.call.arguments",
            span.attributes,
        )
        self.assertNotIn(
            "gen_ai.tool.call.result",
            span.attributes,
        )

    def test_function_that_throws_exception(self):
        def somefunction(arg=None):
            raise Exception("Something went wrong")

        wrapped_somefunction = self.wrap(somefunction)
        try:
            wrapped_somefunction(12345)
        except Exception:
            span = self.otel.get_span_named("execute_tool somefunction")
            self.assertEqual(span.attributes["error.type"], "Exception")

    @patch.dict(
        "os.environ",
        {
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "SPAN_AND_EVENT",
        },
    )
    def test_variadic_keyword_arguments(self):
        def weather(**kwargs):
            return "sunny"

        wrapped = self.wrap(weather)
        wrapped(city="Boston", units="celsius")
        span = self.otel.get_span_named("execute_tool weather")
        arguments = json.loads(span.attributes["gen_ai.tool.call.arguments"])
        self.assertEqual(
            arguments, {"kwargs": {"city": "Boston", "units": "celsius"}}
        )

    @patch.dict(
        "os.environ",
        {
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "SPAN_AND_EVENT",
        },
    )
    def test_variadic_positional_arguments(self):
        def calculate(*args):
            return sum(args)

        wrapped = self.wrap(calculate)
        wrapped(10, 20)
        span = self.otel.get_span_named("execute_tool calculate")
        arguments = json.loads(span.attributes["gen_ai.tool.call.arguments"])
        self.assertEqual(arguments, {"args": [10, 20]})


@pytest.fixture(params=["sync", "async"])
def invoke_tool(
    request, monkeypatch, tracer_provider, logger_provider, meter_provider
) -> Callable[..., Awaitable[object]]:
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )

    async def invoke(
        function: Callable[..., object], *args: object, **kwargs: object
    ) -> object:
        handler = TelemetryHandler(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        )
        if request.param == "async":

            @functools.wraps(function)
            async def async_function(
                *args: object, **kwargs: object
            ) -> object:
                return function(*args, **kwargs)

            wrapped = tool_call_wrapper.wrapped_tool(async_function, handler)
            return await wrapped(*args, **kwargs)
        wrapped = tool_call_wrapper.wrapped_tool(function, handler)
        return wrapped(*args, **kwargs)

    return invoke


@pytest.mark.parametrize(
    "args, kwargs, expected",
    [
        (
            ("Boston",),
            {},
            {"city": "Boston"},
        ),
        (
            ("Boston", 1, "extra"),
            {"units": "celsius", "options": {"rain": True}, "limit": None},
            {
                "city": "Boston",
                "extra": [1, "extra"],
                "units": "celsius",
                "kwargs": {"options": {"rain": True}, "limit": None},
            },
        ),
    ],
)
@pytest.mark.asyncio
async def test_bound_arguments(
    invoke_tool, span_exporter, args, kwargs, expected
) -> None:
    result = {"forecast": "sunny"}

    def weather(
        city: str,
        /,
        *extra: object,
        units: str = "fahrenheit",
        **kwargs: object,
    ) -> dict[str, str]:
        return result

    assert await invoke_tool(weather, *args, **kwargs) is result
    (span,) = span_exporter.get_finished_spans()
    raw = span.attributes[GenAI.GEN_AI_TOOL_CALL_ARGUMENTS]
    assert type(raw) is str
    assert json.loads(raw) == expected
    assert json.loads(span.attributes[GenAI.GEN_AI_TOOL_CALL_RESULT]) == result
    assert span.status.status_code == StatusCode.UNSET


@pytest.mark.asyncio
async def test_empty_arguments(invoke_tool, span_exporter) -> None:
    def tool(default: str = "not explicitly passed") -> str:
        return default

    assert await invoke_tool(tool) == "not explicitly passed"
    (span,) = span_exporter.get_finished_spans()
    assert json.loads(span.attributes[GenAI.GEN_AI_TOOL_CALL_ARGUMENTS]) == {}


@pytest.mark.asyncio
async def test_arguments_use_shared_serializer(
    invoke_tool, span_exporter
) -> None:
    @dataclass
    class Request:
        city: str
        data: bytes

    request = Request(city="Boston", data=b"abc")

    def tool(request_arg: Request) -> str:
        assert request_arg is request
        request_arg.city = "Seattle"
        return "sunny"

    assert await invoke_tool(tool, request) == "sunny"
    assert request.city == "Seattle"
    (span,) = span_exporter.get_finished_spans()
    assert json.loads(span.attributes[GenAI.GEN_AI_TOOL_CALL_ARGUMENTS]) == {
        "request_arg": {"city": "Boston", "data": "YWJj"}
    }


@pytest.mark.parametrize(
    "error_type, expected_error_type",
    [
        (ValueError, "ValueError"),
        (asyncio.CancelledError, "asyncio.exceptions.CancelledError"),
    ],
)
@pytest.mark.asyncio
async def test_arguments_captured_when_tool_fails(
    invoke_tool,
    span_exporter,
    error_type: type[BaseException],
    expected_error_type: str,
) -> None:
    error = error_type("tool failed")

    def tool(city: str) -> None:
        raise error

    with pytest.raises(error_type) as caught:
        await invoke_tool(tool, city="Boston")
    assert caught.value is error
    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[Error.ERROR_TYPE] == expected_error_type
    assert json.loads(span.attributes[GenAI.GEN_AI_TOOL_CALL_ARGUMENTS]) == {
        "city": "Boston"
    }
    assert GenAI.GEN_AI_TOOL_CALL_RESULT not in span.attributes


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.asyncio
async def test_arguments_preserve_call_time_values(
    invoke_tool, span_exporter, fails: bool
) -> None:
    argument = {"items": ["original input"]}
    error = ValueError("tool failed")

    def tool(payload: dict[str, list[str]]) -> str:
        assert payload is argument
        payload["items"].clear()
        payload["new"] = ["added by tool"]
        if fails:
            raise error
        return "done"

    if fails:
        with pytest.raises(ValueError) as caught:
            await invoke_tool(tool, argument)
        assert caught.value is error
    else:
        assert await invoke_tool(tool, argument) == "done"
    assert argument == {"items": [], "new": ["added by tool"]}
    (span,) = span_exporter.get_finished_spans()
    assert json.loads(span.attributes[GenAI.GEN_AI_TOOL_CALL_ARGUMENTS]) == {
        "payload": {"items": ["original input"]}
    }


@pytest.mark.asyncio
async def test_argument_copy_failure_does_not_prevent_tool_execution(
    invoke_tool, span_exporter, caplog
) -> None:
    class NonCopyable:
        def __deepcopy__(self, memo: dict[int, object]) -> NonCopyable:
            raise RuntimeError("cannot copy")

    argument = NonCopyable()

    def tool(value: NonCopyable) -> str:
        assert value is argument
        return "done"

    assert await invoke_tool(tool, argument) == "done"
    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.UNSET
    assert GenAI.GEN_AI_TOOL_CALL_ARGUMENTS not in span.attributes
    assert json.loads(span.attributes[GenAI.GEN_AI_TOOL_CALL_RESULT]) == "done"
    assert "Failed to snapshot tool arguments" in caplog.text


@pytest.mark.asyncio
async def test_arguments_not_bound_when_capture_disabled(
    invoke_tool, span_exporter, monkeypatch
) -> None:
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "NO_CONTENT"
    )
    argument = object()

    def tool(value: object) -> object:
        return value

    with patch.object(
        tool_call_wrapper, "bind_arguments", side_effect=AssertionError
    ):
        assert await invoke_tool(tool, argument) is argument
    (span,) = span_exporter.get_finished_spans()
    assert GenAI.GEN_AI_TOOL_CALL_ARGUMENTS not in span.attributes
    assert GenAI.GEN_AI_TOOL_CALL_RESULT not in span.attributes
