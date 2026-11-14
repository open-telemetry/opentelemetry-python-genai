# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real DSPy client against the GenAI conformance HTTP server."""

from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import dspy
import pytest
from dspy.utils.exceptions import LMInvalidRequestError
from genai_mock_server import app
from werkzeug.serving import make_server

from opentelemetry.instrumentation.genai.dspy import DSPyInstrumentor
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import error_attributes
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.trace import StatusCode

MODEL = "openai/gpt-4o-mini"
PROMPT = "Describe the weather."
ANSWER = "This is a response from the mock server."
ENGINES = ["lm15"] if hasattr(dspy, "lm15") else []
if importlib.util.find_spec("litellm"):
    ENGINES.append("litellm")


@dataclass
class MockServer:
    url: str = ""
    requests: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, environ: dict[str, Any], start_response: Any) -> Any:
        size = int(environ.get("CONTENT_LENGTH") or 0)
        raw = environ["wsgi.input"].read(size)
        environ["wsgi.input"] = io.BytesIO(raw)
        if environ["REQUEST_METHOD"] == "POST":
            self.requests.append(
                {"path": environ["PATH_INFO"], "body": json.loads(raw)}
            )
        return app(environ, start_response)


@pytest.fixture
def mock_server() -> Iterator[MockServer]:
    recorded = MockServer()
    server = make_server("127.0.0.1", 0, recorded, threaded=True)
    recorded.url = f"http://127.0.0.1:{server.server_port}/v1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield recorded
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def make_lm(server: MockServer, engine: str = "lm15") -> dspy.LM:
    return dspy.LM(
        MODEL,
        **({"engine": engine} if hasattr(dspy, "lm15") else {}),
        api_base=server.url,
        api_key="local-test-key",
        cache=False,
        num_retries=0,
        timeout=5,
        temperature=0.2,
        max_tokens=40,
    )


def call_lm(lm: dspy.LM, *, asynchronous: bool, explicit: bool) -> Any:
    argument: Any = PROMPT
    if explicit:
        if hasattr(dspy, "lm15"):
            argument = dspy.lm15.Request(
                model=MODEL,
                messages=[
                    dspy.lm15.Message(
                        role="user", parts=[dspy.lm15.TextPart(text=PROMPT)]
                    )
                ],
                config=dspy.lm15.Config(temperature=0.2, max_tokens=40),
            )
        else:
            argument = dspy.LMRequest.from_call(
                model=MODEL, prompt=PROMPT, temperature=0.2, max_tokens=40
            )
    if asynchronous:
        return asyncio.run(lm.acall(argument))
    return lm(argument)


def assert_http(server: MockServer) -> None:
    assert len(server.requests) == 1
    request = server.requests[0]
    assert request["path"] == "/v1/chat/completions"
    assert request["body"]["model"] == "gpt-4o-mini"
    assert request["body"]["messages"] == [{"role": "user", "content": PROMPT}]
    assert request["body"]["temperature"] == 0.2
    assert (
        request["body"].get(
            "max_tokens", request["body"].get("max_completion_tokens")
        )
        == 40
    )


def assert_result(result: Any, explicit: bool) -> None:
    if explicit:
        assert result.text == ANSWER
    else:
        assert result == [ANSWER]


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("explicit", [False, True], ids=["list", "request"])
@pytest.mark.parametrize("engine", ENGINES)
def test_http_inference(
    mock_server: MockServer,
    tracer_provider: Any,
    logger_provider: Any,
    meter_provider: Any,
    span_exporter: Any,
    asynchronous: bool,
    explicit: bool,
    engine: str,
) -> None:
    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ):
        result = call_lm(
            make_lm(mock_server, engine),
            asynchronous=asynchronous,
            explicit=explicit,
        )

    assert_result(result, explicit)
    assert_http(mock_server)
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "chat gpt-4o-mini"
    attributes = span.attributes
    assert attributes[GenAI.GEN_AI_OPERATION_NAME] == "chat"
    assert attributes[GenAI.GEN_AI_PROVIDER_NAME] == "openai"
    assert attributes[GenAI.GEN_AI_REQUEST_MODEL] == "gpt-4o-mini"
    assert attributes[GenAI.GEN_AI_RESPONSE_MODEL] == "gpt-4o-mini"
    assert attributes[GenAI.GEN_AI_RESPONSE_ID] == "chatcmpl-mock-001"
    assert attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)
    assert attributes[GenAI.GEN_AI_USAGE_INPUT_TOKENS] == 25
    assert attributes[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] == 12
    assert type(attributes[GenAI.GEN_AI_USAGE_INPUT_TOKENS]) is int
    assert type(attributes[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS]) is int
    assert attributes[GenAI.GEN_AI_REQUEST_TEMPERATURE] == 0.2
    assert attributes[GenAI.GEN_AI_REQUEST_MAX_TOKENS] == 40
    inputs = json.loads(attributes[GenAI.GEN_AI_INPUT_MESSAGES])
    outputs = json.loads(attributes[GenAI.GEN_AI_OUTPUT_MESSAGES])
    assert len(inputs) == len(outputs) == 1
    assert inputs[0]["role"] == "user"
    assert inputs[0]["parts"] == [{"type": "text", "content": PROMPT}]
    assert outputs[0]["role"] == "assistant"
    assert outputs[0]["parts"] == [{"type": "text", "content": ANSWER}]
    assert outputs[0]["finish_reason"] == "stop"


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("explicit", [False, True], ids=["list", "request"])
@pytest.mark.parametrize("engine", ENGINES)
def test_http_after_uninstrumentation(
    mock_server: MockServer,
    tracer_provider: Any,
    logger_provider: Any,
    meter_provider: Any,
    span_exporter: Any,
    asynchronous: bool,
    explicit: bool,
    engine: str,
) -> None:
    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        result = call_lm(
            make_lm(mock_server, engine),
            asynchronous=asynchronous,
            explicit=explicit,
        )
        assert_result(result, explicit)
        assert_http(mock_server)
        assert len(span_exporter.get_finished_spans()) == 1

    span_exporter.clear()
    mock_server.requests.clear()
    result = call_lm(
        make_lm(mock_server, engine),
        asynchronous=asynchronous,
        explicit=explicit,
    )
    assert_result(result, explicit)
    assert_http(mock_server)
    assert span_exporter.get_finished_spans() == ()


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.skipif(
    not hasattr(dspy, "lm15"), reason="Native backend requires DSPy 3.4"
)
def test_native_http_error(
    mock_server: MockServer,
    instrument_dspy: DSPyInstrumentor,
    span_exporter: Any,
    asynchronous: bool,
) -> None:
    lm = make_lm(mock_server)
    with pytest.raises(LMInvalidRequestError, match="mock server rejected"):
        if asynchronous:
            asyncio.run(lm.acall("[MOCK_BAD_REQUEST]"))
        else:
            lm("[MOCK_BAD_REQUEST]")

    assert len(mock_server.requests) == 1
    assert mock_server.requests[0]["path"] == "/v1/chat/completions"
    assert mock_server.requests[0]["body"]["messages"] == [
        {"role": "user", "content": "[MOCK_BAD_REQUEST]"}
    ]
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR
    assert (
        spans[0].attributes[error_attributes.ERROR_TYPE]
        == "dspy.utils.exceptions.LMInvalidRequestError"
    )
    assert GenAI.GEN_AI_USAGE_INPUT_TOKENS not in spans[0].attributes
    assert GenAI.GEN_AI_USAGE_OUTPUT_TOKENS not in spans[0].attributes
