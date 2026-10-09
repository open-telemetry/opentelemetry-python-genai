# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any

import pytest
from botocore.exceptions import ParamValidationError
from botocore.stub import Stubber

from opentelemetry.instrumentation.genai.bedrock import patch
from opentelemetry.instrumentation.genai.bedrock.patch import (
    _handle_converse,
    _handle_invoke_model,
)
from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.trace import StatusCode
from opentelemetry.util.genai.handler import TelemetryHandler


def _bedrock_client() -> Any:
    return SimpleNamespace(
        meta=SimpleNamespace(
            endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com"
        )
    )


def _cancelled_call(*_args: Any, **_kwargs: Any) -> Any:
    raise asyncio.CancelledError()


def test_converse_records_cancelled_error(
    tracer_provider,
    span_exporter,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)

    with pytest.raises(asyncio.CancelledError):
        _handle_converse(
            _cancelled_call,
            _bedrock_client(),
            (),
            {},
            {
                "modelId": "amazon.nova-micro-v1:0",
                "messages": [],
            },
            handler,
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert (
        spans[0].attributes[ErrorAttributes.ERROR_TYPE]
        == "asyncio.exceptions.CancelledError"
    )


def test_invoke_model_records_cancelled_error(
    tracer_provider,
    span_exporter,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)

    with pytest.raises(asyncio.CancelledError):
        _handle_invoke_model(
            _cancelled_call,
            _bedrock_client(),
            (),
            {},
            {
                "modelId": "anthropic.claude-v2",
                "body": "{}",
            },
            handler,
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert (
        spans[0].attributes[ErrorAttributes.ERROR_TYPE]
        == "asyncio.exceptions.CancelledError"
    )


_CONVERSE_RESPONSE: dict[str, Any] = {
    "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
    "metrics": {"latencyMs": 1},
}


class _SdkError(Exception):
    pass


_SDK_ERROR = _SdkError()


def _failing_call(*_args: Any, **_kwargs: Any) -> Any:
    raise _SDK_ERROR


async def _async_failing_call(*_args: Any, **_kwargs: Any) -> Any:
    raise _SDK_ERROR


def _raise(*_args: Any, **_kwargs: Any) -> None:
    raise TypeError("unexpected request shape")


@pytest.mark.parametrize(
    "handle, extractor, api_params",
    [
        (
            patch._handle_converse,
            "extract_converse_request",
            {"modelId": "amazon.nova-micro-v1:0"},
        ),
        (
            patch._handle_async_converse,
            "extract_converse_request",
            {"modelId": "amazon.nova-micro-v1:0"},
        ),
        (
            patch._handle_invoke_model,
            "extract_invoke_model_request",
            {"modelId": "anthropic.claude-v2"},
        ),
        (
            patch._handle_async_invoke_model,
            "extract_invoke_model_request",
            {"modelId": "anthropic.claude-v2"},
        ),
        (
            patch._handle_invoke_model,
            "extract_embedding_request",
            {"modelId": "amazon.titan-embed-text-v2:0"},
        ),
        (
            patch._handle_async_invoke_model,
            "extract_embedding_request",
            {"modelId": "amazon.titan-embed-text-v2:0"},
        ),
        (
            patch._handle_invoke_agent,
            "extract_invoke_agent_request",
            {"agentId": "agent"},
        ),
        (
            patch._handle_async_invoke_agent,
            "extract_invoke_agent_request",
            {"agentId": "agent"},
        ),
        (
            patch._handle_retrieve,
            "extract_retrieve_request",
            {"knowledgeBaseId": "kb"},
        ),
        (
            patch._handle_async_retrieve,
            "extract_retrieve_request",
            {"knowledgeBaseId": "kb"},
        ),
    ],
)
def test_request_extraction_error_does_not_fail_the_call(
    tracer_provider, span_exporter, monkeypatch, handle, extractor, api_params
) -> None:
    monkeypatch.setattr(patch, extractor, _raise)
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    is_async = inspect.iscoroutinefunction(handle)

    with pytest.raises(_SdkError) as raised:
        result = handle(
            _async_failing_call if is_async else _failing_call,
            _bedrock_client(),
            (),
            {},
            api_params,
            handler,
        )
        if is_async:
            asyncio.run(result)

    assert raised.value is _SDK_ERROR
    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes[ErrorAttributes.ERROR_TYPE].endswith("_SdkError")


def test_request_extraction_error_keeps_a_successful_call(
    bedrock_client, instrument_with_content, span_exporter, monkeypatch
) -> None:
    monkeypatch.setattr(patch, "extract_converse_request", _raise)
    with Stubber(bedrock_client) as stubber:
        stubber.add_response("converse", _CONVERSE_RESPONSE)
        response = bedrock_client.converse(
            modelId="amazon.nova-micro-v1:0",
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
        )

    assert {
        key: value
        for key, value in response.items()
        if key != "ResponseMetadata"
    } == _CONVERSE_RESPONSE
    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.UNSET


def test_converse_malformed_request_reaches_botocore_validation(
    bedrock_client, instrument_with_content, span_exporter
) -> None:
    # A document block whose format is a list is invalid for botocore, but
    # extraction sees it first.
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "document": {
                        "format": [],
                        "name": "doc",
                        "source": {"bytes": b"x"},
                    }
                }
            ],
        }
    ]
    with pytest.raises(ParamValidationError):
        bedrock_client.converse(
            modelId="amazon.nova-micro-v1:0", messages=messages
        )

    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes[ErrorAttributes.ERROR_TYPE].endswith(
        "ParamValidationError"
    )

    # The failed call must not leave its invocation open for the next one.
    span_exporter.clear()
    with Stubber(bedrock_client) as stubber:
        stubber.add_response("converse", _CONVERSE_RESPONSE)
        bedrock_client.converse(
            modelId="amazon.nova-micro-v1:0",
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
        )
    assert len(span_exporter.get_finished_spans()) == 1
