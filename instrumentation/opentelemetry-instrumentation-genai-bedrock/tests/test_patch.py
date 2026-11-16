# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from botocore.exceptions import ParamValidationError
from botocore.stub import Stubber

from opentelemetry import context as context_api
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
    before = context_api.get_current()
    with pytest.raises(ParamValidationError):
        bedrock_client.converse(
            modelId="amazon.nova-micro-v1:0", messages=messages
        )
    assert context_api.get_current() is before

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
