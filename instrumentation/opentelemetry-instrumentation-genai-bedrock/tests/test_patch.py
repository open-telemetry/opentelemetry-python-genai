# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from opentelemetry.instrumentation.genai.bedrock.patch import (
    _handle_converse,
    _handle_invoke_model,
)
from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
)
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
