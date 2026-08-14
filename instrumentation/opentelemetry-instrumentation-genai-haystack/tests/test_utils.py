# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared assertion helpers for Haystack instrumentation tests."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


def assert_chat_span_attributes(  # pylint: disable=too-many-arguments
    span: ReadableSpan,
    *,
    request_model: str,
    operation_name: str = "chat",
    provider: str | None = None,
    response_model: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    finish_reasons: Sequence[str] | None = None,
) -> None:
    attributes = span.attributes or {}
    assert span.name == f"{operation_name} {request_model}"
    assert attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == operation_name
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == request_model
    if provider is not None:
        assert attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == provider
    if response_model is not None:
        assert (
            attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == response_model
        )
    if input_tokens is not None:
        assert (
            attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
            == input_tokens
        )
    if output_tokens is not None:
        assert (
            attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
            == output_tokens
        )
    if finish_reasons is not None:
        assert (
            tuple(finish_reasons)
            == attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS]
        )


def assert_error_recorded(span: ReadableSpan, error_type: str) -> None:
    assert not span.status.is_ok
    assert (span.attributes or {}).get(
        ErrorAttributes.ERROR_TYPE
    ) == error_type


def load_messages_attribute(
    span: ReadableSpan, attribute: str
) -> list[Mapping[str, Any]]:
    value = (span.attributes or {}).get(attribute)
    assert isinstance(value, str), (
        f"expected {attribute} to be a JSON string, got {value!r}"
    )
    parsed = json.loads(value)
    assert isinstance(parsed, list)
    return parsed


def message_part_types(message: Mapping[str, Any]) -> list[str]:
    return [part["type"] for part in message["parts"]]


def text_content(message: Mapping[str, Any]) -> str | None:
    for part in message["parts"]:
        if part["type"] == "text":
            return part["content"]
    return None
