# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Portkey AI prompt completions instrumentation."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from portkey_ai import Portkey

try:
    from portkey_ai import AsyncPortkey
except ImportError:
    AsyncPortkey = None  # type: ignore[assignment,misc]

from opentelemetry.instrumentation.genai.portkey import PortkeyInstrumentor
from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes import server_attributes
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.trace import StatusCode

_has_async_portkey = AsyncPortkey is not None


def _create_mock_prompt_completion(
    id_val: str = "promptcmpl-123",
    model: str = "gpt-4o",
    text: str = "Rendered template response",
    role: str = "assistant",
    finish_reason: str = "stop",
    prompt_tokens: int = 12,
    completion_tokens: int = 6,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=id_val,
        model=model,
        choices=[
            SimpleNamespace(
                index=0,
                text=text,
                finish_reason=finish_reason,
                message=SimpleNamespace(role=role, content=text),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def test_sync_prompt_completions_basic(
    tracer_provider, logger_provider, meter_provider, span_exporter
):
    with instrument(
        PortkeyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        p = Portkey(
            api_key="test_pk",
            provider="openai",
            base_url="https://api.portkey.ai/v1",
        )
        mock_resp = _create_mock_prompt_completion()
        p.prompts.completions._post = MagicMock(return_value=mock_resp)

        res = p.prompts.completions.create(
            prompt_id="pp-customer-service-v1",
            variables={"user_name": "Alice"},
            temperature=0.3,
            max_tokens=200,
        )

        assert res.id == "promptcmpl-123"

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "chat"
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
            == GenAIAttributes.GenAiOperationNameValues.CHAT.value
        )
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME)
            == "openai"
        )
        assert (
            span.attributes.get(server_attributes.SERVER_ADDRESS)
            == "api.portkey.ai"
        )
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_PROMPT_NAME)
            == "pp-customer-service-v1"
        )
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE)
            == 0.3
        )
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS)
            == 200
        )
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_ID)
            == "promptcmpl-123"
        )
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_MODEL)
            == "gpt-4o"
        )
        assert span.attributes.get(
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
        ) == ("stop",)
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
            == 12
        )
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS)
            == 6
        )


@pytest.mark.skipif(
    not _has_async_portkey,
    reason="AsyncPortkey not available in this version of portkey-ai",
)
@pytest.mark.asyncio
async def test_async_prompt_completions_basic(
    tracer_provider, logger_provider, meter_provider, span_exporter
):
    with instrument(
        PortkeyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        ap = AsyncPortkey(
            api_key="test_pk",
            provider="anthropic",
        )
        mock_resp = _create_mock_prompt_completion(
            id_val="promptcmpl-async-1",
            model="claude-3-haiku-20240307",
            text="Async prompt output",
            prompt_tokens=15,
            completion_tokens=5,
        )
        ap.prompts.completions._post = AsyncMock(return_value=mock_resp)

        res = await ap.prompts.completions.create(
            prompt_id="pp-async-summary",
            variables={"doc": "text to summarize"},
        )

        assert res.id == "promptcmpl-async-1"

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "chat"
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME)
            == "anthropic"
        )
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_PROMPT_NAME)
            == "pp-async-summary"
        )
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_MODEL)
            == "claude-3-haiku-20240307"
        )
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
            == 15
        )
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS)
            == 5
        )


def test_sync_prompt_completions_content_capture(
    tracer_provider, logger_provider, meter_provider, span_exporter
):
    with instrument(
        PortkeyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ):
        p = Portkey(api_key="test_pk")
        mock_resp = _create_mock_prompt_completion(
            text="Prompt output content"
        )
        p.prompts.completions._post = MagicMock(return_value=mock_resp)

        p.prompts.completions.create(
            prompt_id="pp-test-1",
            variables={"var": "val"},
        )

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        output_messages = json.loads(
            span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        )
        assert len(output_messages) == 1
        assert (
            output_messages[0]["parts"][0]["content"]
            == "Prompt output content"
        )


def test_sync_prompt_completions_error_handling(
    tracer_provider, logger_provider, meter_provider, span_exporter
):
    with instrument(
        PortkeyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        p = Portkey(api_key="test_pk")
        p.prompts.completions._post = MagicMock(
            side_effect=ConnectionError("Prompt API unreachable")
        )

        with pytest.raises(ConnectionError, match="Prompt API unreachable"):
            p.prompts.completions.create(
                prompt_id="pp-failing",
            )

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR
        assert (
            span.attributes.get(ErrorAttributes.ERROR_TYPE)
            == "ConnectionError"
        )


@pytest.mark.skipif(
    not _has_async_portkey,
    reason="AsyncPortkey not available in this version of portkey-ai",
)
@pytest.mark.asyncio
async def test_async_prompt_completions_error_handling(
    tracer_provider, logger_provider, meter_provider, span_exporter
):
    with instrument(
        PortkeyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        ap = AsyncPortkey(api_key="test_pk")
        ap.prompts.completions._post = AsyncMock(
            side_effect=TimeoutError("Async prompt timeout")
        )

        with pytest.raises(TimeoutError, match="Async prompt timeout"):
            await ap.prompts.completions.create(
                prompt_id="pp-async-timeout",
            )

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR
        assert (
            span.attributes.get(ErrorAttributes.ERROR_TYPE) == "TimeoutError"
        )
