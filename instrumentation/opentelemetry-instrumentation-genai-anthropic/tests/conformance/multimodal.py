# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: Anthropic multimodal chat input."""

from __future__ import annotations

import json
import os
from typing import Any
from unittest import mock

from anthropic import Anthropic

from opentelemetry.instrumentation.genai.anthropic import AnthropicInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument

from ..conftest import (
    _IMAGE_URL,
    _NESTED_IMAGE_URL,
    _PDF_URL,
    _REAL_PDF_B64,
    _REAL_PNG_B64,
    multimodal_input_message,
)

_IMAGE_FILE_ID = "file_01Y2U724wHRo2D9zsfYpvQhK"
_PDF_FILE_ID = "file_01Y6v8cAFmoHdUxx9G1vVdcJ"


class MultimodalScenario(Scenario):
    expected_spans = {"chat": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        key_override = (
            {}
            if os.getenv("ANTHROPIC_API_KEY")
            else {"ANTHROPIC_API_KEY": "test_anthropic_api_key"}
        )
        with mock.patch.dict(os.environ, key_override):
            with instrument(
                AnthropicInstrumentor(),
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
                content_capture="SPAN_ONLY",
            ):
                msg = multimodal_input_message()
                msg["content"].extend(
                    [
                        {
                            "type": "image",
                            "source": {
                                "type": "file",
                                "file_id": _IMAGE_FILE_ID,
                            },
                        },
                        {
                            "type": "document",
                            "source": {
                                "type": "file",
                                "file_id": _PDF_FILE_ID,
                            },
                        },
                    ]
                )
                with vcr.use_cassette(
                    "test_chat_anthropic_multimodal_image_llm_call.yaml"
                ):
                    Anthropic().messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=100,
                        messages=[msg],
                        extra_headers={
                            "anthropic-beta": "files-api-2025-04-14"
                        },
                    )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        chat_spans = [
            entry["span"]
            for entry in report["samples"]
            if "span" in entry
            and _attr(entry["span"], "gen_ai.operation.name") == "chat"
        ]
        assert chat_spans, "no chat span emitted"

        input_messages = json.loads(
            _attr(chat_spans[0], "gen_ai.input.messages")
        )
        assert len(input_messages) == 1
        assert input_messages[0]["role"] == "user"

        parts = input_messages[0]["parts"]
        assert len(parts) == 10
        assert parts[0] == {
            "type": "text",
            "content": "Describe these images.",
        }
        assert parts[1] == {
            "type": "blob",
            "mime_type": "image/png",
            "modality": "image",
            "content": _REAL_PNG_B64,
        }
        assert parts[2] == {
            "type": "uri",
            "mime_type": None,
            "modality": "image",
            "uri": _IMAGE_URL,
        }
        assert parts[3] == {
            "type": "blob",
            "mime_type": "application/pdf",
            "modality": "document",
            "content": _REAL_PDF_B64,
        }
        assert parts[4] == {
            "type": "uri",
            "mime_type": None,
            "modality": "document",
            "uri": _PDF_URL,
        }
        assert parts[5] == {
            "type": "blob",
            "mime_type": "text/plain",
            "modality": "document",
            "content": "RG9jdW1lbnQgdGV4dA==",
        }
        assert parts[6] == {
            "type": "text",
            "content": "Nested text",
        }
        assert parts[7] == {
            "type": "uri",
            "mime_type": None,
            "modality": "image",
            "uri": _NESTED_IMAGE_URL,
        }
        assert parts[8] == {
            "type": "file",
            "mime_type": None,
            "modality": "image",
            "file_id": _IMAGE_FILE_ID,
        }
        assert parts[9] == {
            "type": "file",
            "mime_type": None,
            "modality": "document",
            "file_id": _PDF_FILE_ID,
        }


def _attr(span: dict[str, Any], name: str) -> Any:
    for attr in span["attributes"]:
        if attr["name"] == name:
            return attr["value"]
    return None
