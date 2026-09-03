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
                with vcr.use_cassette(
                    "test_chat_anthropic_multimodal_image_llm_call.yaml"
                ):
                    Anthropic().messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=100,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Describe these images.",
                                    },
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": "QUJD",
                                        },
                                    },
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "url",
                                            "url": "https://example.com/image.png",
                                        },
                                    },
                                    {
                                        "type": "document",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "application/pdf",
                                            "data": "QUJD",
                                        },
                                    },
                                    {
                                        "type": "document",
                                        "source": {
                                            "type": "url",
                                            "url": "https://example.com/document.pdf",
                                        },
                                    },
                                    {
                                        "type": "document",
                                        "source": {
                                            "type": "text",
                                            "media_type": "text/plain",
                                            "data": "Document text",
                                        },
                                    },
                                    {
                                        "type": "document",
                                        "title": "Reference",
                                        "context": "Use the nested content.",
                                        "citations": {"enabled": True},
                                        "source": {
                                            "type": "content",
                                            "content": [
                                                {
                                                    "type": "text",
                                                    "text": "Nested text",
                                                },
                                                {
                                                    "type": "image",
                                                    "source": {
                                                        "type": "url",
                                                        "url": (
                                                            "https://example.com/nested.png"
                                                        ),
                                                    },
                                                },
                                            ],
                                        },
                                    },
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "file",
                                            "file_id": "file-image",
                                        },
                                    },
                                    {
                                        "type": "document",
                                        "source": {
                                            "type": "file",
                                            "file_id": "file-document",
                                        },
                                    },
                                ],
                            }
                        ],
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
        assert len(parts) == 9
        assert parts[0] == {
            "type": "text",
            "content": "Describe these images.",
        }
        assert parts[1] == {
            "type": "blob",
            "mime_type": "image/png",
            "modality": "image",
            "content": "QUJD",
        }
        assert parts[2] == {
            "type": "uri",
            "mime_type": None,
            "modality": "image",
            "uri": "https://example.com/image.png",
        }
        assert parts[3] == {
            "type": "blob",
            "mime_type": "application/pdf",
            "modality": "document",
            "content": "QUJD",
        }
        assert parts[4] == {
            "type": "uri",
            "mime_type": "application/pdf",
            "modality": "document",
            "uri": "https://example.com/document.pdf",
        }
        assert parts[5] == {
            "type": "blob",
            "mime_type": "text/plain",
            "modality": "document",
            "content": "RG9jdW1lbnQgdGV4dA==",
        }
        assert parts[6] == {
            "type": "document",
            "value": {
                "parts": [
                    {"content": "Nested text", "type": "text"},
                    {
                        "mime_type": None,
                        "modality": "image",
                        "uri": "https://example.com/nested.png",
                        "type": "uri",
                    },
                ],
                "title": "Reference",
                "context": "Use the nested content.",
                "citations": {"enabled": True},
            },
        }
        assert parts[7] == {
            "type": "file",
            "mime_type": None,
            "modality": "image",
            "file_id": "file-image",
        }
        assert parts[8] == {
            "type": "file",
            "mime_type": None,
            "modality": "document",
            "file_id": "file-document",
        }


def _attr(span: dict[str, Any], name: str) -> Any:
    for attr in span["attributes"]:
        if attr["name"] == name:
            return attr["value"]
    return None
