# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario for an image ``uri`` part on a ``chat`` input."""

from __future__ import annotations

from typing import Any

import pytest

from opentelemetry.instrumentation.genai.smolagents import (
    SmolagentsInstrumentor,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument

from ..test_utils import vllm_model
from ._helpers import attr, chat_spans, part_fields
from .inference import IN_PROCESS_GAPS

_IMAGE_URL = (
    "https://fastly.picsum.photos/id/237/200/300.jpg"
    "?hmac=TmmQSbShHz9CdQm0NkEjx1Dyh_Y984R9LpNrpvH2D_U"
)


class MultimodalScenario(Scenario):
    """A vision-capable ``VLLMModel`` taking an image alongside the text.

    A VLM keeps the message content as parts instead of flattening it to text,
    which is what puts an image part on the span.
    """

    expected_spans = {"chat": 1}
    expected_metrics = ("gen_ai.client.operation.duration",)
    expected_violations = IN_PROCESS_GAPS

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        with instrument(
            SmolagentsInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            # vllm_model fakes the vllm modules for the duration of the call.
            with pytest.MonkeyPatch.context() as monkeypatch:
                model = vllm_model(monkeypatch)
                model._is_vlm = True
                model.flatten_messages_as_text = False
                model.generate(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "What breed is this dog?",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": _IMAGE_URL},
                                },
                            ],
                        }
                    ]
                )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        input_parts = {
            fields
            for span in chat_spans(report)
            for fields in part_fields(attr(span, "gen_ai.input.messages"))
        }
        assert ("uri", "image") in input_parts, (
            f"expected an image uri input part, saw {input_parts}"
        )
