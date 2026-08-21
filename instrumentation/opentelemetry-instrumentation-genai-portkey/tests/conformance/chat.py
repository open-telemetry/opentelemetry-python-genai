# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: Portkey AI chat (inference)."""

from __future__ import annotations

import os
from typing import Any
from unittest import mock

from portkey_ai import Portkey

from opentelemetry.instrumentation.genai.portkey import PortkeyInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument


class ChatScenario(Scenario):
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
            if os.getenv("PORTKEY_API_KEY")
            else {"PORTKEY_API_KEY": "test_portkey_api_key"}
        )
        with mock.patch.dict(os.environ, key_override):
            with instrument(
                PortkeyInstrumentor(),
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
                content_capture="SPAN_ONLY",
            ):
                with vcr.use_cassette("chat_conformance.yaml"):
                    client = Portkey(api_key=os.environ["PORTKEY_API_KEY"])
                    client.chat.completions.create(
                        model="gpt-4o-mini",
                        max_tokens=10,
                        messages=[
                            {
                                "role": "user",
                                "content": "Say hello in one word.",
                            }
                        ],
                    )
