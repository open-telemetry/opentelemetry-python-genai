# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: bedrock invoke_model with a seekable body."""

from __future__ import annotations

import io
import json
from typing import Any

import boto3

from opentelemetry.instrumentation.genai.bedrock import BedrockInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.instrumentor import instrument

from .invoke_model import InvokeModelScenario


class InvokeModelFileBodyScenario(InvokeModelScenario):
    """The chat scenario, with the request body as a seekable file object.

    The request is serialized the same way, so the same cassette answers it;
    what changes is that the bytes reach the SDK through a ``read`` call the
    instrumentation records rather than a string it could parse directly.
    """

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        with instrument(
            BedrockInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            with vcr.use_cassette("test_invoke_model_llama_conformance.yaml"):
                client = boto3.client(
                    "bedrock-runtime", region_name="us-east-1"
                )
                response = client.invoke_model(
                    modelId="meta.llama3-8b-instruct-v1:0",
                    body=io.BytesIO(
                        json.dumps(
                            {
                                "prompt": "Say this is a test",
                                "max_gen_len": 10,
                                "temperature": 0.8,
                                "top_p": 0.9,
                            }
                        ).encode()
                    ),
                )
                response["body"].read()
