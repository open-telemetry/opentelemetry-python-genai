# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: bedrock invoke_model (embeddings) with a seekable body."""

from __future__ import annotations

import io
import json
from typing import Any

import boto3

from opentelemetry.instrumentation.genai.bedrock import BedrockInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.instrumentor import instrument

from .embedding import EmbeddingScenario

#: The body the SDK reads. The recorded response carries no vector of its own,
#: so the dimension count can only come from these bytes.
_INPUT_TEXT = "Hello world"
_DIMENSIONS = 512
_BODY = {"inputText": _INPUT_TEXT, "dimensions": _DIMENSIONS}


class EmbeddingFileBodyScenario(EmbeddingScenario):
    """The embedding scenario, with the request body as a seekable file object.

    The response is recorded, so the dimension count on the span can only have
    come from the request body the instrumentation recorded a read of.
    """

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        attributes = {
            attr["name"]: attr["value"]
            for entry in report["samples"]
            if "span" in entry
            for attr in entry["span"]["attributes"]
        }
        dimension_count = attributes.get("gen_ai.embeddings.dimension.count")
        assert dimension_count == _DIMENSIONS, (
            "the dimension count should come from the file-like request body; "
            f"expected {_BODY['dimensions']}, saw {dimension_count}"
        )

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
            with vcr.use_cassette(
                "test_embedding_request_body_is_recorded.yaml"
            ):
                client = boto3.client(
                    "bedrock-runtime", region_name="us-east-1"
                )
                response = client.invoke_model(
                    modelId="amazon.titan-embed-text-v2:0",
                    body=io.BytesIO(json.dumps(_BODY).encode()),
                )
                response["body"].read()
