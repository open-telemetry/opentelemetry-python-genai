# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
Prototype demonstration script for semantic-conventions-genai PR #475.

This script demonstrates that when an upstream framework (e.g. an agent framework,
DSPy, or gateway) initiates an inference span and an inference event in the
OpenTelemetry context:
1. Google GenAI instrumentation detects the existing inference span and event.
2. It suppresses emitting duplicate inference spans and duplicate events.
3. It enriches the existing span and event with transport metadata (`server.address`)
   and custom downstream attributes (`google_genai.inference_suppressed`).
4. Exactly 1 span and 1 event are recorded in total.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add google-genai test utilities for mocking responses without real network calls
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(
    0,
    str(
        REPO_ROOT
        / "instrumentation"
        / "opentelemetry-instrumentation-google-genai"
    ),
)

import google.genai
from google.genai.models import Models
from tests.generate_content.util import (
    convert_to_response,
    create_response,
)

from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv.attributes import server_attributes
from opentelemetry.util.genai import (
    get_current_inference_event,
    get_current_inference_span,
)
from opentelemetry.util.genai.handler import TelemetryHandler


def main() -> None:
    print(
        "==================================================================="
    )
    print("PROTOTYPING INFERENCE SPAN & EVENT DEDUPLICATION (PR #475)")
    print(
        "===================================================================\n"
    )

    # 1. Enable GenAI event emission
    os.environ["OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT"] = "true"

    # 2. Setup OpenTelemetry TracerProvider and LoggerProvider with In-Memory Exporters
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    log_exporter = InMemoryLogRecordExporter()
    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(
        SimpleLogRecordProcessor(log_exporter)
    )

    # Mock the client's generate_content call to simulate a successful API response
    original_generate_content = Models.generate_content
    Models.generate_content = lambda *args, **kwargs: convert_to_response(
        create_response(text="Hello from Gemini!")
    )

    # 3. Instrument google-genai
    instrumentor = GoogleGenAiSdkInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
    )

    try:
        client = google.genai.Client(vertexai=False, api_key="test-api-key")

        # 4. An upstream framework (agent framework, DSPy, gateway) starts an inference invocation
        upstream_handler = TelemetryHandler(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
        )

        print("[Step 1] Upstream framework starts an inference invocation...")
        with upstream_handler.inference(
            provider="upstream-agent-framework",
            request_model="gemini-2.0-flash",
        ):
            # Check what's in context
            active_span = get_current_inference_span()
            active_event = get_current_inference_event()
            assert active_event is not None
            print(f"  -> Active inference span in context: {active_span}")
            print(
                f"  -> Active inference event in context: {active_event.event_name}"
            )
            print(
                f"  -> Initial event attributes: {dict(active_event.attributes or {})}"
            )

            print("\n[Step 2] Downstream Google GenAI client is invoked...")
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents="Hello, how are you?",
            )
            print(f"  -> Client response received: {response.text}")
            print(
                f"  -> Event attributes after Google GenAI execution: {dict(active_event.attributes or {})}"
            )

        print(
            "\n[Step 3] Upstream invocation ended. Checking exported telemetry..."
        )
        finished_spans = span_exporter.get_finished_spans()
        finished_logs = log_exporter.get_finished_logs()

        print(f"\nFinished spans count: {len(finished_spans)}")
        for i, span in enumerate(finished_spans, 1):
            print(f"  Span #{i}: '{span.name}'")
            print(
                f"    gen_ai.provider.name:               {span.attributes.get('gen_ai.provider.name')}"
            )
            print(
                f"    server.address:                     {span.attributes.get(server_attributes.SERVER_ADDRESS)}"
            )
            print(
                f"    google_genai.inference_suppressed:  {span.attributes.get('google_genai.inference_suppressed')}"
            )

        print(f"\nFinished log events count: {len(finished_logs)}")
        for i, log in enumerate(finished_logs, 1):
            record = log.log_record
            print(f"  Event #{i}: '{record.event_name}'")
            print(
                f"    gen_ai.provider.name:               {record.attributes.get('gen_ai.provider.name')}"
            )
            print(
                f"    server.address:                     {record.attributes.get(server_attributes.SERVER_ADDRESS)}"
            )
            print(
                f"    google_genai.inference_suppressed:  {record.attributes.get('google_genai.inference_suppressed')}"
            )

        # 5. Assertions
        assert len(finished_spans) == 1, (
            f"Expected 1 span, got {len(finished_spans)}"
        )
        assert len(finished_logs) == 1, (
            f"Expected 1 event, got {len(finished_logs)}"
        )

        span = finished_spans[0]
        event = finished_logs[0].log_record

        assert span.attributes is not None
        assert event.attributes is not None

        assert (
            span.attributes.get(server_attributes.SERVER_ADDRESS)
            == "generativelanguage.googleapis.com"
        )
        assert (
            event.attributes.get(server_attributes.SERVER_ADDRESS)
            == "generativelanguage.googleapis.com"
        )
        assert (
            span.attributes.get("gen_ai.provider.name")
            == "upstream-agent-framework"
        )
        assert (
            event.attributes.get("gen_ai.provider.name")
            == "upstream-agent-framework"
        )
        assert span.attributes.get("google_genai.inference_suppressed") is True
        assert (
            event.attributes.get("google_genai.inference_suppressed") is True
        )

        print(
            "\n==================================================================="
        )
        print("SUCCESS: Exactly 1 span and 1 event recorded!")
        print(
            "Both enriched with 'server.address = generativelanguage.googleapis.com'"
        )
        print("Both enriched with 'google_genai.inference_suppressed = True'")
        print("Duplicate inference telemetry was successfully suppressed.")
        print(
            "==================================================================="
        )

    finally:
        Models.generate_content = original_generate_content
        instrumentor.uninstrument()


if __name__ == "__main__":
    main()
