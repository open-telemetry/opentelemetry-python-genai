# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

# pylint: skip-file
"""Manual OpenAI Agents instrumentation example."""

from __future__ import annotations

from agents import (
    Agent,
    RunConfig,
    Runner,
    function_tool,
)
from dotenv import load_dotenv

from opentelemetry import trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.instrumentation.genai.openai import OpenAIInstrumentor
from opentelemetry.instrumentation.genai.openai_agents import (
    OpenAIAgentsInstrumentor,
)
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def configure_otel() -> tuple[TracerProvider, MeterProvider, LoggerProvider]:
    """Configure OpenTelemetry providers and install the instrumentor."""

    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(tracer_provider)

    meter_provider = MeterProvider(
        metric_readers=[
            PeriodicExportingMetricReader(OTLPMetricExporter()),
        ],
    )
    set_meter_provider(meter_provider)

    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(OTLPLogExporter())
    )
    set_logger_provider(logger_provider)

    OpenAIInstrumentor().instrument(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    )
    OpenAIAgentsInstrumentor().instrument(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    )
    return tracer_provider, meter_provider, logger_provider


@function_tool
def get_weather(city: str) -> str:
    """Return a canned weather response for the requested city."""

    return f"The forecast for {city} is sunny with pleasant temperatures."


def main() -> None:
    load_dotenv()
    tracer_provider, meter_provider, logger_provider = configure_otel()
    weather_specialist = Agent(
        name="weather_specialist",
        instructions=(
            "You answer weather questions. Always call the get_weather tool "
            "for the requested city, then summarize the result in one short "
            "sentence with a packing suggestion."
        ),
        tools=[get_weather],
        model="gpt-4o-mini",
    )
    triage_agent = Agent(
        name="triage",
        instructions=(
            "You are a triage agent. If the user asks about weather, "
            "hand off to weather_specialist. Otherwise answer briefly yourself."
        ),
        handoffs=[weather_specialist],
        model="gpt-4o-mini",
    )

    try:
        # ``RunConfig.workflow_name`` is the agents library's own knob for
        # naming the workflow. The instrumentation reads it and emits the
        # value as the ``gen_ai.workflow.name`` attribute on the workflow
        # span — without it, the default "Agent workflow" is used.
        result = Runner.run_sync(
            triage_agent,
            "I'm visiting Barcelona this weekend. How should I pack?",
            run_config=RunConfig(workflow_name="weather-triage"),
        )
        print("Agent response:")
        print(result.final_output)
    finally:
        tracer_provider.shutdown()
        meter_provider.shutdown()
        logger_provider.shutdown()


if __name__ == "__main__":
    main()
