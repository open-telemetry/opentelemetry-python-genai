# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for suppressing CrewAI's built-in usage telemetry."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from unittest.mock import Mock

import pytest

from opentelemetry import trace
from opentelemetry.instrumentation.genai.crewai import CrewAIInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider


def test_conftest_disables_crewai_telemetry_before_import() -> None:
    """The env must be set before ``crewai`` is first imported.

    Older CrewAI releases otherwise build their telemetry provider at import
    and install it as the global ``TracerProvider``.
    """
    from crewai.telemetry.constants import CREWAI_TELEMETRY_SERVICE_NAME
    from crewai.telemetry.telemetry import Telemetry

    assert Telemetry().ready is False
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        assert (
            provider.resource.attributes.get(SERVICE_NAME)
            != CREWAI_TELEMETRY_SERVICE_NAME
        )


def _run_choke_point() -> tuple[object, bool]:
    """Call the choke point with a self that CrewAI would consider live.

    The conftest disables CrewAI telemetry through the environment, which makes
    the real singleton short-circuit regardless of our patch.
    """
    from crewai.telemetry.telemetry import Telemetry

    telemetry = Mock()
    telemetry._should_execute_telemetry.return_value = True
    operation = Mock(return_value="span")
    result = Telemetry._safe_telemetry_operation(telemetry, operation)
    return result, operation.called


def test_crewai_telemetry_operations_are_dropped_by_default(
    instrument_crewai,
) -> None:
    result, called = _run_choke_point()
    assert result is None
    assert called is False


def test_crewai_telemetry_operations_run_when_kept(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
) -> None:
    instrumentor = CrewAIInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        disable_crewai_telemetry=False,
    )
    try:
        result, called = _run_choke_point()
    finally:
        instrumentor.uninstrument()
    assert result == "span"
    assert called is True


# Runs a crew with CrewAI telemetry enabled in the environment. The OTLP
# exporter on the telemetry singleton's provider is shut down and replaced by
# an in-memory exporter so nothing is sent to CrewAI's collector, and the span
# names are printed. Older CrewAI installs that provider globally and emits
# through the global tracer, so the provider is reused rather than swapped.
_CREW_SCRIPT = """
import json, sys
from crewai import Agent, BaseLLM, Crew, Process, Task
from crewai.telemetry.telemetry import Telemetry
from opentelemetry.instrumentation.genai.crewai import CrewAIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


class ScriptedLLM(BaseLLM):
    def __init__(self):
        super().__init__(model="scripted-model")

    def call(self, *args, **kwargs):
        return "Thought: done.\\nFinal Answer: ok"

    def supports_function_calling(self):
        return False

    def supports_stop_words(self):
        return False


exporter = InMemorySpanExporter()
telemetry = Telemetry()
assert telemetry.ready, "CrewAI telemetry must be enabled for this script"
telemetry.provider.shutdown()
telemetry.provider.add_span_processor(SimpleSpanProcessor(exporter))

CrewAIInstrumentor().instrument(
    tracer_provider=TracerProvider(),
    disable_crewai_telemetry=json.loads(sys.argv[1]),
)
agent = Agent(
    role="Worker", goal="g", backstory="b", llm=ScriptedLLM(), verbose=False
)
crew = Crew(
    agents=[agent],
    tasks=[Task(description="d", expected_output="o", agent=agent)],
    process=Process.sequential,
    verbose=False,
)
crew.kickoff()
print("SPANS=" + json.dumps(sorted({s.name for s in exporter.get_finished_spans()})))
"""


def _crewai_span_names(disable: bool) -> list[str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in (
            "CREWAI_DISABLE_TELEMETRY",
            "CREWAI_DISABLE_TRACKING",
            "OTEL_SDK_DISABLED",
        )
    }
    env["CREWAI_STORAGE_DIR"] = tempfile.mkdtemp(prefix="crewai-telemetry-")
    # Skips CrewAI's first-run cloud tracing prompt; unrelated to usage telemetry.
    env["CREWAI_TESTING"] = "true"
    completed = subprocess.run(
        [sys.executable, "-c", _CREW_SCRIPT, json.dumps(disable)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    line = next(
        line
        for line in completed.stdout.splitlines()
        if line.startswith("SPANS=")
    )
    return json.loads(line[len("SPANS=") :])


@pytest.mark.parametrize("disable", [True, False])
def test_crew_run_crewai_telemetry_spans(disable: bool) -> None:
    span_names = _crewai_span_names(disable)
    if disable:
        assert span_names == []
    else:
        assert "Crew Created" in span_names
