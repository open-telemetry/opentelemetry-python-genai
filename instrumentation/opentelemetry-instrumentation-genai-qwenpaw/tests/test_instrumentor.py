# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Instrument/uninstrument lifecycle for the QwenPaw instrumentor."""

from __future__ import annotations

from types import ModuleType

from opentelemetry.instrumentation.genai.qwenpaw import QwenPawInstrumentor


def test_instrumentor_targets_qwenpaw_distribution():
    assert QwenPawInstrumentor().instrumentation_dependencies() == (
        "qwenpaw >= 1.1.0, < 2.0.0",
    )


def test_uninstrument_unwraps_query_handler(monkeypatch):
    runner_module = ModuleType("qwenpaw.app.runner.runner")
    runner_module.AgentRunner = type("AgentRunner", (), {})
    unwrap_calls = []

    monkeypatch.setattr(
        "opentelemetry.instrumentation.genai.qwenpaw.import_module",
        lambda name: runner_module,
    )
    monkeypatch.setattr(
        "opentelemetry.instrumentation.genai.qwenpaw.unwrap",
        lambda cls, attr: unwrap_calls.append((cls, attr)),
    )

    instrumentor = QwenPawInstrumentor()
    instrumentor._is_instrumented_by_opentelemetry = True
    instrumentor.uninstrument()

    assert unwrap_calls == [(runner_module.AgentRunner, "query_handler")]


def test_instrument_uninstrument_roundtrip(
    runner_module, tracer_provider, logger_provider, meter_provider
):
    del runner_module
    instrumentor = QwenPawInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )
    instrumentor.uninstrument()
