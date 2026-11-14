# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Instrument/uninstrument lifecycle for the QwenPaw instrumentor."""

from __future__ import annotations

from importlib.metadata import Distribution, PackageNotFoundError
from types import ModuleType

import pytest

from opentelemetry.instrumentation.genai.qwenpaw import QwenPawInstrumentor


def test_instrumentor_targets_qwenpaw_distribution():
    assert QwenPawInstrumentor().instrumentation_dependencies() == (
        "qwenpaw >= 1.1.0, < 3.0.0; python_version < '3.14'",
    )


def test_uninstrument_unwraps_query_handler(monkeypatch):
    monkeypatch.setattr(
        "opentelemetry.instrumentation.genai.qwenpaw._runtime_v2",
        lambda: False,
    )
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
    tracer_provider, logger_provider, meter_provider
):
    instrumentor = QwenPawInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )
    instrumentor.uninstrument()


def test_instrument_uninstrument_without_qwenpaw_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from_name = Distribution.from_name

    def from_name_without_qwenpaw(name: str) -> Distribution:
        if name == "qwenpaw":
            raise PackageNotFoundError(name)
        return from_name(name)

    monkeypatch.setattr(Distribution, "from_name", from_name_without_qwenpaw)

    instrumentor = QwenPawInstrumentor()
    instrumentor.instrument(skip_dep_check=True)
    try:
        assert instrumentor.is_instrumented_by_opentelemetry
    finally:
        instrumentor.uninstrument()
