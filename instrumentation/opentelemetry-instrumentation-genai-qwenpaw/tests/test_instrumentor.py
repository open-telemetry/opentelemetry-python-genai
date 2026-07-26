# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-distribution instrumentor plugins and instrument/uninstrument lifecycle."""

from __future__ import annotations

from types import ModuleType

import pytest

from opentelemetry.instrumentation.genai.qwenpaw import (
    CoPawInstrumentor,
    QwenPawInstrumentor,
)


def test_qwenpaw_instrumentor_targets_qwenpaw_distribution():
    assert QwenPawInstrumentor().instrumentation_dependencies() == (
        "qwenpaw >= 1.1.0, < 2.0.0",
    )
    assert QwenPawInstrumentor._runner_module == "qwenpaw.app.runner.runner"


def test_copaw_instrumentor_targets_legacy_copaw_distribution():
    assert CoPawInstrumentor().instrumentation_dependencies() == (
        "copaw >= 0.1.0, <= 1.0.2",
    )
    assert CoPawInstrumentor._runner_module == "copaw.app.runner.runner"


@pytest.mark.parametrize(
    "instrumentor_cls_under_test",
    [QwenPawInstrumentor, CoPawInstrumentor],
    ids=lambda cls: cls.__name__,
)
def test_uninstrument_unwraps_query_handler(
    monkeypatch, instrumentor_cls_under_test
):
    runner_module = ModuleType(instrumentor_cls_under_test._runner_module)
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

    instrumentor = instrumentor_cls_under_test()
    instrumentor._is_instrumented_by_opentelemetry = True
    instrumentor.uninstrument()

    assert unwrap_calls == [(runner_module.AgentRunner, "query_handler")]


def test_instrument_uninstrument_roundtrip(
    instrumentor_cls, tracer_provider, logger_provider, meter_provider
):
    instrumentor = instrumentor_cls()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )
    instrumentor.uninstrument()
