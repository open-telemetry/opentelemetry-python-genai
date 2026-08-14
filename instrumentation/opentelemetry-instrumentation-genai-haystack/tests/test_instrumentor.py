# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Sanity tests for the ``HaystackInstrumentor`` itself: entry point, and
that instrument/uninstrument actually wrap and unwrap the methods this
package documents."""

from haystack import Pipeline

from opentelemetry.instrumentation.genai.haystack import HaystackInstrumentor
from opentelemetry.util._importlib_metadata import entry_points


def test_entrypoint_for_opentelemetry_instrument():
    (instrumentor_entrypoint,) = entry_points(
        group="opentelemetry_instrumentor", name="haystack"
    )
    instrumentor = instrumentor_entrypoint.load()()
    assert isinstance(instrumentor, HaystackInstrumentor)


def test_instrument_and_uninstrument_wrap_and_unwrap_expected_methods(
    tracer_provider,
):
    original_pipeline_run = Pipeline.run
    original_pipeline_run_async = getattr(Pipeline, "run_async", None)
    original_pipeline_run_async_generator = getattr(Pipeline, "run_async_generator", None)

    instrumentor = HaystackInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    try:
        assert Pipeline.run is not original_pipeline_run
        if original_pipeline_run_async:
            assert getattr(Pipeline, "run_async") is not original_pipeline_run_async
        if original_pipeline_run_async_generator:
            assert getattr(Pipeline, "run_async_generator") is not original_pipeline_run_async_generator
    finally:
        instrumentor.uninstrument()

    assert Pipeline.run == original_pipeline_run
    if original_pipeline_run_async:
        assert getattr(Pipeline, "run_async") == original_pipeline_run_async
    if original_pipeline_run_async_generator:
        assert getattr(Pipeline, "run_async_generator") == original_pipeline_run_async_generator
