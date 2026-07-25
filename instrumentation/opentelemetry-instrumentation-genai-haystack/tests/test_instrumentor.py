# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Sanity tests for the ``HaystackInstrumentor`` itself: entry point, and
that instrument/uninstrument actually wrap and unwrap the methods this
package documents."""

import haystack
from haystack.components.generators.chat.openai import OpenAIChatGenerator

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
    original_pipeline_run = haystack.Pipeline.run
    original_pipeline_run_async = haystack.Pipeline.run_async
    original_chat_generator_run = OpenAIChatGenerator.run

    instrumentor = HaystackInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    try:
        assert haystack.Pipeline.run is not original_pipeline_run
        assert haystack.Pipeline.run_async is not original_pipeline_run_async
        assert OpenAIChatGenerator.run is not original_chat_generator_run
    finally:
        instrumentor.uninstrument()

    assert haystack.Pipeline.run == original_pipeline_run
    assert haystack.Pipeline.run_async == original_pipeline_run_async
    assert OpenAIChatGenerator.run == original_chat_generator_run
