# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Sanity tests for the ``HaystackInstrumentor`` itself: entry point, and
that instrument/uninstrument actually wrap and unwrap the methods this
package documents."""

from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.core.component.component import _Component

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
    original_component_register = _Component._component
    original_chat_generator_run = OpenAIChatGenerator.run

    instrumentor = HaystackInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    try:
        assert _Component._component is not original_component_register
        assert OpenAIChatGenerator.run is not original_chat_generator_run
    finally:
        instrumentor.uninstrument()

    assert _Component._component == original_component_register
    assert OpenAIChatGenerator.run == original_chat_generator_run
