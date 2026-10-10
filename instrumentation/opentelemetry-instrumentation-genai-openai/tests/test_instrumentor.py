# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import openai
import wrapt

from opentelemetry.instrumentation.genai.openai import OpenAIInstrumentor
from opentelemetry.instrumentation.genai.openai.package import _instruments


def test_instrumentation_dependencies_exposed() -> None:
    instrumentor = OpenAIInstrumentor()
    assert instrumentor.instrumentation_dependencies() == _instruments


def test_uninstrument_after_reinstantiation() -> None:
    instrumentor = OpenAIInstrumentor()
    instrumentor.instrument()
    try:
        assert isinstance(
            openai.resources.chat.completions.Completions.create,
            wrapt.BoundFunctionWrapper,
        )
        if hasattr(openai.resources.chat.completions.Completions, "parse"):
            assert isinstance(
                openai.resources.chat.completions.Completions.parse,
                wrapt.BoundFunctionWrapper,
            )
    finally:
        # Re-instantiate the singleton: __init__ should not reset _parse_supported
        OpenAIInstrumentor().uninstrument()

    assert not isinstance(
        openai.resources.chat.completions.Completions.create,
        wrapt.BoundFunctionWrapper,
    )
    if hasattr(openai.resources.chat.completions.Completions, "parse"):
        assert not isinstance(
            openai.resources.chat.completions.Completions.parse,
            wrapt.BoundFunctionWrapper,
        )
