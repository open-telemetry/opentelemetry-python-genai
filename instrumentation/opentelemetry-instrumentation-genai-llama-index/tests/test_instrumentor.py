# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from opentelemetry.instrumentation.genai.llama_index import (
    LlamaIndexInstrumentor,
)
from opentelemetry.instrumentation.genai.llama_index.package import (
    _instruments,
)


def test_instrumentation_dependencies_exposed() -> None:
    instrumentor = LlamaIndexInstrumentor()
    assert instrumentor.instrumentation_dependencies() == _instruments


def test_instrument_initializes_handler() -> None:
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.instrument()
    try:
        assert instrumentor._handler is not None
    finally:
        instrumentor.uninstrument()


def test_double_instrument_is_noop() -> None:
    instrumentor = LlamaIndexInstrumentor()
    try:
        instrumentor.instrument()
        first = instrumentor._handler
        instrumentor.instrument()
        assert instrumentor._handler is first
    finally:
        instrumentor.uninstrument()


def test_double_uninstrument_is_noop() -> None:
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.instrument()
    instrumentor.uninstrument()
    instrumentor.uninstrument()
