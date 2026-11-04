# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
import textwrap

from llama_index.core.instrumentation import get_dispatcher

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
        assert instrumentor._span_handler in get_dispatcher().span_handlers
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


def test_uninstrument_removes_span_handler() -> None:
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.instrument()
    span_handler = instrumentor._span_handler
    instrumentor.uninstrument()
    assert span_handler not in get_dispatcher().span_handlers


def test_uninstrument_after_reinstantiation() -> None:
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.instrument()
    span_handler = instrumentor._span_handler

    # Re-instantiate the singleton: __init__ should not reset _span_handler
    LlamaIndexInstrumentor().uninstrument()
    assert span_handler not in get_dispatcher().span_handlers


# These tests run in a fresh interpreter: llama_index.core is already imported
# in this one, which makes the post-import hook fire immediately.
_SETUP = """
from opentelemetry.instrumentation.genai.llama_index import (
    LlamaIndexInstrumentor,
)

instrumentor = LlamaIndexInstrumentor()
"""


def _run_in_fresh_interpreter(test_body: str) -> None:
    script = _SETUP + textwrap.dedent(test_body)
    subprocess.run([sys.executable, "-c", script], check=True)


def test_instrument_does_not_import_llama_index() -> None:
    # Importing llama_index.core during auto-instrumentation startup crashes
    # pydantic-core on Python 3.15, so instrument() must defer the import.
    _run_in_fresh_interpreter(
        """
        import sys

        instrumentor.instrument()

        assert "llama_index.core" not in sys.modules
        """
    )


def test_uninstrument_before_import_does_not_attach() -> None:
    # uninstrument must remove the registered hook().
    _run_in_fresh_interpreter(
        """
        instrumentor.instrument()
        instrumentor.uninstrument()

        from llama_index.core.instrumentation import get_dispatcher

        handlers = [type(h).__name__ for h in get_dispatcher().span_handlers]
        assert handlers == ["NullSpanHandler"]
        assert instrumentor._span_handler is None
        """
    )


def test_reinstrument_before_import_attaches_once() -> None:
    # Each instrument() registers a hook, so later calls must not register
    # a duplicate hook.
    _run_in_fresh_interpreter(
        """
        instrumentor.instrument()
        instrumentor.uninstrument()
        instrumentor.instrument()

        from llama_index.core.instrumentation import get_dispatcher

        handlers = [type(h).__name__ for h in get_dispatcher().span_handlers]
        assert handlers == ["NullSpanHandler", "LlamaIndexSpanHandler"]
        """
    )