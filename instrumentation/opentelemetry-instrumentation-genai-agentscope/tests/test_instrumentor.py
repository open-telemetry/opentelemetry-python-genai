# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AgentScope instrumentor lifecycle."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import agentscope.tracing._trace as agentscope_tracing_trace
import wrapt

from opentelemetry.instrumentation.genai.agentscope import (
    AgentScopeInstrumentor,
)
from opentelemetry.sdk.trace import TracerProvider


class TestAgentScopeInstrumentor(unittest.TestCase):
    def setUp(self):
        self.tracer_provider = TracerProvider()
        self.instrumentor = AgentScopeInstrumentor()

    def tearDown(self):
        try:
            self.instrumentor.uninstrument()
        except Exception:
            pass

    def test_init(self):
        self.assertIsNotNone(self.instrumentor)
        self.assertIsNone(self.instrumentor._handler)

    def test_instrumentation_dependencies(self):
        dependencies = self.instrumentor.instrumentation_dependencies()
        self.assertIsInstance(dependencies, tuple)
        self.assertTrue(any("agentscope" in dep for dep in dependencies))

    def test_instrument_creates_handler(self):
        self.instrumentor.instrument(tracer_provider=self.tracer_provider)
        self.assertIsNotNone(self.instrumentor._handler)

    def test_uninstrument_clears_handler(self):
        self.instrumentor.instrument(tracer_provider=self.tracer_provider)
        self.instrumentor.uninstrument()
        self.assertIsNone(self.instrumentor._handler)

    def test_uninstrument_without_instrument(self):
        try:
            self.instrumentor.uninstrument()
        except Exception as e:  # noqa: BLE001
            self.fail(f"uninstrument() raised unexpectedly: {e}")

    def test_uninstrument_exception_handling(self):
        self.instrumentor.instrument(tracer_provider=self.tracer_provider)
        with patch(
            "builtins.__import__", side_effect=ImportError("no module")
        ):
            try:
                self.instrumentor.uninstrument()
            except Exception as e:  # noqa: BLE001
                self.fail(f"uninstrument() raised unexpectedly: {e}")

    def test_instrument_multiple_times(self):
        self.instrumentor.instrument(tracer_provider=self.tracer_provider)
        self.instrumentor.instrument(tracer_provider=self.tracer_provider)
        self.assertIsNotNone(self.instrumentor._handler)

    def test_check_tracing_enabled_patch(self):
        original_func = getattr(
            agentscope_tracing_trace, "_check_tracing_enabled", None
        )
        if original_func is None:
            self.skipTest("_check_tracing_enabled not present in agentscope")

        # wrap_function_wrapper installs a wrapt.FunctionWrapper. Check for that
        # explicitly rather than ObjectProxy: wrapt 2.x no longer makes
        # FunctionWrapper a subclass of ObjectProxy.
        self.assertFalse(isinstance(original_func, wrapt.FunctionWrapper))

        self.instrumentor.instrument(tracer_provider=self.tracer_provider)

        patched_func = agentscope_tracing_trace._check_tracing_enabled
        self.assertTrue(isinstance(patched_func, wrapt.FunctionWrapper))
        self.assertFalse(patched_func())

        self.instrumentor.uninstrument()

        restored_func = agentscope_tracing_trace._check_tracing_enabled
        self.assertFalse(isinstance(restored_func, wrapt.FunctionWrapper))


if __name__ == "__main__":
    unittest.main()
