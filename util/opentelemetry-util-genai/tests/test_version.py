# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import textwrap
import unittest

from opentelemetry.util.genai.version import __version__


class TestVersion(unittest.TestCase):
    def test_version_exists(self):
        """Test that version is defined and is a string."""
        self.assertIsInstance(__version__, str)
        self.assertTrue(len(__version__) > 0)

    def test_version_format(self):
        """Test that version follows expected format."""
        # Should be in format like "0.1b0.dev" or similar
        self.assertRegex(__version__, r"^\d+\.\d+.*")

    def test_core_apis_without_optional_dependencies(self):
        """Core util-genai APIs work in a fresh process without wrapt or opentelemetry-instrumentation."""
        script = textwrap.dedent(
            """
            import sys
            sys.modules["wrapt"] = None
            sys.modules["opentelemetry.instrumentation"] = None

            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
                InMemorySpanExporter,
            )
            from opentelemetry.util.genai import (
                completion_hook,
                environment_variables,
                handler,
                invocation,
                types,
                utils,
                version,
            )

            exporter = InMemorySpanExporter()
            tracer_provider = TracerProvider()
            tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
            telemetry_handler = handler.TelemetryHandler(
                tracer_provider=tracer_provider
            )
            with telemetry_handler.inference(
                "openai", request_model="gpt-4o"
            ) as inf:
                inf.input_messages = [
                    types.InputMessage(
                        role="user",
                        parts=[types.TextPart(content="hello")],
                    )
                ]
                inf.response_model_name = "gpt-4o"
                inf.finish_reasons = ["stop"]

            spans = exporter.get_finished_spans()
            assert len(spans) == 1, f"expected 1 span, got {len(spans)}"
            assert "wrapt" not in sys.modules or sys.modules["wrapt"] is None
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
