# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from opentelemetry.instrumentation.genai.dspy import DSPyInstrumentor
from opentelemetry.instrumentation.genai.dspy.version import __version__
from opentelemetry.test_util_genai.scope import TelemetryHandlerScopeTest


class TestInstrumentationScope(TelemetryHandlerScopeTest):
    instrumentor_class = DSPyInstrumentor
    instrumentation_scope_name = "opentelemetry.instrumentation.genai.dspy"
    instrumentation_scope_version = __version__
