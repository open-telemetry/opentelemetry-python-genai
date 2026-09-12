# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from opentelemetry.instrumentation.genai.smolagents import SmolagentsInstrumentor
from opentelemetry.instrumentation.genai.smolagents.version import __version__
from opentelemetry.test_util_genai.scope import TelemetryHandlerScopeTest


class TestInstrumentationScope(TelemetryHandlerScopeTest):
    instrumentor_class = SmolagentsInstrumentor
    instrumentation_scope_name = (
        "opentelemetry.instrumentation.genai.smolagents"
    )
    instrumentation_scope_version = __version__
