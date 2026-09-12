# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from opentelemetry.instrumentation.genai.agno import AgnoInstrumentor
from opentelemetry.instrumentation.genai.agno.version import __version__
from opentelemetry.test_util_genai.scope import TelemetryHandlerScopeTest


class TestInstrumentationScope(TelemetryHandlerScopeTest):
    instrumentor_class = AgnoInstrumentor
    instrumentation_scope_name = "opentelemetry.instrumentation.genai.agno"
    instrumentation_scope_version = __version__
