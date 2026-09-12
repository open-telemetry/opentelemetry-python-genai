# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from opentelemetry.instrumentation.genai.crewai import CrewAIInstrumentor
from opentelemetry.instrumentation.genai.crewai.version import __version__
from opentelemetry.test_util_genai.scope import TelemetryHandlerScopeTest


class TestInstrumentationScope(TelemetryHandlerScopeTest):
    instrumentor_class = CrewAIInstrumentor
    instrumentation_scope_name = "opentelemetry.instrumentation.genai.crewai"
    instrumentation_scope_version = __version__
