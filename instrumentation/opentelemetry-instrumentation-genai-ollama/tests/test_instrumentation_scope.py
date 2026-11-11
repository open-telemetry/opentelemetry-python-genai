# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from opentelemetry.instrumentation.genai.ollama import OllamaInstrumentor
from opentelemetry.instrumentation.genai.ollama.version import __version__
from opentelemetry.test_util_genai.scope import TelemetryHandlerScopeTest


class TestInstrumentationScope(TelemetryHandlerScopeTest):
    instrumentor_class = OllamaInstrumentor
    instrumentation_scope_name = "opentelemetry.instrumentation.genai.ollama"
    instrumentation_scope_version = __version__
