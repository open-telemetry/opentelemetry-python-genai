# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from opentelemetry.instrumentation.genai.anthropic import AnthropicInstrumentor
from opentelemetry.instrumentation.genai.anthropic.version import __version__
from opentelemetry.test_util_genai.scope import TelemetryHandlerScopeTest


class TestInstrumentationScope(TelemetryHandlerScopeTest):
    instrumentor_class = AnthropicInstrumentor
    instrumentation_scope_name = "opentelemetry.instrumentation.genai.anthropic"
    instrumentation_scope_version = __version__
