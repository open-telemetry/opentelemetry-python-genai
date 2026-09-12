# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from opentelemetry.instrumentation.genai.qwen_agent import QwenAgentInstrumentor
from opentelemetry.instrumentation.genai.qwen_agent.version import __version__
from opentelemetry.test_util_genai.scope import TelemetryHandlerScopeTest


class TestInstrumentationScope(TelemetryHandlerScopeTest):
    instrumentor_class = QwenAgentInstrumentor
    instrumentation_scope_name = (
        "opentelemetry.instrumentation.genai.qwen_agent"
    )
    instrumentation_scope_version = __version__
