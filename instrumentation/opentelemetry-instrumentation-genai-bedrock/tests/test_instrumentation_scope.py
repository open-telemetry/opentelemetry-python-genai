# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from opentelemetry.instrumentation.genai.bedrock import BedrockInstrumentor
from opentelemetry.instrumentation.genai.bedrock.version import __version__
from opentelemetry.test_util_genai.scope import TelemetryHandlerScopeTest


class TestInstrumentationScope(TelemetryHandlerScopeTest):
    instrumentor_class = BedrockInstrumentor
    instrumentation_scope_name = "opentelemetry.instrumentation.genai.bedrock"
    instrumentation_scope_version = __version__
