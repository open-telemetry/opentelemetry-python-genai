# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from opentelemetry.instrumentation.genai.qwenpaw import QwenPawInstrumentor
from opentelemetry.instrumentation.genai.qwenpaw.version import __version__
from opentelemetry.test_util_genai.scope import TelemetryHandlerScopeTest


class TestInstrumentationScope(TelemetryHandlerScopeTest):
    instrumentor_class = QwenPawInstrumentor
    instrumentation_scope_name = "opentelemetry.instrumentation.genai.qwenpaw"
    instrumentation_scope_version = __version__
