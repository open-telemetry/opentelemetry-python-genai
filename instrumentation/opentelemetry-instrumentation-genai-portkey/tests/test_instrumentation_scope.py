# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from opentelemetry.instrumentation.genai.portkey import PortkeyInstrumentor
from opentelemetry.instrumentation.genai.portkey.version import __version__
from opentelemetry.test_util_genai.scope import TelemetryHandlerScopeTest


class TestInstrumentationScope(TelemetryHandlerScopeTest):
    instrumentor_class = PortkeyInstrumentor
    instrumentation_scope_name = "opentelemetry.instrumentation.genai.portkey"
    instrumentation_scope_version = __version__
