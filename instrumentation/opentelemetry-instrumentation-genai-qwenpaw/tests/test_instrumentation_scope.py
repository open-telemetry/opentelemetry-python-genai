# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from opentelemetry.instrumentation.genai.qwenpaw import QwenPawInstrumentor
from opentelemetry.instrumentation.genai.qwenpaw.version import __version__
from opentelemetry.test_util_genai.scope import TelemetryHandlerScopeTest


@pytest.mark.usefixtures("runner_module")
class TestInstrumentationScope(TelemetryHandlerScopeTest):
    instrumentor_class = QwenPawInstrumentor
    instrumentation_scope_name = "opentelemetry.instrumentation.genai.qwenpaw"
    instrumentation_scope_version = __version__
