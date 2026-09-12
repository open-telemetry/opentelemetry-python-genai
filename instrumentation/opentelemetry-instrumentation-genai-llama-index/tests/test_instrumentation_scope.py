# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from opentelemetry.instrumentation.genai.llama_index import LlamaIndexInstrumentor
from opentelemetry.instrumentation.genai.llama_index.version import __version__
from opentelemetry.test_util_genai.scope import TelemetryHandlerScopeTest


class TestInstrumentationScope(TelemetryHandlerScopeTest):
    instrumentor_class = LlamaIndexInstrumentor
    instrumentation_scope_name = (
        "opentelemetry.instrumentation.genai.llama_index"
    )
    instrumentation_scope_version = __version__
