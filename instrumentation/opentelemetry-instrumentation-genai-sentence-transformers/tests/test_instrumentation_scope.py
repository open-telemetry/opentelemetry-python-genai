# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from opentelemetry.instrumentation.genai.sentence_transformers import (
    SentenceTransformersInstrumentor,
)
from opentelemetry.instrumentation.genai.sentence_transformers.version import (
    __version__,
)
from opentelemetry.test_util_genai.scope import TelemetryHandlerScopeTest


class TestInstrumentationScope(TelemetryHandlerScopeTest):
    instrumentor_class = SentenceTransformersInstrumentor
    instrumentation_scope_name = (
        "opentelemetry.instrumentation.genai.sentence_transformers"
    )
    instrumentation_scope_version = __version__
