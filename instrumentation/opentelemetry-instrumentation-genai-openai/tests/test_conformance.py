# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-scenario conformance tests for openai-v2.

Each scenario runs the instrumentation against a recorded API call and
validates the emitted telemetry against the GenAI semantic conventions via
Weaver live-check.
"""

from __future__ import annotations

from typing import Any

import pytest

# Skip collection when weaver_live_check isn't installed (non-conformance envs).
pytest.importorskip("opentelemetry.test.weaver_live_check")

from opentelemetry.test.weaver_live_check import WeaverLiveCheck
from opentelemetry.test_util_genai.conformance import (
    Scenario,
    run_conformance,
)

from .conformance.embedding import EmbeddingScenario
from .conformance.inference import InferenceScenario
from .conformance.inference_streaming import InferenceStreamingScenario
from .conformance.responses_conversation import ResponsesConversationScenario
from .conformance.responses_fetch import ResponsesFetchScenario
from .conformance.responses_stream import ResponsesStreamScenario
from .conformance.responses_streaming import ResponsesStreamingScenario
from .conformance.tool_calling import ToolCallingScenario


@pytest.mark.parametrize(
    "scenario",
    [
        InferenceScenario(),
        InferenceStreamingScenario(),
        EmbeddingScenario(),
        ToolCallingScenario(),
        ResponsesConversationScenario(),
        ResponsesFetchScenario(),
        ResponsesStreamScenario(),
        ResponsesStreamingScenario(),
    ],
    ids=lambda s: type(s).__name__,
)
def test_conformance(
    scenario: Scenario, vcr: Any, weaver_live_check: WeaverLiveCheck
) -> None:
    run_conformance(scenario, vcr=vcr, weaver=weaver_live_check)
