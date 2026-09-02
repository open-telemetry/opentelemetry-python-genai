# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-scenario conformance tests for bedrock."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("opentelemetry.test.weaver_live_check")
pytest.importorskip("opentelemetry.exporter.otlp.proto.grpc")

from opentelemetry.test.weaver_live_check import WeaverLiveCheck
from opentelemetry.test_util_genai.conformance import (
    Scenario,
    run_conformance,
)

from .conformance.inference import InferenceScenario
from .conformance.inference_streaming import InferenceStreamingScenario
from .conformance.tool_calling import ToolCallingScenario


@pytest.mark.parametrize(
    "scenario",
    [
        InferenceScenario(),
        InferenceStreamingScenario(),
        ToolCallingScenario(),
    ],
    ids=lambda s: type(s).__name__,
)
def test_conformance(
    scenario: Scenario, vcr: Any, weaver_live_check: WeaverLiveCheck
) -> None:
    run_conformance(scenario, vcr=vcr, weaver=weaver_live_check)
