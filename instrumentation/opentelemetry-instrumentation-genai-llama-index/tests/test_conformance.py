# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("opentelemetry.test.weaver_live_check")
pytest.importorskip("opentelemetry.exporter.otlp.proto.grpc")

from opentelemetry.test.weaver_live_check import WeaverLiveCheck
from opentelemetry.test_util_genai.conformance import Scenario, run_conformance

from .conformance.agent import AgentScenario
from .conformance.retrieval import RetrievalScenario
from .conformance.workflow import WorkflowScenario


@pytest.mark.parametrize(
    "scenario",
    [AgentScenario(), WorkflowScenario(), RetrievalScenario()],
    ids=lambda scenario: type(scenario).__name__,
)
def test_conformance(
    scenario: Scenario, vcr: Any, weaver_live_check: WeaverLiveCheck
) -> None:
    run_conformance(scenario, vcr=vcr, weaver=weaver_live_check)
