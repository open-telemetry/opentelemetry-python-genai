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


@pytest.mark.parametrize("scenario", [AgentScenario()])
def test_conformance(
    scenario: Scenario, vcr: Any, weaver_live_check: WeaverLiveCheck
) -> None:
    run_conformance(scenario, vcr=vcr, weaver=weaver_live_check)
