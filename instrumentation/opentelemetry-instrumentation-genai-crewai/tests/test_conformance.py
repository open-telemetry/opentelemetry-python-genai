# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry.test.weaver_live_check")
pytest.importorskip("opentelemetry.exporter.otlp.proto.grpc")

from opentelemetry.test.weaver_live_check import WeaverLiveCheck
from opentelemetry.test_util_genai.conformance import (
    Scenario,
    run_conformance,
)

from .conformance.inference import InferenceScenario


@pytest.mark.parametrize("scenario", [InferenceScenario()])
def test_conformance(
    scenario: Scenario,
    weaver_live_check: WeaverLiveCheck,
) -> None:
    run_conformance(scenario, vcr=None, weaver=weaver_live_check)
