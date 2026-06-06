# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-scenario conformance tests for openai-agents."""

from __future__ import annotations

from typing import Any

import pytest

# Skip collection when weaver_live_check or OTLP exporters aren't installed
# (non-conformance envs).
pytest.importorskip("opentelemetry.test.weaver_live_check")
pytest.importorskip("opentelemetry.exporter.otlp.proto.grpc")

from opentelemetry.test.weaver_live_check import WeaverLiveCheck  # noqa: E402
from opentelemetry.test_util_genai.conformance import (  # noqa: E402
    Scenario,
    run_conformance,
)

from .conformance.orchestration import OrchestrationScenario


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param(
            OrchestrationScenario(),
            marks=pytest.mark.skip(
                reason=(
                    "openai-agents instrumentation has multiple semconv gaps "
                    "surfaced by this scenario; tracked in "
                    "https://github.com/open-telemetry/opentelemetry-python-genai/issues/86"
                )
            ),
        ),
    ],
    ids=lambda s: type(s).__name__,
)
def test_conformance(
    scenario: Scenario, vcr: Any, weaver_live_check: WeaverLiveCheck
) -> None:
    run_conformance(scenario, vcr=vcr, weaver=weaver_live_check)
