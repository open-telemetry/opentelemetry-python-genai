# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-scenario conformance tests for qwenpaw."""

from __future__ import annotations

from importlib.metadata import version
from typing import Any

import pytest

# Skip collection when weaver_live_check or OTLP exporters aren't installed
# (non-conformance envs).
pytest.importorskip("opentelemetry.test.weaver_live_check")
pytest.importorskip("opentelemetry.exporter.otlp.proto.grpc")

from opentelemetry.test.weaver_live_check import WeaverLiveCheck
from opentelemetry.test_util_genai.conformance import (
    Scenario,
    run_conformance,
)

from .conformance.invoke_agent import InvokeAgentScenario
from .conformance.runtime_error import RuntimeErrorScenario
from .conformance.runtime_messages import RuntimeMessagesScenario


@pytest.mark.parametrize(
    "scenario",
    [
        InvokeAgentScenario(),
        RuntimeErrorScenario(),
        *(
            [RuntimeMessagesScenario()]
            if int(version("qwenpaw").split(".")[0]) >= 2
            else []
        ),
    ],
    ids=lambda s: type(s).__name__,
)
def test_conformance(
    scenario: Scenario, vcr: Any, weaver_live_check: WeaverLiveCheck
) -> None:
    run_conformance(scenario, vcr=vcr, weaver=weaver_live_check)
