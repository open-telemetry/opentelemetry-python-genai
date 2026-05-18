# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-scenario conformance tests for langchain."""

from __future__ import annotations

from typing import Any

import pytest

# Skip collection when weaver_live_check isn't installed (non-conformance envs).
pytest.importorskip("opentelemetry.test.weaver_live_check")

from opentelemetry.test.weaver_live_check import WeaverLiveCheck  # noqa: E402
from opentelemetry.test_util_genai.conformance import (  # noqa: E402
    Scenario,
    run_conformance,
)

from .conformance.inference import InferenceScenario

pytestmark = pytest.mark.conformance


@pytest.mark.parametrize(
    "scenario",
    [
        InferenceScenario(),
    ],
    ids=lambda s: type(s).__name__,
)
def test_conformance(
    scenario: Scenario, vcr: Any, weaver_live_check: WeaverLiveCheck
) -> None:
    run_conformance(scenario, vcr=vcr, weaver=weaver_live_check)
