# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-scenario conformance tests for anthropic."""

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
from .conformance.tool_calling import ToolCallingScenario

_LEGACY_SYSTEM_SKIP = pytest.mark.skip(
    reason="anthropic emits legacy gen_ai.system in experimental mode"
)


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param(InferenceScenario(), marks=_LEGACY_SYSTEM_SKIP),
        pytest.param(ToolCallingScenario(), marks=_LEGACY_SYSTEM_SKIP),
    ],
    ids=lambda s: type(s).__name__,
)
def test_conformance(
    scenario: Scenario, vcr: Any, weaver_live_check: WeaverLiveCheck
) -> None:
    run_conformance(scenario, vcr=vcr, weaver=weaver_live_check)
