# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("opentelemetry.test.weaver_live_check")

from opentelemetry.test.weaver_live_check import WeaverLiveCheck  # noqa: E402
from opentelemetry.test_util_genai.conformance import (  # noqa: E402
    Scenario,
    run_conformance,
)

from .conformance.embedding import EmbeddingScenario  # noqa: E402
from .conformance.inference import InferenceScenario  # noqa: E402
from .conformance.invoke_agent import InvokeAgentScenario  # noqa: E402
from .conformance.invoke_workflow import WorkflowScenario  # noqa: E402
from .conformance.retrieval import RetrievalScenario  # noqa: E402
from .conformance.tool_calling import ToolCallingScenario  # noqa: E402


@pytest.mark.parametrize(
    "scenario",
    [
        InferenceScenario(),
        EmbeddingScenario(),
        RetrievalScenario(),
        ToolCallingScenario(),
        WorkflowScenario(),
        InvokeAgentScenario(),
    ],
    ids=lambda s: type(s).__name__,
)
def test_conformance(
    scenario: Scenario, vcr, weaver_live_check: WeaverLiveCheck
) -> None:
    run_conformance(scenario, vcr=vcr, weaver=weaver_live_check)
