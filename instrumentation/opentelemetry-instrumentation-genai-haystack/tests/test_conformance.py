# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Type

import pytest

from opentelemetry.test_util_genai.conformance import Scenario

from .conformance.invoke_workflow import WorkflowScenario

SCENARIOS: list[Type[Scenario]] = [
    WorkflowScenario,
]

@pytest.mark.parametrize('scenario_cls', SCENARIOS, ids=lambda c: c.__name__)
def test_scenario(
    scenario_cls: Type[Scenario],
    tracer_provider,
    meter_provider,
    logger_provider,
    vcr,
) -> None:
    scenario_cls().run(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
        vcr=vcr,
    )
