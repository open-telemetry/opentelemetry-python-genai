# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

pytest.importorskip("genai_conformance")

from opentelemetry.conformance import ConformanceSession, scenarios

pytest_plugins = ["opentelemetry.test_util_genai.common_conformance"]

DIRECTORY = Path(__file__).parent / "conformance"


@pytest.mark.parametrize("scenario_name", scenarios(DIRECTORY))
def test_scenario(
    scenario_name: str, common_session: ConformanceSession
) -> None:
    result = common_session.run(scenario_name)
    assert not result.failures, "\n".join(result.failures)
    assert not result.violations, "\n".join(result.violations)
