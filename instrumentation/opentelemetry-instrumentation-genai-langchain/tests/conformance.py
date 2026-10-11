# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("genai_conformance")


from opentelemetry.conformance import ConformanceSession, scenarios
from opentelemetry.test_util_genai.common_conformance import (
    assert_message_part,
    messages,
)

pytest_plugins = ["opentelemetry.test_util_genai.common_conformance"]

DIRECTORY = Path(__file__).parent / "conformance"


@pytest.mark.parametrize("scenario_name", scenarios(DIRECTORY))
def test_scenario(
    scenario_name: str, common_session: ConformanceSession
) -> None:
    result = common_session.run(scenario_name)
    assert not result.failures, "\n".join(result.failures)
    assert not result.violations, "\n".join(result.violations)
    multimodal_parts = {
        "multimodal": ("blob", "image"),
        "multimodal_image_block": ("blob", "image"),
        "multimodal_responses": ("blob", "image"),
        "multimodal_anthropic": ("blob", "image"),
        "multimodal_anthropic_file": ("file", "image"),
    }
    if scenario_name in multimodal_parts:
        part_type, modality = multimodal_parts[scenario_name]
        assert_message_part(
            result, "gen_ai.input.messages", part_type, modality
        )
    if scenario_name == "tool_calling":
        assert_message_part(result, "gen_ai.output.messages", "tool_call")
        assert_message_part(
            result, "gen_ai.input.messages", "tool_call_response"
        )
    if scenario_name in {"inference", "streaming", "tool_calling"}:
        assert any(
            message["role"] == "system"
            for message in messages(result, "gen_ai.input.messages")
        )
