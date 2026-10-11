# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: streaming workflow run for Agno."""

from __future__ import annotations

from typing import Any

from agno.workflow.workflow import Workflow


def step1(step_input: Any) -> str:
    return "step1 stream output"


workflow = Workflow(
    name="test-conformance-workflow-streaming",
    steps=[step1],
    session_id="session-workflow-streaming",
)
stream = workflow.run("hello workflow streaming conformance", stream=True)
for _ in stream:
    pass
