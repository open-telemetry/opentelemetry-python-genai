# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: basic workflow run for Agno."""

from agno.workflow.workflow import Workflow

workflow = Workflow(
    name="test-conformance-workflow",
    steps=[],
    session_id="session-workflow",
)
workflow.run("hello workflow conformance")
