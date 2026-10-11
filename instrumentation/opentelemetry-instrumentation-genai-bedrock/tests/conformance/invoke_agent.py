# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: bedrock invoke_agent."""

from __future__ import annotations

import boto3

client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")
response = client.invoke_agent(
    agentId="AGENT1234567",
    agentAliasId="ALIAS1234567",
    sessionId="conformance-session-1",
    inputText="What is the capital of France?",
)
for _ in response["completion"]:
    pass
