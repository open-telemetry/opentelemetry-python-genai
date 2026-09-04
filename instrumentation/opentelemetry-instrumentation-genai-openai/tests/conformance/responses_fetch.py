# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: OpenAI Responses API fetch by id."""

from openai import OpenAI

RESPONSE_ID = "resp_0f4faba17dcd0f1e0069e2f3e4907881909179832ba1237025"

OpenAI().responses.retrieve(RESPONSE_ID)
