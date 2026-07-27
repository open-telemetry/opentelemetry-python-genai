# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared ``ExpectedViolation`` declarations for gaps that recur across
several conformance scenarios in this package. See the package README's
"Known limitations" section for the full rationale behind each.
"""

from __future__ import annotations

from opentelemetry.test_util_genai.conformance import ExpectedViolation

# Haystack's OpenAIChatGenerator (_convert_chat_completion_to_chat_message)
# never copies the OpenAI response's `id` into the ChatMessage it builds, and
# `run()` returns no other place to find it -- the response id is genuinely
# unrecoverable from this instrumentation, not just unpopulated.
MISSING_RESPONSE_ID = ExpectedViolation(
    "genai_expected_attribute_missing", "gen_ai.response.id"
)

# Haystack's SDK-backed generators/embedders construct their underlying SDK
# client lazily (`self.client`/`self.async_client` start as None) via
# `warm_up()`, which `Pipeline.run()` calls automatically before running its
# components -- so server.address/port is already populated for
# Pipeline-driven calls. A component called *standalone* (not through a
# Pipeline) only gets it starting on the instance's second call, since
# nothing else triggers warm_up() first. Every standalone-call conformance
# scenario constructs a fresh instance and calls it exactly once, so this is
# unavoidable here without instrumentation code forcing early client
# construction (e.g. calling `component.warm_up()` ourselves) -- deliberately
# not done, since warm_up() also warms up any configured tools, which for
# some Tool/Toolset implementations (e.g. an MCP-backed Toolset) can mean
# arbitrary, instrumentation-inappropriate I/O.
MISSING_SERVER_ADDRESS = ExpectedViolation(
    "genai_expected_attribute_missing", "server.address"
)

# tool_call.id correlation would require hooking the private
# haystack.components.agents.tool_calling._make_context_bound_invoke -- see
# patch.py's Tool.invoke wrapper comment.
MISSING_TOOL_CALL_ID = ExpectedViolation(
    "genai_expected_attribute_missing", "gen_ai.tool.call.id"
)
