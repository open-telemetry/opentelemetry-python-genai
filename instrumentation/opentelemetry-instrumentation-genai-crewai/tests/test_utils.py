# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from opentelemetry.instrumentation.genai.crewai.utils import (
    get_crew_name,
    get_tool_name,
    serialize_bound_arguments,
)


class _Crew:
    name = "Research Crew"


class _Tool:
    name = "search"


def _sample(agent, *, topic="otel"):
    return topic


class _Agent:
    role = "researcher"
    goal = "find references"
    backstory = "knows where to look"
    verbose = False
    allow_delegation = False
    max_iter = 3
    max_rpm = None


def test_get_crew_name_prefers_explicit_name():
    assert get_crew_name(_Crew()) == "Research Crew"


def test_get_tool_name_prefers_tool_name():
    assert get_tool_name(_Tool()) == "search"


def test_serialize_bound_arguments_uses_agent_getter():
    value = serialize_bound_arguments(_sample, (_Agent(),), {"topic": "genai"})

    assert value["agent"]["role"] == "researcher"
    assert value["topic"] == "genai"
