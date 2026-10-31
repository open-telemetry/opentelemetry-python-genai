# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.genai.langchain.agent_context import (
    claim_graph,
    wrap_astream,
    wrap_stream,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("positional_config", [False, True])
@pytest.mark.parametrize(
    "metadata",
    [
        {"workflow_name": "runtime_workflow"},
        {"agent_name": "runtime_agent"},
        {"agent_type": "worker"},
        {"otel_agent_span": True},
    ],
)
async def test_graph_announcement_uses_invocation_metadata(
    async_mode, positional_config, metadata
):
    graph = SimpleNamespace(
        name="compiled_workflow",
        config={"metadata": {"workflow_name": "bound_workflow"}},
    )
    config = {"metadata": metadata}
    args = ({}, config) if positional_config else ({},)
    kwargs = {} if positional_config else {"config": config}
    announcements = []

    def stream(*args, **kwargs):
        announcements.append(claim_graph())
        yield 1

    async def astream(*args, **kwargs):
        announcements.append(claim_graph())
        yield 1

    if async_mode:
        assert [
            chunk async for chunk in wrap_astream(astream, graph, args, kwargs)
        ] == [1]
    else:
        assert list(wrap_stream(stream, graph, args, kwargs)) == [1]

    assert len(announcements) == 1
    announcement = announcements[0]
    assert announcement.metadata == {
        "workflow_name": "bound_workflow",
        **metadata,
    }
    assert announcement.is_agent == ("workflow_name" not in metadata)
    assert announcement.name == (
        metadata.get("agent_name")
        if announcement.is_agent
        else "compiled_workflow"
    )


def test_workflow_announcement_carries_compiled_name():
    graph = SimpleNamespace(name="compiled_workflow", config=None)
    announcements = []

    def stream():
        announcements.append(claim_graph())
        yield 1

    assert list(wrap_stream(stream, graph, (), {})) == [1]
    assert announcements[0].is_agent is False
    assert announcements[0].name == "compiled_workflow"


def test_graph_announcement_ignores_enclosing_task_metadata():
    graph = SimpleNamespace(
        name="child",
        config={"metadata": {"workflow_name": "child_override"}},
    )
    inherited_config = {
        "configurable": {"__pregel_task_id": "parent-task"},
        "metadata": {
            "agent_name": "parent",
            "workflow_name": "parent_override",
        },
    }
    announcements = []

    def stream(*args, **kwargs):
        announcements.append(claim_graph())
        yield 1

    assert list(
        wrap_stream(stream, graph, (), {"config": inherited_config})
    ) == [1]
    assert announcements[0].is_agent is False
    assert announcements[0].name == "child"
    assert announcements[0].metadata == {"workflow_name": "child_override"}
